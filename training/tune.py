#!/usr/bin/env python3
"""
Hyperparameter Tuning with Optuna (d3rlpy)
==========================================
Searches over training hyperparameters using Optuna with ASHA pruning.
Each trial runs a shortened d3rlpy training session and reports mean
episode reward. Best params can be loaded by train.py --params-from.

Usage
-----
    # Run 50 trials:
    python training/tune.py

    # Show best trial:
    python training/tune.py --show-best

    # Auto-launch 10-seed baseline after tuning:
    python training/tune.py --auto-seeds

    # Override trials or step budget:
    python training/tune.py --n-trials 20 --steps-per-trial 200000

Requirements
------------
    pip install optuna d3rlpy gymnasium
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))


# ── study settings ───────────────────────────────────────────────────────────

STUDY_NAME = 'baseline-hparam-search'
STORAGE_PATH = 'sqlite:///optuna_baseline.db'


# ── reward-tracking wrapper ──────────────────────────────────────────────────

class RewardTracker:
    """
    Wraps a gymnasium env to track episode returns.

    d3rlpy's fit_online doesn't expose episode rewards in its callback,
    so we track them here and read from the callback.
    """

    def __init__(self, env):
        self.env = env
        self.episode_returns: deque = deque(maxlen=200)
        self._current_return = 0.0

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        self._current_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_return += reward
        if terminated or truncated:
            self.episode_returns.append(self._current_return)
            self._current_return = 0.0
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def mean_return(self, last_n: int = 100) -> float:
        """Mean of last N episode returns."""
        if not self.episode_returns:
            return 0.0
        recent = list(self.episode_returns)[-last_n:]
        return float(np.mean(recent))


# ── objective ────────────────────────────────────────────────────────────────

def objective(trial, steps_per_trial: int, use_wandb: bool, num_envs: int = 1) -> float:
    """
    Single Optuna trial: sample hyperparams, run shortened training,
    return mean episode reward.
    """
    import d3rlpy
    from d3rlpy.algos.qlearning.explorers import NormalNoise
    from d3rlpy.logging import FileAdapterFactory, WanDBAdapterFactory

    from baseline_encoder_factory import TransformerEncoderFactory
    from gym_env import BaselineGymEnv

    # ── sample hyperparameters ───────────────────────────────────────────
    actor_lr = trial.suggest_float('actor_lr', 1e-5, 1e-3, log=True)
    critic_lr = trial.suggest_float('critic_lr', 1e-5, 1e-3, log=True)
    awac_lambda = trial.suggest_float('awac_lambda', 0.1, 5.0)
    tau = trial.suggest_float('tau', 0.001, 0.05, log=True)
    batch_size = trial.suggest_int('batch_size', 64, 512, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    explore_noise = trial.suggest_float('explore_noise', 0.01, 0.3)

    t_window = 8
    encoder_factory = TransformerEncoderFactory(t_window=t_window)

    algo = d3rlpy.algos.AWACConfig(
        batch_size=batch_size,
        gamma=gamma,
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        tau=tau,
        lam=awac_lambda,
    ).create(device='cpu')

    raw_env = BaselineGymEnv(t_window=t_window)
    env = RewardTracker(raw_env)

    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100_000, env=env)
    explorer = NormalNoise(std=explore_noise)

    if use_wandb:
        logger_adapter = WanDBAdapterFactory(project='rlbot-baseline-tuning')
    else:
        logger_adapter = FileAdapterFactory(
            root_dir=f'models/tune/trial_{trial.number}'
        )

    pruning_interval = 50_000  # check every 50k steps

    parallel = num_envs > 1

    # Shared reward tracker for both paths
    episode_returns: deque = env.episode_returns  # reuse RewardTracker's deque

    def _mean_return(last_n: int = 100) -> float:
        if not episode_returns:
            return 0.0
        recent = list(episode_returns)[-last_n:]
        return float(np.mean(recent))

    def callback(algo_obj, epoch, total_step):
        import optuna
        if total_step > 0 and total_step % pruning_interval < max(1000, num_envs):
            mean_r = _mean_return(100)
            if len(episode_returns) > 0:
                trial.report(mean_r, step=total_step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    envs = None
    _wandb_run = None
    try:
        if parallel:
            from train import (
                SubprocVecEnv, fit_online_parallel, TrainConfig as _TC,
            )

            # Initialize W&B for parallel path (sequential path gets it from d3rlpy)
            if use_wandb:
                try:
                    import wandb
                    _wandb_run = wandb.init(
                        project='rlbot-baseline-tuning',
                        name=f'tune_trial_{trial.number}',
                        group='optuna',
                        config={
                            'trial_number': trial.number,
                            'actor_lr': actor_lr,
                            'critic_lr': critic_lr,
                            'awac_lambda': awac_lambda,
                            'tau': tau,
                            'batch_size': batch_size,
                            'gamma': gamma,
                            'explore_noise': explore_noise,
                            'num_envs': num_envs,
                            'steps_per_trial': steps_per_trial,
                        },
                        reinit=True,
                    )
                except ImportError:
                    pass

            envs = SubprocVecEnv(num_envs=num_envs, t_window=t_window)
            # Build algo before parallel loop
            algo.build_with_env(raw_env)

            # Parallel path uses on_episode_complete to feed the shared deque
            def _on_ep_complete(ep_return: float):
                episode_returns.append(ep_return)

            # Build a minimal TrainConfig for fit_online_parallel
            par_config = _TC(
                total_steps=steps_per_trial,
                num_envs=num_envs,
                t_window=t_window,
                random_steps=5_000,
                update_interval=1,
                n_steps_per_epoch=10_000,
                batch_size=batch_size,
                explore_noise=explore_noise,
            )
            fit_online_parallel(
                algo=algo,
                config=par_config,
                envs=envs,
                buffer=buffer,
                explorer=explorer,
                callback=callback,
                on_episode_complete=_on_ep_complete,
            )
        else:
            algo.fit_online(
                env=env,
                buffer=buffer,
                explorer=explorer,
                n_steps=steps_per_trial,
                n_steps_per_epoch=10_000,
                random_steps=5_000,
                experiment_name=f'tune_trial_{trial.number}',
                logger_adapter=logger_adapter,
                show_progress=False,
                callback=callback,
            )
    except Exception as e:
        try:
            import optuna
            if isinstance(e, optuna.exceptions.TrialPruned):
                raise
        except ImportError:
            pass
        print(f'Trial {trial.number} failed: {e}', file=sys.stderr)
        return float('-inf')
    finally:
        if envs is not None:
            envs.close()
        if _wandb_run is not None:
            import wandb
            wandb.finish()
        raw_env.close()

    mean_reward = _mean_return(100)
    assert not np.isnan(mean_reward), \
        f"Trial {trial.number}: NaN reward — training diverged"
    return mean_reward


# ── auto-seed launcher ──────────────────────────────────────────────────────

def launch_seeds(study, n_seeds: int = 10, extra_args: list = None) -> list:
    """
    Launch baseline training with best params from Optuna study.

    Returns list of (seed, subprocess.Popen) tuples.
    """
    best = study.best_params
    Path('models/baseline').mkdir(parents=True, exist_ok=True)

    print(f'\n=== Launching {n_seeds} seeds with best params ===')
    for k, v in best.items():
        print(f'  {k}: {v}')
    print()

    procs = []
    for seed in range(n_seeds):
        cmd = [
            sys.executable, 'training/train.py',
            '--seed', str(seed),
            '--actor-lr', str(best.get('actor_lr', 3e-4)),
            '--critic-lr', str(best.get('critic_lr', 3e-4)),
            '--awac-lambda', str(best.get('awac_lambda', 1.0)),
            '--tau', str(best.get('tau', 0.005)),
            '--batch-size', str(int(best.get('batch_size', 256))),
            '--gamma', str(best.get('gamma', 0.99)),
            '--explore-noise', str(best.get('explore_noise', 0.1)),
        ]
        if extra_args:
            cmd.extend(extra_args)

        log_path = f'models/baseline/seed_{seed}.log'
        log_file = open(log_path, 'w')
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT
        )
        procs.append((seed, proc, log_file))
        print(f'  Seed {seed}: PID {proc.pid}, log: {log_path}')

    return procs


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import optuna
        from optuna.pruners import SuccessiveHalvingPruner
        from optuna.samplers import TPESampler
    except ImportError:
        print('Optuna required: pip install optuna', file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Baseline hyperparameter tuning.')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--steps-per-trial', type=int, default=500_000)
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Parallel RLGym-sim environments per trial (default: 1)')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--show-best', action='store_true')
    parser.add_argument('--study-name', default=STUDY_NAME)
    parser.add_argument('--storage', default=STORAGE_PATH)
    # Auto-seed launching
    parser.add_argument('--auto-seeds', action='store_true',
                        help='Auto-launch baseline seeds after tuning completes')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of seeds to launch (default: 10)')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for all seed processes to complete')
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        pruner=SuccessiveHalvingPruner(
            min_resource=50_000,
            reduction_factor=3,
        ),
        sampler=TPESampler(seed=42),
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    if args.show_best:
        if not study.trials:
            print('No completed trials yet.')
        else:
            best = study.best_trial
            print(f'\nBest trial #{best.number}')
            print(f'  Reward: {best.value:.4f}')
            print('  Hyperparameters:')
            for k, v in best.params.items():
                print(f'    {k}: {v}')
            print(f'\nUse with: python training/train.py '
                  f'--params-from {args.storage.replace("sqlite:///", "")}')
        return

    n_existing = len([t for t in study.trials if t.state.is_finished()])
    print(f'Study "{args.study_name}" — {n_existing} completed trials.')
    print(f'Running {args.n_trials} more ({args.steps_per_trial:,} steps each)...')

    study.optimize(
        lambda trial: objective(
            trial, args.steps_per_trial, not args.no_wandb, args.num_envs,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print('\n=== Tuning complete ===')
    best = study.best_trial
    print(f'Best trial #{best.number}  reward={best.value:.4f}')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    # ── auto-launch seeds ────────────────────────────────────────────────
    if args.auto_seeds:
        extra = []
        if args.no_wandb:
            extra.append('--no-wandb')
        procs = launch_seeds(study, n_seeds=args.n_seeds, extra_args=extra)

        if args.wait:
            print(f'\nWaiting for {len(procs)} seed processes...')
            for seed, proc, log_file in procs:
                proc.wait()
                log_file.close()
                status = 'OK' if proc.returncode == 0 else f'FAILED ({proc.returncode})'
                print(f'  Seed {seed}: {status}')
            print('All seeds complete.')
        else:
            print(f'\n{len(procs)} seeds launched in background.')
            print('Monitor with: tail -f models/baseline/seed_0.log')


if __name__ == '__main__':
    main()
