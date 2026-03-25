#!/usr/bin/env python3
"""
Hyperparameter Tuning with Optuna (d3rlpy)
==========================================
Searches over training hyperparameters using Optuna with ASHA pruning.
Each trial runs a shortened d3rlpy training session and reports mean
episode reward. Best params can be loaded by train.py --params-from.

The study is stored in SQLite for crash-resistant resumability.

Usage
-----
    # Run 50 trials:
    python training/tune.py

    # Override trials or step budget:
    python training/tune.py --n-trials 20 --steps-per-trial 200000

    # Show best trial:
    python training/tune.py --show-best

    # Resume from existing study:
    python training/tune.py --storage sqlite:///optuna_baseline.db

Requirements
------------
    pip install optuna d3rlpy gymnasium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))


# ── study settings ───────────────────────────────────────────────────────────

STUDY_NAME = 'baseline-hparam-search'
STORAGE_PATH = 'sqlite:///optuna_baseline.db'


# ── objective ────────────────────────────────────────────────────────────────

def objective(trial, steps_per_trial: int, use_wandb: bool) -> float:
    """
    Single Optuna trial: sample hyperparams, run shortened training,
    return mean episode reward.
    """
    import numpy as np
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

    env = BaselineGymEnv(t_window=t_window)
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100_000, env=env)
    explorer = NormalNoise(std=explore_noise)

    if use_wandb:
        logger_adapter = WanDBAdapterFactory(project='rlbot-baseline-tuning')
    else:
        logger_adapter = FileAdapterFactory(
            root_dir=f'models/tune/trial_{trial.number}'
        )

    # Track rewards for pruning
    episode_rewards = []
    pruning_interval = 50_000  # check every 50k steps

    def callback(algo_obj, epoch, total_step):
        import optuna
        # Report at pruning intervals
        if total_step > 0 and total_step % pruning_interval < 1000:
            if episode_rewards:
                mean_reward = float(np.mean(episode_rewards[-100:]))
                trial.report(mean_reward, step=total_step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    try:
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
        # Check if it's an Optuna pruning signal
        try:
            import optuna
            if isinstance(e, optuna.exceptions.TrialPruned):
                raise
        except ImportError:
            pass
        print(f'Trial {trial.number} failed: {e}', file=sys.stderr)
        return float('-inf')
    finally:
        env.close()

    # Return mean reward from last 100 episodes
    if episode_rewards:
        return float(np.mean(episode_rewards[-100:]))
    return 0.0


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
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--show-best', action='store_true')
    parser.add_argument('--study-name', default=STUDY_NAME)
    parser.add_argument('--storage', default=STORAGE_PATH)
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
            print(f'\nUse with: python training/train.py --params-from {args.storage.replace("sqlite:///", "")}')
        return

    n_existing = len([t for t in study.trials if t.state.is_finished()])
    print(f'Study "{args.study_name}" — {n_existing} completed trials.')
    print(f'Running {args.n_trials} more ({args.steps_per_trial:,} steps each)...')

    study.optimize(
        lambda trial: objective(trial, args.steps_per_trial, not args.no_wandb),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print('\n=== Tuning complete ===')
    best = study.best_trial
    print(f'Best trial #{best.number}  reward={best.value:.4f}')
    for k, v in best.params.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
