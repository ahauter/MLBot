#!/usr/bin/env python3
"""
Unified Training Script (d3rlpy)
================================
Single entry point for all RL training. Uses d3rlpy for algorithm
implementations (AWAC, SAC, TD3, CQL, etc.) with our custom transformer
encoder and self-play opponent management.

Usage
-----
    # Baseline training (AWAC, sparse reward, self-play):
    python training/train.py

    # Specific seed:
    python training/train.py --seed 3

    # Swap algorithm:
    python training/train.py --algo SAC

    # Use Optuna-tuned hyperparameters:
    python training/train.py --params-from optuna_baseline.db

    # Short test run:
    python training/train.py --total-steps 10000 --eval-interval 5000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

import d3rlpy
from d3rlpy.algos.qlearning.explorers import NormalNoise
from d3rlpy.logging import WanDBAdapterFactory, FileAdapterFactory

from baseline_encoder_factory import TransformerEncoderFactory
from gym_env import BaselineGymEnv
from self_play import OpponentPool
from encoder import N_TOKENS, TOKEN_FEATURES


# ── configuration ────────────────────────────────────────────────────────────

ALGO_MAP = {
    'AWAC': d3rlpy.algos.AWACConfig,
    'SAC': d3rlpy.algos.SACConfig,
    'TD3': d3rlpy.algos.TD3Config,
    'CQL': d3rlpy.algos.CQLConfig,
    'IQL': d3rlpy.algos.IQLConfig,
    'TD3PlusBC': d3rlpy.algos.TD3PlusBCConfig,
}


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # ── seed & identity ──────────────────────────────────────────────────────
    seed: int = 0
    algo: str = 'AWAC'

    # ── budget ───────────────────────────────────────────────────────────────
    total_steps: int = 50_000_000
    eval_interval: int = 200_000       # env steps between Psyonix evaluations
    snapshot_interval: int = 10_000    # env steps between self-play snapshots

    # ── architecture ─────────────────────────────────────────────────────────
    t_window: int = 8
    obs_dim: int = 8 * N_TOKENS * TOKEN_FEATURES  # 800

    # ── AWAC / algorithm hyperparameters ─────────────────────────────────────
    batch_size: int = 256
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    awac_lambda: float = 1.0          # AWAC advantage temperature
    n_critics: int = 2
    explore_noise: float = 0.1        # Gaussian exploration std

    # ── replay buffer ────────────────────────────────────────────────────────
    buffer_capacity: int = 1_000_000
    random_steps: int = 10_000        # random actions before training starts

    # ── d3rlpy fit_online settings ───────────────────────────────────────────
    update_interval: int = 1          # gradient updates per env step
    n_steps_per_epoch: int = 10_000   # steps per d3rlpy epoch (logging granularity)

    # ── self-play ────────────────────────────────────────────────────────────
    max_snapshots: int = 20

    # ── convergence ──────────────────────────────────────────────────────────
    rookie_target_wr: float = 0.60    # win rate target vs Psyonix Rookie
    consecutive_evals_required: int = 2

    # ── paths ────────────────────────────────────────────────────────────────
    model_dir: str = 'models/baseline'
    snapshot_dir: str = 'models/baseline/snapshots'

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_project: str = 'rlbot-baseline'
    wandb_tags: list = field(default_factory=lambda: ['baseline', 'no-intervention'])
    no_wandb: bool = False


# ── seed setup ───────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── d3rlpy algorithm builder ────────────────────────────────────────────────

def build_algo(config: TrainConfig) -> d3rlpy.algos.QLearningAlgoBase:
    """Build a d3rlpy algorithm from config."""
    encoder_factory = TransformerEncoderFactory(t_window=config.t_window)

    algo_cls = ALGO_MAP.get(config.algo)
    if algo_cls is None:
        raise ValueError(
            f'Unknown algorithm: {config.algo}. '
            f'Available: {list(ALGO_MAP.keys())}'
        )

    # Build kwargs common to all algorithms
    common = dict(
        batch_size=config.batch_size,
        gamma=config.gamma,
        actor_learning_rate=config.actor_lr,
        critic_learning_rate=config.critic_lr,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        tau=config.tau,
    )

    # Algorithm-specific params
    if config.algo == 'AWAC':
        common['lam'] = config.awac_lambda
        common['n_action_samples'] = 1
    if hasattr(algo_cls, '__dataclass_fields__') and 'n_critics' in algo_cls.__dataclass_fields__:
        common['n_critics'] = config.n_critics

    # Filter to only params the config class accepts
    valid_fields = {f.name for f in dataclasses.fields(algo_cls)}
    filtered = {k: v for k, v in common.items() if k in valid_fields}

    algo_config = algo_cls(**filtered)
    return algo_config.create(device=('cuda:0' if torch.cuda.is_available() else 'cpu'))


# ── callback ─────────────────────────────────────────────────────────────────

class TrainingCallback:
    """
    Callback for d3rlpy's fit_online. Handles self-play snapshots,
    evaluation, convergence detection, and W&B eval logging.
    """

    def __init__(self, config: TrainConfig, env: BaselineGymEnv, pool: OpponentPool):
        self.config = config
        self.env = env
        self.pool = pool
        self.start_time = time.time()
        self.consecutive_wins = 0
        self.converged = False
        self._last_snapshot_step = 0
        self._last_eval_step = 0
        self._wandb = None

        # Initialize W&B for eval/metadata logging (separate from d3rlpy's logger)
        if not config.no_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                pass

    def log_metadata(self):
        """Log run metadata to W&B."""
        if self._wandb is None or self._wandb.run is None:
            return
        self._wandb.run.config.update({
            'meta/algorithm': self.config.algo,
            'meta/seed': self.config.seed,
            'meta/reward_components': 1,
            'meta/intervention': 'none',
            'meta/reference_bot_primary': 'Psyonix_Rookie',
            'meta/observation_dim': self.config.obs_dim,
            'meta/step_budget': self.config.total_steps,
            'meta/t_window': self.config.t_window,
            'meta/d3rlpy_version': d3rlpy.__version__,
        })

    def __call__(self, algo, epoch: int, total_step: int) -> None:
        # ── self-play snapshot ───────────────────────────────────────────
        if total_step - self._last_snapshot_step >= self.config.snapshot_interval:
            self._save_snapshot(algo, total_step)
            self._last_snapshot_step = total_step

        # ── evaluation ───────────────────────────────────────────────────
        if total_step - self._last_eval_step >= self.config.eval_interval:
            self._run_eval(algo, total_step)
            self._last_eval_step = total_step

    def _save_snapshot(self, algo, total_step: int) -> None:
        """Save current policy as self-play opponent snapshot."""
        self.pool.save_snapshot(algo, total_step)

        # Update the env's opponent from the pool
        if self.pool.num_snapshots() > 0:
            opponent_algo = self.pool.sample_opponent()
            self.env.set_opponent(opponent_algo)

    def _run_eval(self, algo, total_step: int) -> None:
        """Run evaluation against Psyonix tiers."""
        wall_clock = time.time() - self.start_time

        # For now, log placeholder — real Psyonix eval requires live RLBot
        # TODO: integrate training/evaluate.py when RLBot is available
        print(f'\n[step {total_step:,}] Evaluation checkpoint '
              f'(wall clock: {wall_clock/3600:.1f}h)')

        eval_metrics = {
            'eval/steps': total_step,
            'eval/wall_clock_seconds': int(wall_clock),
        }

        # Try to run evaluation if evaluate module is available
        try:
            from evaluate import run_evaluation
            model_dir = Path(self.config.model_dir) / 'eval_temp'
            model_dir.mkdir(parents=True, exist_ok=True)
            algo.save(str(model_dir / 'd3rlpy_model'))

            win_rates = run_evaluation(str(model_dir))
            eval_metrics.update({
                'eval/win_rate_beginner': win_rates.get('Beginner', 0.0),
                'eval/win_rate_rookie': win_rates.get('Rookie', 0.0),
                'eval/win_rate_pro': win_rates.get('Pro', 0.0),
                'eval/win_rate_allstar': win_rates.get('Allstar', 0.0),
            })

            rookie_wr = win_rates.get('Rookie', 0.0)
            print(f'  Rookie win rate: {rookie_wr:.1%}')

            # Convergence check
            if rookie_wr >= self.config.rookie_target_wr:
                self.consecutive_wins += 1
                print(f'  Target met ({self.consecutive_wins}/'
                      f'{self.config.consecutive_evals_required} consecutive)')
                if self.consecutive_wins >= self.config.consecutive_evals_required:
                    self.converged = True
                    print('  CONVERGED — stopping training.')
            else:
                self.consecutive_wins = 0

        except (ImportError, Exception) as e:
            print(f'  Evaluation skipped: {e}')

        # Log to W&B
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.log(eval_metrics, step=total_step)


# ── main training function ───────────────────────────────────────────────────

def train(config: TrainConfig) -> None:
    """Run training with the given configuration."""
    assert config.eval_interval > 0, "eval_interval must be positive"
    assert config.total_steps > 0, "total_steps must be positive"
    assert config.batch_size > 0, "batch_size must be positive"
    assert config.algo in ALGO_MAP, f"Unknown algo: {config.algo}"

    set_seed(config.seed)

    model_dir = Path(config.model_dir) / f'seed_{config.seed}'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    print(f'Training config:')
    print(f'  Algorithm:  {config.algo}')
    print(f'  Seed:       {config.seed}')
    print(f'  Steps:      {config.total_steps:,}')
    print(f'  Device:     {"cuda" if torch.cuda.is_available() else "cpu"}')
    print(f'  Eval every: {config.eval_interval:,} steps')
    print(f'  Model dir:  {model_dir}')

    # ── environment ──────────────────────────────────────────────────────
    env = BaselineGymEnv(t_window=config.t_window)

    # ── d3rlpy algorithm ─────────────────────────────────────────────────
    algo = build_algo(config)

    # ── self-play opponent pool ──────────────────────────────────────────
    # algo_builder creates a fresh algo instance for loading opponent weights
    def _algo_builder():
        a = build_algo(config)
        a.build_with_env(env)
        return a

    pool = OpponentPool(
        snapshot_dir=config.snapshot_dir,
        algo_builder=_algo_builder,
        max_snapshots=config.max_snapshots,
    )

    # ── callback ─────────────────────────────────────────────────────────
    callback = TrainingCallback(config, env, pool)

    # ── W&B logging ──────────────────────────────────────────────────────
    if config.no_wandb:
        logger_adapter = FileAdapterFactory(root_dir=str(model_dir / 'logs'))
    else:
        logger_adapter = WanDBAdapterFactory(project=config.wandb_project)

    # ── exploration ──────────────────────────────────────────────────────
    explorer = NormalNoise(mean=0.0, std=config.explore_noise)

    # ── replay buffer ────────────────────────────────────────────────────
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=config.buffer_capacity,
        env=env,
    )

    # ── run metadata ─────────────────────────────────────────────────────
    callback.log_metadata()

    # ── train ────────────────────────────────────────────────────────────
    print(f'\nStarting training...\n')
    algo.fit_online(
        env=env,
        buffer=buffer,
        explorer=explorer,
        n_steps=config.total_steps,
        n_steps_per_epoch=config.n_steps_per_epoch,
        update_interval=config.update_interval,
        random_steps=config.random_steps,
        experiment_name=f'{config.algo}_seed{config.seed}',
        logger_adapter=logger_adapter,
        show_progress=True,
        callback=callback,
    )

    # ── save final model ─────────────────────────────────────────────────
    algo.save(str(model_dir / 'final_model'))
    print(f'\nTraining complete. Model saved to {model_dir}')

    if callback.converged:
        print(f'Converged at Rookie win rate >= {config.rookie_target_wr:.0%}')
    else:
        print(f'Did not converge within {config.total_steps:,} steps.')

    env.close()


# ── Optuna parameter loading ────────────────────────────────────────────────

def load_params_from_optuna(db_path: str) -> dict:
    """Load best hyperparameters from an Optuna study database."""
    try:
        import optuna
    except ImportError:
        raise ImportError('optuna required: pip install optuna')

    study = optuna.load_study(
        study_name='baseline-hparam-search',
        storage=f'sqlite:///{db_path}',
    )
    if not study.best_trial:
        raise RuntimeError(f'No completed trials in {db_path}')

    print(f'Loaded best params from Optuna (trial #{study.best_trial.number}):')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    return study.best_params


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train RL agent with d3rlpy.')

    # Core
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algo', default='AWAC', choices=list(ALGO_MAP.keys()))
    parser.add_argument('--total-steps', type=int, default=50_000_000)

    # Hyperparameters
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--awac-lambda', type=float, default=1.0)
    parser.add_argument('--explore-noise', type=float, default=0.1)
    parser.add_argument('--buffer-capacity', type=int, default=1_000_000)
    parser.add_argument('--random-steps', type=int, default=10_000)

    # Architecture
    parser.add_argument('--t-window', type=int, default=8)

    # Training loop
    parser.add_argument('--eval-interval', type=int, default=200_000)
    parser.add_argument('--snapshot-interval', type=int, default=10_000)
    parser.add_argument('--n-steps-per-epoch', type=int, default=10_000)

    # Paths
    parser.add_argument('--model-dir', default='models/baseline')

    # W&B
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', default='rlbot-baseline')

    # Optuna integration
    parser.add_argument('--params-from', default=None,
                        help='Load best hyperparams from Optuna SQLite DB')

    args = parser.parse_args()

    config = TrainConfig(
        seed=args.seed,
        algo=args.algo,
        total_steps=args.total_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        awac_lambda=args.awac_lambda,
        explore_noise=args.explore_noise,
        buffer_capacity=args.buffer_capacity,
        random_steps=args.random_steps,
        t_window=args.t_window,
        eval_interval=args.eval_interval,
        snapshot_interval=args.snapshot_interval,
        n_steps_per_epoch=args.n_steps_per_epoch,
        model_dir=args.model_dir,
        no_wandb=args.no_wandb,
        wandb_project=args.wandb_project,
    )

    # Override with Optuna-tuned params if requested
    if args.params_from:
        params = load_params_from_optuna(args.params_from)
        param_map = {
            'actor_lr': 'actor_lr',
            'critic_lr': 'critic_lr',
            'awac_lambda': 'awac_lambda',
            'tau': 'tau',
            'batch_size': 'batch_size',
            'gamma': 'gamma',
            'explore_noise': 'explore_noise',
        }
        for optuna_key, config_key in param_map.items():
            if optuna_key in params:
                setattr(config, config_key, params[optuna_key])

    train(config)


if __name__ == '__main__':
    main()
