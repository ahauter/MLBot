#!/usr/bin/env python3
"""
Hyperparameter Tuning with Optuna + ASHA Pruning
=================================================
Searches over AWAC training hyperparameters using Optuna with the
Asynchronous Successive Halving (ASHA) pruner for early stopping of
underperforming trials.

Each trial runs a shortened training session and reports its smoothed
episode reward. Trials showing no improvement are pruned at regular
checkpoints (every 200 episodes).

The Optuna study is stored in a local SQLite database so it survives
process crashes — just re-run this script to resume from where you left off.

Each trial also logs to W&B (if configured) under the same project,
grouped so the W&B parallel coordinates plot shows the full search.

Usage
-----
    # Run 50 Optuna trials (resumable):
    python training/tune.py

    # Override number of trials or episodes per trial:
    python training/tune.py --n-trials 20 --episodes-per-trial 1000

    # Disable W&B (stdout only):
    python training/tune.py --no-wandb

    # Show best trial from a completed study:
    python training/tune.py --show-best

Requirements
------------
    pip install optuna wandb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from train_config import TrainConfig
from train_all import RLGymEnv, discover_all_configs, train
from logger import ExperimentLogger


# ── Optuna study settings ─────────────────────────────────────────────────────

STUDY_NAME    = 'awac-hparam-search'
STORAGE_PATH  = 'sqlite:///optuna_mlbot.db'
PRUNE_INTERVAL = 200    # report intermediate reward every N episodes
PRUNE_MIN_RESOURCE = 200    # minimum episodes before pruning is allowed
PRUNE_REDUCTION_FACTOR = 3  # ASHA halving factor


# ── objective function ────────────────────────────────────────────────────────

def objective(trial, episodes_per_trial: int, use_wandb: bool) -> float:
    """
    Single Optuna trial: sample hyperparameters, run training, return reward.

    The smoothed 100-episode mean reward is used as the objective.
    Trials are pruned at PRUNE_INTERVAL-episode checkpoints if they are
    significantly below the current best.
    """
    import optuna

    config = TrainConfig(
        lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        awac_beta=trial.suggest_float('awac_beta', 0.1, 5.0),
        awac_max_weight=trial.suggest_float('awac_max_weight', 5.0, 100.0, log=True),
        gamma=trial.suggest_float('gamma', 0.95, 0.999),
        explore_std=trial.suggest_float('explore_std', 0.01, 0.3),
        max_episodes=episodes_per_trial,
        save_every=episodes_per_trial + 1,   # skip mid-trial checkpoints
        model_dir=f'models/trial_{trial.number}/',
        wandb_project='mlbot',
        wandb_run_name=f'trial-{trial.number:03d}',
        wandb_group='optuna-study',
    )

    configs = discover_all_configs()
    if not configs:
        raise RuntimeError('No scenario configs found — cannot run trial.')

    env   = RLGymEnv()
    logger = ExperimentLogger(config, enabled=use_wandb, group='optuna-study')

    smoothed_reward = train(
        all_configs=configs,
        env=env,
        config=config,
        logger=logger,
        trial=trial,
    )
    return smoothed_reward


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import optuna
        from optuna.pruners import ASHAPruner
        from optuna.samplers import TPESampler
    except ImportError:
        print('Optuna is not installed. Run:  pip install optuna', file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description='AWAC hyperparameter tuning via Optuna.')
    parser.add_argument('--n-trials',          default=50,   type=int,
                        help='Number of Optuna trials to run (default: 50)')
    parser.add_argument('--episodes-per-trial', default=2000, type=int,
                        help='Training episodes per trial (default: 2000)')
    parser.add_argument('--no-wandb',          action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--show-best',         action='store_true',
                        help='Print best trial from existing study and exit')
    parser.add_argument('--study-name',        default=STUDY_NAME)
    parser.add_argument('--storage',           default=STORAGE_PATH,
                        help='Optuna storage URL (default: sqlite:///optuna_mlbot.db)')
    args = parser.parse_args()

    # Suppress Optuna's verbose per-trial logs (W&B covers it)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        pruner=ASHAPruner(
            min_resource=PRUNE_MIN_RESOURCE,
            reduction_factor=PRUNE_REDUCTION_FACTOR,
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
            print(f'  Smoothed reward: {best.value:.4f}')
            print('  Hyperparameters:')
            for k, v in best.params.items():
                print(f'    {k}: {v}')
        return

    n_existing = len([t for t in study.trials
                      if t.state.is_finished()])
    print(f'Study "{args.study_name}" — {n_existing} completed trials so far.')
    print(f'Running {args.n_trials} more trials ({args.episodes_per_trial} ep each)...')
    print(f'Storage: {args.storage}')

    study.optimize(
        lambda trial: objective(
            trial,
            episodes_per_trial=args.episodes_per_trial,
            use_wandb=not args.no_wandb,
        ),
        n_trials=args.n_trials,
        n_jobs=1,           # RL envs are not thread-safe; run sequentially
        show_progress_bar=True,
    )

    print('\n=== Tuning complete ===')
    best = study.best_trial
    print(f'Best trial #{best.number}  reward={best.value:.4f}')
    for k, v in best.params.items():
        print(f'  {k}: {v}')
    print(f'\nTo resume: python training/tune.py --storage {args.storage}')
    print('To view:   python training/tune.py --show-best')


if __name__ == '__main__':
    main()
