#!/usr/bin/env python3
"""
Experiment Runner
=================
Loads an ExperimentConfig from YAML, tunes hyperparameters per condition,
then runs training with AxisTracker across all seeds. Prints a summary
table with mean +/- 95% CI for convergence steps.

Every condition is automatically tuned before training begins. Tuning
results are cached in the Optuna DB — re-running skips conditions that
already have completed trials.

Usage
-----
    # Full experiment (tune + train, 5 seeds per condition):
    python training/run_experiment.py --config training/experiments/00_naive_baseline.yaml

    # Override seeds:
    python training/run_experiment.py --config training/experiments/12_dense_reward_sweep.yaml --seeds 0,1

    # Control tuning budget:
    python training/run_experiment.py --config training/experiments/00_naive_baseline.yaml \\
        --tune-trials 20 --tune-steps 200000

    # Dry run (print configs, no tuning or training):
    python training/run_experiment.py --config training/experiments/00_naive_baseline.yaml --dry-run

    # Skip tuning (use cached or default params):
    python training/run_experiment.py --config training/experiments/00_naive_baseline.yaml --skip-tuning
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from axis_tracker import AxisTracker
from experiment_config import ExperimentConfig


# ── HP param keys that map from Optuna → TrainConfig ─────────────────────────

_TUNED_PARAM_MAP = {
    'actor_lr': 'actor_lr',
    'critic_lr': 'critic_lr',
    'awac_lambda': 'awac_lambda',
    'tau': 'tau',
    'batch_size': 'batch_size',
    'gamma': 'gamma',
    'explore_noise': 'explore_noise',
}


def _make_train_config(cfg_dict: dict):
    """
    Create a TrainConfig from a dict, ignoring unknown keys.

    Imports train.py lazily so dry-run mode works without heavy deps.
    """
    from train import TrainConfig
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    filtered = {k: v for k, v in cfg_dict.items()
                if k in valid_fields and not k.startswith('_')}
    return TrainConfig(**filtered)


def _setup_axis_tracker(budget: dict, cfg_dict: dict) -> AxisTracker:
    """Create an AxisTracker pre-populated with initial resource values."""
    tracker = AxisTracker()

    reward_type = cfg_dict.get('reward_type', 'sparse')
    if reward_type == 'sparse':
        tracker.set_reward_components(0)
    elif reward_type == 'dense':
        weights = cfg_dict.get('dense_reward_weights')
        if weights:
            active = sum(1 for w in weights.values() if w != 0.0)
            tracker.set_reward_components(active)
        else:
            tracker.set_reward_components(7)

    tracker.set_pretrain_hours(budget.get('pretrain_gpu_hours', 0.0))

    return tracker


def _apply_tuned_params(cfg_dict: dict, tuned: dict) -> None:
    """Merge Optuna best params into a config dict (in-place)."""
    for optuna_key, config_key in _TUNED_PARAM_MAP.items():
        if optuna_key in tuned:
            cfg_dict[config_key] = tuned[optuna_key]


def tune_all_conditions(
    exp: ExperimentConfig,
    all_configs: list,
    tune_trials: int,
    tune_steps: int,
    num_envs: int,
    no_wandb: bool,
    storage: str,
    reset_tuning: bool,
) -> dict:
    """
    Tune hyperparameters for every unique condition in the experiment.

    Returns dict mapping condition_name → best_params dict.
    """
    from tune import tune_for_condition, load_tuned_params

    # Deduplicate conditions (many seeds share same condition)
    seen = {}
    for cond_name, seed, cfg_dict in all_configs:
        if cond_name not in seen:
            seen[cond_name] = dict(cfg_dict)
            seen[cond_name].pop('_budget', None)

    print(f'\n{"=" * 60}')
    print(f'TUNING PHASE — {len(seen)} condition(s)')
    print(f'{"=" * 60}')

    tuned_params = {}
    for cond_name, cfg_dict in seen.items():
        study_key = f'{exp.name}/{cond_name}'

        # Check for cached results first
        if not reset_tuning:
            cached = load_tuned_params(study_key, storage=storage)
            if cached:
                print(f'\n  [{cond_name}] Using cached tuned params '
                      f'({len(cached)} params from previous run)')
                for k, v in cached.items():
                    print(f'    {k}: {v}')
                tuned_params[cond_name] = cached
                continue

        print(f'\n  [{cond_name}] Running {tune_trials} tuning trials...')
        best = tune_for_condition(
            condition_name=study_key,
            condition_config=cfg_dict,
            n_trials=tune_trials,
            steps_per_trial=tune_steps,
            num_envs=num_envs,
            use_wandb=not no_wandb,
            storage=storage,
            reset=reset_tuning,
        )
        tuned_params[cond_name] = best

    print(f'\n{"=" * 60}')
    print('TUNING COMPLETE')
    print(f'{"=" * 60}')
    for cond, params in tuned_params.items():
        summary = ', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                            for k, v in params.items())
        print(f'  {cond}: {summary}')
    print()

    return tuned_params


def run_single(cfg_dict: dict, budget: dict, no_wandb: bool = False) -> dict:
    """
    Run a single training seed.

    Returns
    -------
    dict with keys: seed, converged, final_step, wall_seconds, axis_costs
    """
    from train import train as do_train

    if no_wandb:
        cfg_dict['no_wandb'] = True

    train_config = _make_train_config(cfg_dict)
    tracker = _setup_axis_tracker(budget, cfg_dict)

    start = time.time()
    do_train(train_config, axis_tracker=tracker)
    elapsed = time.time() - start

    return {
        'seed': cfg_dict['seed'],
        'converged': tracker.sim_steps < budget.get('sim_steps', float('inf')),
        'final_step': tracker.sim_steps,
        'wall_seconds': elapsed,
        'axis_costs': tracker.as_dict(prefix=''),
    }


def print_summary(results_by_condition: dict) -> None:
    """Print aggregated results per condition."""
    print('\n' + '=' * 80)
    print('EXPERIMENT SUMMARY')
    print('=' * 80)

    for cond_name, results in results_by_condition.items():
        steps = [r['final_step'] for r in results]
        converged = [r['converged'] for r in results]
        walls = [r['wall_seconds'] for r in results]

        n = len(steps)
        mean_steps = np.mean(steps)
        ci_steps = 1.96 * np.std(steps, ddof=1) / np.sqrt(n) if n > 1 else 0
        conv_rate = np.mean(converged)

        print(f'\n  Condition: {cond_name}')
        print(f'    Seeds:          {n}')
        print(f'    Converged:      {sum(converged)}/{n} ({conv_rate:.0%})')
        print(f'    Steps (mean):   {mean_steps:,.0f} ± {ci_steps:,.0f} (95% CI)')
        print(f'    Wall time:      {np.mean(walls)/3600:.1f}h mean')

        if results:
            ax = results[0]['axis_costs']
            ax_keys = sorted(ax.keys())
            for k in ax_keys:
                vals = [r['axis_costs'][k] for r in results]
                mean_v = np.mean(vals)
                print(f'    {k}: {mean_v:,.1f}')

    print('\n' + '=' * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run an RL experiment: tune hyperparameters per condition, then train.')
    parser.add_argument('--config', required=True, help='Path to experiment YAML')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seed list (overrides YAML)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without tuning or training')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override TrainConfig fields: key=value ...')

    # Tuning control
    parser.add_argument('--tune-trials', type=int, default=30,
                        help='Optuna trials per condition (default: 30)')
    parser.add_argument('--tune-steps', type=int, default=500_000,
                        help='Env steps per tuning trial (default: 500k)')
    parser.add_argument('--tune-envs', type=int, default=1,
                        help='Parallel envs during tuning (default: 1)')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip tuning, use default or cached params')
    parser.add_argument('--reset-tuning', action='store_true',
                        help='Discard cached tuning results and re-tune')
    parser.add_argument('--optuna-storage', default='sqlite:///optuna_baseline.db',
                        help='Optuna storage path')

    args = parser.parse_args()

    # Load experiment config
    exp = ExperimentConfig.from_yaml(args.config)
    print(f'Experiment: {exp.name}')
    print(f'Description: {exp.description}')
    print(f'Intervention: {exp.intervention}')

    # Override seeds if requested
    if args.seeds:
        exp.seeds = [int(s) for s in args.seeds.split(',')]

    # Parse overrides
    overrides = {}
    for ov in args.override:
        k, v = ov.split('=', 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        overrides[k] = v

    # Generate all (condition, seed, config) tuples
    all_configs = exp.to_train_configs()
    n_conditions = len(set(c[0] for c in all_configs))
    print(f'Total runs: {len(all_configs)} '
          f'({n_conditions} conditions x {len(exp.seeds)} seeds)')

    if args.dry_run:
        print('\n--- DRY RUN ---')
        for cond_name, seed, cfg_dict in all_configs:
            cfg_dict.update(overrides)
            budget = cfg_dict.pop('_budget', {})
            print(f'\n[{cond_name}] seed={seed}')
            print(f'  budget: {json.dumps(budget, indent=4)}')
            for k in ['algo', 'reward_type', 'total_steps', 'replay_seed_dir',
                       'pretrained_encoder_path', 'dense_reward_weights',
                       'intervention', 'model_dir']:
                if k in cfg_dict:
                    print(f'  {k}: {cfg_dict[k]}')
        return

    # ── Phase 1: Tune ─────────────────────────────────────────────────────
    tuned_params = {}
    if not args.skip_tuning:
        tuned_params = tune_all_conditions(
            exp=exp,
            all_configs=all_configs,
            tune_trials=args.tune_trials,
            tune_steps=args.tune_steps,
            num_envs=args.tune_envs,
            no_wandb=args.no_wandb,
            storage=args.optuna_storage,
            reset_tuning=args.reset_tuning,
        )
    else:
        # Try to load cached params even when skipping
        from tune import load_tuned_params
        for cond_name, seed, cfg_dict in all_configs:
            if cond_name not in tuned_params:
                cached = load_tuned_params(
                    f'{exp.name}/{cond_name}', storage=args.optuna_storage)
                if cached:
                    tuned_params[cond_name] = cached
        if tuned_params:
            print(f'\nLoaded cached tuned params for {len(tuned_params)} condition(s)')
        else:
            print('\nNo cached tuned params found — using defaults')

    # ── Phase 2: Train ────────────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('TRAINING PHASE')
    print(f'{"=" * 60}')

    results_by_condition = defaultdict(list)

    for i, (cond_name, seed, cfg_dict) in enumerate(all_configs, 1):
        cfg_dict.update(overrides)
        budget = cfg_dict.pop('_budget', {})

        # Apply tuned hyperparameters for this condition
        if cond_name in tuned_params:
            _apply_tuned_params(cfg_dict, tuned_params[cond_name])

        print(f'\n{"─" * 60}')
        print(f'Run {i}/{len(all_configs)}: {cond_name} / seed {seed}')
        if cond_name in tuned_params:
            print(f'  (using tuned params)')
        print(f'{"─" * 60}')

        result = run_single(cfg_dict, budget, no_wandb=args.no_wandb)
        results_by_condition[cond_name].append(result)

        print(f'  -> steps={result["final_step"]:,} '
              f'wall={result["wall_seconds"]/3600:.2f}h '
              f'converged={result["converged"]}')

    # Print summary
    print_summary(dict(results_by_condition))

    # Save results to JSON
    output_dir = Path('models') / exp.name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.json'

    save_data = {
        'tuned_params': tuned_params,
        'results': {cond: results for cond, results in results_by_condition.items()},
    }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
