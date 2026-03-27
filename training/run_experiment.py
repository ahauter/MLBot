#!/usr/bin/env python3
"""
Experiment Runner
=================
Loads an ExperimentConfig from YAML, generates TrainConfig per
(condition, seed), runs training with AxisTracker, and prints a
summary table with mean ± 95% CI for convergence steps.

Usage
-----
    # Full experiment (5 seeds per condition):
    python training/run_experiment.py --config training/experiments/00_naive_baseline.yaml

    # Override seeds:
    python training/run_experiment.py --config training/experiments/12_dense_reward_sweep.yaml --seeds 0,1

    # Dry run (print configs, no training):
    python training/run_experiment.py --config training/experiments/00_naive_baseline.yaml --dry-run

    # Override training params:
    python training/run_experiment.py --config training/experiments/00_naive_baseline.yaml \\
        --override total_steps=100000 eval_interval=50000 --no-wandb
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
    """Create an AxisTracker pre-populated with Axis 2/4/5 values from config."""
    tracker = AxisTracker()

    # Axis 4: count active reward components from config
    reward_type = cfg_dict.get('reward_type', 'sparse')
    if reward_type == 'sparse':
        tracker.set_reward_components(0)
    elif reward_type == 'dense':
        weights = cfg_dict.get('dense_reward_weights')
        if weights:
            active = sum(1 for w in weights.values() if w != 0.0)
            tracker.set_reward_components(active)
        else:
            tracker.set_reward_components(7)  # all default components

    # Axis 5: pre-training hours (set by budget, actual value logged post-hoc)
    tracker.set_pretrain_hours(budget.get('pretrain_gpu_hours', 0.0))

    return tracker


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

        # Axis costs summary
        if results:
            ax = results[0]['axis_costs']
            ax_keys = sorted(ax.keys())
            for k in ax_keys:
                vals = [r['axis_costs'][k] for r in results]
                mean_v = np.mean(vals)
                print(f'    {k}: {mean_v:,.1f}')

    print('\n' + '=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Run an RL experiment from YAML config.')
    parser.add_argument('--config', required=True, help='Path to experiment YAML')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seed list (overrides YAML)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without running training')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override TrainConfig fields: key=value ...')
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
        # Try to cast to int/float
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
    print(f'Total runs: {len(all_configs)} '
          f'({len(set(c[0] for c in all_configs))} conditions × '
          f'{len(exp.seeds)} seeds)')

    if args.dry_run:
        print('\n--- DRY RUN ---')
        for cond_name, seed, cfg_dict in all_configs:
            cfg_dict.update(overrides)
            budget = cfg_dict.pop('_budget', {})
            print(f'\n[{cond_name}] seed={seed}')
            print(f'  budget: {json.dumps(budget, indent=4)}')
            # Show key config fields
            for k in ['algo', 'reward_type', 'total_steps', 'replay_seed_dir',
                       'pretrained_encoder_path', 'dense_reward_weights',
                       'intervention', 'model_dir']:
                if k in cfg_dict:
                    print(f'  {k}: {cfg_dict[k]}')
        return

    # Run experiments
    results_by_condition = defaultdict(list)

    for i, (cond_name, seed, cfg_dict) in enumerate(all_configs, 1):
        cfg_dict.update(overrides)
        budget = cfg_dict.pop('_budget', {})

        print(f'\n{"─" * 60}')
        print(f'Run {i}/{len(all_configs)}: {cond_name} / seed {seed}')
        print(f'{"─" * 60}')

        result = run_single(cfg_dict, budget, no_wandb=args.no_wandb)
        results_by_condition[cond_name].append(result)

        print(f'  → steps={result["final_step"]:,} '
              f'wall={result["wall_seconds"]/3600:.2f}h '
              f'converged={result["converged"]}')

    # Print summary
    print_summary(dict(results_by_condition))

    # Save results to JSON
    output_dir = Path('models') / exp.name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(
            {cond: results for cond, results in results_by_condition.items()},
            f, indent=2, default=str,
        )
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
