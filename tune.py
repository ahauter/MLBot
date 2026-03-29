#!/usr/bin/env python3
"""
YAML-Based Hyperparameter Tuning with Optuna
=============================================
Algorithm-agnostic hyperparameter search driven by YAML config stubs.

Flow:
  1. Load a stub YAML config (specifies algorithm class + search space)
  2. Optuna samples hyperparameters from the search space
  3. Each trial creates an Algorithm instance, runs a short training loop
  4. Best params are emitted as a complete YAML config
  5. Optionally auto-launch multi-seed training with the best config

Usage
-----
    # Run 50 trials with a stub config:
    python training/tune.py --config configs/ppo_tune_stub.yaml

    # Show best trial from a previous study:
    python training/tune.py --config configs/ppo_tune_stub.yaml --show-best

    # Emit tuned config and auto-launch 5 seeds:
    python training/tune.py --config configs/ppo_tune_stub.yaml --auto-seeds 5

    # Override trials or step budget:
    python training/tune.py --config configs/ppo_tune_stub.yaml --n-trials 20 --steps-per-trial 200000

Requirements
------------
    pip install optuna pyyaml gymnasium torch
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))


# ── utility functions ────────────────────────────────────────────────────────

def load_class(dotted_path: str):
    """Import a class from a dotted module path, e.g. 'training.train.PPOAlgorithm'."""
    module_path, class_name = dotted_path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), class_name)


def sample_from_search_space(trial, search_space: dict) -> dict:
    """Sample Optuna params from a search space dict.

    Each entry in search_space maps a dotted param path to a spec dict:
        'algorithm.params.lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True}
        'algorithm.params.batch_size': {'type': 'int', 'low': 64, 'high': 512}
        'algorithm.params.activation': {'type': 'categorical', 'choices': ['relu', 'tanh']}
    """
    overrides = {}
    for param_path, spec in search_space.items():
        if spec['type'] == 'float':
            val = trial.suggest_float(
                param_path, spec['low'], spec['high'],
                log=spec.get('log', False),
            )
        elif spec['type'] == 'int':
            val = trial.suggest_int(
                param_path, spec['low'], spec['high'],
                log=spec.get('log', False),
            )
        elif spec['type'] == 'categorical':
            val = trial.suggest_categorical(param_path, spec['choices'])
        else:
            raise ValueError(f"Unknown search space type: {spec['type']}")
        overrides[param_path] = val
    return overrides


def apply_overrides(config: dict, overrides: dict) -> None:
    """Set nested keys by dotted path.

    Example: 'algorithm.params.lr' -> config['algorithm']['params']['lr']
    """
    for dotted_path, value in overrides.items():
        keys = dotted_path.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value


def load_config(yaml_path: str) -> dict:
    """Load a stub YAML config and resolve the algorithm class.

    Merges class defaults with YAML overrides for both params and search_space.
    Attaches the resolved class object under config['algorithm']['cls'].
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Resolve algorithm class
    algo_class_path = config.get('algorithm', {}).get('class')
    if algo_class_path is None:
        raise ValueError("Config must specify algorithm.class (dotted path)")

    AlgoCls = load_class(algo_class_path)
    config.setdefault('algorithm', {})['cls'] = AlgoCls

    # Merge class defaults with YAML overrides for params
    class_defaults = {}
    if hasattr(AlgoCls, 'default_params'):
        class_defaults = AlgoCls.default_params()
    yaml_params = config.get('algorithm', {}).get('params', {})
    config['algorithm']['params'] = {**class_defaults, **yaml_params}

    # Build search space: class defaults + YAML overrides
    class_search_space = {}
    if hasattr(AlgoCls, 'default_search_space'):
        class_search_space = AlgoCls.default_search_space()
    yaml_search_space = config.get('search_space', {})
    config['search_space'] = {**class_search_space, **yaml_search_space}

    return config


# ── reward tracking ──────────────────────────────────────────────────────────

class RewardTracker:
    """Wraps a gymnasium env to track episode returns for Optuna pruning.

    Algorithm-agnostic: works with any env that follows the gymnasium API.
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

def objective(trial, stub_config: dict, steps_per_trial: int, device: str) -> float:
    """Single Optuna trial: sample hyperparams, run short training, return mean reward.

    Creates an Algorithm instance from the config, runs collection/update loops,
    and reports intermediate values for Optuna pruning.
    """
    import optuna
    from gym_env import BaselineGymEnv

    config = copy.deepcopy(stub_config)

    # Sample hyperparameters from search space
    overrides = sample_from_search_space(trial, config.get('search_space', {}))
    apply_overrides(config, overrides)
    config['total_steps'] = steps_per_trial

    # Resolve algorithm class
    AlgoCls = config['algorithm']['cls']

    # Create a single agent (no population, for speed)
    agent = AlgoCls(agent_id=0, config=config, device=device)

    # Create environment
    t_window = config.get('t_window', 8)
    reward_class_path = config.get('reward', {}).get('class', None)
    reward_type = 'sparse'
    if reward_class_path and 'Dense' in reward_class_path:
        reward_type = 'dense'

    raw_env = BaselineGymEnv(t_window=t_window, reward_type=reward_type)
    env = RewardTracker(raw_env)

    pruning_interval = 50_000
    total_step = 0
    obs, _ = env.reset()

    try:
        while total_step < steps_per_trial:
            # Collect a transition via Algorithm ABC interface
            action_result = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action_result.action)
            done = terminated or truncated

            # Store transition (Algorithm manages its own buffer)
            agent.store_transition(obs, action_result, reward, next_obs, done, info)

            obs = next_obs
            total_step += 1

            if done:
                obs, _ = env.reset()

            # Update when the algorithm says it has enough data
            if agent.should_update():
                agent.update()

            # Report to Optuna for pruning
            if total_step > 0 and total_step % pruning_interval == 0:
                mean_r = env.mean_return(100)
                if len(env.episode_returns) > 0:
                    trial.report(mean_r, step=total_step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f'Trial {trial.number} failed: {e}', file=sys.stderr)
        return float('-inf')
    finally:
        raw_env.close()
        del agent
        gc.collect()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    mean_reward = env.mean_return(100)
    if np.isnan(mean_reward):
        print(f"Trial {trial.number}: NaN reward -- training diverged",
              file=sys.stderr)
        return float('-inf')
    return mean_reward


# ── YAML emission ────────────────────────────────────────────────────────────

def emit_complete_yaml(stub_config: dict, best_params: dict, output_path: str) -> None:
    """Write a complete YAML config with best params, no search_space key.

    The output YAML is ready to be passed to `python training/train.py --config`.
    """
    config = copy.deepcopy(stub_config)
    apply_overrides(config, best_params)

    # Remove search space (tuning is done)
    config.pop('search_space', None)

    # Remove non-serializable cls references
    for section in ('algorithm', 'opponent_pool', 'environment', 'reward',
                    'replay', 'feedback', 'evaluation', 'logger'):
        if section in config and isinstance(config[section], dict):
            config[section].pop('cls', None)

    # Set full training budget
    config['total_steps'] = config.get('total_steps', 50_000_000)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f'Wrote tuned config to: {output_path}')


def launch_seeds(output_yaml: str, n_seeds: int) -> None:
    """Launch multi-seed training runs sequentially with the tuned config."""
    print(f'\n=== Launching {n_seeds} seeds with config: {output_yaml} ===')
    for seed in range(n_seeds):
        cmd = [
            sys.executable, 'training/train.py',
            '--config', output_yaml,
            '--seed', str(seed),
        ]
        log_dir = Path('models/tuned')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'seed_{seed}.log'

        print(f'  Seed {seed}/{n_seeds - 1}: log -> {log_path}')
        with open(log_path, 'w') as log_file:
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        print(f'  Seed {seed} complete.')
    print('All seeds complete.')


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import optuna
        from optuna.pruners import SuccessiveHalvingPruner
        from optuna.samplers import TPESampler
    except ImportError:
        print('Optuna required: pip install optuna', file=sys.stderr)
        sys.exit(1)

    import torch

    parser = argparse.ArgumentParser(
        description='YAML-based hyperparameter tuning with Optuna')
    parser.add_argument('--config', required=True,
                        help='Stub YAML config path')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--steps-per-trial', type=int, default=500_000,
                        help='Environment steps per trial (default: 500000)')
    parser.add_argument('--output', default=None,
                        help='Output YAML path (default: configs/<name>_tuned_best.yaml)')
    parser.add_argument('--show-best', action='store_true',
                        help='Show best trial and exit')
    parser.add_argument('--auto-seeds', type=int, default=0,
                        help='Auto-launch N seeds after tuning (0 = disabled)')
    parser.add_argument('--study-name', default='yaml-hparam-search',
                        help='Optuna study name')
    parser.add_argument('--storage', default=None,
                        help='Optuna storage URL (default: sqlite:///<config_stem>_optuna.db)')
    parser.add_argument('--reset', action='store_true',
                        help='Delete existing study and start from scratch')
    parser.add_argument('--device', default=None,
                        help='Torch device (default: auto-detect)')
    args = parser.parse_args()

    # Derive defaults
    config_stem = Path(args.config).stem
    if args.storage is None:
        args.storage = f'sqlite:///{config_stem}_optuna.db'
    if args.output is None:
        args.output = f'configs/{config_stem}_tuned_best.yaml'
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load stub config
    stub_config = load_config(args.config)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.reset:
        try:
            optuna.delete_study(study_name=args.study_name,
                                storage=args.storage)
            print(f'Deleted existing study "{args.study_name}".')
        except KeyError:
            print(f'No existing study "{args.study_name}" -- starting fresh.')

    study = optuna.create_study(
        direction='maximize',
        pruner=SuccessiveHalvingPruner(
            min_resource=50_000,
            reduction_factor=3,
        ),
        sampler=TPESampler(seed=42),
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=not args.reset,
    )

    # Show best and exit
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
        return

    # Run optimization
    n_existing = len([t for t in study.trials if t.state.is_finished()])
    print(f'Study "{args.study_name}" -- {n_existing} completed trials.')
    print(f'Running {args.n_trials} more ({args.steps_per_trial:,} steps each, '
          f'device={args.device})...')
    search_space = stub_config.get('search_space', {})
    if search_space:
        print(f'Search space ({len(search_space)} params):')
        for param_path, spec in search_space.items():
            print(f'  {param_path}: {spec}')
    else:
        print('Warning: empty search space -- trials will all use default params.')

    study.optimize(
        lambda trial: objective(trial, stub_config, args.steps_per_trial, args.device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Report results
    print('\n=== Tuning complete ===')
    best = study.best_trial
    print(f'Best trial #{best.number}  reward={best.value:.4f}')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    # Emit complete YAML
    emit_complete_yaml(stub_config, best.params, args.output)

    # Auto-launch seeds
    if args.auto_seeds > 0:
        launch_seeds(args.output, args.auto_seeds)


if __name__ == '__main__':
    main()
