"""
Experiment Configuration System
================================
YAML-backed dataclasses that define a complete experiment: intervention,
resource budgets, sweep points, seeds, and W&B metadata.

An experiment config generates one ``TrainConfig`` per (sweep_point, seed)
combination. The runner (``run_experiment.py``) iterates these and calls
``train()`` for each.

Usage
-----
    from experiment_config import ExperimentConfig
    exp = ExperimentConfig.from_yaml('training/experiments/00_naive_baseline.yaml')
    for condition, seed, train_cfg in exp.to_train_configs():
        train(train_cfg)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ── axis budget ───────────────────────────────────────────────────────────────

@dataclass
class AxisBudget:
    """Resource limits for each of the five research axes."""

    sim_steps: int = 50_000_000        # Axis 1: max simulation env steps
    num_replays: int = 0               # Axis 2: replay files to load
    num_labels: int = 0                # Axis 3: human feedback labels
    reward_components: int = 0         # Axis 4: active dense-reward components (0 = sparse)
    pretrain_gpu_hours: float = 0.0    # Axis 5: pre-training compute


# ── sweep point ───────────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    """One condition in a parameter sweep."""

    name: str                          # e.g. "replays_100"
    overrides: Dict = field(default_factory=dict)  # TrainConfig field overrides
    budget_overrides: Dict = field(default_factory=dict)  # AxisBudget field overrides


# ── experiment config ─────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Full experiment definition.

    A single-condition experiment has ``sweep=None``.
    A sweep experiment has one ``SweepPoint`` per condition; each point
    can override TrainConfig fields and AxisBudget fields independently.
    """

    name: str
    description: str
    intervention: str                  # 'none', 'dense_reward', 'offline_replays', etc.
    base_config: Dict                  # TrainConfig field values (algo, reward_type, ...)
    budget: AxisBudget = field(default_factory=AxisBudget)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    sweep: Optional[List[SweepPoint]] = None
    wandb_project: str = 'rlbot-baseline'
    wandb_tags: List[str] = field(default_factory=list)

    # ── config generation ─────────────────────────────────────────────────

    def to_train_configs(self) -> List[Tuple[str, int, dict]]:
        """
        Generate ``(condition_name, seed, config_dict)`` tuples.

        Each *config_dict* contains all fields needed to construct a
        ``TrainConfig``. The caller (run_experiment.py) is responsible for
        creating the actual ``TrainConfig`` from this dict — keeping
        this module free of a hard import on ``train.py``.
        """
        conditions = self._sweep_conditions()
        results = []
        for cond_name, cond_overrides, cond_budget in conditions:
            for seed in self.seeds:
                cfg = dict(self.base_config)
                cfg.update(cond_overrides)
                cfg['seed'] = seed
                cfg['wandb_project'] = self.wandb_project
                cfg['wandb_group'] = f'{self.name}/{cond_name}'
                cfg['wandb_run_name'] = f'{cond_name}/seed_{seed}'
                cfg['wandb_tags'] = list(self.wandb_tags) + [
                    self.intervention, cond_name,
                ]
                cfg['model_dir'] = f'models/{self.name}/{cond_name}'
                cfg['intervention'] = self.intervention

                # Stash budget in config dict for the runner to extract
                cfg['_budget'] = {
                    'sim_steps': cond_budget.sim_steps,
                    'num_replays': cond_budget.num_replays,
                    'num_labels': cond_budget.num_labels,
                    'reward_components': cond_budget.reward_components,
                    'pretrain_gpu_hours': cond_budget.pretrain_gpu_hours,
                }
                results.append((cond_name, seed, cfg))
        return results

    def _sweep_conditions(self) -> List[Tuple[str, Dict, AxisBudget]]:
        """Return list of (name, config_overrides, budget) per condition."""
        if self.sweep is None:
            return [('baseline', {}, copy.deepcopy(self.budget))]

        conditions = []
        for point in self.sweep:
            budget = copy.deepcopy(self.budget)
            for k, v in point.budget_overrides.items():
                if hasattr(budget, k):
                    setattr(budget, k, v)
            conditions.append((point.name, dict(point.overrides), budget))
        return conditions

    # ── YAML serialisation ────────────────────────────────────────────────

    def to_yaml(self, path: str | Path) -> None:
        """Serialize experiment config to a YAML file."""
        data = {
            'name': self.name,
            'description': self.description,
            'intervention': self.intervention,
            'base_config': self.base_config,
            'budget': {
                'sim_steps': self.budget.sim_steps,
                'num_replays': self.budget.num_replays,
                'num_labels': self.budget.num_labels,
                'reward_components': self.budget.reward_components,
                'pretrain_gpu_hours': self.budget.pretrain_gpu_hours,
            },
            'seeds': self.seeds,
            'wandb_project': self.wandb_project,
            'wandb_tags': self.wandb_tags,
        }
        if self.sweep is not None:
            data['sweep'] = [
                {
                    'name': p.name,
                    'overrides': p.overrides,
                    'budget_overrides': p.budget_overrides,
                }
                for p in self.sweep
            ]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ExperimentConfig':
        """Load experiment config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        budget_data = data.get('budget', {})
        budget = AxisBudget(
            sim_steps=budget_data.get('sim_steps', 50_000_000),
            num_replays=budget_data.get('num_replays', 0),
            num_labels=budget_data.get('num_labels', 0),
            reward_components=budget_data.get('reward_components', 0),
            pretrain_gpu_hours=budget_data.get('pretrain_gpu_hours', 0.0),
        )

        sweep = None
        if 'sweep' in data and data['sweep']:
            sweep = [
                SweepPoint(
                    name=s['name'],
                    overrides=s.get('overrides', {}),
                    budget_overrides=s.get('budget_overrides', {}),
                )
                for s in data['sweep']
            ]

        return cls(
            name=data['name'],
            description=data['description'],
            intervention=data['intervention'],
            base_config=data.get('base_config', {}),
            budget=budget,
            seeds=data.get('seeds', [0, 1, 2, 3, 4]),
            sweep=sweep,
            wandb_project=data.get('wandb_project', 'rlbot-baseline'),
            wandb_tags=data.get('wandb_tags', []),
        )
