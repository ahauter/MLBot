"""
Evaluation Pipeline Configuration
==================================
Centralized dataclass for all eval pipeline parameters.

Reads eval-specific settings from ``config['evaluation']['params']``
and the shared environment class from the top-level ``config['env_class']``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EvalConfig:
    """Configuration for the simulation-based evaluation pipeline."""

    eval_interval: int = 50_000
    checkpoint_dir: str = 'checkpoints'
    episodes_per_tier: int = 100
    episode_timeout_steps: int = 3_000
    skill_target_tier: str = 'Rookie'
    skill_target_win_rate: float = 0.60
    t_window: int = 8

    # Dotted path to the gymnasium.Env class used for evaluation.
    # Read from the top-level config key (shared with training).
    # None means fall back to training.environments.baseline_env.BaselineGymEnv.
    env_class: Optional[str] = None

    # tier_name -> snapshot_path (None = random opponent)
    tier_opponents: Dict[str, Optional[str]] = field(default_factory=lambda: {
        'Beginner': None,
        'Rookie': None,
        'Pro': None,
        'Allstar': None,
    })

    @classmethod
    def from_config(cls, config: dict) -> 'EvalConfig':
        """Build from the training YAML config dict.

        Eval-specific params come from ``config['evaluation']['params']``.
        The env class comes from the top-level ``config['env_class']``.
        """
        defaults = cls()
        eval_section = config.get('evaluation', {})
        params = eval_section.get('params', {})
        return cls(
            eval_interval=config.get('eval_interval', defaults.eval_interval),
            checkpoint_dir=params.get('checkpoint_dir', defaults.checkpoint_dir),
            episodes_per_tier=params.get('episodes_per_tier', defaults.episodes_per_tier),
            episode_timeout_steps=params.get('episode_timeout_steps', defaults.episode_timeout_steps),
            skill_target_tier=params.get('skill_target_tier', defaults.skill_target_tier),
            skill_target_win_rate=params.get('skill_target_win_rate', defaults.skill_target_win_rate),
            t_window=config.get('t_window', defaults.t_window),
            env_class=config.get('env_class', defaults.env_class),
            tier_opponents=params.get('tier_opponents', defaults.tier_opponents),
        )

    # Ordered tier list — easiest to hardest
    TIER_ORDER = ['Beginner', 'Rookie', 'Pro', 'Allstar']
