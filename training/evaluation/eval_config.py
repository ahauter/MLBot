"""
Evaluation Pipeline Configuration
==================================
Centralized dataclass for all eval pipeline parameters.
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

    # tier_name -> snapshot_path (None = random opponent)
    tier_opponents: Dict[str, Optional[str]] = field(default_factory=lambda: {
        'Beginner': None,
        'Rookie': None,
        'Pro': None,
        'Allstar': None,
    })

    @classmethod
    def from_config(cls, config: dict) -> 'EvalConfig':
        """Build from the training YAML config dict."""
        defaults = cls()
        eval_section = config.get('evaluation', {})
        return cls(
            eval_interval=config.get('eval_interval', defaults.eval_interval),
            checkpoint_dir=eval_section.get('checkpoint_dir', defaults.checkpoint_dir),
            episodes_per_tier=eval_section.get('episodes_per_tier', defaults.episodes_per_tier),
            episode_timeout_steps=eval_section.get('episode_timeout_steps', defaults.episode_timeout_steps),
            skill_target_tier=eval_section.get('skill_target_tier', defaults.skill_target_tier),
            skill_target_win_rate=eval_section.get('skill_target_win_rate', defaults.skill_target_win_rate),
            t_window=config.get('t_window', defaults.t_window),
            tier_opponents=eval_section.get('tier_opponents', defaults.tier_opponents),
        )

    # Ordered tier list — easiest to hardest
    TIER_ORDER = ['Beginner', 'Rookie', 'Pro', 'Allstar']
