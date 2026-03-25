"""
Training Configuration
======================
Central dataclass for all AWAC training hyperparameters.
Pass a TrainConfig to train() instead of individual keyword args.
"""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class TrainConfig:
    # ── AWAC ──────────────────────────────────────────────────────────────────
    awac_beta: float = 1.0          # temperature: higher → closer to pure BC
    awac_max_weight: float = 20.0   # clip exp(A/β) to prevent gradient explosion
    gamma: float = 0.99             # discount factor

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr: float = 3e-4

    # ── Exploration ───────────────────────────────────────────────────────────
    explore_std: float = 0.1        # Gaussian noise std on analog actions

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buffer_capacity: int = 500_000
    buffer_batch_size: int = 256        # off-policy batch size per episode update
    expert_replay_ratio: float = 0.5    # fraction of off-policy batch from expert buffer
                                        # (0.0 = sim only, 1.0 = expert only)

    # ── Training loop ─────────────────────────────────────────────────────────
    max_episodes: int = 10_000
    save_every: int = 500
    model_dir: str = 'models/'

    # ── Logging / W&B ─────────────────────────────────────────────────────────
    wandb_project: str = 'mlbot'
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None   # useful for grouping Optuna trials
