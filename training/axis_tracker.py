"""
Five-Axis Resource Cost Tracker
================================
Tracks cumulative resource consumption across the five research axes
during a single training run.

Axes
----
1. Simulation      — environment steps consumed
2. Real-world data — replay files loaded / processed
3. Human feedback  — labels or rank comparisons consumed
4. Reward eng.     — number of active dense-reward components
5. Pre-training    — GPU-hours of self-supervised pre-training

All metrics are logged under the ``resources_consumed/`` prefix in W&B
so they appear in a dedicated dashboard section.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AxisTracker:
    """Accumulates and logs resource costs on all five axes."""

    # Axis 1: cumulative environment steps
    sim_steps: int = 0

    # Axis 2: total replay files loaded into the buffer
    replays_loaded: int = 0

    # Axis 3: total human labels / rank comparisons consumed
    labels_consumed: int = 0

    # Axis 4: number of active reward-shaping components (0 = sparse only)
    reward_components: int = 0

    # Axis 5: GPU-hours spent on pre-training (0 if N/A)
    pretrain_gpu_hours: float = 0.0

    # ── mutators ──────────────────────────────────────────────────────────

    def record_sim_steps(self, n: int) -> None:
        """Increment simulation step counter."""
        self.sim_steps += n

    def record_replays(self, n: int) -> None:
        """Increment replay-files-loaded counter."""
        self.replays_loaded += n

    def record_labels(self, n: int) -> None:
        """Increment human-labels-consumed counter."""
        self.labels_consumed += n

    def set_reward_components(self, n: int) -> None:
        """Set the number of active dense-reward components."""
        self.reward_components = n

    def set_pretrain_hours(self, h: float) -> None:
        """Set pre-training GPU-hours (typically set once at run start)."""
        self.pretrain_gpu_hours = h

    # ── serialisation ─────────────────────────────────────────────────────

    def as_dict(self, prefix: str = 'resources_consumed/') -> dict:
        """Return current axis values as a flat dict suitable for W&B."""
        return {
            f'{prefix}axis1_sim_steps': self.sim_steps,
            f'{prefix}axis2_replays_loaded': self.replays_loaded,
            f'{prefix}axis3_labels_consumed': self.labels_consumed,
            f'{prefix}axis4_reward_components': self.reward_components,
            f'{prefix}axis5_pretrain_gpu_hours': self.pretrain_gpu_hours,
        }

    def log(self, wandb_mod, step: int) -> None:
        """Log current axis values to W&B (no-op if wandb_mod is None)."""
        if wandb_mod is None or wandb_mod.run is None:
            return
        wandb_mod.log(self.as_dict(), step=step)
