"""Sparse (goal-only) reward function. The baseline zero-point."""
from __future__ import annotations

import numpy as np

from training.abstractions import RewardFunction


class SparseRewardFunction(RewardFunction):
    """Goals only: +1 for scoring, -1 for conceding, 0 otherwise."""

    def compute_reward(self, obs: np.ndarray, action: np.ndarray,
                       next_obs: np.ndarray, done: bool, info: dict) -> float:
        if done:
            return float(info.get('goal', 0))
        return 0.0

    def on_reset(self) -> None:
        pass

    def get_metrics(self) -> dict:
        return {'num_components': 1}
