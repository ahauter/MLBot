"""
Frozen Opponent Self-Play
=========================
Performance-gated self-play: the opponent is a frozen snapshot that only
updates when the live agent demonstrates mastery.

Swap condition (over rolling 100-episode window):
  - Mean outcome score > 0.5  (goal=+1, concede=-1, timeout=0)

Classes
-------
EpisodeOutcomeTracker
    Rolling window tracker for mean-outcome score.
FrozenOpponentPool(OpponentPool)
    Pool of up to 5 frozen snapshots with performance-gated promotion.
OutcomeTrackingEnv
    BaselineGymEnv wrapper that records episode outcomes (sequential path).
"""
from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from self_play import OpponentPool


class EpisodeOutcomeTracker:
    """
    Tracks episode outcomes over a rolling window for swap decisions.

    Parameters
    ----------
    window_size : int
        Number of episodes in the rolling window (default 100).
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._outcomes: deque = deque(maxlen=window_size)

    def record(self, goal: int) -> None:
        """Record an episode outcome.

        Parameters
        ----------
        goal : int
            1 = agent scored (win), -1 = opponent scored (loss), 0 = timeout (draw).
        """
        self._outcomes.append(goal)

    @property
    def episode_count(self) -> int:
        return len(self._outcomes)

    def score(self) -> float:
        """Mean outcome over the rolling window. goal=+1, concede=-1, timeout=0."""
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    def should_swap(self, score_threshold: float = 0.5) -> bool:
        """Return True if mean outcome exceeds the threshold."""
        if len(self._outcomes) < self.window_size:
            return False
        return self.score() > score_threshold

    def reset(self) -> None:
        """Clear the outcome window (e.g. after a swap event)."""
        self._outcomes.clear()


class FrozenOpponentPool(OpponentPool):
    """
    Frozen opponent pool with performance-gated promotion.

    Maintains up to ``max_snapshots`` (default 5) policy snapshots.
    Opponents are sampled uniformly from the pool. New snapshots are added
    only when the live agent's mean outcome score exceeds ``score_threshold``
    over a rolling window of episodes (goal=+1, concede=-1, timeout=0).

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory where snapshots are stored.
    algo_builder : callable
        Function that creates a fresh d3rlpy algo instance.
    max_snapshots : int
        Maximum snapshots to retain (default 5).
    window_size : int
        Episode window for outcome tracking (default 100).
    score_threshold : float
        Mean-outcome threshold for promotion (default 0.1).
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        algo_builder=None,
        max_snapshots: int = 5,
        window_size: int = 100,
        score_threshold: float = 0.1,
    ):
        super().__init__(snapshot_dir, algo_builder, max_snapshots)
        self.score_threshold = score_threshold
        self.tracker = EpisodeOutcomeTracker(window_size=window_size)
        self.swap_count = 0

    def save_snapshot(self, algo, step: int) -> Path:
        """Save a snapshot (promotion). Evicts oldest if pool is full."""
        snap_dir = self._save_snapshot_to_dir(algo, step)
        self.swap_count += 1
        self.tracker.reset()
        return snap_dir

    def sample_opponent(self):
        """Load a uniformly random snapshot as frozen opponent.

        Returns None if no snapshots exist (opponent will be random).
        """
        snapshots = self._list_snapshots()
        if not snapshots:
            return None
        snap_dir = random.choice(snapshots)
        return self._load_snapshot(snap_dir)

    def latest(self):
        snapshots = self._list_snapshots()
        if not snapshots:
            return None
        return self._load_snapshot(snapshots[-1])

    def num_snapshots(self) -> int:
        return len(self._list_snapshots())

    def on_episode_end(self, goal: int) -> None:
        """Record episode outcome for swap condition tracking."""
        self.tracker.record(goal)

    def should_swap(self, total_step: int) -> bool:
        """Return True if mean outcome score exceeds the threshold."""
        return self.tracker.should_swap(self.score_threshold)

    def log_swap_event(self, total_step: int) -> None:
        """Log a swap event to stdout. W&B logging is handled by the main thread."""
        s = self.tracker.score()
        n = self.num_snapshots()
        print(
            f"[step {total_step:,}] SWAP #{self.swap_count}: "
            f"score={s:.2f}, pool_size={n}"
        )

    def get_metrics(self) -> dict:
        """Return current metrics for MetricsRegistry collection.

        Called by the main thread at each logging point. Read-only access
        to pool state — safe to call from any thread.
        """
        return {
            'swap_count': self.swap_count,
            'pool_size': self.num_snapshots(),
            'score': self.tracker.score(),
        }


class OutcomeTrackingEnv:
    """
    Wrapper around BaselineGymEnv that records episode outcomes.

    Used in the sequential training path (d3rlpy fit_online) where the
    callback doesn't receive per-episode data directly.

    Parameters
    ----------
    *args, **kwargs
        Passed through to BaselineGymEnv.
    """

    def __init__(self, *args, **kwargs):
        from gym_env import BaselineGymEnv
        self._env = BaselineGymEnv(*args, **kwargs)
        self.episode_outcomes: deque = deque(maxlen=200)

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        if done:
            self.episode_outcomes.append(info.get('goal', 0))
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def close(self):
        self._env.close()

    def set_opponent(self, opponent):
        self._env.set_opponent(opponent)

    def load_opponent_from_path(self, path):
        self._env.load_opponent_from_path(path)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space
