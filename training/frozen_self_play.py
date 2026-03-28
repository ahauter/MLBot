"""
Frozen Opponent Self-Play
=========================
Performance-gated self-play: the opponent is a frozen snapshot that only
updates when the live agent demonstrates offensive mastery.

Swap condition (over rolling 100-episode window):
  - Win rate > 70%  (wins / decisive games, excluding draws)
  - Goals scored per episode > 0.5

This breaks the defensive equilibrium that arises in co-evolving self-play
with sparse rewards, creating a stepwise curriculum without reward shaping.

Classes
-------
EpisodeOutcomeTracker
    Rolling window tracker for win rate and goals-per-episode.
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

    def win_rate(self) -> float:
        """Win rate over decisive games (excludes draws).

        Returns 0.0 if no decisive games in the window.
        """
        if not self._outcomes:
            return 0.0
        wins = sum(1 for g in self._outcomes if g == 1)
        losses = sum(1 for g in self._outcomes if g == -1)
        decisive = wins + losses
        if decisive == 0:
            return 0.0
        return wins / decisive

    def goals_per_episode(self) -> float:
        """Average goals scored by the agent per episode."""
        if not self._outcomes:
            return 0.0
        goals = sum(1 for g in self._outcomes if g == 1)
        return goals / len(self._outcomes)

    def should_swap(
        self,
        win_rate_threshold: float = 0.7,
        goals_threshold: float = 0.5,
    ) -> Tuple[bool, Optional[str]]:
        """Check whether the frozen opponent should be promoted.

        Returns
        -------
        (should_swap, warning_or_none)
            (True, None) if both conditions met.
            (False, warning_str) if win rate met but goals too low (defensive play).
            (False, None) otherwise.
        """
        if len(self._outcomes) < self.window_size:
            return False, None

        wr = self.win_rate()
        gpe = self.goals_per_episode()

        if wr > win_rate_threshold:
            if gpe > goals_threshold:
                return True, None
            else:
                return False, (
                    f"Defensive play detected: win_rate={wr:.2f} > {win_rate_threshold} "
                    f"but goals_per_episode={gpe:.2f} <= {goals_threshold}"
                )
        return False, None

    def reset(self) -> None:
        """Clear the outcome window (e.g. after a swap event)."""
        self._outcomes.clear()


class FrozenOpponentPool(OpponentPool):
    """
    Frozen opponent pool with performance-gated promotion.

    Maintains up to ``max_snapshots`` (default 5) policy snapshots.
    Opponents are sampled uniformly from the pool. New snapshots are added
    only when the live agent meets the swap condition (>70% win rate AND
    >0.5 goals/episode over 100 episodes).

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
    win_rate_threshold : float
        Win rate threshold for promotion (default 0.7).
    goals_threshold : float
        Goals-per-episode threshold for promotion (default 0.5).
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        algo_builder=None,
        max_snapshots: int = 5,
        window_size: int = 100,
        win_rate_threshold: float = 0.7,
        goals_threshold: float = 0.5,
    ):
        super().__init__(snapshot_dir, algo_builder, max_snapshots)
        self.win_rate_threshold = win_rate_threshold
        self.goals_threshold = goals_threshold
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
        """Return True if the agent has mastered the current frozen opponent.

        Checks the rolling window for win rate > threshold AND
        goals per episode > threshold. Logs a warning if defensive play
        is detected (high win rate but low scoring).
        """
        do_swap, warning = self.tracker.should_swap(
            self.win_rate_threshold, self.goals_threshold
        )
        if warning:
            print(f"[step {total_step:,}] WARNING: {warning}")
            # Log to W&B if available
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'frozen_self_play/defensive_play_warning': 1,
                        'frozen_self_play/win_rate_at_warning': self.tracker.win_rate(),
                        'frozen_self_play/goals_per_ep_at_warning': self.tracker.goals_per_episode(),
                    }, step=total_step)
            except ImportError:
                pass
        return do_swap

    def log_swap_event(self, total_step: int) -> None:
        """Log a swap event to stdout and W&B."""
        wr = self.tracker.win_rate()
        gpe = self.tracker.goals_per_episode()
        n = self.num_snapshots()
        print(
            f"[step {total_step:,}] SWAP #{self.swap_count}: "
            f"win_rate={wr:.2f}, goals_per_ep={gpe:.2f}, pool_size={n}"
        )
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'frozen_self_play/swap_event': self.swap_count,
                    'frozen_self_play/win_rate_at_swap': wr,
                    'frozen_self_play/goals_per_ep_at_swap': gpe,
                    'frozen_self_play/pool_size': n,
                }, step=total_step)
        except ImportError:
            pass


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
