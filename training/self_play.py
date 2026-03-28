"""
Self-Play Opponent Pool (d3rlpy)
================================
Abstract base class for self-play opponent pools, plus the periodic
(naive baseline) implementation.

During training, the current d3rlpy algo is saved as a snapshot.
When collecting episodes, an opponent is sampled from the pool and loaded
as a frozen d3rlpy model that calls .predict() to get actions.

Usage
-----
    pool = PeriodicOpponentPool('models/baseline/snapshots', algo_builder=fn)
    pool.save_snapshot(algo, step=10000)

    opponent_algo = pool.sample_opponent()
    env.set_opponent(opponent_algo)  # uses algo.predict() for actions
"""
from __future__ import annotations

import random
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

# Snapshot file name (d3rlpy save_model format)
_MODEL_FILE = 'model.pt'


class OpponentPool(ABC):
    """
    Abstract base class for self-play opponent pools.

    Subclasses define when to swap opponents (periodic, performance-gated, etc.)
    and how many snapshots to retain.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory where snapshots are stored.
    algo_builder : callable
        Function that creates a fresh d3rlpy algo instance (same architecture).
        Called as algo_builder() -> d3rlpy algo.
    max_snapshots : int
        Maximum number of snapshots to keep. Oldest are deleted first.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        algo_builder=None,
        max_snapshots: int = 20,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots
        self._algo_builder = algo_builder

    @abstractmethod
    def save_snapshot(self, algo, step: int) -> Path:
        """Save a snapshot of the current policy."""
        ...

    @abstractmethod
    def sample_opponent(self):
        """Load a random snapshot as a frozen opponent. Returns None if empty."""
        ...

    @abstractmethod
    def latest(self):
        """Load the most recent snapshot. Returns None if empty."""
        ...

    @abstractmethod
    def num_snapshots(self) -> int:
        """Return the number of available snapshots."""
        ...

    @abstractmethod
    def should_swap(self, total_step: int) -> bool:
        """Return True if an opponent swap should happen now."""
        ...

    def on_episode_end(self, goal: int) -> None:
        """Hook called after each episode with goal outcome.

        Parameters
        ----------
        goal : int
            1 = blue (agent) scored, -1 = orange (opponent) scored, 0 = timeout.
        """

    # ── shared internals ──────────────────────────────────────────────────

    def _list_snapshots(self) -> List[Path]:
        """Return snapshot dirs sorted by step number (ascending)."""
        if not self.snapshot_dir.exists():
            return []
        return [
            d for d in sorted(self.snapshot_dir.iterdir())
            if d.is_dir() and (d / _MODEL_FILE).exists()
        ]

    def _load_snapshot(self, snap_dir: Path):
        """
        Load a d3rlpy model snapshot.

        Creates a fresh algo instance via algo_builder, then loads the
        saved weights. Returns the algo ready for .predict().
        """
        assert self._algo_builder is not None, \
            "OpponentPool requires algo_builder to load snapshots"

        algo = self._algo_builder()
        model_path = str(snap_dir / _MODEL_FILE)
        algo.load_model(model_path)

        # Verify the loaded model can predict
        obs_dim = algo.impl.observation_shape[0]
        test_obs = np.zeros((1, obs_dim), dtype=np.float32)
        test_action = algo.predict(test_obs)
        assert test_action is not None, \
            f"Loaded model from {snap_dir} cannot predict"
        assert test_action.shape[1] == 8, \
            f"Bad action dim from loaded model: {test_action.shape}"

        return algo

    def _save_snapshot_to_dir(self, algo, step: int) -> Path:
        """Save model weights to a step-named subdirectory and clean up old ones."""
        snap_dir = self.snapshot_dir / f'step_{step:010d}'
        snap_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(snap_dir / _MODEL_FILE)
        algo.save_model(model_path)
        assert (snap_dir / _MODEL_FILE).exists(), \
            f"Failed to save snapshot: {model_path}"
        self._cleanup()
        return snap_dir

    def _cleanup(self) -> None:
        """Delete oldest snapshots if over max_snapshots."""
        snapshots = self._list_snapshots()
        while len(snapshots) > self.max_snapshots:
            oldest = snapshots.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)


class PeriodicOpponentPool(OpponentPool):
    """
    Saves snapshots on a fixed step interval and samples uniformly.

    This is the naive baseline self-play strategy: save every N steps,
    randomly pick a past version as opponent.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory where snapshots are stored.
    algo_builder : callable
        Function that creates a fresh d3rlpy algo instance.
    max_snapshots : int
        Maximum snapshots to retain (default 20, oldest evicted first).
    snapshot_interval : int
        Environment steps between snapshots (default 10000).
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        algo_builder=None,
        max_snapshots: int = 20,
        snapshot_interval: int = 10_000,
    ):
        super().__init__(snapshot_dir, algo_builder, max_snapshots)
        self.snapshot_interval = snapshot_interval
        self._last_snapshot_step = 0

    def save_snapshot(self, algo, step: int) -> Path:
        return self._save_snapshot_to_dir(algo, step)

    def sample_opponent(self):
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

    def should_swap(self, total_step: int) -> bool:
        if total_step - self._last_snapshot_step >= self.snapshot_interval:
            self._last_snapshot_step = total_step
            return True
        return False
