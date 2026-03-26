"""
Self-Play Opponent Pool (d3rlpy)
================================
Manages a rotating pool of d3rlpy model snapshots for self-play training.

During training, the current d3rlpy algo is periodically saved as a snapshot.
When collecting episodes, an opponent is sampled from the pool and loaded
as a frozen d3rlpy model that calls .predict() to get actions.

Usage
-----
    pool = OpponentPool('models/baseline/snapshots', config)
    pool.save_snapshot(algo, step=10000)

    opponent_algo = pool.sample_opponent()
    env.set_opponent(opponent_algo)  # uses algo.predict() for actions
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

# Snapshot file name (d3rlpy save_model format)
_MODEL_FILE = 'model.pt'


class OpponentPool:
    """
    Maintains a directory of d3rlpy model snapshots for self-play.

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

    def save_snapshot(self, algo, step: int) -> Path:
        """
        Save a snapshot of the current d3rlpy algo at the given step.

        Parameters
        ----------
        algo : d3rlpy algo
            The current training algorithm (must be built/initialized).
        step : int
            Environment step count for naming.
        """
        snap_dir = self.snapshot_dir / f'step_{step:010d}'
        snap_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(snap_dir / _MODEL_FILE)
        algo.save_model(model_path)
        assert (snap_dir / _MODEL_FILE).exists(), \
            f"Failed to save snapshot: {model_path}"
        self._cleanup()
        return snap_dir

    def sample_opponent(self):
        """
        Load a random snapshot as a frozen opponent.

        Returns None if no snapshots exist yet (opponent will be random).
        """
        snapshots = self._list_snapshots()
        if not snapshots:
            return None
        snap_dir = random.choice(snapshots)
        return self._load_snapshot(snap_dir)

    def latest(self):
        """Load the most recent snapshot."""
        snapshots = self._list_snapshots()
        if not snapshots:
            return None
        return self._load_snapshot(snapshots[-1])

    def num_snapshots(self) -> int:
        return len(self._list_snapshots())

    # ── internals ──────────────────────────────────────────────────────────

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

    def _cleanup(self) -> None:
        """Delete oldest snapshots if over max_snapshots."""
        snapshots = self._list_snapshots()
        while len(snapshots) > self.max_snapshots:
            oldest = snapshots.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
