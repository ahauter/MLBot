"""
Self-Play Opponent Pool
=======================
Abstract base class for self-play opponent pools, plus concrete
implementations for the abstracted training framework.

Snapshots are saved as PyTorch state_dicts (encoder + policy weights).
The training loop calls save_snapshot() / sample_opponent() through
the ABC interface.

Usage
-----
    pool = HistoricalOpponentPool('models/snapshots')
    pool.save_snapshot(algorithm, step=10000)
    path = pool.sample_opponent_path()  # returns snapshot dir path
"""
from __future__ import annotations

import random
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

# Snapshot file names
_ENCODER_FILE = 'encoder.pt'
_POLICY_FILE = 'policy.pt'


class OpponentPool(ABC):
    """
    Abstract base class for self-play opponent pools.

    Subclasses define when to swap opponents (periodic, performance-gated, etc.)
    and how many snapshots to retain.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory where snapshots are stored.
    max_snapshots : int
        Maximum number of snapshots to keep. Oldest are deleted first.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        max_snapshots: int = 20,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots

    @classmethod
    def default_params(cls) -> dict:
        return {'max_snapshots': 20}

    @abstractmethod
    def save_snapshot(self, algo, step: int) -> Path:
        """Save a snapshot of the current policy."""
        ...

    @abstractmethod
    def sample_opponent(self):
        """Sample an opponent. Returns snapshot path or None if empty."""
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

    def get_metrics(self) -> dict:
        """Return metrics for MetricsRegistry."""
        return {'pool_size': self.num_snapshots()}

    # ── shared internals ──────────────────────────────────────────────────

    def _list_snapshots(self) -> List[Path]:
        """Return snapshot dirs sorted by step number (ascending)."""
        if not self.snapshot_dir.exists():
            return []
        return [
            d for d in sorted(self.snapshot_dir.iterdir())
            if d.is_dir() and (d / _ENCODER_FILE).exists()
        ]

    def _save_weights(self, algo, step: int) -> Path:
        """Save encoder+policy state_dicts to a step-named subdirectory."""
        snap_dir = self.snapshot_dir / f'step_{step:010d}'
        snap_dir.mkdir(parents=True, exist_ok=True)
        weights = algo.get_weights()
        torch.save(weights['encoder'], str(snap_dir / _ENCODER_FILE))
        torch.save(weights['policy'], str(snap_dir / _POLICY_FILE))
        self._cleanup()
        return snap_dir

    def _cleanup(self) -> None:
        """Delete oldest snapshots if over max_snapshots."""
        snapshots = self._list_snapshots()
        while len(snapshots) > self.max_snapshots:
            oldest = snapshots.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)


class HistoricalOpponentPool(OpponentPool):
    """
    Exponential recency-weighted sampling for population-based training.

    More recent snapshots are sampled with higher probability. The decay
    rate controls how quickly older snapshots become unlikely.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory where snapshots are stored.
    decay_rate : float
        Exponential decay rate. Higher = more uniform sampling.
        0.95 means each older snapshot is 0.95x as likely as the next newer one.
    max_snapshots : int
        Maximum snapshots to retain.
    snapshot_interval : int
        Minimum steps between snapshots.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        decay_rate: float = 0.95,
        max_snapshots: int = 50,
        snapshot_interval: int = 50_000,
        **kwargs,
    ):
        super().__init__(snapshot_dir, max_snapshots)
        self.decay_rate = decay_rate
        self.snapshot_interval = snapshot_interval
        self._last_snapshot_step = 0

    @classmethod
    def default_params(cls) -> dict:
        return {
            'decay_rate': 0.95,
            'max_snapshots': 50,
            'snapshot_interval': 50_000,
        }

    def save_snapshot(self, algo, step: int) -> Path:
        self._last_snapshot_step = step
        return self._save_weights(algo, step)

    def sample_opponent(self) -> Optional[str]:
        """Sample a snapshot path with exponential recency weighting."""
        return self.sample_opponent_path()

    def sample_opponent_path(self) -> Optional[str]:
        """Return path to a sampled snapshot directory, or None if empty."""
        snapshots = self._list_snapshots()
        if not snapshots:
            return None
        n = len(snapshots)
        # Weights: newest=1.0, next=decay_rate, next=decay_rate^2, ...
        weights = [self.decay_rate ** (n - 1 - i) for i in range(n)]
        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = random.choices(snapshots, weights=probs, k=1)[0]
        return str(chosen)

    def num_snapshots(self) -> int:
        return len(self._list_snapshots())

    def should_swap(self, total_step: int) -> bool:
        if total_step - self._last_snapshot_step >= self.snapshot_interval:
            return True
        return False

    def on_episode_end(self, goal: int) -> None:
        pass  # Population manages scores externally

    def get_metrics(self) -> dict:
        return {
            'pool_size': self.num_snapshots(),
            'decay_rate': self.decay_rate,
        }


def load_opponent_from_snapshot(snap_dir: str | Path, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Load encoder+policy state_dicts from a snapshot directory.

    Returns dict with 'encoder' and 'policy' state_dicts.
    Used by gym_env workers to load opponents for inference.
    """
    snap_dir = Path(snap_dir)
    return {
        'encoder': torch.load(str(snap_dir / _ENCODER_FILE), map_location=device),
        'policy': torch.load(str(snap_dir / _POLICY_FILE), map_location=device),
    }
