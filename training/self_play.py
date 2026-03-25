"""
Self-Play Opponent Pool
=======================
Manages a rotating pool of policy snapshots for self-play training.

During training, the current policy is periodically saved as a snapshot.
When collecting episodes, an opponent is sampled from the pool.

Usage
-----
    pool = OpponentPool('models/baseline/snapshots')
    pool.save_snapshot(encoder, policy_head, step=10000)

    opp_enc, opp_pol = pool.sample_opponent(device='cpu')
    env.set_opponent(opp_enc, opp_pol)
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))

from encoder import SharedTransformerEncoder
from policy_head import PolicyHead


class OpponentPool:
    """
    Maintains a directory of policy snapshots for self-play.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory where snapshots are stored.
    max_snapshots : int
        Maximum number of snapshots to keep. Oldest are deleted first.
    """

    def __init__(self, snapshot_dir: str | Path, max_snapshots: int = 20):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots

    def save_snapshot(
        self,
        encoder: SharedTransformerEncoder,
        policy_head: PolicyHead,
        step: int,
    ) -> Path:
        """Save a snapshot of the current policy at the given step."""
        snap_dir = self.snapshot_dir / f'step_{step:010d}'
        snap_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), str(snap_dir / 'encoder.pt'))
        torch.save(policy_head.state_dict(), str(snap_dir / 'policy.pt'))
        self._cleanup()
        return snap_dir

    def sample_opponent(
        self, device: str = 'cpu',
    ) -> Tuple[Optional[SharedTransformerEncoder], Optional[PolicyHead]]:
        """
        Load a random snapshot as a frozen opponent.

        Returns (None, None) if no snapshots exist yet (opponent will be random).
        """
        snapshots = self._list_snapshots()
        if not snapshots:
            return None, None

        snap_dir = random.choice(snapshots)
        return self._load_snapshot(snap_dir, device)

    def latest(
        self, device: str = 'cpu',
    ) -> Tuple[Optional[SharedTransformerEncoder], Optional[PolicyHead]]:
        """Load the most recent snapshot."""
        snapshots = self._list_snapshots()
        if not snapshots:
            return None, None
        return self._load_snapshot(snapshots[-1], device)

    def num_snapshots(self) -> int:
        return len(self._list_snapshots())

    # ── internals ──────────────────────────────────────────────────────────

    def _list_snapshots(self) -> List[Path]:
        """Return snapshot dirs sorted by step number (ascending)."""
        if not self.snapshot_dir.exists():
            return []
        dirs = [
            d for d in sorted(self.snapshot_dir.iterdir())
            if d.is_dir() and (d / 'encoder.pt').exists()
        ]
        return dirs

    def _load_snapshot(
        self, snap_dir: Path, device: str,
    ) -> Tuple[SharedTransformerEncoder, PolicyHead]:
        encoder = SharedTransformerEncoder()
        encoder.load_state_dict(
            torch.load(str(snap_dir / 'encoder.pt'), map_location=device)
        )
        encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False

        policy = PolicyHead()
        policy.load_state_dict(
            torch.load(str(snap_dir / 'policy.pt'), map_location=device)
        )
        policy.to(device).eval()
        for p in policy.parameters():
            p.requires_grad = False

        return encoder, policy

    def _cleanup(self) -> None:
        """Delete oldest snapshots if over max_snapshots."""
        snapshots = self._list_snapshots()
        while len(snapshots) > self.max_snapshots:
            oldest = snapshots.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
