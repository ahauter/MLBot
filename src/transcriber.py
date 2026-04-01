"""
Game Transcriber
================
Records every tick of a live RLBot match from one bot's perspective.
Used by both MLBot (AI) and HumanProxyBot to capture training data.

Each tick stores:
  - tokens:  (N_TOKENS, TOKEN_FEATURES) from state_to_tokens()
  - action:  (8,) the bot's own action
  - reward:  sparse +1/-1/0 from score deltas
  - done:    True on goal frames

On save, writes training/transcripts/<bot_name>_<timestamp>.npz
"""
from __future__ import annotations

import atexit
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from encoder import state_to_tokens, N_TOKENS, TOKEN_FEATURES

_REPO = Path(__file__).parent.parent
TRANSCRIPT_DIR = _REPO / 'training' / 'transcripts'


class GameTranscriber:
    """Records game ticks for offline SAC training.

    Parameters
    ----------
    bot_name : str
        Identifier for this bot (e.g. 'mlbot', 'humanproxy').
    transcript_dir : str or Path, optional
        Directory to write .npz files.  Defaults to training/transcripts/.
    """

    def __init__(
        self,
        bot_name: str,
        transcript_dir: Optional[str] = None,
    ):
        self.bot_name = bot_name
        self.transcript_dir = Path(transcript_dir) if transcript_dir else TRANSCRIPT_DIR
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        self._tokens: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

        # Score tracking for sparse reward
        self._prev_own_score: int = 0
        self._prev_opp_score: int = 0
        self._car_idx: Optional[int] = None

        self._saved = False

    def record_tick(
        self,
        packet,
        car_idx: int,
        action: np.ndarray,
    ) -> None:
        """Record one tick of game data.

        Parameters
        ----------
        packet : GameTickPacket
            The RLBot game packet for this tick.
        car_idx : int
            This bot's car index (0 = blue, 1 = orange).
        action : np.ndarray
            The 8-float action vector this bot outputted.
        """
        self._car_idx = car_idx
        opp_idx = car_idx ^ 1

        # Extract tokens from this bot's perspective (normalized, perspective-aware)
        tokens = state_to_tokens(packet, car_idx)  # (1, N_TOKENS, TOKEN_FEATURES)
        self._tokens.append(tokens[0])  # (N_TOKENS, TOKEN_FEATURES)
        self._actions.append(np.asarray(action, dtype=np.float32).copy())

        # Sparse reward from score deltas
        own_team = 0 if car_idx == 0 else 1
        opp_team = 1 if car_idx == 0 else 0
        own_score = int(packet.teams[own_team].score)
        opp_score = int(packet.teams[opp_team].score)

        reward = 0.0
        done = False
        if own_score > self._prev_own_score:
            reward = 1.0
            done = True
        elif opp_score > self._prev_opp_score:
            reward = -1.0
            done = True

        self._prev_own_score = own_score
        self._prev_opp_score = opp_score

        self._rewards.append(reward)
        self._dones.append(done)

    def save(self) -> Optional[Path]:
        """Write recorded data to .npz and return the file path.

        Safe to call multiple times — only writes once.
        Returns None if no ticks were recorded.
        """
        if self._saved:
            return None
        self._saved = True

        if not self._tokens:
            print(f'[transcriber:{self.bot_name}] No ticks recorded — nothing saved.')
            return None

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = self.transcript_dir / f'{self.bot_name}_{ts}.npz'

        np.savez_compressed(
            out_path,
            tokens=np.stack(self._tokens).astype(np.float32),    # (T, N_TOKENS, TOKEN_FEATURES)
            actions=np.stack(self._actions).astype(np.float32),   # (T, 8)
            rewards=np.array(self._rewards, dtype=np.float32),    # (T,)
            dones=np.array(self._dones, dtype=np.bool_),          # (T,)
        )

        n_goals = sum(1 for r in self._rewards if r != 0.0)
        print(f'[transcriber:{self.bot_name}] Saved {len(self._tokens)} ticks '
              f'({n_goals} goals) → {out_path}')
        return out_path

    @property
    def num_ticks(self) -> int:
        return len(self._tokens)
