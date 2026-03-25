"""
SequenceReplayBuffer
====================
Off-policy replay buffer for AWAC training with the spatiotemporal encoder.

Stores per-step transitions and samples contiguous windows of length T_WINDOW
that always stay within a single episode.

Storage layout (circular buffer):
    obs_buf        (capacity, N_max, TOKEN_FEATURES)  — single-step token matrix
    act_buf        (capacity, ACTION_DIM)
    rew_buf        (capacity,)
    done_buf       (capacity,)                        — True on last step of episode
    episode_buf    (capacity,)  int32                 — episode ID; windows crossing
                                                        boundaries are rejected

Usage
-----
    buf = SequenceReplayBuffer(capacity=500_000, t_window=4, action_dim=8)

    # After each episode:
    buf.add_episode(trajectory)   # trajectory = list of (obs, action, reward, done)
                                  # obs: (N, TOKEN_FEATURES) numpy array

    # During AWAC gradient update:
    windows, actions, returns = buf.sample(batch_size=256, gamma=0.99)
    # windows : (batch, T_WINDOW, N, TOKEN_FEATURES) torch.Tensor
    # actions : (batch, ACTION_DIM) torch.Tensor
    # returns : (batch,) torch.Tensor   — MC returns at the last step of each window
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


class SequenceReplayBuffer:

    def __init__(
        self,
        capacity:   int,
        t_window:   int,
        action_dim: int = 8,
        token_features: int = 10,
    ) -> None:
        self.capacity       = capacity
        self.t_window       = t_window
        self.action_dim     = action_dim
        self.token_features = token_features

        # Obs stored as object arrays to handle variable N per game mode
        self._obs: List[np.ndarray | None]  = [None] * capacity   # each (N, F)
        self._act     = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rew     = np.zeros(capacity, dtype=np.float32)
        self._done    = np.zeros(capacity, dtype=bool)
        self._episode = np.zeros(capacity, dtype=np.int32)

        self._ptr      = 0     # next write position
        self._size     = 0     # number of valid entries
        self._ep_id    = 0     # current episode counter

    # ── writing ───────────────────────────────────────────────────────────────

    def add_episode(
        self,
        trajectory: List[Tuple[np.ndarray, np.ndarray, float, bool]],
    ) -> None:
        """
        Add a complete episode to the buffer.

        trajectory: list of (obs, action, reward, done)
            obs    : (N, TOKEN_FEATURES) float32
            action : (ACTION_DIM,)       float32
            reward : scalar
            done   : True only on the final step
        """
        ep = self._ep_id
        self._ep_id += 1
        for obs, action, reward, done in trajectory:
            idx = self._ptr % self.capacity
            self._obs[idx]     = obs.astype(np.float32)
            self._act[idx]     = action
            self._rew[idx]     = reward
            self._done[idx]    = done
            self._episode[idx] = ep
            self._ptr  += 1
            self._size  = min(self._size + 1, self.capacity)

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        batch_size: int,
        gamma: float = 0.99,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of contiguous T_WINDOW-step windows.

        Returns
        -------
        windows : (batch, T_WINDOW, N, TOKEN_FEATURES)
        actions : (batch, ACTION_DIM)   — action taken at the last step of each window
        returns : (batch,)              — MC discounted return from the last step
        """
        valid = self._valid_endpoints()
        if len(valid) == 0:
            raise RuntimeError(
                'Not enough data in replay buffer to sample windows of length '
                f'{self.t_window}.  Add more episodes first.'
            )
        chosen = np.random.choice(valid, size=batch_size, replace=True)

        T = self.t_window
        cap = self.capacity

        # Determine N from the first chosen window (assume homogeneous within batch)
        first_obs = self._obs[chosen[0]]
        N, F = first_obs.shape

        windows = np.zeros((batch_size, T, N, F), dtype=np.float32)
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        returns = np.zeros(batch_size, dtype=np.float32)

        for b, end_idx in enumerate(chosen):
            for t in range(T):
                src = (end_idx - (T - 1 - t)) % cap
                windows[b, t] = self._obs[src]
            actions[b] = self._act[end_idx]
            returns[b] = self._mc_return(end_idx, gamma)

        return (
            torch.tensor(windows, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _valid_endpoints(self) -> np.ndarray:
        """
        Return buffer indices that are valid as the last step of a T-length window.

        Validity requires:
          - all T steps in the window belong to the same episode
          - none of the first T-1 steps in the window are done=True
            (a done in the middle means an episode ended there)

        Uses numpy vectorised ops so it runs in milliseconds even on a full
        500 K-step buffer (vs. ~2 s for the equivalent pure-Python loop).
        """
        if self._size < self.t_window:
            return np.array([], dtype=np.intp)

        T   = self.t_window
        cap = self.capacity

        # All candidate endpoint indices in the written portion of the buffer
        indices = np.arange(T - 1, self._size, dtype=np.intp) % cap
        eps     = self._episode[indices]

        mask = np.ones(len(indices), dtype=bool)
        for k in range(1, T):
            prev = (indices - k) % cap
            mask &= (self._episode[prev] == eps)
            mask &= ~self._done[prev]

        return indices[mask]

    def _mc_return(self, end_idx: int, gamma: float) -> float:
        """
        Compute MC return looking forward from end_idx until done or buffer end.
        Stops at episode boundary.
        """
        ep  = self._episode[end_idx]
        cap = self.capacity
        G   = 0.0
        # Scan forward through stored transitions in the same episode
        # (limited look-ahead to keep this O(horizon) not O(capacity))
        max_horizon = 300
        for h in range(max_horizon):
            idx = (end_idx + h) % cap
            if h > 0 and self._episode[idx] != ep:
                break
            G += (gamma ** h) * float(self._rew[idx])
            if self._done[idx]:
                break
        return G

    def __len__(self) -> int:
        return self._size
