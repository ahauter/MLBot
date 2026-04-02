"""
Trajectory Dataset for SE3 Pretraining
=======================================
Converts replay .npz files into fixed-length trajectory windows of SE3 raw
state sequences for pretraining the spectral field encoder.

Each window is a (window_len, RAW_STATE_DIM) tensor where previous-frame
values (velocities, angular velocities, scalars) are computed from consecutive
frame differences.

Car angular velocity is derived from Euler angle finite differences (not
available directly in replay token format). has_flip and on_ground are
approximated from z position.

Usage:
    dataset = TrajectoryDataset('training/data/parsed/', window_len=64)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import MAX_ANG_VEL, CEILING_Z
from se3_field import (
    RAW_STATE_DIM, D_AMP,
    _BALL_OFF, _EGO_OFF, _OPP_OFF, _PAD_OFF, _GS_OFF,
    _PREV_VEL_OFF, _PREV_EGO_VEL_OFF, _PREV_OPP_VEL_OFF,
    _PREV_ANG_VEL_OFF, _PREV_EGO_ANG_VEL_OFF, _PREV_OPP_ANG_VEL_OFF,
    _PREV_SCALARS_OFF, _PREV_OPP_SCALARS_OFF,
    euler_to_quaternion_batch,
)

# Approximate on_ground threshold (normalised z)
_ON_GROUND_Z_THRESH = 17.5 / CEILING_Z

# Token layout constants
_N_TOKENS = 10
_TOKEN_FEATURES = 10


class TrajectoryDataset(Dataset):
    """Yields fixed-length trajectory windows as (window_len, RAW_STATE_DIM) tensors.

    Parameters
    ----------
    parsed_dir : str or Path
        Directory containing .npz replay files from collect_replays.py.
    window_len : int
        Number of consecutive frames per window.
    stride : int
        Step between window starts (overlap = window_len - stride).
    min_episode_len : int
        Skip episodes shorter than this.
    """

    def __init__(
        self,
        parsed_dir: str | Path,
        window_len: int = 64,
        stride: int = 32,
        min_episode_len: int = 100,
    ):
        self.window_len = window_len
        self.stride = stride
        self.windows: List[np.ndarray] = []

        parsed_dir = Path(parsed_dir)
        npz_files = sorted(parsed_dir.glob('*.npz'))

        for path in npz_files:
            data = np.load(path)
            if 'tokens' not in data:
                continue
            tokens = data['tokens']    # (T, 2, N_TOKENS, TOKEN_FEATURES)
            dones = data.get('dones', np.zeros((len(tokens), 2), dtype=bool))
            T = len(tokens)

            for player in (0, 1):
                # Split into episodes
                ep_start = 0
                for t in range(T):
                    if dones[t, player]:
                        self._process_episode(
                            tokens[ep_start:t + 1, player], min_episode_len)
                        ep_start = t + 1
                # Final segment
                if ep_start < T:
                    self._process_episode(
                        tokens[ep_start:, player], min_episode_len)

    def _process_episode(self, tokens: np.ndarray, min_len: int) -> None:
        """Convert an episode's tokens to raw state sequence and slice into windows."""
        if len(tokens) < min_len:
            return
        raw_seq = self._tokens_to_raw_state_sequence(tokens)
        T = len(raw_seq)
        for start in range(0, T - self.window_len + 1, self.stride):
            self.windows.append(raw_seq[start:start + self.window_len])

    def _tokens_to_raw_state_sequence(self, tokens: np.ndarray) -> np.ndarray:
        """Convert (T, N_TOKENS, TOKEN_FEATURES) to (T, RAW_STATE_DIM).

        Computes prev velocities, angular velocities, and scalars from
        consecutive frame differences. First frame gets zero prev values.
        """
        T = len(tokens)
        raw_seq = np.zeros((T, RAW_STATE_DIM), dtype=np.float32)

        prev_ball_vel = np.zeros(3, dtype=np.float32)
        prev_ego_vel = np.zeros(3, dtype=np.float32)
        prev_opp_vel = np.zeros(3, dtype=np.float32)
        prev_ball_ang_vel = np.zeros(3, dtype=np.float32)
        prev_ego_ang_vel = np.zeros(3, dtype=np.float32)
        prev_opp_ang_vel = np.zeros(3, dtype=np.float32)
        prev_ego_scalars = np.zeros(3, dtype=np.float32)
        prev_opp_scalars = np.zeros(3, dtype=np.float32)
        prev_ego_euler = np.zeros(3, dtype=np.float32)
        prev_opp_euler = np.zeros(3, dtype=np.float32)

        for t in range(T):
            tok = tokens[t]  # (N_TOKENS, TOKEN_FEATURES)
            raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)

            # Ball: pos(3) + vel(3) + ang_vel(3)
            raw[_BALL_OFF:_BALL_OFF + 3] = tok[0, :3]
            raw[_BALL_OFF + 3:_BALL_OFF + 6] = tok[0, 3:6]
            raw[_BALL_OFF + 6:_BALL_OFF + 9] = tok[0, 6:9]

            # Ego: pos(3) + vel(3) + quat(4) + ang_vel(3) + boost(1) + has_flip(1) + on_ground(1)
            raw[_EGO_OFF:_EGO_OFF + 3] = tok[1, :3]
            raw[_EGO_OFF + 3:_EGO_OFF + 6] = tok[1, 3:6]
            ego_euler = tok[1, 6:9].copy()
            yaw = ego_euler[0] * math.pi
            pitch = ego_euler[1] * math.pi
            roll = ego_euler[2] * math.pi
            ego_q = euler_to_quaternion_batch(
                np.array([yaw]), np.array([pitch]), np.array([roll]))[0]
            raw[_EGO_OFF + 6:_EGO_OFF + 10] = ego_q
            # Angular velocity from Euler finite differences
            ego_d_euler = (ego_euler - prev_ego_euler) * math.pi
            ego_ang_vel = np.clip(
                ego_d_euler / (MAX_ANG_VEL * (1.0 / 120.0)), -1.0, 1.0)
            raw[_EGO_OFF + 10:_EGO_OFF + 13] = ego_ang_vel
            raw[_EGO_OFF + 13] = tok[1, 9]  # boost
            ego_on_ground = float(tok[1, 2] < _ON_GROUND_Z_THRESH)
            raw[_EGO_OFF + 14] = ego_on_ground  # has_flip
            raw[_EGO_OFF + 15] = ego_on_ground  # on_ground

            # Opponent
            raw[_OPP_OFF:_OPP_OFF + 3] = tok[2, :3]
            raw[_OPP_OFF + 3:_OPP_OFF + 6] = tok[2, 3:6]
            opp_euler = tok[2, 6:9].copy()
            opp_q = euler_to_quaternion_batch(
                np.array([opp_euler[0] * math.pi]),
                np.array([opp_euler[1] * math.pi]),
                np.array([opp_euler[2] * math.pi]))[0]
            raw[_OPP_OFF + 6:_OPP_OFF + 10] = opp_q
            opp_d_euler = (opp_euler - prev_opp_euler) * math.pi
            opp_ang_vel = np.clip(
                opp_d_euler / (MAX_ANG_VEL * (1.0 / 120.0)), -1.0, 1.0)
            raw[_OPP_OFF + 10:_OPP_OFF + 13] = opp_ang_vel
            raw[_OPP_OFF + 13] = 0.0  # opp boost hidden
            opp_on_ground = float(tok[2, 2] < _ON_GROUND_Z_THRESH)
            raw[_OPP_OFF + 14] = opp_on_ground
            raw[_OPP_OFF + 15] = opp_on_ground

            # Boost pads: active flags only
            for i in range(6):
                raw[_PAD_OFF + i] = tok[3 + i, 3]

            # Game state
            raw[_GS_OFF:_GS_OFF + 3] = tok[9, :3]

            # Previous velocities
            raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 3] = prev_ball_vel
            raw[_PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3] = prev_ego_vel
            raw[_PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3] = prev_opp_vel

            # Previous angular velocities
            raw[_PREV_ANG_VEL_OFF:_PREV_ANG_VEL_OFF + 3] = prev_ball_ang_vel
            raw[_PREV_EGO_ANG_VEL_OFF:_PREV_EGO_ANG_VEL_OFF + 3] = prev_ego_ang_vel
            raw[_PREV_OPP_ANG_VEL_OFF:_PREV_OPP_ANG_VEL_OFF + 3] = prev_opp_ang_vel

            # Previous scalars
            raw[_PREV_SCALARS_OFF:_PREV_SCALARS_OFF + 3] = prev_ego_scalars
            raw[_PREV_OPP_SCALARS_OFF:_PREV_OPP_SCALARS_OFF + 3] = prev_opp_scalars

            raw_seq[t] = raw

            # Update prev values for next iteration
            prev_ball_vel = raw[_BALL_OFF + 3:_BALL_OFF + 6].copy()
            prev_ego_vel = raw[_EGO_OFF + 3:_EGO_OFF + 6].copy()
            prev_opp_vel = raw[_OPP_OFF + 3:_OPP_OFF + 6].copy()
            prev_ball_ang_vel = raw[_BALL_OFF + 6:_BALL_OFF + 9].copy()
            prev_ego_ang_vel = ego_ang_vel.copy()
            prev_opp_ang_vel = opp_ang_vel.copy()
            prev_ego_scalars = np.array(
                [raw[_EGO_OFF + 13], raw[_EGO_OFF + 14], raw[_EGO_OFF + 15]],
                dtype=np.float32)
            prev_opp_scalars = np.array(
                [raw[_OPP_OFF + 13], raw[_OPP_OFF + 14], raw[_OPP_OFF + 15]],
                dtype=np.float32)
            prev_ego_euler = ego_euler.copy()
            prev_opp_euler = opp_euler.copy()

        return raw_seq

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.windows[idx])
