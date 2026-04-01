"""
SE3Bot — RLBot Agent using SE(3) Spectral Fields
=================================================
Loads SE3Encoder + SE3Policy at startup, then on every tick:
  1. Extract raw state from GamePacket (positions, velocities, Euler→quaternion)
  2. Pack [raw_state | prev_coefficients] → 185-dim
  3. SE3Encoder.forward → 128-dim updated coefficients
  4. SE3Policy.forward → 8-float action
  5. Translate to ControllerState
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # prevent CUDA DLL load in bot subprocess

import numpy as np
import torch

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

_SRC = Path(__file__).parent
_REPO = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import FIELD_X, FIELD_Y, CEILING_Z, MAX_VEL, MAX_ANG_VEL, MAX_BOOST, MAX_SCORE, MAX_TIME
from se3_field import (
    SE3Encoder, SE3_OBS_DIM, RAW_STATE_DIM, COEFF_DIM,
    euler_to_quaternion, make_initial_coefficients, pack_observation,
    _BALL_OFF, _EGO_OFF, _OPP_OFF, _PAD_OFF, _GS_OFF, _PREV_VEL_OFF,
)
from se3_policy import SE3Policy

MODEL_DIR = _REPO / 'models'

_N_BIG_PADS = 6


class SE3Bot(Bot):

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.encoder: SE3Encoder = None
        self.policy: SE3Policy = None
        self._prev_coeff: np.ndarray = None
        self._prev_ball_vel: np.ndarray = None

    def initialize(self) -> None:
        enc_path = MODEL_DIR / 'se3_encoder.pt'
        if enc_path.exists():
            self.encoder = SE3Encoder.load_from(str(enc_path))
        else:
            self.encoder = SE3Encoder()

        self.policy = SE3Policy()
        head_path = MODEL_DIR / 'se3_policy.pt'
        if head_path.exists():
            self.policy.load(str(head_path))

        self.encoder.eval()
        self.policy.eval()

        self._prev_coeff = make_initial_coefficients()
        self._prev_ball_vel = np.zeros(3, dtype=np.float32)

    def get_output(self, packet: GamePacket) -> ControllerState:
        raw_state = self._extract_raw_state(packet)
        packed = pack_observation(raw_state, self._prev_coeff)

        with torch.no_grad():
            packed_t = torch.tensor(packed[np.newaxis], dtype=torch.float32)
            coeff = self.encoder(packed_t).numpy()[0]  # (128,)

        self._prev_coeff = coeff.copy()
        self._prev_ball_vel = raw_state[_BALL_OFF + 3:_BALL_OFF + 6].copy()

        action, _ = self.policy.act(coeff[np.newaxis])
        return self._translate_controls(action)

    def _extract_raw_state(self, packet: GamePacket) -> np.ndarray:
        """Extract 57-dim raw state from GamePacket."""
        opp_idx = self.index ^ 1
        ball = packet.balls[0].physics
        own = packet.players[self.index].physics
        opp = packet.players[opp_idx].physics
        own_boost = float(packet.players[self.index].boost)

        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)

        # Ball: pos(3) + vel(3)
        raw[_BALL_OFF + 0] = ball.location.x / FIELD_X
        raw[_BALL_OFF + 1] = ball.location.y / FIELD_Y
        raw[_BALL_OFF + 2] = ball.location.z / CEILING_Z
        raw[_BALL_OFF + 3] = ball.velocity.x / MAX_VEL
        raw[_BALL_OFF + 4] = ball.velocity.y / MAX_VEL
        raw[_BALL_OFF + 5] = ball.velocity.z / MAX_VEL

        # Ego: pos(3) + vel(3) + quat(4) + boost(1)
        raw[_EGO_OFF + 0] = own.location.x / FIELD_X
        raw[_EGO_OFF + 1] = own.location.y / FIELD_Y
        raw[_EGO_OFF + 2] = own.location.z / CEILING_Z
        raw[_EGO_OFF + 3] = own.velocity.x / MAX_VEL
        raw[_EGO_OFF + 4] = own.velocity.y / MAX_VEL
        raw[_EGO_OFF + 5] = own.velocity.z / MAX_VEL
        ego_q = euler_to_quaternion(
            float(own.rotation.yaw),
            float(own.rotation.pitch),
            float(own.rotation.roll),
        ).numpy()
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = ego_q
        raw[_EGO_OFF + 10] = own_boost / MAX_BOOST

        # Opponent: pos(3) + vel(3) + quat(4)
        raw[_OPP_OFF + 0] = opp.location.x / FIELD_X
        raw[_OPP_OFF + 1] = opp.location.y / FIELD_Y
        raw[_OPP_OFF + 2] = opp.location.z / CEILING_Z
        raw[_OPP_OFF + 3] = opp.velocity.x / MAX_VEL
        raw[_OPP_OFF + 4] = opp.velocity.y / MAX_VEL
        raw[_OPP_OFF + 5] = opp.velocity.z / MAX_VEL
        opp_q = euler_to_quaternion(
            float(opp.rotation.yaw),
            float(opp.rotation.pitch),
            float(opp.rotation.roll),
        ).numpy()
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = opp_q

        # Boost pads (6 big pads)
        big_pads = getattr(packet, 'boost_pad_states', None)
        if big_pads is not None:
            for i in range(min(len(big_pads), _N_BIG_PADS)):
                off = _PAD_OFF + i * 4
                pad = big_pads[i]
                raw[off + 0] = getattr(pad, 'x', 0.0) / FIELD_X
                raw[off + 1] = getattr(pad, 'y', 0.0) / FIELD_Y
                raw[off + 2] = getattr(pad, 'z', 0.0) / CEILING_Z
                raw[off + 3] = float(getattr(pad, 'is_active', 0))

        # Game state
        blue_score = float(packet.teams[0].score)
        orange_score = float(packet.teams[1].score)
        score_diff = (blue_score - orange_score) if self.index == 0 else (orange_score - blue_score)
        time_rem = float(getattr(packet.match_info, 'seconds_remaining', 0.0))
        overtime = float(getattr(packet.match_info, 'is_overtime', False))
        raw[_GS_OFF + 0] = np.clip(score_diff / MAX_SCORE, -1.0, 1.0)
        raw[_GS_OFF + 1] = np.clip(time_rem / MAX_TIME, 0.0, 1.0)
        raw[_GS_OFF + 2] = overtime

        # Previous ball velocity (for contact detection)
        raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 3] = self._prev_ball_vel

        return raw

    def _translate_controls(self, action: np.ndarray) -> ControllerState:
        ctrl = ControllerState()
        ctrl.throttle = float(np.clip(action[0], -1.0, 1.0))
        ctrl.steer = float(np.clip(action[1], -1.0, 1.0))
        ctrl.pitch = float(np.clip(action[2], -1.0, 1.0))
        ctrl.yaw = float(np.clip(action[3], -1.0, 1.0))
        ctrl.roll = float(np.clip(action[4], -1.0, 1.0))
        ctrl.jump = bool(action[5] >= 0.5)
        ctrl.boost = bool(action[6] >= 0.5)
        ctrl.handbrake = bool(action[7] >= 0.5)
        return ctrl
