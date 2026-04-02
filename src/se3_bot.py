"""
SE3Bot — RLBot Agent using SE(3) Spectral Fields
=================================================
Loads SE3Encoder + SE3Policy at startup, then on every tick:
  1. Extract raw state from GamePacket (positions, velocities, ang_vel, Euler→quaternion)
  2. Pack [raw_state | prev_coefficients] → SE3_OBS_DIM
  3. SE3Encoder.encode_for_policy → 26-dim physical summary
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
    _BALL_OFF, _EGO_OFF, _OPP_OFF, _PAD_OFF, _GS_OFF,
    _PREV_VEL_OFF, _PREV_EGO_VEL_OFF, _PREV_OPP_VEL_OFF,
    _PREV_ANG_VEL_OFF, _PREV_EGO_ANG_VEL_OFF, _PREV_OPP_ANG_VEL_OFF,
    _PREV_SCALARS_OFF, _PREV_OPP_SCALARS_OFF,
)
from se3_policy import SE3Policy

MODEL_DIR = _REPO / 'models'

_N_BIG_PADS = 6

# Approximate on_ground threshold (normalised z, ~17 uu / CEILING_Z)
_ON_GROUND_Z_THRESH = 17.5 / CEILING_Z


class SE3Bot(Bot):

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.encoder: SE3Encoder = None
        self.policy: SE3Policy = None
        self._prev_coeff: np.ndarray = None
        self._prev_ball_vel: np.ndarray = None
        self._prev_ego_vel: np.ndarray = None
        self._prev_opp_vel: np.ndarray = None
        self._prev_ball_ang_vel: np.ndarray = None
        self._prev_ego_ang_vel: np.ndarray = None
        self._prev_opp_ang_vel: np.ndarray = None
        self._prev_ego_scalars: np.ndarray = None  # (boost, has_flip, on_ground)
        self._prev_opp_scalars: np.ndarray = None

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
        self._prev_ego_vel = np.zeros(3, dtype=np.float32)
        self._prev_opp_vel = np.zeros(3, dtype=np.float32)
        self._prev_ball_ang_vel = np.zeros(3, dtype=np.float32)
        self._prev_ego_ang_vel = np.zeros(3, dtype=np.float32)
        self._prev_opp_ang_vel = np.zeros(3, dtype=np.float32)
        self._prev_ego_scalars = np.zeros(3, dtype=np.float32)
        self._prev_opp_scalars = np.zeros(3, dtype=np.float32)

    def get_output(self, packet: GamePacket) -> ControllerState:
        raw_state = self._extract_raw_state(packet)
        packed = pack_observation(raw_state, self._prev_coeff)

        with torch.no_grad():
            packed_t = torch.tensor(packed[np.newaxis], dtype=torch.float32)
            embed = self.encoder.encode_for_policy(packed_t)
            coeff = self.encoder(packed_t).numpy()[0]

        self._prev_coeff = coeff.copy()
        # Cache velocities
        self._prev_ball_vel = raw_state[_BALL_OFF + 3:_BALL_OFF + 6].copy()
        self._prev_ego_vel = raw_state[_EGO_OFF + 3:_EGO_OFF + 6].copy()
        self._prev_opp_vel = raw_state[_OPP_OFF + 3:_OPP_OFF + 6].copy()
        # Cache angular velocities
        self._prev_ball_ang_vel = raw_state[_BALL_OFF + 6:_BALL_OFF + 9].copy()
        self._prev_ego_ang_vel = raw_state[_EGO_OFF + 10:_EGO_OFF + 13].copy()
        self._prev_opp_ang_vel = raw_state[_OPP_OFF + 10:_OPP_OFF + 13].copy()
        # Cache scalars
        self._prev_ego_scalars = raw_state[_EGO_OFF + 13:_EGO_OFF + 16].copy()
        self._prev_opp_scalars = raw_state[_OPP_OFF + 13:_OPP_OFF + 16].copy()

        action, _ = self.policy.act(embed.numpy())
        return self._translate_controls(action)

    def _extract_raw_state(self, packet: GamePacket) -> np.ndarray:
        """Extract 74-dim raw state from GamePacket."""
        opp_idx = self.index ^ 1
        ball = packet.balls[0].physics
        own = packet.players[self.index].physics
        opp_phys = packet.players[opp_idx].physics
        own_player = packet.players[self.index]
        opp_player = packet.players[opp_idx]

        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)

        # Ball: pos(3) + vel(3) + ang_vel(3)
        raw[_BALL_OFF + 0] = ball.location.x / FIELD_X
        raw[_BALL_OFF + 1] = ball.location.y / FIELD_Y
        raw[_BALL_OFF + 2] = ball.location.z / CEILING_Z
        raw[_BALL_OFF + 3] = ball.velocity.x / MAX_VEL
        raw[_BALL_OFF + 4] = ball.velocity.y / MAX_VEL
        raw[_BALL_OFF + 5] = ball.velocity.z / MAX_VEL
        raw[_BALL_OFF + 6] = ball.angular_velocity.x / MAX_ANG_VEL
        raw[_BALL_OFF + 7] = ball.angular_velocity.y / MAX_ANG_VEL
        raw[_BALL_OFF + 8] = ball.angular_velocity.z / MAX_ANG_VEL

        # Ego: pos(3) + vel(3) + quat(4) + ang_vel(3) + boost(1) + has_flip(1) + on_ground(1)
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
        raw[_EGO_OFF + 10] = own.angular_velocity.x / MAX_ANG_VEL
        raw[_EGO_OFF + 11] = own.angular_velocity.y / MAX_ANG_VEL
        raw[_EGO_OFF + 12] = own.angular_velocity.z / MAX_ANG_VEL
        raw[_EGO_OFF + 13] = float(own_player.boost) / MAX_BOOST
        # on_ground: approximate from z position
        ego_on_ground = float(raw[_EGO_OFF + 2] < _ON_GROUND_Z_THRESH)
        raw[_EGO_OFF + 14] = ego_on_ground  # has_flip ≈ on_ground (conservative)
        raw[_EGO_OFF + 15] = ego_on_ground  # on_ground

        # Opponent: same layout
        raw[_OPP_OFF + 0] = opp_phys.location.x / FIELD_X
        raw[_OPP_OFF + 1] = opp_phys.location.y / FIELD_Y
        raw[_OPP_OFF + 2] = opp_phys.location.z / CEILING_Z
        raw[_OPP_OFF + 3] = opp_phys.velocity.x / MAX_VEL
        raw[_OPP_OFF + 4] = opp_phys.velocity.y / MAX_VEL
        raw[_OPP_OFF + 5] = opp_phys.velocity.z / MAX_VEL
        opp_q = euler_to_quaternion(
            float(opp_phys.rotation.yaw),
            float(opp_phys.rotation.pitch),
            float(opp_phys.rotation.roll),
        ).numpy()
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = opp_q
        raw[_OPP_OFF + 10] = opp_phys.angular_velocity.x / MAX_ANG_VEL
        raw[_OPP_OFF + 11] = opp_phys.angular_velocity.y / MAX_ANG_VEL
        raw[_OPP_OFF + 12] = opp_phys.angular_velocity.z / MAX_ANG_VEL
        raw[_OPP_OFF + 13] = 0.0  # opponent boost hidden
        opp_on_ground = float(raw[_OPP_OFF + 2] < _ON_GROUND_Z_THRESH)
        raw[_OPP_OFF + 14] = opp_on_ground  # has_flip
        raw[_OPP_OFF + 15] = opp_on_ground  # on_ground

        # Boost pads (6 big pads) — active flags only
        big_pads = getattr(packet, 'boost_pad_states', None)
        if big_pads is not None:
            for i in range(min(len(big_pads), _N_BIG_PADS)):
                raw[_PAD_OFF + i] = float(getattr(big_pads[i], 'is_active', 0))

        # Game state
        blue_score = float(packet.teams[0].score)
        orange_score = float(packet.teams[1].score)
        score_diff = (blue_score - orange_score) if self.index == 0 else (orange_score - blue_score)
        time_rem = float(getattr(packet.match_info, 'seconds_remaining', 0.0))
        overtime = float(getattr(packet.match_info, 'is_overtime', False))
        raw[_GS_OFF + 0] = np.clip(score_diff / MAX_SCORE, -1.0, 1.0)
        raw[_GS_OFF + 1] = np.clip(time_rem / MAX_TIME, 0.0, 1.0)
        raw[_GS_OFF + 2] = overtime

        # Previous velocities
        raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 3] = self._prev_ball_vel
        raw[_PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3] = self._prev_ego_vel
        raw[_PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3] = self._prev_opp_vel

        # Previous angular velocities
        raw[_PREV_ANG_VEL_OFF:_PREV_ANG_VEL_OFF + 3] = self._prev_ball_ang_vel
        raw[_PREV_EGO_ANG_VEL_OFF:_PREV_EGO_ANG_VEL_OFF + 3] = self._prev_ego_ang_vel
        raw[_PREV_OPP_ANG_VEL_OFF:_PREV_OPP_ANG_VEL_OFF + 3] = self._prev_opp_ang_vel

        # Previous scalars (boost, has_flip, on_ground)
        raw[_PREV_SCALARS_OFF:_PREV_SCALARS_OFF + 3] = self._prev_ego_scalars
        raw[_PREV_OPP_SCALARS_OFF:_PREV_OPP_SCALARS_OFF + 3] = self._prev_opp_scalars

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
