"""
SE(3) Spectral Field Representation
====================================
Probabilistic scene representation where each object is an SE(3) spectral
field — a frequency-domain function over position + quaternion space.

The field geometry (spatial frequencies k_spatial and quaternion bases) is
learned via backprop.  Coefficients carry temporal memory and are updated
online at 120 Hz via a residual rule.

Quaternion convention: (w, x, y, z), unit norm.  Double-cover is preserved
(q and -q produce different inner-product signs).  Euler→quaternion
conversion enforces w >= 0 for consistent sign from game observations.

Architecture
------------
    Env: raw_state (57) + prev_coefficients (128) = 185-dim obs
         ↓
    SE3Encoder.forward(packed_obs)       ← k_spatial, quaternions are nn.Parameters
         ↓
    128-dim updated coefficients         (differentiable w.r.t. field geometry)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── import normalisation constants from the existing encoder ─────────────────

from encoder import FIELD_X, FIELD_Y, CEILING_Z, MAX_VEL, MAX_ANG_VEL, MAX_BOOST

# ── constants ────────────────────────────────────────────────────────────────

OBJECTS: List[str] = [
    'ball',
    'ego',
    'team',
    'opponents',
    'stadium',
    'goal_team',
    'goal_opponents',
    'boost',
]

K = 8                              # spectral components per field
N_OBJECTS = len(OBJECTS)            # 8
COEFF_DIM = N_OBJECTS * K * 3 * 2  # 384  (x/y/z × real/imag per component)
RAW_STATE_DIM = 57                 # see layout table in plan
SE3_OBS_DIM = RAW_STATE_DIM + COEFF_DIM  # 441

# Object indices in OBJECTS list
_BALL = 0
_EGO = 1
_TEAM = 2
_OPP = 3
_STADIUM = 4
_GOAL_TEAM = 5
_GOAL_OPP = 6
_BOOST = 7

# Raw state offsets
_BALL_OFF = 0    # pos(3) + vel(3) = 6
_EGO_OFF = 6     # pos(3) + vel(3) + quat(4) + boost(1) = 11
_OPP_OFF = 17    # pos(3) + vel(3) + quat(4) = 10
_PAD_OFF = 27    # 6 pads × (pos(3) + active(1)) = 24
_GS_OFF = 51     # score_diff + time_rem + overtime = 3
_PREV_VEL_OFF = 54  # prev ball vel(3)

# Identity quaternion (w, x, y, z)
_QUAT_IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])

# Contact detection
CONTACT_THRESHOLD = 50.0
DT = 1.0 / 120.0  # 120 Hz tick


# ── quaternion utilities ─────────────────────────────────────────────────────

def euler_to_quaternion(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    """Convert Euler angles (radians, ZYX convention) to unit quaternion (w,x,y,z).

    Enforces w >= 0 (canonical hemisphere) so the spectral inner product
    gets a consistent sign from game observations.
    """
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = torch.tensor([w, x, y, z], dtype=torch.float32)
    if q[0] < 0:
        q = -q
    return q / q.norm()


def euler_to_quaternion_batch(yaw: np.ndarray, pitch: np.ndarray,
                              roll: np.ndarray) -> np.ndarray:
    """Numpy batch version for env use. Returns (N, 4) float32."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = np.stack([w, x, y, z], axis=-1).astype(np.float32)
    # Canonical hemisphere: flip if w < 0
    mask = q[..., 0] < 0
    q[mask] = -q[mask]
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return q / norms


def normalise_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalise quaternion(s) to unit length. Works on any shape (..., 4)."""
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def quaternion_inner(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Batched quaternion inner product.

    q1: (..., K, 4)   quaternion basis vectors
    q2: (..., 4)       observed quaternion (broadcast)
    returns: (..., K)  alignment per component, raw sign preserved
    """
    return (q1 * q2.unsqueeze(-2)).sum(dim=-1)


def quaternion_exponential(theta: float, q_hat: torch.Tensor) -> torch.Tensor:
    """e^(theta * q_hat) where q_hat is a pure unit quaternion [0, x, y, z]."""
    w = torch.tensor([math.cos(theta)])
    xyz = q_hat[1:] * math.sin(theta)
    return torch.cat([w, xyz])


# ── contact detection ────────────────────────────────────────────────────────

def detect_contact(vel_prev: torch.Tensor, vel_curr: torch.Tensor,
                   dt: float = DT,
                   threshold: float = CONTACT_THRESHOLD) -> torch.Tensor:
    """Detect contact via velocity discontinuity.

    Works batched: vel_prev/vel_curr shape (..., 3).
    Returns bool tensor of shape (...).
    """
    dv = (vel_curr - vel_prev) / dt
    return dv.norm(dim=-1) > threshold


def detect_contact_np(vel_prev: np.ndarray, vel_curr: np.ndarray,
                      dt: float = DT,
                      threshold: float = CONTACT_THRESHOLD) -> np.ndarray:
    """Numpy version for env use."""
    dv = (vel_curr - vel_prev) / dt
    return np.linalg.norm(dv, axis=-1) > threshold


# ── SE3 Encoder (nn.Module with learned field geometry) ──────────────────────

class SE3Encoder(nn.Module):
    """
    Learned SE(3) spectral field encoder.

    Parameters (learned via backprop):
        k_spatial:   [N_OBJECTS, K, 3]  spatial frequency vectors
        quaternions: [N_OBJECTS, K, 4]  orientation basis (unit quaternions)
        update_lr:   [N_OBJECTS]        per-object coefficient update rate

    Forward pass:
        Input:  (batch, 185) = [raw_state (57) | prev_coefficients (128)]
        Output: (batch, 128) = updated coefficients (differentiable w.r.t. params)

    The coefficient update is a single-step residual rule, providing a
    gradient path from the policy loss to the field geometry via truncated
    BPTT (depth 1).
    """

    def __init__(self):
        super().__init__()

        # Learned spatial frequencies — initialised small (low frequency)
        self.k_spatial = nn.Parameter(
            torch.randn(N_OBJECTS, K, 3) * 0.1)

        # Learned quaternion basis — normalised after each optimiser step
        _q = torch.randn(N_OBJECTS, K, 4)
        _q = _q / _q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.quaternions = nn.Parameter(_q)

        # Learned per-object update rate (log-space for positivity)
        self.log_lr = nn.Parameter(torch.full((N_OBJECTS,), math.log(0.05)))

    @property
    def d_model(self) -> int:
        """Output dimensionality (for interface compatibility)."""
        return COEFF_DIM

    def forward(self, packed_obs: torch.Tensor) -> torch.Tensor:
        """
        packed_obs: (batch, 185)
        returns:    (batch, 128) updated coefficients
        """
        batch = packed_obs.shape[0]
        device = packed_obs.device

        # Split raw state and previous coefficients
        raw = packed_obs[:, :RAW_STATE_DIM]               # (batch, 57)
        prev_coeff = packed_obs[:, RAW_STATE_DIM:]         # (batch, 384)
        prev_coeff = prev_coeff.reshape(batch, N_OBJECTS, K, 3, 2)

        # Effective learning rates (positive via exp)
        lr = torch.exp(self.log_lr).unsqueeze(0)           # (1, N_OBJECTS)

        # Normalise quaternion basis (projection, no grad break)
        q_basis = self.quaternions / self.quaternions.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)          # (N_OBJECTS, K, 4)

        # Parse raw state into per-object positions and orientations
        ball_pos = raw[:, _BALL_OFF:_BALL_OFF + 3]         # (batch, 3)
        ball_vel = raw[:, _BALL_OFF + 3:_BALL_OFF + 6]     # (batch, 3)
        ego_pos = raw[:, _EGO_OFF:_EGO_OFF + 3]
        ego_quat = raw[:, _EGO_OFF + 6:_EGO_OFF + 10]     # (batch, 4)
        opp_pos = raw[:, _OPP_OFF:_OPP_OFF + 3]
        opp_quat = raw[:, _OPP_OFF + 6:_OPP_OFF + 10]
        prev_ball_vel = raw[:, _PREV_VEL_OFF:_PREV_VEL_OFF + 3]

        # Identity quaternion for objects without orientation
        id_q = _QUAT_IDENTITY.to(device).unsqueeze(0).expand(batch, 4)

        # Assemble per-object positions and quaternions via torch.cat
        # (no in-place indexed assignment — preserves autograd graph)

        # Boost: mean position of active pads
        pad_data = raw[:, _PAD_OFF:_PAD_OFF + 24].reshape(batch, 6, 4)
        pad_pos = pad_data[:, :, :3]             # (batch, 6, 3)
        pad_active = pad_data[:, :, 3:4]         # (batch, 6, 1)
        active_sum = pad_active.sum(dim=1).clamp(min=1.0)  # (batch, 1)
        boost_pos = (pad_pos * pad_active).sum(dim=1) / active_sum  # (batch, 3)

        zeros3 = torch.zeros(batch, 3, device=device)
        goal_team_pos = torch.zeros(batch, 3, device=device)
        goal_team_pos = goal_team_pos + torch.tensor([0.0, -1.0, 0.0], device=device)
        goal_opp_pos = torch.zeros(batch, 3, device=device)
        goal_opp_pos = goal_opp_pos + torch.tensor([0.0, 1.0, 0.0], device=device)

        # positions: (batch, N_OBJECTS, 3) — ordered by OBJECTS list
        positions = torch.stack([
            ball_pos,       # _BALL
            ego_pos,        # _EGO
            ego_pos,        # _TEAM (1v1: team = ego)
            opp_pos,        # _OPP
            zeros3,         # _STADIUM (centred at origin)
            goal_team_pos,  # _GOAL_TEAM
            goal_opp_pos,   # _GOAL_OPP
            boost_pos,      # _BOOST
        ], dim=1)           # (batch, 8, 3)

        # orientations: (batch, N_OBJECTS, 4)
        orientations = torch.stack([
            id_q,           # _BALL
            ego_quat,       # _EGO
            ego_quat,       # _TEAM
            opp_quat,       # _OPP
            id_q,           # _STADIUM
            id_q,           # _GOAL_TEAM
            id_q,           # _GOAL_OPP
            id_q,           # _BOOST
        ], dim=1)           # (batch, 8, 4)

        # ── contact detection (ball) ─────────────────────────────────────
        contact = detect_contact(prev_ball_vel, ball_vel)   # (batch,)

        # ── spectral field update (batched, differentiable) ──────────────
        # Spatial components: cos (real) and sin (imaginary)
        # k_spatial: (N_OBJECTS, K, 3), positions: (batch, N_OBJECTS, 3)
        # -> phase, spatial_cos, spatial_sin: (batch, N_OBJECTS, K)
        phase = torch.einsum('okd,bod->bok', self.k_spatial, positions)
        spatial_cos = torch.cos(phase)
        spatial_sin = torch.sin(phase)

        # Orientation component: quaternion inner product
        # q_basis: (N_OBJECTS, K, 4), orientations: (batch, N_OBJECTS, 4)
        # -> orient: (batch, N_OBJECTS, K)
        orient = torch.einsum('okd,bod->bok', q_basis, orientations)

        # Target: full 3D position vector (batch, N_OBJECTS, 3)
        target = positions

        # Field evaluation: sum over K components → (batch, N_OBJECTS, 3)
        # basis: (batch, N_OBJECTS, K) unsqueeze(-1) → (batch, N_OBJECTS, K, 1)
        # prev_coeff[..., 0]: (batch, N_OBJECTS, K, 3)
        basis_cos = (spatial_cos * orient).unsqueeze(-1)   # (batch, N_OBJECTS, K, 1)
        basis_sin = (spatial_sin * orient).unsqueeze(-1)
        predicted_real = (basis_cos * prev_coeff[:, :, :, :, 0]).sum(dim=2)  # (batch, N_OBJECTS, 3)
        predicted_imag = (basis_sin * prev_coeff[:, :, :, :, 1]).sum(dim=2)
        residual_real = target - predicted_real            # (batch, N_OBJECTS, 3)
        residual_imag = target - predicted_imag

        # Delta: outer product of basis (K) × residual (3) → (batch, N_OBJECTS, K, 3)
        # lr: (1, N_OBJECTS) → (1, N_OBJECTS, 1, 1) after two unsqueezes
        scaled_res_real = (lr.unsqueeze(-1) * residual_real).unsqueeze(2)  # (batch, N_OBJECTS, 1, 3)
        scaled_res_imag = (lr.unsqueeze(-1) * residual_imag).unsqueeze(2)
        delta_real = scaled_res_real * (spatial_cos * orient).unsqueeze(-1)  # (batch, N_OBJECTS, K, 3)
        delta_imag = scaled_res_imag * (spatial_sin * orient).unsqueeze(-1)

        new_coeff_real = prev_coeff[:, :, :, :, 0] + delta_real   # (batch, N_OBJECTS, K, 3)
        new_coeff_imag = prev_coeff[:, :, :, :, 1] + delta_imag

        # Reset ball coefficients on contact (mask multiply, no in-place)
        # contact_keep: 0.0 if contact, 1.0 if no contact → (batch, 1)
        contact_keep = (1.0 - contact.float()).unsqueeze(-1)

        # obj_keep: (batch, N_OBJECTS, 1) — ball uses contact_keep, others 1
        obj_keep_list = [
            contact_keep.unsqueeze(1) if i == _BALL
            else torch.ones(batch, 1, 1, device=device)
            for i in range(N_OBJECTS)
        ]
        obj_keep = torch.cat(obj_keep_list, dim=1)  # (batch, N_OBJECTS, 1)

        # obj_keep: (batch, N_OBJECTS, 1) → broadcast over K and 3 axes
        obj_keep = obj_keep.unsqueeze(-1)                  # (batch, N_OBJECTS, 1, 1)
        new_coeff_real = new_coeff_real * obj_keep
        new_coeff_imag = new_coeff_imag * obj_keep

        # Stack and flatten: (batch, N_OBJECTS, K, 3, 2) → (batch, 384)
        new_coeff = torch.stack([new_coeff_real, new_coeff_imag], dim=-1)
        return new_coeff.reshape(batch, COEFF_DIM)

    @torch.no_grad()
    def normalise_quaternions_(self) -> None:
        """Project quaternion parameters back to unit sphere. Call after optimizer.step()."""
        self.quaternions.data = normalise_quaternion(self.quaternions.data)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(
            torch.load(path, map_location='cpu', weights_only=True))

    @classmethod
    def load_from(cls, path: str) -> 'SE3Encoder':
        model = cls()
        model.load(path)
        return model


# ── numpy helpers for env-side coefficient updates (no grad) ─────────────────

def make_initial_coefficients() -> np.ndarray:
    """Zero-initialised coefficients for episode start."""
    return np.zeros(COEFF_DIM, dtype=np.float32)


def update_coefficients_np(
    k_spatial: np.ndarray,      # (N_OBJECTS, K, 3)
    quaternions: np.ndarray,    # (N_OBJECTS, K, 4) unit
    lr: np.ndarray,             # (N_OBJECTS,)
    prev_coeff: np.ndarray,     # (COEFF_DIM,) = 128
    raw_state: np.ndarray,      # (RAW_STATE_DIM,) = 57
) -> np.ndarray:
    """Non-differentiable coefficient update for env stepping.

    Mirrors SE3Encoder.forward logic but in numpy for subprocess envs.
    Returns updated coefficients (128,).
    """
    coeff = prev_coeff.reshape(N_OBJECTS, K, 3, 2).copy()

    # Parse raw state
    ball_pos = raw_state[_BALL_OFF:_BALL_OFF + 3]
    ball_vel = raw_state[_BALL_OFF + 3:_BALL_OFF + 6]
    ego_pos = raw_state[_EGO_OFF:_EGO_OFF + 3]
    ego_quat = raw_state[_EGO_OFF + 6:_EGO_OFF + 10]
    opp_pos = raw_state[_OPP_OFF:_OPP_OFF + 3]
    opp_quat = raw_state[_OPP_OFF + 6:_OPP_OFF + 10]
    prev_ball_vel = raw_state[_PREV_VEL_OFF:_PREV_VEL_OFF + 3]

    id_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Per-object positions and orientations
    positions = np.zeros((N_OBJECTS, 3), dtype=np.float32)
    orientations = np.tile(id_q, (N_OBJECTS, 1))

    positions[_BALL] = ball_pos
    positions[_EGO] = ego_pos
    orientations[_EGO] = ego_quat
    positions[_TEAM] = ego_pos
    orientations[_TEAM] = ego_quat
    positions[_OPP] = opp_pos
    orientations[_OPP] = opp_quat
    positions[_GOAL_TEAM, 1] = -1.0
    positions[_GOAL_OPP, 1] = 1.0

    # Boost: mean of active pads
    pad_data = raw_state[_PAD_OFF:_PAD_OFF + 24].reshape(6, 4)
    pad_pos = pad_data[:, :3]
    pad_active = pad_data[:, 3:4]
    active_sum = max(pad_active.sum(), 1.0)
    positions[_BOOST] = (pad_pos * pad_active).sum(axis=0) / active_sum

    # Contact detection
    contact = detect_contact_np(prev_ball_vel, ball_vel)

    # Update each object
    for obj in range(N_OBJECTS):
        pos = positions[obj]
        ori = orientations[obj]

        # Spatial: cos (real) and sin (imaginary) bases
        phase = k_spatial[obj] @ pos          # (K,)
        spatial_cos = np.cos(phase)
        spatial_sin = np.sin(phase)

        # Orientation: dot product  shape (K,)
        orient = (quaternions[obj] * ori).sum(axis=-1)

        target = pos                                     # (3,) — x, y, z

        # Real update: (K,) @ (K, 3) → (3,); outer (K,) × (3,) → (K, 3)
        predicted_real = (spatial_cos * orient) @ coeff[obj, :, :, 0]
        residual_real = target - predicted_real
        coeff[obj, :, :, 0] += np.outer(lr[obj] * spatial_cos * orient, residual_real)

        # Imaginary update
        predicted_imag = (spatial_sin * orient) @ coeff[obj, :, :, 1]
        residual_imag = target - predicted_imag
        coeff[obj, :, :, 1] += np.outer(lr[obj] * spatial_sin * orient, residual_imag)

    # Reset ball on contact
    if contact:
        coeff[_BALL] = 0.0

    return coeff.ravel()


def pack_observation(raw_state: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Pack raw state and coefficients into the 185-dim observation."""
    return np.concatenate([raw_state, coefficients]).astype(np.float32)
