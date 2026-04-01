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
    Env: raw_state (63) + prev_coefficients (576) = 639-dim obs
         ↓
    SE3Encoder.forward(packed_obs)       ← k_spatial, quaternions are nn.Parameters
         ↓
    576-dim updated coefficients         (internal, for env persistence)

    SE3Encoder.encode_for_policy(packed_obs)
         ↓
    82-dim physical summary              (pos/vel/accel + context, LayerNorm'd)

Coefficient layout (internal, 3 channels per spectral component):
    channel 0: position real (cosine)
    channel 1: position imaginary (sine)
    channel 2: acceleration residual (observed_accel - gravity)
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
N_CHANNELS = 3                     # real, imaginary, acceleration residual
COEFF_DIM = N_OBJECTS * K * 3 * N_CHANNELS  # 576
RAW_STATE_DIM = 63                 # 57 base + prev ego vel(3) + prev opp vel(3)
EMBED_DIM = 82                     # physical summary for policy
SE3_OBS_DIM = RAW_STATE_DIM + COEFF_DIM  # 639
COEFF_CLIP = 10.0                  # clamp coefficients to [-CLIP, CLIP]

# Gravity: normalised per-tick velocity change on z-axis
# Rocket League gravity ≈ 650 uu/s², MAX_VEL=2300, tick=1/120
GRAVITY_DV_Z = -650.0 / (2300.0 * 120.0)  # ≈ -0.002355 per tick

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
_PREV_EGO_VEL_OFF = 57  # prev ego vel(3)
_PREV_OPP_VEL_OFF = 60  # prev opp vel(3)

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
        log_lr:      [N_OBJECTS]        per-object coefficient update rate
        W_interact:  [N_OBJECTS, N_OBJECTS]  pairwise coupling
        output_norm: LayerNorm(EMBED_DIM)  normalises physical summary

    forward(packed_obs):
        Input:  (batch, SE3_OBS_DIM=639)
        Output: (batch, COEFF_DIM=576) updated coefficients (for env persistence)

    encode_for_policy(packed_obs):
        Input:  (batch, SE3_OBS_DIM=639)
        Output: (batch, EMBED_DIM=82) normalised physical summary
    """

    def __init__(self):
        super().__init__()

        # Learned spatial frequencies — unit-scale so cos/sin have real variance
        self.k_spatial = nn.Parameter(
            torch.randn(N_OBJECTS, K, 3) * 1.0)

        # Learned quaternion basis — normalised after each optimiser step
        _q = torch.randn(N_OBJECTS, K, 4)
        _q = _q / _q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.quaternions = nn.Parameter(_q)

        # Learned per-object update rate (log-space for positivity)
        self.log_lr = nn.Parameter(torch.full((N_OBJECTS,), math.log(0.05)))

        # Learned pairwise interaction weights (zero-init = no coupling at start)
        self.W_interact = nn.Parameter(torch.zeros(N_OBJECTS, N_OBJECTS))

        # LayerNorm on physical summary output
        self.output_norm = nn.LayerNorm(EMBED_DIM)

    @property
    def d_model(self) -> int:
        """Output dimensionality for policy (EMBED_DIM)."""
        return EMBED_DIM

    def _update_coefficients(self, packed_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core coefficient update. Returns (new_coeff_flat, basis_cos, basis_sin, raw, new_coeff_5d).

        new_coeff_flat: (batch, COEFF_DIM=576) for env persistence
        basis_cos/sin:  (batch, N_OBJECTS, K, 1) for reconstruction
        raw:            (batch, RAW_STATE_DIM) parsed raw state
        new_coeff_5d:   (batch, N_OBJECTS, K, 3, 3) structured coefficients
        """
        batch = packed_obs.shape[0]
        device = packed_obs.device

        # Split raw state and previous coefficients
        raw = packed_obs[:, :RAW_STATE_DIM]                # (batch, 63)
        prev_coeff = packed_obs[:, RAW_STATE_DIM:]          # (batch, 576)
        prev_coeff = prev_coeff.reshape(batch, N_OBJECTS, K, 3, N_CHANNELS)

        # Effective learning rates (positive via exp)
        lr = torch.exp(self.log_lr).unsqueeze(0)           # (1, N_OBJECTS)

        # Normalise quaternion basis (projection, no grad break)
        q_basis = self.quaternions / self.quaternions.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)          # (N_OBJECTS, K, 4)

        # Parse raw state into per-object positions and orientations
        ball_pos = raw[:, _BALL_OFF:_BALL_OFF + 3]         # (batch, 3)
        ball_vel = raw[:, _BALL_OFF + 3:_BALL_OFF + 6]     # (batch, 3)
        ego_pos = raw[:, _EGO_OFF:_EGO_OFF + 3]
        ego_vel = raw[:, _EGO_OFF + 3:_EGO_OFF + 6]
        ego_quat = raw[:, _EGO_OFF + 6:_EGO_OFF + 10]     # (batch, 4)
        opp_pos = raw[:, _OPP_OFF:_OPP_OFF + 3]
        opp_vel = raw[:, _OPP_OFF + 3:_OPP_OFF + 6]
        opp_quat = raw[:, _OPP_OFF + 6:_OPP_OFF + 10]
        prev_ball_vel = raw[:, _PREV_VEL_OFF:_PREV_VEL_OFF + 3]
        prev_ego_vel = raw[:, _PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3]
        prev_opp_vel = raw[:, _PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3]

        # Identity quaternion for objects without orientation
        id_q = _QUAT_IDENTITY.to(device).unsqueeze(0).expand(batch, 4)

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

        # ── spectral basis computation ───────────────────────────────────
        phase = torch.einsum('okd,bod->bok', self.k_spatial, positions)
        spatial_cos = torch.cos(phase)
        spatial_sin = torch.sin(phase)
        orient = torch.einsum('okd,bod->bok', q_basis, orientations)

        basis_cos = (spatial_cos * orient).unsqueeze(-1)   # (batch, N_OBJECTS, K, 1)
        basis_sin = (spatial_sin * orient).unsqueeze(-1)

        # ── Position channels (0=real, 1=imag): complex LMS ─────────────
        target = positions                                  # (batch, N_OBJECTS, 3)
        coeff_a = prev_coeff[:, :, :, :, 0]                # (batch, N_OBJECTS, K, 3)
        coeff_b = prev_coeff[:, :, :, :, 1]
        predicted = (basis_cos * coeff_a - basis_sin * coeff_b).sum(dim=2)

        residual = target - predicted                       # (batch, N_OBJECTS, 3)

        # Cross-object coupling on position residuals
        W_eff = torch.eye(N_OBJECTS, device=device) + self.W_interact
        residual = torch.einsum('ij,bjd->bid', W_eff, residual)

        # LMS update for position channels
        scaled_res = (lr.unsqueeze(-1) * residual).unsqueeze(2)
        delta_real = scaled_res * basis_cos
        delta_imag = -scaled_res * basis_sin

        new_coeff_real = coeff_a + delta_real
        new_coeff_imag = coeff_b + delta_imag

        # ── Acceleration channel (2): gravity-residual LMS ───────────────
        # Compute per-tick delta_v for moveable objects, subtract gravity
        gravity_vec = torch.tensor([0.0, 0.0, GRAVITY_DV_Z], device=device)
        # delta_v = vel[t] - vel[t-1]
        ball_dv = ball_vel - prev_ball_vel - gravity_vec
        ego_dv = ego_vel - prev_ego_vel - gravity_vec
        opp_dv = opp_vel - prev_opp_vel - gravity_vec

        # Build per-object accel target: (batch, N_OBJECTS, 3)
        # Static objects get zero accel target
        accel_target = torch.zeros(batch, N_OBJECTS, 3, device=device)
        # Use torch.cat approach for autograd safety
        accel_list = [
            ball_dv.unsqueeze(1),   # _BALL
            ego_dv.unsqueeze(1),    # _EGO
            ego_dv.unsqueeze(1),    # _TEAM = ego in 1v1
            opp_dv.unsqueeze(1),    # _OPP
            zeros3.unsqueeze(1),    # _STADIUM
            zeros3.unsqueeze(1),    # _GOAL_TEAM
            zeros3.unsqueeze(1),    # _GOAL_OPP
            zeros3.unsqueeze(1),    # _BOOST
        ]
        accel_target = torch.cat(accel_list, dim=1)  # (batch, 8, 3)

        coeff_c = prev_coeff[:, :, :, :, 2]           # (batch, N_OBJECTS, K, 3)
        accel_predicted = (basis_cos * coeff_c).sum(dim=2)  # (batch, N_OBJECTS, 3)
        accel_residual = accel_target - accel_predicted

        # Cross-object coupling on accel residuals (same W_interact)
        accel_residual = torch.einsum('ij,bjd->bid', W_eff, accel_residual)

        scaled_accel_res = (lr.unsqueeze(-1) * accel_residual).unsqueeze(2)
        delta_accel = scaled_accel_res * basis_cos
        new_coeff_accel = coeff_c + delta_accel

        # ── Contact reset: zero all channels for ball ────────────────────
        contact_keep = (1.0 - contact.float()).unsqueeze(-1)
        obj_keep_list = [
            contact_keep.unsqueeze(1) if i == _BALL
            else torch.ones(batch, 1, 1, device=device)
            for i in range(N_OBJECTS)
        ]
        obj_keep = torch.cat(obj_keep_list, dim=1).unsqueeze(-1)  # (batch, N_OBJECTS, 1, 1)
        new_coeff_real = new_coeff_real * obj_keep
        new_coeff_imag = new_coeff_imag * obj_keep
        new_coeff_accel = new_coeff_accel * obj_keep

        # Stack all 3 channels: (batch, N_OBJECTS, K, 3, 3) → (batch, 576)
        new_coeff = torch.stack([new_coeff_real, new_coeff_imag, new_coeff_accel], dim=-1)
        new_coeff = torch.clamp(new_coeff, -COEFF_CLIP, COEFF_CLIP)

        return new_coeff.reshape(batch, COEFF_DIM), basis_cos, basis_sin, raw, new_coeff

    def forward(self, packed_obs: torch.Tensor) -> torch.Tensor:
        """Update coefficients (for env persistence / buffer storage).

        packed_obs: (batch, SE3_OBS_DIM=639)
        returns:    (batch, COEFF_DIM=576) updated coefficients
        """
        coeff_flat, _, _, _, _ = self._update_coefficients(packed_obs)
        return coeff_flat

    def encode_for_policy(self, packed_obs: torch.Tensor) -> torch.Tensor:
        """Update coefficients and return physical summary for policy.

        packed_obs: (batch, SE3_OBS_DIM=639)
        returns:    (batch, EMBED_DIM=82) normalised physical summary
        """
        coeff_flat, basis_cos, basis_sin, raw, new_coeff_5d = self._update_coefficients(packed_obs)
        batch = packed_obs.shape[0]
        device = packed_obs.device

        # new_coeff_5d: (batch, N_OBJECTS, K, 3, 3)
        ca = new_coeff_5d[:, :, :, :, 0]   # real
        cb = new_coeff_5d[:, :, :, :, 1]   # imag
        cc = new_coeff_5d[:, :, :, :, 2]   # accel

        # Reconstruct position: Re[f] = Σ_k (a·cos - b·sin)
        pos_recon = (basis_cos * ca - basis_sin * cb).sum(dim=2)   # (batch, 8, 3)
        # Reconstruct velocity: Im[f] = Σ_k (a·sin + b·cos)
        vel_recon = (basis_sin * ca + basis_cos * cb).sum(dim=2)   # (batch, 8, 3)
        # Reconstruct accel residual: Σ_k (c·cos)
        accel_recon = (basis_cos * cc).sum(dim=2)                  # (batch, 8, 3)

        # Game context from raw state
        boost = raw[:, _EGO_OFF + 10:_EGO_OFF + 11]   # (batch, 1)
        game_state = raw[:, _GS_OFF:_GS_OFF + 3]      # (batch, 3) score_diff, time, overtime

        # Pad active flags
        pad_data = raw[:, _PAD_OFF:_PAD_OFF + 24].reshape(batch, 6, 4)
        pad_active = pad_data[:, :, 3]                 # (batch, 6)

        # Assemble: 24 + 24 + 24 + 1 + 3 + 6 = 82
        embed = torch.cat([
            pos_recon.reshape(batch, -1),     # 24
            vel_recon.reshape(batch, -1),     # 24
            accel_recon.reshape(batch, -1),   # 24
            boost,                             # 1
            game_state,                        # 3
            pad_active,                        # 6
        ], dim=-1)                             # (batch, 82)

        return self.output_norm(embed)

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
    prev_coeff: np.ndarray,     # (COEFF_DIM,) = 576
    raw_state: np.ndarray,      # (RAW_STATE_DIM,) = 63
    W_interact: Optional[np.ndarray] = None,  # (N_OBJECTS, N_OBJECTS) or None
) -> np.ndarray:
    """Non-differentiable coefficient update for env stepping.

    Mirrors SE3Encoder._update_coefficients logic in numpy.
    3 channels: real (pos cos), imag (pos sin), accel residual.
    Returns updated coefficients (576,).
    """
    coeff = prev_coeff.reshape(N_OBJECTS, K, 3, N_CHANNELS).copy()

    # Parse raw state
    ball_pos = raw_state[_BALL_OFF:_BALL_OFF + 3]
    ball_vel = raw_state[_BALL_OFF + 3:_BALL_OFF + 6]
    ego_pos = raw_state[_EGO_OFF:_EGO_OFF + 3]
    ego_vel = raw_state[_EGO_OFF + 3:_EGO_OFF + 6]
    ego_quat = raw_state[_EGO_OFF + 6:_EGO_OFF + 10]
    opp_pos = raw_state[_OPP_OFF:_OPP_OFF + 3]
    opp_vel = raw_state[_OPP_OFF + 3:_OPP_OFF + 6]
    opp_quat = raw_state[_OPP_OFF + 6:_OPP_OFF + 10]
    prev_ball_vel = raw_state[_PREV_VEL_OFF:_PREV_VEL_OFF + 3]
    prev_ego_vel = raw_state[_PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3]
    prev_opp_vel = raw_state[_PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3]

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

    # ── Acceleration targets (gravity residual) ──────────────────────
    gravity_vec = np.array([0.0, 0.0, GRAVITY_DV_Z], dtype=np.float32)
    accel_targets = np.zeros((N_OBJECTS, 3), dtype=np.float32)
    accel_targets[_BALL] = ball_vel - prev_ball_vel - gravity_vec
    accel_targets[_EGO] = ego_vel - prev_ego_vel - gravity_vec
    accel_targets[_TEAM] = ego_vel - prev_ego_vel - gravity_vec
    accel_targets[_OPP] = opp_vel - prev_opp_vel - gravity_vec

    # ── Pre-pass: compute per-object bases and residuals ─────────────
    all_basis_cos = np.zeros((N_OBJECTS, K), dtype=np.float32)
    all_basis_sin = np.zeros((N_OBJECTS, K), dtype=np.float32)
    all_pos_residuals = np.zeros((N_OBJECTS, 3), dtype=np.float32)
    all_accel_residuals = np.zeros((N_OBJECTS, 3), dtype=np.float32)

    for obj in range(N_OBJECTS):
        pos = positions[obj]
        ori = orientations[obj]

        phase = k_spatial[obj] @ pos                     # (K,)
        s_cos = np.cos(phase)
        s_sin = np.sin(phase)
        orient = (quaternions[obj] * ori).sum(axis=-1)   # (K,)

        basis_cos = s_cos * orient                       # (K,)
        basis_sin = s_sin * orient

        all_basis_cos[obj] = basis_cos
        all_basis_sin[obj] = basis_sin

        # Position residual: Re[f] = Σ_k (a_k·cos·q - b_k·sin·q)
        predicted = (basis_cos @ coeff[obj, :, :, 0]
                     - basis_sin @ coeff[obj, :, :, 1])  # (3,)
        all_pos_residuals[obj] = pos - predicted

        # Accel residual: Σ_k (c_k·cos·q)
        accel_pred = basis_cos @ coeff[obj, :, :, 2]     # (3,)
        all_accel_residuals[obj] = accel_targets[obj] - accel_pred

    # ── Cross-object coupling ────────────────────────────────────────
    if W_interact is not None:
        W_eff = np.eye(N_OBJECTS, dtype=np.float32) + W_interact
        all_pos_residuals = W_eff @ all_pos_residuals    # (N_OBJECTS, 3)
        all_accel_residuals = W_eff @ all_accel_residuals

    # ── LMS update (all 3 channels) ─────────────────────────────────
    for obj in range(N_OBJECTS):
        bc = all_basis_cos[obj]                          # (K,)
        bs = all_basis_sin[obj]

        # Position: real channel
        coeff[obj, :, :, 0] += np.outer(lr[obj] * bc, all_pos_residuals[obj])
        # Position: imag channel
        coeff[obj, :, :, 1] -= np.outer(lr[obj] * bs, all_pos_residuals[obj])
        # Acceleration channel
        coeff[obj, :, :, 2] += np.outer(lr[obj] * bc, all_accel_residuals[obj])

    # Reset ball on contact (all channels)
    if contact:
        coeff[_BALL] = 0.0

    # Clamp to prevent unbounded growth
    np.clip(coeff, -COEFF_CLIP, COEFF_CLIP, out=coeff)

    return coeff.ravel()


def pack_observation(raw_state: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Pack raw state and coefficients into SE3_OBS_DIM observation."""
    return np.concatenate([raw_state, coefficients]).astype(np.float32)
