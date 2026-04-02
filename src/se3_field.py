"""
SE(3) Spectral Field Representation
====================================
Probabilistic scene representation where each object is an SE(3) spectral
field — a frequency-domain function over R^9 amplitude space (position +
angular velocity + boost + has_flip + on_ground).

The field geometry (spatial frequencies k_spatial and quaternion bases) is
learned via backprop.  Coefficients carry temporal memory and are updated
online at 120 Hz via a residual rule.

Quaternion convention: (w, x, y, z), unit norm.  Double-cover is preserved
(q and -q produce different inner-product signs).  Euler→quaternion
conversion enforces w >= 0 for consistent sign from game observations.

Architecture
------------
    Env: raw_state (74) + prev_coefficients (1080) = 1154-dim obs
         ↓
    SE3Encoder.forward(packed_obs)       ← k_spatial, quaternions are nn.Parameters
         ↓
    1080-dim updated coefficients        (internal, for env persistence)

    SE3Encoder.encode_for_policy(packed_obs)
         ↓
    26-dim physical summary              (interaction conv + context, LayerNorm'd)

Coefficient layout (internal, 3 channels per spectral component):
    channel 0: amplitude real (cosine)
    channel 1: amplitude imaginary (sine)
    channel 2: acceleration residual (observed_accel - gravity)

Amplitude dimensions (D_AMP=9):
    [0:3] position (x, y, z)
    [3:6] angular velocity (wx, wy, wz)
    [6]   boost amount
    [7]   has_flip flag
    [8]   on_ground flag
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── import normalisation constants from the existing encoder ─────────────────

from encoder import FIELD_X, FIELD_Y, CEILING_Z, MAX_VEL, MAX_ANG_VEL, MAX_BOOST

# ── constants ────────────────────────────────────────────────────────────────

OBJECTS: List[str] = [
    'ball',
    'ego',
    'team',
    'opponent',
    'stadium',
]

K = 8                              # spectral components per field
N_OBJECTS = len(OBJECTS)            # 5
D_AMP = 9                          # amplitude dimension (pos + ang_vel + boost + has_flip + on_ground)
N_CHANNELS = 3                     # real, imaginary, acceleration residual
COEFF_DIM = N_OBJECTS * K * D_AMP * N_CHANNELS  # 1080
COEFF_CLIP = 10.0                  # clamp coefficients to [-CLIP, CLIP]

# Interaction conv dimensions
D_FIELD = 16                       # per-object projected field vector
D_OUTER = 4                        # rank-reduced outer product dimension
CONV_OUT = 16                      # channels out of interaction conv
ACCEL_CTX_DIM = 4                  # ego_delta_norm + opp_delta_norm + ego_surprise_norm + opp_surprise_norm
CONTEXT_DIM = 10                   # ego_boost(1) + game_state(3) + pad_active(6)
EMBED_DIM = CONV_OUT + ACCEL_CTX_DIM + CONTEXT_DIM  # 30

# Raw state layout (74 dims)
RAW_STATE_DIM = 74
ACCEL_HIST_DIM = 2 * N_OBJECTS * D_AMP  # 90: prev_accel_res(45) + accel_ema(45)
SE3_OBS_DIM = RAW_STATE_DIM + COEFF_DIM + ACCEL_HIST_DIM  # 1244

# Gravity: normalised per-tick velocity change on z-axis
# Rocket League gravity ≈ 650 uu/s², MAX_VEL=2300, tick=1/120
GRAVITY_DV_Z = -650.0 / (2300.0 * 120.0)  # ≈ -0.002355 per tick

# Object indices in OBJECTS list
_BALL = 0
_EGO = 1
_TEAM = 2
_OPP = 3
_STADIUM = 4

# Raw state offsets
_BALL_OFF = 0          # pos(3) + vel(3) + ang_vel(3) = 9
_EGO_OFF = 9           # pos(3) + vel(3) + quat(4) + ang_vel(3) + boost(1) + has_flip(1) + on_ground(1) = 16
_OPP_OFF = 25          # same layout as ego = 16
_PAD_OFF = 41          # 6 × active(1) = 6
_GS_OFF = 47           # score_diff(1) + time_rem(1) + overtime(1) = 3
_PREV_VEL_OFF = 50     # prev ball vel(3) + ego vel(3) + opp vel(3) = 9
_PREV_EGO_VEL_OFF = 53
_PREV_OPP_VEL_OFF = 56
_PREV_ANG_VEL_OFF = 59  # prev ball ang_vel(3) + ego ang_vel(3) + opp ang_vel(3) = 9
_PREV_EGO_ANG_VEL_OFF = 62
_PREV_OPP_ANG_VEL_OFF = 65
_PREV_SCALARS_OFF = 68  # prev ego(boost, has_flip, on_ground)(3) + opp(3) = 6
_PREV_OPP_SCALARS_OFF = 71

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
    Learned SE(3) spectral field encoder with R^9 amplitude vectors and
    interaction convolutions.

    Parameters (learned via backprop):
        k_spatial:        [N_OBJECTS, K, D_AMP]  frequency vectors in R^9
        quaternions:      [N_OBJECTS, K, 4]       orientation basis (unit quaternions)
        log_lr:           [N_OBJECTS]             per-object coefficient update rate
        W_interact:       [N_OBJECTS, N_OBJECTS]  pairwise coupling
        field_proj:       Linear(K*D_AMP*N_CHANNELS, D_FIELD)  per-object projection
        proj_left/right:  Linear(D_FIELD, D_OUTER)  for rank-reduced outer products
        interaction_conv: Conv2d stack on object-pair features
        output_norm:      LayerNorm(EMBED_DIM)

    forward(packed_obs):
        Input:  (batch, SE3_OBS_DIM=1154)
        Output: (batch, COEFF_DIM=1080) updated coefficients

    encode_for_policy(packed_obs):
        Input:  (batch, SE3_OBS_DIM=1154)
        Output: (batch, EMBED_DIM=26) normalised physical summary
    """

    def __init__(self, momentum_mode: str = 'both'):
        """
        Parameters
        ----------
        momentum_mode : str
            'both' | 'delta_only' | 'surprise_only' | 'none'
            Controls which action momentum signals feed into the policy embedding.
            EMA state always updates regardless (needed for accel_hist persistence).
        """
        super().__init__()
        assert momentum_mode in ('both', 'delta_only', 'surprise_only', 'none'), \
            f"Invalid momentum_mode: {momentum_mode!r}"
        self.momentum_mode = momentum_mode

        # Learned frequency vectors in R^D_AMP — phase = k · amplitude
        self.k_spatial = nn.Parameter(
            torch.randn(N_OBJECTS, K, D_AMP) * 1.0)

        # Learned quaternion basis — normalised after each optimiser step
        _q = torch.randn(N_OBJECTS, K, 4)
        _q = _q / _q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.quaternions = nn.Parameter(_q)

        # Learned per-object update rate (log-space for positivity)
        self.log_lr = nn.Parameter(torch.full((N_OBJECTS,), math.log(0.05)))

        # Learned pairwise interaction weights (zero-init = no coupling at start)
        self.W_interact = nn.Parameter(torch.zeros(N_OBJECTS, N_OBJECTS))

        # Interaction conv modules
        _coeff_flat_dim = K * D_AMP * N_CHANNELS  # 8 * 9 * 3 = 216
        self.field_proj = nn.Linear(_coeff_flat_dim, D_FIELD)
        self.proj_left = nn.Linear(D_FIELD, D_OUTER)
        self.proj_right = nn.Linear(D_FIELD, D_OUTER)

        _conv_in = 1 + D_OUTER * D_OUTER + D_OUTER * D_OUTER  # 1 + 16 + 16 = 33
        self.interaction_conv = nn.Sequential(
            nn.Conv2d(_conv_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, CONV_OUT, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Acceleration momentum projections
        self.delta_proj = nn.Linear(D_AMP, D_FIELD)      # accel delta → field dim
        self.surprise_proj = nn.Linear(D_AMP, D_FIELD)   # accel surprise → field dim
        self.log_accel_alpha = nn.Parameter(              # learned EMA decay per object
            torch.full((N_OBJECTS,), 0.0))                # sigmoid(0) = 0.5

        # LayerNorm on physical summary output
        self.output_norm = nn.LayerNorm(EMBED_DIM)

    @property
    def d_model(self) -> int:
        """Output dimensionality for policy (EMBED_DIM)."""
        return EMBED_DIM

    def _update_coefficients(self, packed_obs: torch.Tensor):
        """Core coefficient update with D_AMP=9 amplitude vectors.

        Returns (new_coeff_flat, basis_cos, basis_sin, raw, new_coeff_5d,
                 accel_delta, accel_surprise, new_accel_hist).

        new_coeff_flat:   (batch, COEFF_DIM=1080) for env persistence
        basis_cos/sin:    (batch, N_OBJECTS, K) scalar basis per component
        raw:              (batch, RAW_STATE_DIM=74) parsed raw state
        new_coeff_5d:     (batch, N_OBJECTS, K, D_AMP, N_CHANNELS) structured coefficients
        accel_delta:      (batch, N_OBJECTS, D_AMP) simple diff of accel residual
        accel_surprise:   (batch, N_OBJECTS, D_AMP) residual vs EMA prediction
        new_accel_hist:   (batch, ACCEL_HIST_DIM=90) updated history for env persistence
        """
        batch = packed_obs.shape[0]
        device = packed_obs.device

        # Split raw state, previous coefficients, and accel history
        raw = packed_obs[:, :RAW_STATE_DIM]                 # (batch, 74)
        prev_coeff = packed_obs[:, RAW_STATE_DIM:RAW_STATE_DIM + COEFF_DIM]  # (batch, 1080)
        prev_coeff = prev_coeff.reshape(batch, N_OBJECTS, K, D_AMP, N_CHANNELS)
        accel_hist = packed_obs[:, RAW_STATE_DIM + COEFF_DIM:]  # (batch, 90)
        _half = N_OBJECTS * D_AMP  # 45
        prev_accel_res = accel_hist[:, :_half].reshape(batch, N_OBJECTS, D_AMP)
        prev_accel_ema = accel_hist[:, _half:].reshape(batch, N_OBJECTS, D_AMP)

        # Effective learning rates (positive via exp)
        lr = torch.exp(self.log_lr).unsqueeze(0)            # (1, N_OBJECTS)

        # Normalise quaternion basis (projection, no grad break)
        q_basis = self.quaternions / self.quaternions.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)           # (N_OBJECTS, K, 4)

        # ── Parse raw state ──────────────────────────────────────────────
        ball_pos = raw[:, _BALL_OFF:_BALL_OFF + 3]
        ball_vel = raw[:, _BALL_OFF + 3:_BALL_OFF + 6]
        ball_ang_vel = raw[:, _BALL_OFF + 6:_BALL_OFF + 9]

        ego_pos = raw[:, _EGO_OFF:_EGO_OFF + 3]
        ego_vel = raw[:, _EGO_OFF + 3:_EGO_OFF + 6]
        ego_quat = raw[:, _EGO_OFF + 6:_EGO_OFF + 10]
        ego_ang_vel = raw[:, _EGO_OFF + 10:_EGO_OFF + 13]
        ego_boost = raw[:, _EGO_OFF + 13:_EGO_OFF + 14]
        ego_has_flip = raw[:, _EGO_OFF + 14:_EGO_OFF + 15]
        ego_on_ground = raw[:, _EGO_OFF + 15:_EGO_OFF + 16]

        opp_pos = raw[:, _OPP_OFF:_OPP_OFF + 3]
        opp_vel = raw[:, _OPP_OFF + 3:_OPP_OFF + 6]
        opp_quat = raw[:, _OPP_OFF + 6:_OPP_OFF + 10]
        opp_ang_vel = raw[:, _OPP_OFF + 10:_OPP_OFF + 13]
        opp_boost = raw[:, _OPP_OFF + 13:_OPP_OFF + 14]
        opp_has_flip = raw[:, _OPP_OFF + 14:_OPP_OFF + 15]
        opp_on_ground = raw[:, _OPP_OFF + 15:_OPP_OFF + 16]

        prev_ball_vel = raw[:, _PREV_VEL_OFF:_PREV_VEL_OFF + 3]
        prev_ego_vel = raw[:, _PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3]
        prev_opp_vel = raw[:, _PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3]

        prev_ball_ang_vel = raw[:, _PREV_ANG_VEL_OFF:_PREV_ANG_VEL_OFF + 3]
        prev_ego_ang_vel = raw[:, _PREV_EGO_ANG_VEL_OFF:_PREV_EGO_ANG_VEL_OFF + 3]
        prev_opp_ang_vel = raw[:, _PREV_OPP_ANG_VEL_OFF:_PREV_OPP_ANG_VEL_OFF + 3]

        prev_ego_scalars = raw[:, _PREV_SCALARS_OFF:_PREV_SCALARS_OFF + 3]
        prev_opp_scalars = raw[:, _PREV_OPP_SCALARS_OFF:_PREV_OPP_SCALARS_OFF + 3]

        # Identity quaternion for objects without orientation
        id_q = _QUAT_IDENTITY.to(device).unsqueeze(0).expand(batch, 4)

        zeros9 = torch.zeros(batch, D_AMP, device=device)

        # ── Build 9d amplitude targets per object ────────────────────────
        # Ball: [pos(3), ang_vel(3), 0, 0, 0]
        ball_amp = torch.cat([ball_pos, ball_ang_vel,
                              torch.zeros(batch, 3, device=device)], dim=-1)
        # Ego: [pos(3), ang_vel(3), boost, has_flip, on_ground]
        ego_amp = torch.cat([ego_pos, ego_ang_vel,
                             ego_boost, ego_has_flip, ego_on_ground], dim=-1)
        # Team = ego in 1v1
        team_amp = ego_amp
        # Opp: [pos(3), ang_vel(3), boost, has_flip, on_ground]
        opp_amp = torch.cat([opp_pos, opp_ang_vel,
                             opp_boost, opp_has_flip, opp_on_ground], dim=-1)
        # Stadium: zeros
        stadium_amp = zeros9

        # amplitudes: (batch, N_OBJECTS, D_AMP=9)
        amplitudes = torch.stack([
            ball_amp, ego_amp, team_amp, opp_amp, stadium_amp
        ], dim=1)

        # orientations: (batch, N_OBJECTS, 4)
        orientations = torch.stack([
            id_q, ego_quat, ego_quat, opp_quat, id_q
        ], dim=1)

        # ── contact detection (ball) ─────────────────────────────────────
        contact = detect_contact(prev_ball_vel, ball_vel)   # (batch,)

        # ── spectral basis computation ───────────────────────────────────
        # Phase: k_spatial (N_OBJECTS, K, D_AMP) @ amplitudes (batch, N_OBJECTS, D_AMP) → (batch, N_OBJECTS, K)
        phase = torch.einsum('okd,bod->bok', self.k_spatial, amplitudes)
        spatial_cos = torch.cos(phase)                      # (batch, N_OBJECTS, K)
        spatial_sin = torch.sin(phase)
        orient = torch.einsum('okd,bod->bok', q_basis, orientations)  # (batch, N_OBJECTS, K)

        basis_cos = spatial_cos * orient                    # (batch, N_OBJECTS, K)
        basis_sin = spatial_sin * orient

        # ── Amplitude channels (0=real, 1=imag): complex LMS ────────────
        target = amplitudes                                  # (batch, N_OBJECTS, D_AMP)
        coeff_a = prev_coeff[:, :, :, :, 0]                 # (batch, N_OBJECTS, K, D_AMP)
        coeff_b = prev_coeff[:, :, :, :, 1]

        # Reconstruct: Σ_k basis_cos[k] * a[k] - basis_sin[k] * b[k]
        # basis_cos: (batch, N_OBJECTS, K) → unsqueeze(-1) → (batch, N_OBJECTS, K, 1)
        bc = basis_cos.unsqueeze(-1)                         # (batch, N_OBJECTS, K, 1)
        bs = basis_sin.unsqueeze(-1)
        predicted = (bc * coeff_a - bs * coeff_b).sum(dim=2)  # (batch, N_OBJECTS, D_AMP)

        residual = target - predicted                         # (batch, N_OBJECTS, D_AMP)

        # Cross-object coupling on residuals
        W_eff = torch.eye(N_OBJECTS, device=device) + self.W_interact
        residual = torch.einsum('ij,bjd->bid', W_eff, residual)

        # LMS update for amplitude channels
        # lr: (1, N_OBJECTS) → (1, N_OBJECTS, 1) → scaled_res: (batch, N_OBJECTS, D_AMP)
        scaled_res = lr.unsqueeze(-1) * residual              # (batch, N_OBJECTS, D_AMP)
        # scaled_res: (batch, N_OBJECTS, 1, D_AMP) to broadcast with (batch, N_OBJECTS, K, 1)
        delta_real = scaled_res.unsqueeze(2) * bc              # (batch, N_OBJECTS, K, D_AMP)
        delta_imag = -scaled_res.unsqueeze(2) * bs

        new_coeff_real = coeff_a + delta_real
        new_coeff_imag = coeff_b + delta_imag

        # ── Acceleration channel (2): gravity-residual LMS ───────────────
        # 9d acceleration targets: derivatives of all amplitude dims
        gravity_9d = torch.zeros(D_AMP, device=device)
        gravity_9d[2] = GRAVITY_DV_Z  # gravity only on z position

        # Ball accel: d(pos)/dt - gravity for dims 0-2, d(ang_vel)/dt for 3-5, zeros for 6-8
        ball_accel = torch.cat([
            ball_vel - prev_ball_vel - gravity_9d[:3].unsqueeze(0).expand(batch, -1),
            ball_ang_vel - prev_ball_ang_vel,
            torch.zeros(batch, 3, device=device),
        ], dim=-1)

        # Ego accel
        ego_scalars_accel = torch.cat([ego_boost, ego_has_flip, ego_on_ground], dim=-1) - prev_ego_scalars
        ego_accel = torch.cat([
            ego_vel - prev_ego_vel - gravity_9d[:3].unsqueeze(0).expand(batch, -1),
            ego_ang_vel - prev_ego_ang_vel,
            ego_scalars_accel,
        ], dim=-1)

        # Opp accel
        opp_scalars_accel = torch.cat([opp_boost, opp_has_flip, opp_on_ground], dim=-1) - prev_opp_scalars
        opp_accel = torch.cat([
            opp_vel - prev_opp_vel - gravity_9d[:3].unsqueeze(0).expand(batch, -1),
            opp_ang_vel - prev_opp_ang_vel,
            opp_scalars_accel,
        ], dim=-1)

        # Build per-object accel target: (batch, N_OBJECTS, D_AMP)
        accel_target = torch.stack([
            ball_accel,
            ego_accel,
            ego_accel,      # team = ego
            opp_accel,
            zeros9,         # stadium
        ], dim=1)

        coeff_c = prev_coeff[:, :, :, :, 2]                  # (batch, N_OBJECTS, K, D_AMP)
        accel_predicted = (bc * coeff_c).sum(dim=2)           # (batch, N_OBJECTS, D_AMP)
        accel_residual = accel_target - accel_predicted

        # Cross-object coupling on accel residuals (same W_interact)
        accel_residual = torch.einsum('ij,bjd->bid', W_eff, accel_residual)

        scaled_accel_res = lr.unsqueeze(-1) * accel_residual
        delta_accel = scaled_accel_res.unsqueeze(2) * bc
        new_coeff_accel = coeff_c + delta_accel

        # ── Contact reset: zero all channels for ball ────────────────────
        contact_keep = (1.0 - contact.float()).unsqueeze(-1)  # (batch, 1)
        obj_keep_list = [
            contact_keep.unsqueeze(1) if i == _BALL
            else torch.ones(batch, 1, 1, device=device)
            for i in range(N_OBJECTS)
        ]
        obj_keep = torch.cat(obj_keep_list, dim=1).unsqueeze(-1)  # (batch, N_OBJECTS, 1, 1)
        new_coeff_real = new_coeff_real * obj_keep
        new_coeff_imag = new_coeff_imag * obj_keep
        new_coeff_accel = new_coeff_accel * obj_keep

        # Stack: (batch, N_OBJECTS, K, D_AMP, N_CHANNELS) → (batch, COEFF_DIM)
        new_coeff = torch.stack([new_coeff_real, new_coeff_imag, new_coeff_accel], dim=-1)
        new_coeff = torch.clamp(new_coeff, -COEFF_CLIP, COEFF_CLIP)

        # ── Action momentum signals ─────────────────────────────────────
        # accel_residual here is post-coupling: (batch, N_OBJECTS, D_AMP)
        # Simple difference: spikes when action changes
        accel_delta = accel_residual - prev_accel_res

        # EMA with learned per-object decay
        alpha = torch.sigmoid(self.log_accel_alpha)          # (N_OBJECTS,)
        alpha_b = alpha.unsqueeze(0).unsqueeze(-1)           # (1, N_OBJECTS, 1)
        new_accel_ema = alpha_b * accel_residual + (1 - alpha_b) * prev_accel_ema
        accel_surprise = accel_residual - prev_accel_ema     # spikes vs EMA prediction

        # Pack updated history for env persistence
        new_accel_hist = torch.cat([
            accel_residual.reshape(batch, -1),               # becomes prev_accel_res next tick
            new_accel_ema.reshape(batch, -1),                # persisted EMA state
        ], dim=-1)                                           # (batch, ACCEL_HIST_DIM)

        return (new_coeff.reshape(batch, COEFF_DIM), basis_cos, basis_sin, raw, new_coeff,
                accel_delta, accel_surprise, new_accel_hist)

    def forward(self, packed_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update coefficients (for env persistence / buffer storage).

        packed_obs: (batch, SE3_OBS_DIM=1244)
        returns:    (coeff_flat (batch, COEFF_DIM), new_accel_hist (batch, ACCEL_HIST_DIM))
        """
        coeff_flat, _, _, _, _, _, _, new_accel_hist = self._update_coefficients(packed_obs)
        return coeff_flat, new_accel_hist

    def encode_for_policy(self, packed_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update coefficients and return physical summary via interaction conv.

        packed_obs: (batch, SE3_OBS_DIM=1244)
        returns:    (embed (batch, EMBED_DIM=30),
                     coeff_flat (batch, COEFF_DIM=1080),
                     new_accel_hist (batch, ACCEL_HIST_DIM=90))
        """
        (coeff_flat, basis_cos, basis_sin, raw, new_coeff_5d,
         accel_delta, accel_surprise, new_accel_hist) = self._update_coefficients(packed_obs)
        batch = packed_obs.shape[0]
        device = packed_obs.device

        # new_coeff_5d: (batch, N_OBJECTS, K, D_AMP, N_CHANNELS)
        # Reshape per-object: flatten K * D_AMP * N_CHANNELS = 216
        f_flat = new_coeff_5d.reshape(batch, N_OBJECTS, K * D_AMP * N_CHANNELS)

        # Project to per-object field vectors
        f = self.field_proj(f_flat)                            # (batch, N_OBJECTS, D_FIELD)

        # Project action momentum signals and fuse additively
        f_delta = self.delta_proj(accel_delta)                 # (batch, N_OBJECTS, D_FIELD)
        f_surprise = self.surprise_proj(accel_surprise)        # (batch, N_OBJECTS, D_FIELD)

        # Ablation masking (momentum_mode)
        if self.momentum_mode in ('surprise_only', 'none'):
            f_delta = f_delta * 0.0
        if self.momentum_mode in ('delta_only', 'none'):
            f_surprise = f_surprise * 0.0

        f_combined = f + f_delta + f_surprise                  # (batch, N_OBJECTS, D_FIELD)

        # Inner product matrix: spectral alignment
        inner = torch.bmm(f_combined, f_combined.transpose(1, 2))  # (batch, N_OBJECTS, N_OBJECTS)
        inner = inner.unsqueeze(1)                             # (batch, 1, 5, 5)

        # Rank-reduced outer products
        left = self.proj_left(f_combined)                      # (batch, N_OBJECTS, D_OUTER)
        right = self.proj_right(f_combined)                    # (batch, N_OBJECTS, D_OUTER)

        # Left self-outer: (batch, 5, 5, D_OUTER²)
        left_outer = torch.einsum('bid,bjc->bijdc', left, left)  # (batch, 5, 5, 4, 4)
        left_outer = left_outer.reshape(batch, N_OBJECTS, N_OBJECTS, D_OUTER * D_OUTER)
        left_outer = left_outer.permute(0, 3, 1, 2)           # (batch, 16, 5, 5)

        # Right self-outer
        right_outer = torch.einsum('bid,bjc->bijdc', right, right)
        right_outer = right_outer.reshape(batch, N_OBJECTS, N_OBJECTS, D_OUTER * D_OUTER)
        right_outer = right_outer.permute(0, 3, 1, 2)         # (batch, 16, 5, 5)

        # Concatenate: (batch, 33, 5, 5)
        conv_input = torch.cat([inner, left_outer, right_outer], dim=1)

        # Interaction conv → (batch, CONV_OUT, 1, 1)
        conv_out = self.interaction_conv(conv_input)
        conv_out = conv_out.squeeze(-1).squeeze(-1)            # (batch, CONV_OUT=16)

        # Action momentum context: per-object norms
        ego_delta_norm = accel_delta[:, _EGO].norm(dim=-1, keepdim=True)       # (batch, 1)
        opp_delta_norm = accel_delta[:, _OPP].norm(dim=-1, keepdim=True)       # (batch, 1)
        ego_surprise_norm = accel_surprise[:, _EGO].norm(dim=-1, keepdim=True) # (batch, 1)
        opp_surprise_norm = accel_surprise[:, _OPP].norm(dim=-1, keepdim=True) # (batch, 1)

        # Ablation masking for context norms
        if self.momentum_mode in ('surprise_only', 'none'):
            ego_delta_norm = ego_delta_norm * 0.0
            opp_delta_norm = opp_delta_norm * 0.0
        if self.momentum_mode in ('delta_only', 'none'):
            ego_surprise_norm = ego_surprise_norm * 0.0
            opp_surprise_norm = opp_surprise_norm * 0.0

        # Context scalars from raw state
        ego_boost_ctx = raw[:, _EGO_OFF + 13:_EGO_OFF + 14]   # (batch, 1)
        game_state = raw[:, _GS_OFF:_GS_OFF + 3]              # (batch, 3)
        pad_active = raw[:, _PAD_OFF:_PAD_OFF + 6]            # (batch, 6)

        # Assemble: CONV_OUT(16) + ACCEL_CTX(4) + CONTEXT(10) = 30
        embed = torch.cat([conv_out,
                           ego_delta_norm, opp_delta_norm,
                           ego_surprise_norm, opp_surprise_norm,
                           ego_boost_ctx, game_state, pad_active], dim=-1)

        return self.output_norm(embed), coeff_flat, new_accel_hist

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
    k_spatial: np.ndarray,      # (N_OBJECTS, K, D_AMP)
    quaternions: np.ndarray,    # (N_OBJECTS, K, 4) unit
    lr: np.ndarray,             # (N_OBJECTS,)
    prev_coeff: np.ndarray,     # (COEFF_DIM,) = 1080
    raw_state: np.ndarray,      # (RAW_STATE_DIM,) = 74
    W_interact: Optional[np.ndarray] = None,  # (N_OBJECTS, N_OBJECTS) or None
) -> np.ndarray:
    """Non-differentiable coefficient update for env stepping.

    Mirrors SE3Encoder._update_coefficients logic in numpy.
    3 channels: real (amp cos), imag (amp sin), accel residual.
    D_AMP=9: pos(3) + ang_vel(3) + boost(1) + has_flip(1) + on_ground(1).
    Returns updated coefficients (1080,).
    """
    coeff = prev_coeff.reshape(N_OBJECTS, K, D_AMP, N_CHANNELS).copy()

    # ── Parse raw state ──────────────────────────────────────────────
    ball_pos = raw_state[_BALL_OFF:_BALL_OFF + 3]
    ball_vel = raw_state[_BALL_OFF + 3:_BALL_OFF + 6]
    ball_ang_vel = raw_state[_BALL_OFF + 6:_BALL_OFF + 9]

    ego_pos = raw_state[_EGO_OFF:_EGO_OFF + 3]
    ego_vel = raw_state[_EGO_OFF + 3:_EGO_OFF + 6]
    ego_quat = raw_state[_EGO_OFF + 6:_EGO_OFF + 10]
    ego_ang_vel = raw_state[_EGO_OFF + 10:_EGO_OFF + 13]
    ego_boost = raw_state[_EGO_OFF + 13]
    ego_has_flip = raw_state[_EGO_OFF + 14]
    ego_on_ground = raw_state[_EGO_OFF + 15]

    opp_pos = raw_state[_OPP_OFF:_OPP_OFF + 3]
    opp_vel = raw_state[_OPP_OFF + 3:_OPP_OFF + 6]
    opp_quat = raw_state[_OPP_OFF + 6:_OPP_OFF + 10]
    opp_ang_vel = raw_state[_OPP_OFF + 10:_OPP_OFF + 13]
    opp_boost = raw_state[_OPP_OFF + 13]
    opp_has_flip = raw_state[_OPP_OFF + 14]
    opp_on_ground = raw_state[_OPP_OFF + 15]

    prev_ball_vel = raw_state[_PREV_VEL_OFF:_PREV_VEL_OFF + 3]
    prev_ego_vel = raw_state[_PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3]
    prev_opp_vel = raw_state[_PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3]

    prev_ball_ang_vel = raw_state[_PREV_ANG_VEL_OFF:_PREV_ANG_VEL_OFF + 3]
    prev_ego_ang_vel = raw_state[_PREV_EGO_ANG_VEL_OFF:_PREV_EGO_ANG_VEL_OFF + 3]
    prev_opp_ang_vel = raw_state[_PREV_OPP_ANG_VEL_OFF:_PREV_OPP_ANG_VEL_OFF + 3]

    prev_ego_scalars = raw_state[_PREV_SCALARS_OFF:_PREV_SCALARS_OFF + 3]
    prev_opp_scalars = raw_state[_PREV_OPP_SCALARS_OFF:_PREV_OPP_SCALARS_OFF + 3]

    id_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # ── Build 9d amplitude targets per object ────────────────────────
    amplitudes = np.zeros((N_OBJECTS, D_AMP), dtype=np.float32)
    amplitudes[_BALL, :3] = ball_pos
    amplitudes[_BALL, 3:6] = ball_ang_vel
    amplitudes[_EGO, :3] = ego_pos
    amplitudes[_EGO, 3:6] = ego_ang_vel
    amplitudes[_EGO, 6] = ego_boost
    amplitudes[_EGO, 7] = ego_has_flip
    amplitudes[_EGO, 8] = ego_on_ground
    amplitudes[_TEAM] = amplitudes[_EGO]  # team = ego in 1v1
    amplitudes[_OPP, :3] = opp_pos
    amplitudes[_OPP, 3:6] = opp_ang_vel
    amplitudes[_OPP, 6] = opp_boost
    amplitudes[_OPP, 7] = opp_has_flip
    amplitudes[_OPP, 8] = opp_on_ground
    # _STADIUM stays zero

    orientations = np.tile(id_q, (N_OBJECTS, 1))
    orientations[_EGO] = ego_quat
    orientations[_TEAM] = ego_quat
    orientations[_OPP] = opp_quat

    # Contact detection
    contact = detect_contact_np(prev_ball_vel, ball_vel)

    # ── 9d acceleration targets ──────────────────────────────────────
    gravity_9d = np.zeros(D_AMP, dtype=np.float32)
    gravity_9d[2] = GRAVITY_DV_Z

    accel_targets = np.zeros((N_OBJECTS, D_AMP), dtype=np.float32)
    # Ball: pos accel + ang_vel accel + zero scalars
    accel_targets[_BALL, :3] = ball_vel - prev_ball_vel - gravity_9d[:3]
    accel_targets[_BALL, 3:6] = ball_ang_vel - prev_ball_ang_vel
    # Ego
    ego_scalars = np.array([ego_boost, ego_has_flip, ego_on_ground], dtype=np.float32)
    accel_targets[_EGO, :3] = ego_vel - prev_ego_vel - gravity_9d[:3]
    accel_targets[_EGO, 3:6] = ego_ang_vel - prev_ego_ang_vel
    accel_targets[_EGO, 6:9] = ego_scalars - prev_ego_scalars
    # Team = ego
    accel_targets[_TEAM] = accel_targets[_EGO]
    # Opp
    opp_scalars = np.array([opp_boost, opp_has_flip, opp_on_ground], dtype=np.float32)
    accel_targets[_OPP, :3] = opp_vel - prev_opp_vel - gravity_9d[:3]
    accel_targets[_OPP, 3:6] = opp_ang_vel - prev_opp_ang_vel
    accel_targets[_OPP, 6:9] = opp_scalars - prev_opp_scalars

    # ── Pre-pass: compute per-object bases and residuals ─────────────
    all_basis_cos = np.zeros((N_OBJECTS, K), dtype=np.float32)
    all_basis_sin = np.zeros((N_OBJECTS, K), dtype=np.float32)
    all_amp_residuals = np.zeros((N_OBJECTS, D_AMP), dtype=np.float32)
    all_accel_residuals = np.zeros((N_OBJECTS, D_AMP), dtype=np.float32)

    for obj in range(N_OBJECTS):
        amp = amplitudes[obj]                                # (D_AMP,)
        ori = orientations[obj]                              # (4,)

        # Phase: k_spatial[obj] @ amp → (K,)  (k is (K, D_AMP), amp is (D_AMP,))
        phase = k_spatial[obj] @ amp                         # (K,)
        s_cos = np.cos(phase)
        s_sin = np.sin(phase)
        orient = (quaternions[obj] * ori).sum(axis=-1)       # (K,)

        basis_cos = s_cos * orient                           # (K,)
        basis_sin = s_sin * orient

        all_basis_cos[obj] = basis_cos
        all_basis_sin[obj] = basis_sin

        # Amplitude residual: Re[f] = Σ_k (a_k·cos - b_k·sin)
        # coeff[obj]: (K, D_AMP, N_CHANNELS)
        # basis_cos: (K,) → need (K, 1) to broadcast with (K, D_AMP)
        bc = basis_cos[:, np.newaxis]                        # (K, 1)
        bs = basis_sin[:, np.newaxis]
        predicted = (bc * coeff[obj, :, :, 0] - bs * coeff[obj, :, :, 1]).sum(axis=0)  # (D_AMP,)
        all_amp_residuals[obj] = amp - predicted

        # Accel residual: Σ_k (c_k·cos)
        accel_pred = (bc * coeff[obj, :, :, 2]).sum(axis=0)  # (D_AMP,)
        all_accel_residuals[obj] = accel_targets[obj] - accel_pred

    # ── Cross-object coupling ────────────────────────────────────────
    if W_interact is not None:
        W_eff = np.eye(N_OBJECTS, dtype=np.float32) + W_interact
        all_amp_residuals = W_eff @ all_amp_residuals        # (N_OBJECTS, D_AMP)
        all_accel_residuals = W_eff @ all_accel_residuals

    # ── LMS update (all 3 channels) ─────────────────────────────────
    for obj in range(N_OBJECTS):
        bc = all_basis_cos[obj]                              # (K,)
        bs = all_basis_sin[obj]

        # Real channel: coeff[obj, :, :, 0] += lr * bc[:, None] * residual[None, :]
        coeff[obj, :, :, 0] += lr[obj] * bc[:, np.newaxis] * all_amp_residuals[obj][np.newaxis, :]
        # Imag channel
        coeff[obj, :, :, 1] -= lr[obj] * bs[:, np.newaxis] * all_amp_residuals[obj][np.newaxis, :]
        # Acceleration channel
        coeff[obj, :, :, 2] += lr[obj] * bc[:, np.newaxis] * all_accel_residuals[obj][np.newaxis, :]

    # Reset ball on contact (all channels)
    if contact:
        coeff[_BALL] = 0.0

    # Clamp to prevent unbounded growth
    np.clip(coeff, -COEFF_CLIP, COEFF_CLIP, out=coeff)

    return coeff.ravel(), all_accel_residuals


def update_coefficients_with_hist_np(
    k_spatial: np.ndarray,
    quaternions: np.ndarray,
    lr: np.ndarray,
    prev_coeff: np.ndarray,
    raw_state: np.ndarray,
    accel_hist: np.ndarray,
    W_interact: Optional[np.ndarray] = None,
    accel_alpha: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coefficient update + accel momentum history update.

    Returns (new_coeff (COEFF_DIM,), new_accel_hist (ACCEL_HIST_DIM,)).
    """
    new_coeff, accel_residual = update_coefficients_np(
        k_spatial, quaternions, lr, prev_coeff, raw_state, W_interact)

    _half = N_OBJECTS * D_AMP  # 45
    prev_accel_res = accel_hist[:_half].reshape(N_OBJECTS, D_AMP)
    prev_accel_ema = accel_hist[_half:].reshape(N_OBJECTS, D_AMP)

    # Simple difference
    accel_delta = accel_residual - prev_accel_res

    # EMA update
    if accel_alpha is not None:
        alpha = 1.0 / (1.0 + np.exp(-accel_alpha))  # sigmoid
        alpha_r = alpha[:, np.newaxis]  # (N_OBJECTS, 1)
    else:
        alpha_r = np.full((N_OBJECTS, 1), 0.5, dtype=np.float32)
    new_ema = alpha_r * accel_residual + (1.0 - alpha_r) * prev_accel_ema
    accel_surprise = accel_residual - prev_accel_ema

    new_accel_hist = np.concatenate([
        accel_residual.ravel(),
        new_ema.ravel(),
    ]).astype(np.float32)

    return new_coeff, new_accel_hist


def make_initial_accel_hist() -> np.ndarray:
    """Return zero-initialised accel history (ACCEL_HIST_DIM,)."""
    return np.zeros(ACCEL_HIST_DIM, dtype=np.float32)


def pack_observation(raw_state: np.ndarray, coefficients: np.ndarray,
                     accel_hist: Optional[np.ndarray] = None) -> np.ndarray:
    """Pack raw state, coefficients, and accel history into SE3_OBS_DIM observation."""
    if accel_hist is None:
        accel_hist = make_initial_accel_hist()
    return np.concatenate([raw_state, coefficients, accel_hist]).astype(np.float32)
