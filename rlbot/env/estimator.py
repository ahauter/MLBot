"""
State estimator: convert raw token observations to SE3Fields.

Token layout (from src/encoder.py and training/rlgym_env.py):
    Token 0: Ball    [x, y, z, vx, vy, vz, avx, avy, avz, 0]
    Token 1: Own car [x, y, z, vx, vy, vz, yaw, pitch, roll, boost]
    Token 2: Opp car [x, y, z, vx, vy, vz, yaw, pitch, roll, 0]
    Tokens 3-8: Big boost pads [x, y, z, active, 0, 0, 0, 0, 0, 0]
    Token 9: Game state [score_diff, time_rem, overtime, 0, ...]

All token values are pre-normalized (divided by field dims, MAX_VEL, etc).
"""

import torch
from torch import Tensor

from rlbot.constants import (
    N_COEFFS, L_MAX, BIG_PAD_VALUE,
    FIELD_X, FIELD_Y, CEILING_Z,
)
from rlbot.env.fields import (
    SE3Field, make_field, rotate_coefficients,
    fit_spectral_coefficients,
)


def _ball_field(token: Tensor) -> SE3Field:
    """Build SE3Field for the ball from token 0.

    l=0: 1.0 (presence)
    l=1: velocity direction (already normalized by MAX_VEL)
    l=2: angular velocity outer product (shape/spin)
    """
    pos = token[:3]  # already normalized
    vel = token[3:6]  # normalized by MAX_VEL
    ang_vel = token[6:9]  # normalized by MAX_ANG_VEL

    coeffs = torch.zeros(N_COEFFS, dtype=token.dtype)

    # l=0: presence
    coeffs[0] = 1.0

    # l=1: velocity direction
    speed = torch.norm(vel)
    if speed > 1e-6:
        coeffs[1:4] = vel / speed
    # Scale l=0 by speed so we don't lose magnitude
    coeffs[0] = torch.clamp(speed, min=0.01)

    # l=2: spin tensor from angular velocity
    # Symmetric traceless tensor from angular velocity vector
    av = ang_vel
    av_norm = torch.norm(av)
    if av_norm > 1e-6:
        av_hat = av / av_norm
        # Outer product minus trace/3
        outer = torch.outer(av_hat, av_hat)
        traceless = outer - (1.0 / 3.0) * torch.eye(3, dtype=token.dtype)
        coeffs[4] = traceless[0, 1] * av_norm       # xy
        coeffs[5] = traceless[1, 2] * av_norm       # yz
        s = torch.sqrt(torch.tensor(2.0 / 3.0))
        coeffs[6] = traceless[2, 2] * av_norm * s   # z²
        coeffs[7] = traceless[0, 2] * av_norm       # xz
        coeffs[8] = (traceless[0, 0] - traceless[1, 1]) / 2 * av_norm  # x²-y²

    covariance = torch.ones(N_COEFFS, dtype=token.dtype)
    return SE3Field(pos, coeffs, covariance)


def _car_field(token: Tensor) -> SE3Field:
    """Build SE3Field for a car from token 1 or 2.

    l=0: boost amount (normalized)
    l=1: forward direction from Euler angles (Wigner D^1 rotation)
    l=2: rotation quadrupole from Wigner D^2 rotation
    """
    pos = token[:3]  # normalized
    yaw, pitch, roll = token[6], token[7], token[8]
    boost = token[9]  # already normalized by MAX_BOOST

    # Canonical "resting" coefficients: car pointing along +x
    # l=0: boost, l=1: (1,0,0) forward, l=2: canonical quadrupole
    canonical = torch.zeros(N_COEFFS, dtype=token.dtype)
    canonical[0] = torch.clamp(boost, min=0.01)  # l=0 = boost
    canonical[1] = 1.0  # l=1 = forward direction along x
    # l=2 canonical: identity-like quadrupole (elongated along x)
    canonical[8] = 1.0  # x²-y² component = elongated along x

    # Rotate canonical coefficients by car's orientation
    coeffs = rotate_coefficients(canonical, yaw, pitch, roll)

    covariance = torch.ones(N_COEFFS, dtype=token.dtype)
    return SE3Field(pos, coeffs, covariance)


def _boost_density_field(pad_tokens: Tensor) -> SE3Field:
    """Build a single spatial density field from boost pad tokens.

    Each active pad contributes to the spectral coefficients proportional
    to its boost value. The field position is the boost-value-weighted
    centroid of active pads.

    l=0: total available boost energy (normalized)
    l=1: directional bias of boost availability
    l=2: spatial spread of boost distribution

    Args:
        pad_tokens: (N_PADS, 10) pad tokens (tokens 3-8)

    Returns:
        SE3Field representing the boost density.
    """
    positions = pad_tokens[:, :3]   # (N, 3) normalized positions
    active = pad_tokens[:, 3]       # (N,) active flags

    # Weights: active flag × pad value (all big pads = BIG_PAD_VALUE)
    weights = active * BIG_PAD_VALUE  # (N,)
    total_weight = weights.sum()

    coeffs = torch.zeros(N_COEFFS, dtype=pad_tokens.dtype)
    covariance = torch.ones(N_COEFFS, dtype=pad_tokens.dtype)

    if total_weight < 1e-6:
        # No active pads: field at origin with zero energy
        pos = torch.zeros(3, dtype=pad_tokens.dtype)
        return SE3Field(pos, coeffs, covariance)

    # Boost-value-weighted centroid
    centroid = (weights.unsqueeze(-1) * positions).sum(dim=0) / total_weight

    # l=0: total available boost (normalized by max possible = 6 × 100)
    max_total = 6.0 * BIG_PAD_VALUE
    coeffs[0] = total_weight / max_total

    # l=1: directional bias — weighted mean direction from centroid origin
    # This captures "boost is mostly to the left/right/forward/back"
    direction = centroid  # centroid position IS the directional bias
    dir_norm = torch.norm(direction)
    if dir_norm > 1e-6:
        coeffs[1:4] = direction / dir_norm * (total_weight / max_total)

    # l=2: spatial spread of boost distribution
    # Covariance of active pad positions around centroid
    if active.sum() >= 2:
        relative = positions - centroid.unsqueeze(0)  # (N, 3)
        weighted_rel = (weights.unsqueeze(-1) * relative)  # (N, 3)
        cov = (weighted_rel.T @ relative) / total_weight  # (3, 3)

        # Traceless part → l=2 coefficients
        trace = cov[0, 0] + cov[1, 1] + cov[2, 2]
        traceless = cov - (trace / 3.0) * torch.eye(3, dtype=cov.dtype)

        scale = total_weight / max_total
        coeffs[4] = traceless[0, 1] * scale
        coeffs[5] = traceless[1, 2] * scale
        coeffs[6] = traceless[2, 2] * torch.sqrt(torch.tensor(2.0/3.0)) * scale
        coeffs[7] = traceless[0, 2] * scale
        coeffs[8] = (traceless[0, 0] - traceless[1, 1]) / 2 * scale

    return SE3Field(centroid, coeffs, covariance)


def _game_state_features(token: Tensor) -> Tensor:
    """Extract scalar game state features from token 9.

    Returns:
        (3,) tensor: [score_diff, time_remaining, overtime]
    """
    return token[:3].clone()


def tokens_to_fields(tokens: Tensor) -> tuple[list[SE3Field], Tensor]:
    """Convert a single frame of tokens to SE3Fields.

    Args:
        tokens: (N_TOKENS, 10) token array (10 tokens, 10 features each)

    Returns:
        fields: list of 4 SE3Fields [ball, own_car, opp_car, boost]
        game_state: (3,) scalar game state features
    """
    ball = _ball_field(tokens[0])
    own_car = _car_field(tokens[1])
    opp_car = _car_field(tokens[2])
    boost = _boost_density_field(tokens[3:9])  # tokens 3-8
    game_state = _game_state_features(tokens[9])

    return [ball, own_car, opp_car, boost], game_state


def tokens_window_to_fields(window: Tensor) -> tuple[list[SE3Field], Tensor]:
    """Convert a multi-frame token window to SE3Fields.

    Uses exponential recency weighting: most recent frame has highest weight.
    For ball and cars, fits trajectory-based coefficients from position history.

    Args:
        window: (T, N_TOKENS, 10) token window

    Returns:
        fields: list of 4 SE3Fields [ball, own_car, opp_car, boost]
        game_state: (3,) from most recent frame
    """
    T = window.shape[0]

    if T == 1:
        return tokens_to_fields(window[0])

    # Use most recent frame as base for positions and game state
    latest = window[-1]
    game_state = _game_state_features(latest[9])

    # Ball: fit trajectory from position history
    ball_positions = window[:, 0, :3]  # (T, 3) normalized
    ball_coeffs = fit_spectral_coefficients(ball_positions)
    ball_field = make_field(latest[0, :3], ball_coeffs)

    # Cars: use latest frame orientation, but fit velocity from trajectory
    own_positions = window[:, 1, :3]
    own_coeffs = fit_spectral_coefficients(own_positions)
    # Override l=0 with boost from latest frame
    own_coeffs[0] = torch.clamp(latest[1, 9], min=0.01)
    # Override l=1 with forward direction from latest orientation
    yaw, pitch, roll = latest[1, 6], latest[1, 7], latest[1, 8]
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    forward = torch.stack([cp * cy, cp * sy, sp])
    own_coeffs[1:4] = forward
    own_car = make_field(latest[1, :3], own_coeffs)

    opp_positions = window[:, 2, :3]
    opp_coeffs = fit_spectral_coefficients(opp_positions)
    opp_coeffs[0] = 0.01  # opponent boost unknown
    yaw, pitch, roll = latest[2, 6], latest[2, 7], latest[2, 8]
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    forward = torch.stack([cp * cy, cp * sy, sp])
    opp_coeffs[1:4] = forward
    opp_car = make_field(latest[2, :3], opp_coeffs)

    # Boost: use latest frame only (pad states are instantaneous)
    boost = _boost_density_field(latest[3:9])

    return [ball_field, own_car, opp_car, boost], game_state
