"""
Scene assembly: SE3Fields → flat observation vector.

Assembles all entity fields, computes the interaction matrix,
and produces a fixed-size flat tensor suitable for an MLP policy.
"""

import torch
from torch import Tensor

from rlbot.constants import N_COEFFS, N_ENTITIES, DEFAULT_TEMPERATURE, DEFAULT_SPATIAL_SIGMA
from rlbot.env.fields import SE3Field, interaction_matrix
from rlbot.env.stadium import wall_distance_features
from rlbot.env.estimator import tokens_to_fields, tokens_window_to_fields


# Output dimensions
_N_WALL_FEATURES = 6
_PER_ENTITY = N_COEFFS + N_COEFFS + _N_WALL_FEATURES  # coeffs + covariance + wall = 24
_N_INTERACTION = N_ENTITIES * (N_ENTITIES - 1) // 2    # upper triangle = 6
_N_GAME_STATE = 3
OBS_DIM = _PER_ENTITY * N_ENTITIES + _N_INTERACTION + _N_GAME_STATE  # 24*4 + 6 + 3 = 105


def _upper_triangle(mat: Tensor) -> Tensor:
    """Extract upper triangle of a square matrix (excluding diagonal).

    For a 4×4 matrix, returns 6 values.
    """
    n = mat.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    return mat[indices[0], indices[1]]


def assemble_scene(tokens: Tensor,
                   tau: float = DEFAULT_TEMPERATURE,
                   sigma: float = DEFAULT_SPATIAL_SIGMA) -> Tensor:
    """Convert a single frame of tokens to flat observation vector.

    Args:
        tokens: (N_TOKENS, 10) token array
        tau: interaction matrix temperature
        sigma: spatial decay scale

    Returns:
        (OBS_DIM,) flat observation vector
    """
    fields, game_state = tokens_to_fields(tokens)
    return _fields_to_flat(fields, game_state, tau, sigma)


def assemble_scene_windowed(window: Tensor,
                            tau: float = DEFAULT_TEMPERATURE,
                            sigma: float = DEFAULT_SPATIAL_SIGMA) -> Tensor:
    """Convert a multi-frame token window to flat observation vector.

    Args:
        window: (T, N_TOKENS, 10) token window
        tau: interaction matrix temperature
        sigma: spatial decay scale

    Returns:
        (OBS_DIM,) flat observation vector
    """
    fields, game_state = tokens_window_to_fields(window)
    return _fields_to_flat(fields, game_state, tau, sigma)


def _fields_to_flat(fields: list[SE3Field], game_state: Tensor,
                    tau: float, sigma: float) -> Tensor:
    """Convert fields + game state to flat observation vector.

    Layout (105 floats total):
        [0:24]    entity 0 (ball):     9 coefficients + 9 covariance + 6 wall features
        [24:48]   entity 1 (own car):  9 coefficients + 9 covariance + 6 wall features
        [48:72]   entity 2 (opp car):  9 coefficients + 9 covariance + 6 wall features
        [72:96]   entity 3 (boost):    9 coefficients + 9 covariance + 6 wall features
        [96:102]  interaction matrix upper triangle (6 values)
        [102:105] game state (score_diff, time_rem, overtime)
    """
    parts = []

    # Per-entity features
    for field in fields:
        wall_feats = wall_distance_features(field.position)
        parts.append(field.coefficients)  # (9,)
        parts.append(field.covariance)    # (9,)
        parts.append(wall_feats)          # (6,)

    # Interaction matrix upper triangle
    I = interaction_matrix(fields, tau=tau, sigma=sigma)
    parts.append(_upper_triangle(I))  # (6,)

    # Game state
    parts.append(game_state)  # (3,)

    return torch.cat(parts)
