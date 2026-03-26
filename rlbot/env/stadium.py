"""
Static arena geometry: wall distances, pad positions, goal positions.

All functions operate in normalized coordinates (divided by field dimensions).
"""

import torch
from torch import Tensor

from rlbot.constants import (
    FIELD_X, FIELD_Y, CEILING_Z,
    BIG_PAD_POSITIONS, GOAL_BLUE, GOAL_ORANGE,
)

# Precomputed normalized pad positions
PAD_POSITIONS_NORM = BIG_PAD_POSITIONS.clone()
PAD_POSITIONS_NORM[:, 0] /= FIELD_X
PAD_POSITIONS_NORM[:, 1] /= FIELD_Y
PAD_POSITIONS_NORM[:, 2] /= CEILING_Z

# Precomputed normalized goal positions
GOAL_BLUE_NORM = GOAL_BLUE.clone()
GOAL_BLUE_NORM[0] /= FIELD_X
GOAL_BLUE_NORM[1] /= FIELD_Y
GOAL_BLUE_NORM[2] /= CEILING_Z

GOAL_ORANGE_NORM = GOAL_ORANGE.clone()
GOAL_ORANGE_NORM[0] /= FIELD_X
GOAL_ORANGE_NORM[1] /= FIELD_Y
GOAL_ORANGE_NORM[2] /= CEILING_Z

# Max possible distance in normalized arena (diagonal)
_MAX_DIST = torch.sqrt(torch.tensor(1.0 + 1.0 + 1.0))


def wall_distance_features(position: Tensor) -> Tensor:
    """Compute 6 normalized distance features for a position.

    All inputs and outputs in normalized coordinates (pos / field_dims).

    Features:
        [0] distance to nearest side wall (x)
        [1] distance to nearest end wall (y)
        [2] height / distance to floor (z=0) or ceiling (z=1)
        [3] distance to nearest corner (approximated)
        [4] distance to own goal (blue, y=-1)
        [5] distance to opponent goal (orange, y=+1)

    Args:
        position: (3,) normalized position

    Returns:
        (6,) features in approximately [0, 1] range.
    """
    x, y, z = position[0], position[1], position[2]

    # Side wall: distance to nearest x boundary (±1 in normalized)
    d_side = 1.0 - torch.abs(x)

    # End wall: distance to nearest y boundary (±1 in normalized)
    d_end = 1.0 - torch.abs(y)

    # Height: min of floor distance and ceiling distance
    d_height = torch.min(z, 1.0 - z)

    # Corner: distance to nearest of the 4 corners at (±1, ±1)
    # Use min of (d_side, d_end) as a simple proxy
    d_corner = torch.sqrt(d_side**2 + d_end**2) / torch.sqrt(torch.tensor(2.0))

    # Goal distances
    d_own_goal = torch.norm(position - GOAL_BLUE_NORM)
    d_opp_goal = torch.norm(position - GOAL_ORANGE_NORM)

    # Normalize goal distances by max possible distance
    d_own_goal = d_own_goal / _MAX_DIST
    d_opp_goal = d_opp_goal / _MAX_DIST

    return torch.stack([d_side, d_end, d_height, d_corner,
                        d_own_goal, d_opp_goal])
