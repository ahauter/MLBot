"""Shared constants for SE(3) spectral field representation."""

import torch

# --- Spectral field parameters ---
L_MAX = 2                          # max spherical harmonic degree
N_COEFFS = (L_MAX + 1) ** 2       # = 9 coefficients per entity (1 + 3 + 5)
N_ENTITIES = 4                     # ball, own car, opponent car, boost density

# Degree slicing: coefficients are laid out as [l=0 (1), l=1 (3), l=2 (5)]
DEGREE_SLICES = {
    0: slice(0, 1),
    1: slice(1, 4),
    2: slice(4, 9),
}
DEGREE_SIZES = {0: 1, 1: 3, 2: 5}

# --- Arena geometry (unreal units) ---
FIELD_X = 4096.0
FIELD_Y = 5120.0
CEILING_Z = 2044.0

# --- Normalization ---
MAX_VEL = 2300.0
MAX_ANG_VEL = 5.5
MAX_BOOST = 100.0
MAX_SCORE = 10.0
MAX_TIME = 300.0

# --- Goal positions (unnormalized) ---
GOAL_BLUE = torch.tensor([0.0, -FIELD_Y, 0.0])
GOAL_ORANGE = torch.tensor([0.0, FIELD_Y, 0.0])

# --- Big boost pad positions (unnormalized) ---
BIG_PAD_POSITIONS = torch.tensor([
    [-3584.0,     0.0, 73.0],
    [ 3584.0,     0.0, 73.0],
    [-3072.0,  4096.0, 73.0],
    [ 3072.0,  4096.0, 73.0],
    [-3072.0, -4096.0, 73.0],
    [ 3072.0, -4096.0, 73.0],
], dtype=torch.float32)

# Big pad boost value
BIG_PAD_VALUE = 100.0
SMALL_PAD_VALUE = 12.0

# --- Interaction matrix defaults ---
DEFAULT_SPATIAL_SIGMA = 0.5        # in normalized coordinates
DEFAULT_TEMPERATURE = 1.0
