"""
Learned spectral encoder: tokens → SE3Fields → flat observation.

A neural network that maps raw token observations through an SE(3) spectral
bottleneck. Each entity token is projected to (position, l=0, l=1, l=2)
coefficients by a small MLP, then the SE(3) machinery (interaction matrix,
wall features) is applied as differentiable structure on top.

The network is trained end-to-end with the policy — the SE(3) spectral
structure acts as an inductive bias forcing the representation through
a geometrically meaningful bottleneck.
"""

import torch
import torch.nn as nn
from torch import Tensor

from rlbot.constants import N_COEFFS, N_ENTITIES, L_MAX, DEGREE_SLICES
from rlbot.env.fields import SE3Field, interaction_matrix
from rlbot.env.stadium import wall_distance_features


# Token indices
_BALL = 0
_OWN_CAR = 1
_OPP_CAR = 2
_PADS_START = 3
_PADS_END = 9
_GAME_STATE = 9

# Observation layout
_N_WALL_FEATURES = 6
_PER_ENTITY = N_COEFFS + N_COEFFS + _N_WALL_FEATURES  # 9 + 9 + 6 = 24
_N_INTERACTION = N_ENTITIES * (N_ENTITIES - 1) // 2    # 6
_N_GAME_STATE = 3
OBS_DIM = _PER_ENTITY * N_ENTITIES + _N_INTERACTION + _N_GAME_STATE  # 105


class EntityEncoder(nn.Module):
    """Small MLP that maps a single entity token to SE3Field components.

    Input: (token_dim,) raw token features
    Output: position (3,), coefficients (9,), log_covariance (9,)
    """

    def __init__(self, token_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pos_head = nn.Linear(hidden_dim, 3)
        self.coeff_head = nn.Linear(hidden_dim, N_COEFFS)
        self.cov_head = nn.Linear(hidden_dim, N_COEFFS)

    def forward(self, token: Tensor) -> SE3Field:
        """Map token → SE3Field.

        Args:
            token: (..., token_dim) token features

        Returns:
            SE3Field with position, coefficients, covariance
        """
        h = self.net(token)
        position = self.pos_head(h)
        coefficients = self.coeff_head(h)
        # Softplus on log-covariance to ensure positive covariance
        covariance = nn.functional.softplus(self.cov_head(h))
        return SE3Field(position, coefficients, covariance)


class BoostEncoder(nn.Module):
    """Encodes all 6 boost pad tokens into a single aggregate SE3Field.

    Processes each pad through a shared MLP, then aggregates via
    attention-weighted pooling to produce a single density field.
    """

    def __init__(self, token_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.pad_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Attention weight per pad (scalar)
        self.attn = nn.Linear(hidden_dim, 1)
        # Aggregate → field components
        self.pos_head = nn.Linear(hidden_dim, 3)
        self.coeff_head = nn.Linear(hidden_dim, N_COEFFS)
        self.cov_head = nn.Linear(hidden_dim, N_COEFFS)

    def forward(self, pad_tokens: Tensor) -> SE3Field:
        """Map pad tokens → single aggregate SE3Field.

        Args:
            pad_tokens: (..., N_PADS, token_dim) boost pad tokens

        Returns:
            SE3Field representing the aggregate boost density
        """
        h = self.pad_net(pad_tokens)  # (..., N_PADS, hidden)

        # Attention-weighted aggregation
        weights = torch.softmax(self.attn(h), dim=-2)  # (..., N_PADS, 1)
        aggregated = (weights * h).sum(dim=-2)  # (..., hidden)

        position = self.pos_head(aggregated)
        coefficients = self.coeff_head(aggregated)
        covariance = nn.functional.softplus(self.cov_head(aggregated))
        return SE3Field(position, coefficients, covariance)


class SpectralEncoder(nn.Module):
    """Learned encoder: raw tokens → SE3Fields → flat observation.

    Architecture:
        1. Per-entity MLPs map tokens to SE3Field (position, coeffs, covariance)
        2. Interaction matrix computed from learned fields (differentiable)
        3. Wall features computed from learned positions
        4. Everything concatenated into 105-dim flat vector

    The SE(3) structure (interaction affinity, wall geometry) provides
    geometric inductive bias — the network must express its understanding
    through the spectral field format.
    """

    def __init__(self, token_dim: int = 10, hidden_dim: int = 64,
                 tau: float = 1.0, sigma: float = 0.5):
        super().__init__()
        self.tau = tau
        self.sigma = sigma

        # Separate encoders per entity type (different token semantics)
        self.ball_encoder = EntityEncoder(token_dim, hidden_dim)
        self.own_car_encoder = EntityEncoder(token_dim, hidden_dim)
        self.opp_car_encoder = EntityEncoder(token_dim, hidden_dim)
        self.boost_encoder = BoostEncoder(token_dim, hidden_dim)

        # Game state projection (3 inputs → 3 outputs, learnable transform)
        self.game_state_head = nn.Linear(token_dim, _N_GAME_STATE)

    def forward(self, tokens: Tensor) -> Tensor:
        """Convert raw tokens to flat observation via spectral bottleneck.

        Args:
            tokens: (N_TOKENS, token_dim) single frame, or
                    (batch, N_TOKENS, token_dim) batched

        Returns:
            (OBS_DIM,) or (batch, OBS_DIM) flat observation
        """
        batched = tokens.dim() == 3
        if not batched:
            tokens = tokens.unsqueeze(0)

        batch_size = tokens.shape[0]
        device = tokens.device

        # Encode each entity
        ball_field = self.ball_encoder(tokens[:, _BALL])
        own_car_field = self.own_car_encoder(tokens[:, _OWN_CAR])
        opp_car_field = self.opp_car_encoder(tokens[:, _OPP_CAR])
        boost_field = self.boost_encoder(tokens[:, _PADS_START:_PADS_END])

        # Game state
        game_state = self.game_state_head(tokens[:, _GAME_STATE])

        # Assemble per-sample (interaction matrix is not trivially batchable)
        results = []
        for b in range(batch_size):
            fields = [
                SE3Field(ball_field.position[b], ball_field.coefficients[b], ball_field.covariance[b]),
                SE3Field(own_car_field.position[b], own_car_field.coefficients[b], own_car_field.covariance[b]),
                SE3Field(opp_car_field.position[b], opp_car_field.coefficients[b], opp_car_field.covariance[b]),
                SE3Field(boost_field.position[b], boost_field.coefficients[b], boost_field.covariance[b]),
            ]
            flat = _fields_to_flat(fields, game_state[b], self.tau, self.sigma)
            results.append(flat)

        out = torch.stack(results)  # (batch, OBS_DIM)

        if not batched:
            out = out.squeeze(0)
        return out


def _upper_triangle(mat: Tensor) -> Tensor:
    """Extract upper triangle of a square matrix (excluding diagonal)."""
    n = mat.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    return mat[indices[0], indices[1]]


def _fields_to_flat(fields: list[SE3Field], game_state: Tensor,
                    tau: float, sigma: float) -> Tensor:
    """Convert fields + game state to flat observation vector.

    Same layout as scene.py but with learned field values.
    """
    parts = []

    for field in fields:
        wall_feats = wall_distance_features(field.position)
        parts.append(field.coefficients)
        parts.append(field.covariance)
        parts.append(wall_feats)

    I = interaction_matrix(fields, tau=tau, sigma=sigma)
    parts.append(_upper_triangle(I))

    parts.append(game_state)

    return torch.cat(parts)
