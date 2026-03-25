"""
Shared Spatiotemporal Transformer Encoder
==========================================
Converts a sliding window of T game-state snapshots into a 64-dim embedding.

Each snapshot contains N entity tokens (N varies by game mode):
  Standard 1v1 (N=10):
    token 0    ball          [x, y, z, vx, vy, vz, av_x, av_y, av_z, 0       ]
    token 1    own car       [x, y, z, vx, vy, vz, yaw,  pitch, roll, boost   ]
    token 2    opponent car  [x, y, z, vx, vy, vz, yaw,  pitch, roll, 0       ]
                             └─ opponent boost intentionally hidden
    tokens 3-8 big boost pad [x, y, z, active, 0, 0, 0, 0, 0, 0               ]
    token 9    game state    [score_diff, time_rem, overtime, 0, …             ]

TOKEN_FEATURES = 10.  All values individually normalised to [-1, 1].

Entity type IDs (shared contract across all game modes):
    0 = ball
    1 = own car
    2 = opponent car
    3 = boost pad
    4 = game state

ENTITY_TYPE_IDS_1V1 maps the standard 10 slots to these IDs:
    [0, 1, 2, 3, 3, 3, 3, 3, 3, 4]

Temporal window
---------------
forward() accepts either:
  (batch, N, F)        — single-step legacy path (auto-unsqueezed to T=1)
  (batch, T, N, F)     — sliding window of T steps

entity_type_ids (N,) must be supplied by the caller; it varies per game mode.

During AWAC training, pass entity_perm=(N,) int64 permutation to shuffle the
entity axis as data augmentation.  The same permutation is applied across all
T timesteps.  Never use entity_perm at inference.

Adding more entities (teammates, small pads) just means more tokens per step —
no other code changes required.
"""

from __future__ import annotations

import functools
import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# ── normalisation constants ───────────────────────────────────────────────────

FIELD_X      = 4096.0
FIELD_Y      = 5120.0
CEILING_Z    = 2044.0
MAX_VEL      = 2300.0
MAX_ANG_VEL  = 5.5
MAX_BOOST    = 100.0
MAX_SCORE    = 10.0
MAX_TIME     = 300.0

# ── token dimensions ──────────────────────────────────────────────────────────

TOKEN_FEATURES = 10
D_MODEL        = 64

# N_TOKENS=10 kept for backward compatibility with 1v1 tokenisers.
# New code should read N from the token array shape, not this constant.
N_TOKENS = 10

# ── temporal window ───────────────────────────────────────────────────────────

T_WINDOW = 4   # default sliding window size (4 steps × ~66 ms ≈ 266 ms)
T_MAX    = 8   # maximum window size the model supports

# ── entity type IDs ───────────────────────────────────────────────────────────

N_ENTITY_TYPES = 5   # ball=0, own_car=1, opp_car=2, boost_pad=3, game_state=4

# Standard 1v1 mapping: 10 token slots → type IDs
ENTITY_TYPE_IDS_1V1: List[int] = [0, 1, 2, 3, 3, 3, 3, 3, 3, 4]

_N_BIG_PADS = 6


# ── causal mask ───────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=16)
def _causal_mask(T: int, N: int, device_str: str) -> torch.Tensor:
    """
    Block-lower-triangular additive attention mask of shape (T*N, T*N).

    Entry [i, j] = -inf  if token j belongs to a later timestep than token i,
                 = 0.0   otherwise (token can attend).

    Cached by (T, N, device_str) so it is only computed once per unique combo.
    The returned tensor must not be modified in-place.
    """
    size = T * N
    # which timestep does each position belong to?
    step = torch.arange(size, device=device_str) // N   # (size,)
    mask = torch.zeros(size, size, device=device_str)
    mask[step.unsqueeze(1) < step.unsqueeze(0)] = float('-inf')
    return mask


# ── state extraction ──────────────────────────────────────────────────────────

def state_to_tokens(
    packet,
    car_idx: int,
    big_pads: Optional[List] = None,
) -> np.ndarray:
    """
    Extract (1, N_TOKENS, TOKEN_FEATURES) float32 array from a RLBot
    GameTickPacket.

    Parameters
    ----------
    packet   : RLBot GameTickPacket
    car_idx  : index of "own" car (0 = blue, 1 = orange)
    big_pads : list of BoostPad objects from BoostPadTracker.get_full_boosts()
               Pass None to leave pad tokens zeroed.
    """
    opp_idx = car_idx ^ 1

    ball = packet.game_ball.physics
    own  = packet.game_cars[car_idx].physics
    opp  = packet.game_cars[opp_idx].physics

    own_boost = float(packet.game_cars[car_idx].boost)

    ball_token = np.array([
        ball.location.x         / FIELD_X,
        ball.location.y         / FIELD_Y,
        ball.location.z         / CEILING_Z,
        ball.velocity.x         / MAX_VEL,
        ball.velocity.y         / MAX_VEL,
        ball.velocity.z         / MAX_VEL,
        ball.angular_velocity.x / MAX_ANG_VEL,
        ball.angular_velocity.y / MAX_ANG_VEL,
        ball.angular_velocity.z / MAX_ANG_VEL,
        0.0,
    ], dtype=np.float32)

    own_token = np.array([
        own.location.x  / FIELD_X,
        own.location.y  / FIELD_Y,
        own.location.z  / CEILING_Z,
        own.velocity.x  / MAX_VEL,
        own.velocity.y  / MAX_VEL,
        own.velocity.z  / MAX_VEL,
        float(own.rotation.yaw)   / math.pi,
        float(own.rotation.pitch) / math.pi,
        float(own.rotation.roll)  / math.pi,
        own_boost / MAX_BOOST,
    ], dtype=np.float32)

    opp_token = np.array([
        opp.location.x  / FIELD_X,
        opp.location.y  / FIELD_Y,
        opp.location.z  / CEILING_Z,
        opp.velocity.x  / MAX_VEL,
        opp.velocity.y  / MAX_VEL,
        opp.velocity.z  / MAX_VEL,
        float(opp.rotation.yaw)   / math.pi,
        float(opp.rotation.pitch) / math.pi,
        float(opp.rotation.roll)  / math.pi,
        0.0,
    ], dtype=np.float32)

    pad_tokens = []
    if big_pads is not None:
        for pad in big_pads[:_N_BIG_PADS]:
            pad_tokens.append(np.array([
                pad.location.x / FIELD_X,
                pad.location.y / FIELD_Y,
                pad.location.z / CEILING_Z,
                float(pad.is_active),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32))
    while len(pad_tokens) < _N_BIG_PADS:
        pad_tokens.append(np.zeros(TOKEN_FEATURES, dtype=np.float32))

    blue_score   = float(packet.teams[0].score)
    orange_score = float(packet.teams[1].score)
    score_diff   = (blue_score - orange_score) if car_idx == 0 else (orange_score - blue_score)
    time_rem     = float(getattr(packet.game_info, 'game_time_remaining', 0.0))
    overtime     = float(getattr(packet.game_info, 'is_overtime', False))

    gs_token = np.array([
        np.clip(score_diff / MAX_SCORE, -1.0, 1.0),
        np.clip(time_rem   / MAX_TIME,   0.0, 1.0),
        overtime,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)

    tokens = np.stack(
        [ball_token, own_token, opp_token] + pad_tokens + [gs_token],
        axis=0,
    )                            # (N_TOKENS, TOKEN_FEATURES)
    return tokens[np.newaxis, ...]   # (1, N_TOKENS, TOKEN_FEATURES)


# ── RLGym adapter ─────────────────────────────────────────────────────────────

def rlgym_obs_to_tokens(obs: np.ndarray, player_idx: int) -> np.ndarray:
    """
    Map a flat RLGym observation vector to (1, N, TOKEN_FEATURES).

    N is inferred from obs length: N = len(obs) // TOKEN_FEATURES.
    Assumes the observation was produced by TokenObsBuilder.
    """
    if obs.shape[0] % TOKEN_FEATURES != 0:
        raise ValueError(
            f'obs length {obs.shape[0]} is not divisible by '
            f'TOKEN_FEATURES={TOKEN_FEATURES}'
        )
    N = obs.shape[0] // TOKEN_FEATURES
    return obs.reshape(1, N, TOKEN_FEATURES).astype(np.float32)


# ── transformer building blocks ───────────────────────────────────────────────

class _TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder block (LayerNorm before each sublayer)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=0.0, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x        = x + self.drop1(attn_out)
        x        = x + self.drop2(self.ffn(self.norm2(x)))
        return x


# ── main encoder model ────────────────────────────────────────────────────────

class SharedTransformerEncoder(nn.Module):
    """
    Spatiotemporal encoder: sliding T-step window → (batch, D_MODEL) embedding.

    Input:  (batch, T, N, TOKEN_FEATURES)  — window of T steps, N entities each
        or  (batch, N, TOKEN_FEATURES)     — single-step (auto-unsqueezed to T=1)
    Output: (batch, D_MODEL)               — via mean-pool over last-step tokens

    entity_type_ids: (N,) int64 tensor mapping each slot to its type ID.
                     Must be supplied by the caller; varies per game mode.
                     Use ENTITY_TYPE_IDS_1V1 for standard 1v1.

    entity_perm: optional (N,) int64 permutation for AWAC data augmentation.
                 Shuffles the entity axis identically across all T timesteps.
                 Never use at inference.
    """

    N_HEADS  = 4
    N_LAYERS = 2
    FFN_DIM  = 128

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(TOKEN_FEATURES, d_model)

        # Entity type embedding — replaces the old slot-indexed pos_embedding.
        # Keyed by entity type ID (0-4), shared across all game modes.
        self.entity_type_embedding = nn.Embedding(N_ENTITY_TYPES, d_model)

        # Time embedding — one learnable vector per timestep index (0 = oldest).
        self.time_embedding = nn.Embedding(T_MAX, d_model)

        self.transformer_layers = nn.ModuleList([
            _TransformerEncoderLayer(
                d_model=d_model, n_heads=self.N_HEADS, ffn_dim=self.FFN_DIM,
            )
            for _ in range(self.N_LAYERS)
        ])

    def forward(
        self,
        tokens: torch.Tensor,
        entity_type_ids: Union[torch.Tensor, List[int]],
        entity_perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens:          (batch, T, N, F)  or  (batch, N, F)
        entity_type_ids: (N,) — type ID per entity slot; varies by game mode
        entity_perm:     (N,) optional permutation for AWAC shuffling (training only)
        returns:         (batch, D_MODEL)
        """
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(1)          # (batch, 1, N, F)
        batch, T, N, F = tokens.shape

        # Convert entity_type_ids to a tensor on the right device
        if not isinstance(entity_type_ids, torch.Tensor):
            entity_type_ids = torch.tensor(
                entity_type_ids, dtype=torch.long, device=tokens.device)
        else:
            entity_type_ids = entity_type_ids.to(tokens.device)

        # Apply entity permutation (AWAC data augmentation — training only)
        if entity_perm is not None:
            tokens          = tokens[:, :, entity_perm, :]
            entity_type_ids = entity_type_ids[entity_perm]

        x = tokens.reshape(batch, T * N, F)
        x = self.input_projection(x)                          # (batch, T*N, D)

        # Entity type embeddings — tiled across T timesteps
        etype = entity_type_ids.unsqueeze(0).expand(T, N).reshape(1, T * N)
        x = x + self.entity_type_embedding(etype)             # (batch, T*N, D)

        # Time embeddings — index 0=oldest, T-1=most recent
        t_ids = (
            torch.arange(T, device=x.device)
            .unsqueeze(1).expand(T, N).reshape(1, T * N)
        )
        x = x + self.time_embedding(t_ids)                    # (batch, T*N, D)

        # Causal mask: entities at step t cannot attend to step t+1 .. T-1
        mask = _causal_mask(T, N, str(x.device))
        for layer in self.transformer_layers:
            x = layer(x, attn_mask=mask)                      # (batch, T*N, D)

        # Pool over the most-recent timestep's tokens only.
        # History is already encoded into these positions via cross-time attention.
        return x[:, (T - 1) * N:, :].mean(dim=1)             # (batch, D)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))

    @classmethod
    def load_from(cls, path: str) -> 'SharedTransformerEncoder':
        """Load a saved encoder from a .pt checkpoint."""
        model = cls()
        state = torch.load(path, map_location='cpu')
        # Migration: old checkpoints have pos_embedding — drop it silently.
        state.pop('pos_embedding', None)
        model.load_state_dict(state, strict=False)
        return model
