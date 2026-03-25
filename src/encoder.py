"""
Shared Transformer Encoder
==========================
Converts a game-state packet into a 64-dimensional embedding.

State representation — one token per entity:
  token 0    ball          [x, y, z, vx, vy, vz, av_x, av_y, av_z, 0       ]
  token 1    own car       [x, y, z, vx, vy, vz, yaw,  pitch, roll, boost   ]
  token 2    opponent car  [x, y, z, vx, vy, vz, yaw,  pitch, roll, 0       ]
                           └─ opponent boost is intentionally hidden (not
                              observable in real Rocket League matches)
  tokens 3-8 big boost pad [x, y, z, active, 0, 0, 0, 0, 0, 0               ]
  token 9    game state    [score_diff, time_rem, overtime, 0, 0, 0, 0, 0, 0, 0]

TOKEN_FEATURES = 10.  All values are individually normalised to [-1, 1].

Adding more entities (teammates, small pads) just means adding more tokens —
the transformer architecture requires no other code changes.

RLGym integration
-----------------
`rlgym_obs_to_tokens(obs, player_idx)` maps a flat RLGym observation vector
(produced by our custom TokenObsBuilder) to (1, N_TOKENS, TOKEN_FEATURES).
The encoder and all downstream code stay unchanged across both paths.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

# ── normalization constants ───────────────────────────────────────────────────

FIELD_X      = 4096.0
FIELD_Y      = 5120.0
CEILING_Z    = 2044.0
MAX_VEL      = 2300.0
MAX_ANG_VEL  = 5.5      # rad/s (approximate max in Rocket League)
MAX_BOOST    = 100.0
MAX_SCORE    = 10.0     # normalise score diff; clipped to [-1, 1] after division
MAX_TIME     = 300.0    # regulation length in seconds

# ── token dimensions ──────────────────────────────────────────────────────────

TOKEN_FEATURES = 10

# 1 ball + 1 own car + 1 opponent + 6 big boost pads + 1 game-state
N_TOKENS = 10

# Encoder output dimension
D_MODEL = 64

# Number of big boost pads expected — must match N_TOKENS - 3
_N_BIG_PADS = 6


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
    car_idx  : index of "own" car (0 = blue, 1 = orange); opponent = car_idx ^ 1
    big_pads : list of BoostPad objects from BoostPadTracker.get_full_boosts()
               (must have a `.location` Vec3 and `.is_active` bool).
               Pass None to leave pad tokens zeroed (e.g. first tick before
               BoostPadTracker is initialised).

    Token layout
    ------------
    idx  entity        features
    0    ball           x y z  vx vy vz  av_x av_y av_z  0
    1    own car        x y z  vx vy vz  yaw pitch roll   boost
    2    opponent car   x y z  vx vy vz  yaw pitch roll   0  ← boost hidden
    3-8  big boost pad  x y z  active    0 0 0 0 0 0
    9    game state     score_diff time_rem overtime  0…
    """
    opp_idx = car_idx ^ 1

    ball = packet.game_ball.physics
    own  = packet.game_cars[car_idx].physics
    opp  = packet.game_cars[opp_idx].physics

    own_boost = float(packet.game_cars[car_idx].boost)

    # ── token 0: ball ─────────────────────────────────────────────────────────
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

    # ── token 1: own car ──────────────────────────────────────────────────────
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

    # ── token 2: opponent car — boost intentionally hidden ────────────────────
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
        0.0,   # <-- opponent boost is NOT given to the encoder
    ], dtype=np.float32)

    # ── tokens 3-8: big boost pads ────────────────────────────────────────────
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

    # ── token 9: game state ───────────────────────────────────────────────────
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
    Map a flat RLGym observation vector to (1, N_TOKENS, TOKEN_FEATURES).

    Assumes the observation was produced by TokenObsBuilder (training/rlgym_env.py),
    which serialises the token matrix row-by-row into a flat array of length
    N_TOKENS * TOKEN_FEATURES.  player_idx is accepted for API symmetry but is
    not used here — the obs builder already encodes the observation from the
    requesting player's perspective.
    """
    expected = N_TOKENS * TOKEN_FEATURES
    if obs.shape[0] != expected:
        raise ValueError(
            f'Expected obs length {expected} (N_TOKENS={N_TOKENS} × '
            f'TOKEN_FEATURES={TOKEN_FEATURES}), got {obs.shape[0]}'
        )
    return obs.reshape(1, N_TOKENS, TOKEN_FEATURES).astype(np.float32)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training/eval mode is inherited from the parent module via model.train()/eval()
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x        = x + self.drop1(attn_out)
        x        = x + self.drop2(self.ffn(self.norm2(x)))
        return x


# ── main encoder model ────────────────────────────────────────────────────────

class SharedTransformerEncoder(nn.Module):
    """
    Shared encoder used by both cars and both skill heads in every episode.

    Input:  (batch, N_TOKENS, TOKEN_FEATURES)  — normalised token matrix
    Output: (batch, D_MODEL)                   — embedding via mean-pool over tokens
    """

    N_HEADS  = 4
    N_LAYERS = 2
    FFN_DIM  = 128

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(TOKEN_FEATURES, d_model)

        # Learned per-entity-position embeddings (1, N_TOKENS, d_model).
        # Each entity (ball, own car, opponent, each boost pad, game-state) gets
        # its own learned offset so the transformer can distinguish entity types.
        self.pos_embedding = nn.Parameter(torch.randn(1, N_TOKENS, d_model) * 0.02)

        self.transformer_layers = nn.ModuleList([
            _TransformerEncoderLayer(
                d_model=d_model, n_heads=self.N_HEADS, ffn_dim=self.FFN_DIM,
            )
            for _ in range(self.N_LAYERS)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch, N_TOKENS, TOKEN_FEATURES)
        returns: (batch, D_MODEL)
        """
        x = self.input_projection(tokens)   # (batch, N_TOKENS, D_MODEL)
        x = x + self.pos_embedding          # broadcast add entity-type embeddings
        for layer in self.transformer_layers:
            x = layer(x)                   # (batch, N_TOKENS, D_MODEL)
        return x.mean(dim=1)               # mean-pool → (batch, D_MODEL)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))

    @classmethod
    def load_from(cls, path: str) -> 'SharedTransformerEncoder':
        """Load a saved encoder from a .pt checkpoint."""
        model = cls()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model
