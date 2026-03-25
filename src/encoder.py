"""
Shared Transformer Encoder
==========================
Converts a game-state packet into a 64-dimensional embedding.

State representation — one token per entity (expandable):
  token 0  ball          [bx,  by,  bz,  bvx, bvy, bvz, 0,   0      ]
  token 1  own car       [cx,  cy,  cz,  cvx, cvy, cvz, yaw, boost  ]
  token 2  opponent car  [ox,  oy,  oz,  ovx, ovy, ovz, oyaw,oboost ]

All values divided component-wise by NORM_DIVISORS so all inputs are in [-1, 1].
Adding more entities (teammates, boost pads) just means adding more tokens —
the transformer architecture requires no other code changes.

RLGym integration note
----------------------
This file reads raw RLBot GameTickPacket objects.  When RLGym is available,
add an adapter function `rlgym_obs_to_tokens(obs, player_idx)` that maps
RLGym's numpy observation to the same (1, N_TOKENS, 8) format.
The encoder and all downstream code stay unchanged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── normalization constants ───────────────────────────────────────────────────

FIELD_X   = 4096.0
FIELD_Y   = 5120.0
CEILING_Z = 2044.0
MAX_VEL   = 2300.0
MAX_BOOST = 100.0

# Component-wise divisors: [x, y, z, vx, vy, vz, yaw_or_0, boost_or_0]
NORM_DIVISORS = np.array(
    [FIELD_X, FIELD_Y, CEILING_Z, MAX_VEL, MAX_VEL, MAX_VEL, math.pi, MAX_BOOST],
    dtype=np.float32,
)

# Number of tokens fed to the transformer
N_TOKENS = 3   # ball, own car, opponent car — change here to extend

# Encoder output dimension
D_MODEL = 64


# ── state extraction ──────────────────────────────────────────────────────────

def state_to_tokens(packet, car_idx: int) -> np.ndarray:
    """
    Extract (1, N_TOKENS, 8) float32 array from a RLBot GameTickPacket.

    car_idx   index of "own" car (0=blue, 1=orange); opponent = car_idx ^ 1.

    Token layout:
      token 0  ball
      token 1  own car
      token 2  opponent car

    All values normalised by NORM_DIVISORS.
    """
    opp_idx = car_idx ^ 1

    ball = packet.game_ball.physics
    own  = packet.game_cars[car_idx].physics
    opp  = packet.game_cars[opp_idx].physics

    own_boost = float(packet.game_cars[car_idx].boost)
    opp_boost = float(packet.game_cars[opp_idx].boost)

    ball_token = np.array([
        ball.location.x, ball.location.y, ball.location.z,
        ball.velocity.x, ball.velocity.y, ball.velocity.z,
        0.0, 0.0,
    ], dtype=np.float32)

    own_token = np.array([
        own.location.x, own.location.y, own.location.z,
        own.velocity.x, own.velocity.y, own.velocity.z,
        float(own.rotation.yaw), own_boost,
    ], dtype=np.float32)

    opp_token = np.array([
        opp.location.x, opp.location.y, opp.location.z,
        opp.velocity.x, opp.velocity.y, opp.velocity.z,
        float(opp.rotation.yaw), opp_boost,
    ], dtype=np.float32)

    tokens = np.stack([ball_token, own_token, opp_token], axis=0) / NORM_DIVISORS
    return tokens[np.newaxis, ...]   # (1, N_TOKENS, 8)


# ── transformer building blocks ───────────────────────────────────────────────

class _TransformerEncoderLayer(keras.layers.Layer):
    """Pre-norm transformer encoder block (LayerNorm before each sublayer)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.attn  = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=0.0,
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(ffn_dim, activation='relu'),
            keras.layers.Dense(d_model),
        ])
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.drop1 = keras.layers.Dropout(0.1)
        self.drop2 = keras.layers.Dropout(0.1)

    def call(self, x, training=False):
        # Self-attention sublayer
        x_norm   = self.norm1(x)
        attn_out = self.attn(x_norm, x_norm, training=training)
        x        = x + self.drop1(attn_out, training=training)
        # FFN sublayer
        x_norm  = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x       = x + self.drop2(ffn_out, training=training)
        return x


# ── main encoder model ────────────────────────────────────────────────────────

class SharedTransformerEncoder(keras.Model):
    """
    Shared encoder used by both cars and both skill heads in every episode.

    Input:  (batch, N_TOKENS, 8)  — normalised token matrix
    Output: (batch, 64)           — embedding via mean-pool over tokens
    """

    N_HEADS  = 4
    N_LAYERS = 2
    FFN_DIM  = 128

    def __init__(self, d_model: int = D_MODEL, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        self.input_projection = keras.layers.Dense(d_model, name='input_proj')

        # Learned positional / entity-type embeddings (1, N_TOKENS, d_model)
        # Each entity position (ball, own car, opponent) gets its own offset.
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, N_TOKENS, d_model),
            initializer='random_normal',
            trainable=True,
        )

        self.transformer_layers = [
            _TransformerEncoderLayer(
                d_model=d_model, n_heads=self.N_HEADS, ffn_dim=self.FFN_DIM,
                name=f'transformer_{i}',
            )
            for i in range(self.N_LAYERS)
        ]

    def call(self, tokens, training=False):
        """
        tokens: (batch, N_TOKENS, 8)
        returns: (batch, 64)
        """
        x = self.input_projection(tokens)     # (batch, N_TOKENS, 64)
        x = x + self.pos_embedding            # broadcast add entity-type embeddings
        for layer in self.transformer_layers:
            x = layer(x, training=training)   # (batch, N_TOKENS, 64)
        x = tf.reduce_mean(x, axis=1)         # mean-pool → (batch, 64)
        return x

    def save(self, path: str) -> None:
        self.save_weights(path)

    def load(self, path: str) -> None:
        self.load_weights(path)

    @classmethod
    def load_from(cls, path: str) -> 'SharedTransformerEncoder':
        """Load a saved encoder.  Builds with dummy input first."""
        model = cls()
        model(tf.zeros((1, N_TOKENS, 8)))   # build variables
        model.load_weights(path)
        return model
