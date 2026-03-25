"""
d3rlpy Custom Encoder Factory
==============================
Bridges our SharedTransformerEncoder to d3rlpy's EncoderFactory interface.

d3rlpy expects flat observations. Our env provides frame-stacked flat vectors
of shape (T * N * F,). This encoder reshapes them back to (T, N, F) and
applies the spatiotemporal transformer.

Usage
-----
    from baseline_encoder_factory import TransformerEncoderFactory
    from d3rlpy.algos import AWACConfig

    factory = TransformerEncoderFactory(t_window=8)
    config = AWACConfig(
        actor_encoder_factory=factory,
        critic_encoder_factory=factory,
    )
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))

from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction
from d3rlpy.types import Shape

from encoder import (
    SharedTransformerEncoder,
    ENTITY_TYPE_IDS_1V1,
    N_TOKENS,
    TOKEN_FEATURES,
    D_MODEL,
)


class _TransformerEncoder(Encoder):
    """
    d3rlpy Encoder that wraps our SharedTransformerEncoder.

    Input:  (batch, T*N*F) flat vector
    Output: (batch, D_MODEL) embedding
    """

    def __init__(self, t_window: int):
        super().__init__()
        self.t_window = t_window
        self.n_tokens = N_TOKENS
        self.token_features = TOKEN_FEATURES
        self.encoder = SharedTransformerEncoder(d_model=D_MODEL)
        self._entity_ids = None  # lazily created on correct device

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Expected Tensor, got {type(x)}"
        assert x.dim() == 2, f"Expected 2D input (batch, flat), got {x.dim()}D"
        expected_size = self.t_window * self.n_tokens * self.token_features
        assert x.shape[1] == expected_size, \
            f"Input size {x.shape[1]} != expected {expected_size} " \
            f"(T={self.t_window} * N={self.n_tokens} * F={self.token_features})"
        batch = x.shape[0]
        # Reshape flat → (batch, T, N, F)
        tokens = x.view(batch, self.t_window, self.n_tokens, self.token_features)

        if self._entity_ids is None or self._entity_ids.device != x.device:
            self._entity_ids = torch.tensor(
                ENTITY_TYPE_IDS_1V1, dtype=torch.long, device=x.device
            )

        return self.encoder(tokens, self._entity_ids)  # (batch, D_MODEL)


class _TransformerEncoderWithAction(EncoderWithAction):
    """
    d3rlpy EncoderWithAction for the Q-critic: encodes observation + action.

    Input:  obs (batch, T*N*F), action (batch, action_size)
    Output: (batch, D_MODEL + action_size) — concatenated, then projected
    """

    def __init__(self, t_window: int, action_size: int):
        super().__init__()
        self.obs_encoder = _TransformerEncoder(t_window)
        self.fc = nn.Linear(D_MODEL + action_size, D_MODEL)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        obs_emb = self.obs_encoder(x)  # (batch, D_MODEL)
        combined = torch.cat([obs_emb, action], dim=-1)  # (batch, D_MODEL + A)
        return self.activation(self.fc(combined))  # (batch, D_MODEL)


@dataclasses.dataclass()
class TransformerEncoderFactory(EncoderFactory):
    """
    Factory that creates our transformer encoder for d3rlpy algorithms.

    Parameters
    ----------
    t_window : int
        Frame history length (must match the gym env's t_window).
    """

    t_window: int = 8

    def create(self, observation_shape: Shape) -> _TransformerEncoder:
        return _TransformerEncoder(self.t_window)

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> _TransformerEncoderWithAction:
        return _TransformerEncoderWithAction(self.t_window, action_size)

    @staticmethod
    def get_type() -> str:
        return 'transformer_spatiotemporal'
