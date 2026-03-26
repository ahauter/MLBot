"""
spectral_encoder_factory.py
===========================
d3rlpy encoder factory with learned SE(3) spectral bottleneck.

The encoder takes raw flat token observations (100-dim = 10 tokens × 10 features),
passes them through a learned SpectralEncoder (tokens → SE3Fields → 105-dim),
then through an MLP policy head (105 → 64). The SE(3) structure (interaction
matrix, wall features, coefficient decomposition) acts as an inductive bias —
the network must express its understanding through the spectral field format.

The entire pipeline is trained end-to-end via the AWAC loss.

Usage
-----
    from rlbot.training.spectral_encoder_factory import SpectralEncoderFactory
    from d3rlpy.algos import AWACConfig

    factory = SpectralEncoderFactory()
    config = AWACConfig(
        actor_encoder_factory=factory,
        critic_encoder_factory=factory,
    )
"""

from __future__ import annotations

import dataclasses
from typing import Sequence, Union

import torch
import torch.nn as nn

from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction
from d3rlpy.types import Shape

from rlbot.env.spectral_encoder import SpectralEncoder, OBS_DIM

# Token layout constants
_N_TOKENS = 10
_TOKEN_FEATURES = 10
_FLAT_TOKEN_DIM = _N_TOKENS * _TOKEN_FEATURES  # 100

# Output dimension (matches existing D_MODEL)
_OUTPUT_DIM = 64


class _LearnedSpectralEncoder(Encoder):
    """d3rlpy Encoder wrapping the learned spectral pipeline.

    Input:  (batch, 100) flat token vector
    Output: (batch, 64)  policy embedding

    Internally: reshape to (batch, 10, 10) tokens → SpectralEncoder → (batch, 105)
    → MLP → (batch, 64).
    """

    def __init__(self, encoder_hidden: int, policy_hidden: int, policy_layers: int):
        super().__init__()
        # Learned spectral encoder: tokens → 105-dim spectral observation
        self.spectral = SpectralEncoder(
            token_dim=_TOKEN_FEATURES,
            hidden_dim=encoder_hidden,
        )
        # Policy MLP: 105-dim spectral → 64-dim embedding
        layers: list[nn.Module] = []
        in_dim = OBS_DIM
        for _ in range(policy_layers):
            layers.append(nn.Linear(in_dim, policy_hidden))
            layers.append(nn.ReLU())
            in_dim = policy_hidden
        layers.append(nn.Linear(in_dim, _OUTPUT_DIM))
        self.policy_head = nn.Sequential(*layers)

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        batch = x.shape[0]
        # Reshape flat (100,) → (10, 10) tokens
        tokens = x.view(batch, _N_TOKENS, _TOKEN_FEATURES)
        # Spectral bottleneck: tokens → 105-dim structured observation
        spectral_obs = self.spectral(tokens)  # (batch, 105)
        # Policy head
        return self.policy_head(spectral_obs)  # (batch, 64)


class _LearnedSpectralEncoderWithAction(EncoderWithAction):
    """d3rlpy EncoderWithAction for the Q-critic.

    Encodes observation through the spectral pipeline, concatenates action,
    then projects to output dimension.
    """

    def __init__(self, encoder_hidden: int, policy_hidden: int,
                 policy_layers: int, action_size: int):
        super().__init__()
        self.obs_encoder = _LearnedSpectralEncoder(
            encoder_hidden, policy_hidden, policy_layers
        )
        self.fc = nn.Linear(_OUTPUT_DIM + action_size, _OUTPUT_DIM)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        obs_emb = self.obs_encoder(x)  # (batch, 64)
        combined = torch.cat([obs_emb, action], dim=-1)
        return self.activation(self.fc(combined))  # (batch, 64)


@dataclasses.dataclass()
class SpectralEncoderFactory(EncoderFactory):
    """Factory for the learned spectral encoder pipeline.

    Parameters
    ----------
    encoder_hidden : hidden dim for per-entity MLPs in SpectralEncoder
    policy_hidden : hidden dim for the post-bottleneck policy MLP
    policy_layers : number of hidden layers in the policy MLP
    """

    encoder_hidden: int = 64
    policy_hidden: int = 256
    policy_layers: int = 2

    def create(self, observation_shape: Shape) -> _LearnedSpectralEncoder:
        return _LearnedSpectralEncoder(
            self.encoder_hidden, self.policy_hidden, self.policy_layers,
        )

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> _LearnedSpectralEncoderWithAction:
        return _LearnedSpectralEncoderWithAction(
            self.encoder_hidden, self.policy_hidden,
            self.policy_layers, action_size,
        )

    @staticmethod
    def get_type() -> str:
        return "learned_spectral"
