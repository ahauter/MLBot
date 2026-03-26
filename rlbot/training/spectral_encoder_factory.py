"""
spectral_encoder_factory.py
===========================
d3rlpy encoder factory for SE(3) spectral field observations.

The spectral representation (105-dim) already encodes rotational structure,
entity interactions, and spatial features, so a simple MLP suffices —
no transformer needed.

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

# Output dimension — matches existing D_MODEL for compatibility
_OUTPUT_DIM = 64


class _SpectralEncoder(Encoder):
    """MLP encoder for 105-dim spectral observations.

    Architecture: input → hidden → hidden → hidden → output (64)
    """

    def __init__(self, obs_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, _OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self.net(x)


class _SpectralEncoderWithAction(EncoderWithAction):
    """MLP encoder for observations + actions (used by Q-critic).

    Encodes obs through the same MLP, then concatenates action and
    projects back to output dimension.
    """

    def __init__(self, obs_dim: int, action_size: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.obs_encoder = _SpectralEncoder(obs_dim, hidden_dim, n_layers)
        self.fc = nn.Linear(_OUTPUT_DIM + action_size, _OUTPUT_DIM)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        action: torch.Tensor,
    ) -> torch.Tensor:
        obs_emb = self.obs_encoder(x)
        combined = torch.cat([obs_emb, action], dim=-1)
        return self.activation(self.fc(combined))


@dataclasses.dataclass()
class SpectralEncoderFactory(EncoderFactory):
    """Factory that creates MLP encoders for spectral field observations.

    Parameters
    ----------
    hidden_dim : width of hidden layers
    n_layers : number of hidden layers before the output projection
    """

    hidden_dim: int = 256
    n_layers: int = 3

    def create(self, observation_shape: Shape) -> _SpectralEncoder:
        obs_dim = observation_shape[0]
        return _SpectralEncoder(obs_dim, self.hidden_dim, self.n_layers)

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> _SpectralEncoderWithAction:
        obs_dim = observation_shape[0]
        return _SpectralEncoderWithAction(
            obs_dim, action_size, self.hidden_dim, self.n_layers
        )

    @staticmethod
    def get_type() -> str:
        return "spectral_mlp"
