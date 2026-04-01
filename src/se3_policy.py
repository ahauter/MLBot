"""
SE3 Policy Heads
================
MLP policy + value heads for the SE(3) spectral field representation.

Takes 128-dim coefficient observation from SE3Encoder and outputs:
  policy  (batch, 8)  — 5 analog [-1,1] + 3 binary [0,1]
  value   (batch, 1)  — state-value estimate

Hidden dim = 64 (~13K total params).

Action layout (same as PolicyHead):
  [0] throttle    tanh  [-1,  1]
  [1] steer       tanh  [-1,  1]
  [2] pitch       tanh  [-1,  1]
  [3] yaw_ctrl    tanh  [-1,  1]
  [4] roll        tanh  [-1,  1]
  [5] jump        sigmoid → threshold 0.5
  [6] boost       sigmoid → threshold 0.5
  [7] handbrake   sigmoid → threshold 0.5
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from se3_field import COEFF_DIM

_HIDDEN = 64
_ANALOG_DIM = 5
_BINARY_DIM = 3
_ACTION_DIM = 8


class SE3Policy(nn.Module):
    """Deterministic policy for bot inference."""

    def __init__(self, obs_dim: int = COEFF_DIM, hidden: int = _HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.analog_head = nn.Linear(hidden, _ANALOG_DIM)
        self.binary_head = nn.Linear(hidden, _BINARY_DIM)
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs: (batch, 128)
        returns: policy (batch, 8), value (batch, 1)
        """
        h = self.net(obs)
        analog = torch.tanh(self.analog_head(h))
        binary = torch.sigmoid(self.binary_head(h))
        policy = torch.cat([analog, binary], dim=-1)
        value = self.value_head(h)
        return policy, value

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deterministic action for runtime. obs: (1, 128) numpy."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            emb = torch.tensor(obs, dtype=torch.float32, device=device)
            policy, value = self(emb)
        action = policy.cpu().numpy()[0]
        action[:_ANALOG_DIM] = np.clip(action[:_ANALOG_DIM], -1.0, 1.0)
        action[_ANALOG_DIM:] = (action[_ANALOG_DIM:] >= 0.5).astype(np.float32)
        return action, float(value.cpu().numpy()[0, 0])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(
            torch.load(path, map_location='cpu', weights_only=True))


class StochasticSE3Policy(nn.Module):
    """PPO-compatible stochastic policy with sampling, log_prob, entropy."""

    ANALOG_DIM = _ANALOG_DIM
    BINARY_DIM = _BINARY_DIM
    ACTION_DIM = _ACTION_DIM

    def __init__(self, obs_dim: int = COEFF_DIM, hidden: int = _HIDDEN):
        super().__init__()
        self.layer_norm = nn.LayerNorm(obs_dim)
        self.hidden1 = nn.Linear(obs_dim, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.analog_mean = nn.Linear(hidden, _ANALOG_DIM)
        self.analog_log_std = nn.Parameter(torch.zeros(_ANALOG_DIM))
        self.binary_logits = nn.Linear(hidden, _BINARY_DIM)
        self.value_head = nn.Linear(hidden, 1)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(obs)
        h = F.relu(self.hidden1(x))
        return F.relu(self.hidden2(h))

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions, return (action, log_prob, value, entropy).

        obs: (batch, 128)
        """
        h = self._features(obs)

        mean = torch.tanh(self.analog_mean(h))
        std = torch.exp(torch.clamp(self.analog_log_std, -5.0, 2.0))
        analog_dist = torch.distributions.Normal(mean, std)
        analog_action = analog_dist.rsample().clamp(-1.0, 1.0)

        logits = self.binary_logits(h)
        binary_dist = torch.distributions.Bernoulli(logits=logits)
        binary_action = binary_dist.sample()

        action = torch.cat([analog_action, binary_action], dim=-1)

        log_prob = (analog_dist.log_prob(analog_action).sum(dim=-1)
                    + binary_dist.log_prob(binary_action).sum(dim=-1))
        entropy = (analog_dist.entropy().sum(dim=-1)
                   + binary_dist.entropy().sum(dim=-1))
        value = self.value_head(h).squeeze(-1)

        return action, log_prob, value, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under current policy (PPO importance sampling)."""
        h = self._features(obs)

        mean = torch.tanh(self.analog_mean(h))
        std = torch.exp(torch.clamp(self.analog_log_std, -5.0, 2.0))
        analog_dist = torch.distributions.Normal(mean, std)

        logits = self.binary_logits(h)
        binary_dist = torch.distributions.Bernoulli(logits=logits)

        analog_actions = actions[:, :_ANALOG_DIM]
        binary_actions = actions[:, _ANALOG_DIM:]

        log_prob = (analog_dist.log_prob(analog_actions).sum(dim=-1)
                    + binary_dist.log_prob(binary_actions).sum(dim=-1))
        entropy = (analog_dist.entropy().sum(dim=-1)
                   + binary_dist.entropy().sum(dim=-1))
        value = self.value_head(h).squeeze(-1)

        return log_prob, value, entropy

    def act_deterministic(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean analog + threshold binary. For evaluation and opponent inference."""
        h = self._features(obs)
        analog = torch.tanh(self.analog_mean(h))
        binary = (torch.sigmoid(self.binary_logits(h)) >= 0.5).float()
        action = torch.cat([analog, binary], dim=-1)
        value = self.value_head(h).squeeze(-1)
        return action, value

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(
            torch.load(path, map_location='cpu', weights_only=True))
