"""
SkillHead
=========
Thin policy + value head for one named skill.

Takes a D_MODEL-dim embedding from the shared encoder and outputs:
  policy  (batch, 8)  — 5 analog controls in [-1, 1] + 3 binary controls in [0, 1]
  value   (batch, 1)  — state-value estimate for Actor-Critic training

Action layout (matches human_play.py _controls_to_action exactly):
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

from encoder import D_MODEL


class SkillHead(nn.Module):
    """Policy + value head for a single named skill."""

    ANALOG_DIM = 5   # throttle, steer, pitch, yaw_ctrl, roll
    BINARY_DIM = 3   # jump, boost, handbrake
    ACTION_DIM = 8

    def __init__(self, skill_name: str, d_model: int = D_MODEL):
        super().__init__()
        self.skill_name  = skill_name
        self.hidden      = nn.Linear(d_model, 64)
        self.analog_head = nn.Linear(64, self.ANALOG_DIM)
        self.binary_head = nn.Linear(64, self.BINARY_DIM)
        self.value_head  = nn.Linear(64, 1)

    def forward(
        self, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        embedding: (batch, D_MODEL)
        returns:
          policy: (batch, 8)   analog[:5] in [-1,1] via tanh, binary[5:8] in [0,1] via sigmoid
          value:  (batch, 1)
        """
        h      = F.relu(self.hidden(embedding))
        analog = torch.tanh(self.analog_head(h))        # (batch, 5)
        binary = torch.sigmoid(self.binary_head(h))     # (batch, 3)
        policy = torch.cat([analog, binary], dim=-1)    # (batch, 8)
        value  = self.value_head(h)                     # (batch, 1)
        return policy, value

    def act(self, embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Deterministic greedy action for runtime (no exploration noise).

        embedding: (1, D_MODEL) numpy array
        Returns:
          action:     (8,) float32 numpy
          value_est:  scalar float
        """
        self.eval()
        with torch.no_grad():
            emb    = torch.tensor(embedding, dtype=torch.float32)
            policy, value = self(emb)
        action = policy.numpy()[0]                      # (8,)
        # Analog controls are bounded by tanh, but clip for safety
        action[:5] = np.clip(action[:5], -1.0, 1.0)
        # Binary controls: threshold at 0.5
        action[5:] = (action[5:] >= 0.5).astype(np.float32)
        return action, float(value.numpy()[0, 0])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))
