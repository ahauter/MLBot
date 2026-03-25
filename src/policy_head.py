"""
PolicyHead
==========
Single policy + value head shared across all game situations.

Takes a D_MODEL-dim embedding from the shared encoder and outputs:
  policy  (batch, 8)  — 5 analog controls in [-1, 1] + 3 binary controls in [0, 1]
  value   (batch, 1)  — state-value estimate for AWAC training

Action layout:
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


class PolicyHead(nn.Module):
    """Single policy + value head for all game situations."""

    ANALOG_DIM = 5   # throttle, steer, pitch, yaw_ctrl, roll
    BINARY_DIM = 3   # jump, boost, handbrake
    ACTION_DIM = 8

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
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
        device = next(self.parameters()).device
        with torch.no_grad():
            emb    = torch.tensor(embedding, dtype=torch.float32, device=device)
            policy, value = self(emb)
        action = policy.cpu().numpy()[0]                # (8,)
        action[:5] = np.clip(action[:5], -1.0, 1.0)
        action[5:] = (action[5:] >= 0.5).astype(np.float32)
        return action, float(value.cpu().numpy()[0, 0])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))
