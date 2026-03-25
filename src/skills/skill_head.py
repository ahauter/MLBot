"""
SkillHead
=========
Thin policy + value head for one named skill.

Takes a 64-dim embedding from the shared encoder and outputs:
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
import tensorflow as tf
from tensorflow import keras


class SkillHead(keras.Model):
    """Policy + value head for a single named skill."""

    ANALOG_DIM = 5   # throttle, steer, pitch, yaw_ctrl, roll
    BINARY_DIM = 3   # jump, boost, handbrake
    ACTION_DIM = 8

    def __init__(self, skill_name: str, d_model: int = 64, **kwargs):
        super().__init__(name=f'skill_{skill_name}', **kwargs)
        self.skill_name  = skill_name
        self.hidden      = keras.layers.Dense(64, activation='relu', name='hidden')
        self.analog_head = keras.layers.Dense(self.ANALOG_DIM, activation='tanh',
                                              name='analog')
        self.binary_head = keras.layers.Dense(self.BINARY_DIM, activation='sigmoid',
                                              name='binary')
        self.value_head  = keras.layers.Dense(1, name='value')

    def call(self, embedding, training=False):
        """
        embedding: (batch, 64)
        returns:
          policy: (batch, 8)   analog[:5] in [-1,1], binary[5:8] in [0,1]
          value:  (batch, 1)
        """
        h      = self.hidden(embedding)
        analog = self.analog_head(h)                        # (batch, 5)
        binary = self.binary_head(h)                        # (batch, 3)
        policy = tf.concat([analog, binary], axis=-1)       # (batch, 8)
        value  = self.value_head(h)                         # (batch, 1)
        return policy, value

    def act(self, embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Deterministic greedy action for runtime (no exploration noise).

        embedding: (1, 64) numpy array
        Returns:
          action:     (8,) float32 numpy
          value_est:  scalar float
        """
        emb    = tf.constant(embedding, dtype=tf.float32)
        policy, value = self(emb)
        action = policy.numpy()[0]                          # (8,)
        # Analog controls are bounded by tanh, but clip for safety
        action[:5] = np.clip(action[:5], -1.0, 1.0)
        # Binary controls: threshold at 0.5
        action[5:] = (action[5:] >= 0.5).astype(np.float32)
        return action, float(value.numpy()[0, 0])

    def save(self, path: str) -> None:
        self.save_weights(path)

    def load(self, path: str) -> None:
        self.load_weights(path)
