"""
MLBot — RLBot Agent
===================
Loads the shared spatiotemporal encoder and single policy head at startup,
then on every tick:
  1. Append current token snapshot to the sliding observation window
  2. Encode window → 64-dim embedding
  3. PolicyHead.act() → 8-float action
  4. Translate to SimpleControllerState
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # prevent CUDA DLL load in bot subprocess

import numpy as np
import torch

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

_SRC  = Path(__file__).parent
_REPO = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import (
    SharedTransformerEncoder,
    state_to_tokens,
    ENTITY_TYPE_IDS_1V1,
    T_WINDOW,
)
from policy_head import StochasticPolicyHead

MODEL_DIR = _REPO / 'models'


class MyBot(Bot):

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.encoder:     SharedTransformerEncoder = None
        self.policy_head: StochasticPolicyHead      = None
        self._obs_buffer: deque                    = None
        self._entity_type_ids: torch.Tensor        = torch.tensor(
            ENTITY_TYPE_IDS_1V1, dtype=torch.long)

    def initialize(self) -> None:
        """Load encoder and policy head from models/."""
        enc_path = MODEL_DIR / 'encoder.pt'
        if enc_path.exists():
            self.encoder = SharedTransformerEncoder.load_from(str(enc_path))
        else:
            self.encoder = SharedTransformerEncoder()

        self.policy_head = StochasticPolicyHead()
        head_path = MODEL_DIR / 'policy.pt'
        if head_path.exists():
            self.policy_head.load(str(head_path))

        self.encoder.eval()
        self.policy_head.eval()

        # Observation window — filled on first get_output() call
        self._obs_buffer = None

    def get_output(self, packet: GamePacket) -> ControllerState:
        tokens = state_to_tokens(packet, self.index)   # (1, N, TOKEN_FEATURES)
        frame  = tokens[0]                              # (N, TOKEN_FEATURES)

        # Initialise window by replicating the first frame T times
        if self._obs_buffer is None:
            self._obs_buffer = deque(
                [frame.copy() for _ in range(T_WINDOW)], maxlen=T_WINDOW)
        else:
            self._obs_buffer.append(frame)

        window = np.stack(self._obs_buffer)[np.newaxis]  # (1, T_WINDOW, N, F)

        with torch.no_grad():
            embedding = self.encoder(
                torch.tensor(window, dtype=torch.float32),
                self._entity_type_ids,
            ).numpy()   # (1, D_MODEL)

        if self.policy_head is not None:
            with torch.no_grad():
                emb_t = torch.tensor(embedding, dtype=torch.float32)
                action_t, _ = self.policy_head.act_deterministic(emb_t)
                action = action_t[0].cpu().numpy().astype(np.float32)
        else:
            action = np.zeros(8, dtype=np.float32)

        return self.translate_controls(action)

    def translate_controls(self, action: np.ndarray) -> ControllerState:
        """
        Map an 8-float action array to a ControllerState.
          [0] throttle  [1] steer  [2] pitch  [3] yaw  [4] roll
          [5] jump  [6] boost  [7] handbrake
        """
        ctrl           = ControllerState()
        ctrl.throttle  = float(np.clip(action[0], -1.0, 1.0))
        ctrl.steer     = float(np.clip(action[1], -1.0, 1.0))
        ctrl.pitch     = float(np.clip(action[2], -1.0, 1.0))
        ctrl.yaw       = float(np.clip(action[3], -1.0, 1.0))
        ctrl.roll      = float(np.clip(action[4], -1.0, 1.0))
        ctrl.jump      = bool(action[5] >= 0.5)
        ctrl.boost     = bool(action[6] >= 0.5)
        ctrl.handbrake = bool(action[7] >= 0.5)
        return ctrl


if __name__ == "__main__":
    MyBot("austin/mlbot").run()
