"""
MLBot — RLBot Agent
===================
Loads the shared encoder, KNN skill controller, and per-skill heads at startup,
then on every tick:
  1. Encode current game state → 64-dim embedding
  2. KNN lookup → select which SkillHead to activate
  3. SkillHead.act() → 8-float action
  4. Translate to SimpleControllerState

Notes
-----
- Skill discovery is dynamic: skills are read from the KNN index (which is
  built from the YAML configs after training), so no skill names are hardcoded.
- Works for either car index (blue or orange) because state_to_tokens(packet,
  self.index) encodes the bot's own car as token 1 and the opponent as token 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# Add src/ and training/ to path so encoder/skills are importable
_SRC  = Path(__file__).parent
_REPO = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import SharedTransformerEncoder, state_to_tokens
from skills.skill_head import SkillHead
from skills.controller import KNNController

MODEL_DIR = _REPO / 'models'


class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.index:       int                      = index
        self.encoder:     SharedTransformerEncoder = None
        self.controller:  KNNController            = None
        self.skill_heads: dict[str, SkillHead]     = {}

    def initialize_agent(self) -> None:
        """Load encoder, KNN index, and all available skill heads from models/."""
        enc_path = MODEL_DIR / 'encoder.weights.h5'
        if enc_path.exists():
            self.encoder = SharedTransformerEncoder.load_from(str(enc_path))
        else:
            # No trained model yet — use untrained encoder (random behaviour)
            self.encoder = SharedTransformerEncoder()
            self.encoder(tf.zeros((1, 3, 8)))

        self.controller = KNNController(self.encoder, k=3)
        knn_path = MODEL_DIR / 'knn_index.npz'
        if knn_path.exists():
            self.controller.load_index(str(knn_path))
        else:
            # Index not built yet; controller will have no entries
            pass

        # Load whichever skill heads are present in models/
        for skill_name in self.controller.known_skills():
            head = SkillHead(skill_name)
            head(tf.zeros((1, 64)))   # build variables before loading weights
            head_path = MODEL_DIR / f'skill_{skill_name}.weights.h5'
            if head_path.exists():
                head.load_weights(str(head_path))
            self.skill_heads[skill_name] = head

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # 1. Encode current game state for this car's perspective
        tokens    = state_to_tokens(packet, self.index)            # (1, 3, 8)
        embedding = self.encoder(
            tf.constant(tokens, dtype=tf.float32)
        ).numpy()                                                   # (1, 64)

        # 2. Select skill via KNN lookup
        if self.skill_heads:
            skill = self.controller.select_skill(embedding[0])     # (64,) → str
            if skill not in self.skill_heads:
                skill = next(iter(self.skill_heads))
            action, _ = self.skill_heads[skill].act(embedding)     # (8,)
        else:
            # No trained skill heads available — output neutral controls
            action = np.zeros(8, dtype=np.float32)

        return self.translate_controls(action)

    def translate_controls(self, action: np.ndarray) -> SimpleControllerState:
        """
        Map an 8-float action array to a SimpleControllerState.

        Layout matches human_play.py _controls_to_action exactly:
          [0] throttle    [-1,  1]
          [1] steer       [-1,  1]
          [2] pitch       [-1,  1]
          [3] yaw         [-1,  1]
          [4] roll        [-1,  1]
          [5] jump        {0, 1}
          [6] boost       {0, 1}
          [7] handbrake   {0, 1}
        """
        ctrl           = SimpleControllerState()
        ctrl.throttle  = float(np.clip(action[0], -1.0, 1.0))
        ctrl.steer     = float(np.clip(action[1], -1.0, 1.0))
        ctrl.pitch     = float(np.clip(action[2], -1.0, 1.0))
        ctrl.yaw       = float(np.clip(action[3], -1.0, 1.0))
        ctrl.roll      = float(np.clip(action[4], -1.0, 1.0))
        ctrl.jump      = bool(action[5] >= 0.5)
        ctrl.boost     = bool(action[6] >= 0.5)
        ctrl.handbrake = bool(action[7] >= 0.5)
        return ctrl
