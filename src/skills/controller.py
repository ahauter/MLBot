"""
KNN Skill Controller
====================
Selects which SkillHead to activate at runtime by finding the nearest
scenario embedding in the index and returning a majority-vote skill name.

The index is built AFTER training by encoding representative initial states
from each scenario config.  It is saved as an .npz file and loaded by the bot
at startup — no re-training needed when new scenarios are added (just rebuild
the index from the new configs).

Design notes
------------
- Index covers BOTH car perspectives (car_idx=0 and car_idx=1) per config so
  that a bot playing as orange correctly matches orange-appropriate skills.
- Pure NumPy L2 search; fast enough for < 10 000 entries without ANN libraries.
- _make_dummy_packet creates duck-typed packets from ScenarioConfig so the
  index can be built without running RLBot.
"""

from __future__ import annotations

import random as _random
import types
from typing import List, Optional

import numpy as np
import torch


# ── dummy packet builder (no RLBot dependency) ────────────────────────────────

def _make_dummy_packet(config, rng=None):
    """
    Return a duck-typed packet compatible with state_to_tokens().
    Uses RangeOrFixed.sample() to sample randomised initial positions.
    """
    rng = rng or _random

    init = config.initial_state

    def _vec(x, y, z):
        v = types.SimpleNamespace()
        v.x, v.y, v.z = float(x), float(y), float(z)
        return v

    def _rotation(yaw, pitch=0.0, roll=0.0):
        return types.SimpleNamespace(yaw=float(yaw), pitch=float(pitch), roll=float(roll))

    def _physics(loc, vel, yaw, ang_vel=None):
        p = types.SimpleNamespace()
        p.location         = loc
        p.velocity         = vel
        p.angular_velocity = ang_vel or _vec(0.0, 0.0, 0.0)
        p.rotation         = _rotation(yaw)
        return p

    # Ball
    bl = init.ball.location
    bv = init.ball.velocity
    ball_phys = _physics(
        _vec(bl.x.sample(rng), bl.y.sample(rng), bl.z.sample(rng)),
        _vec(bv.x.sample(rng), bv.y.sample(rng), bv.z.sample(rng)),
        0.0,
    )

    # Blue car
    cl = init.blue.location
    blue_phys = _physics(
        _vec(cl.x.sample(rng), cl.y.sample(rng), cl.z.sample(rng)),
        _vec(0.0, 0.0, 0.0),
        init.blue.yaw.sample(rng),
    )

    # Orange car
    ol = init.orange.location
    orange_phys = _physics(
        _vec(ol.x.sample(rng), ol.y.sample(rng), ol.z.sample(rng)),
        _vec(0.0, 0.0, 0.0),
        init.orange.yaw.sample(rng),
    )

    packet = types.SimpleNamespace()
    packet.game_ball = types.SimpleNamespace(physics=ball_phys)
    packet.game_cars = [
        types.SimpleNamespace(physics=blue_phys,   boost=init.blue.boost.sample(rng)),
        types.SimpleNamespace(physics=orange_phys, boost=init.orange.boost.sample(rng)),
    ]
    # Provide stub teams and game_info so state_to_tokens() can read game state
    packet.teams     = [types.SimpleNamespace(score=0), types.SimpleNamespace(score=0)]
    packet.game_info = types.SimpleNamespace(game_time_remaining=300.0, is_overtime=False)
    return packet


# ── controller ────────────────────────────────────────────────────────────────

class KNNController:
    """
    K-nearest-neighbour skill selector.

    Usage
    -----
        # After training:
        ctrl = KNNController(encoder, k=3)
        ctrl.build_index(all_configs, n_samples=20)
        ctrl.save_index('models/knn_index.npz')

        # At bot runtime:
        ctrl = KNNController(encoder, k=3)
        ctrl.load_index('models/knn_index.npz')
        skill = ctrl.select_skill(embedding)
    """

    def __init__(self, encoder, k: int = 3):
        self.encoder = encoder
        self.k       = k
        self._index_embeddings: np.ndarray = np.zeros((0, 64), dtype=np.float32)
        self._index_skills:     List[str]  = []

    def build_index(self, configs: list, n_samples: int = 10) -> None:
        """
        Build the search index from scenario configs.

        For each config, both car perspectives (car_idx=0 and car_idx=1) are
        indexed so that bots playing as orange also get correct skill routing.

        configs:   list of ScenarioConfig objects
        n_samples: number of random initial states to average per (config, car_idx)
        """
        # Import here to avoid circular import at module level
        from encoder import state_to_tokens

        embeddings: List[np.ndarray] = []
        skills:     List[str]        = []

        for cfg in configs:
            for car_idx in (0, 1):
                skill = (cfg.initial_state.blue.skill   if car_idx == 0
                         else cfg.initial_state.orange.skill)
                if not skill:
                    continue

                sample_embs = []
                for _ in range(n_samples):
                    packet = _make_dummy_packet(cfg)
                    tokens = state_to_tokens(packet, car_idx)          # (1, N_TOKENS, TOKEN_FEATURES)
                    self.encoder.eval()
                    with torch.no_grad():
                        emb = self.encoder(
                            torch.tensor(tokens, dtype=torch.float32)
                        ).detach().numpy()                             # (1, D_MODEL)
                    sample_embs.append(emb[0])                         # (64,)

                mean_emb = np.mean(sample_embs, axis=0)                # (64,)
                embeddings.append(mean_emb)
                skills.append(skill)

        if not embeddings:
            raise ValueError('No valid skill entries found in configs.')

        self._index_embeddings = np.stack(embeddings, axis=0)          # (N, 64)
        self._index_skills     = skills

    def select_skill(self, embedding: np.ndarray) -> str:
        """
        Return the majority-vote skill name among the K nearest index entries.

        embedding: (64,) or (1, 64) numpy array
        """
        emb  = embedding.reshape(1, -1)                                # (1, 64)
        dists = np.linalg.norm(self._index_embeddings - emb, axis=1)  # (N,)
        knn_idx = np.argsort(dists)[:self.k]
        votes   = [self._index_skills[i] for i in knn_idx]
        return max(set(votes), key=votes.count)

    def known_skills(self) -> List[str]:
        """Unique skill names in the index, in insertion order."""
        seen = {}
        for s in self._index_skills:
            seen[s] = None
        return list(seen.keys())

    def save_index(self, path: str) -> None:
        np.savez(
            path,
            embeddings=self._index_embeddings,
            skills=np.array(self._index_skills, dtype=object),
        )

    def load_index(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self._index_embeddings = data['embeddings'].astype(np.float32)
        self._index_skills     = list(data['skills'])
