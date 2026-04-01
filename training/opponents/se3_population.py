"""
SE3 Population-Based Training
==============================
Population variant that creates SE3PPOAlgorithm agents instead of
the baseline PPOAlgorithm.  Otherwise identical to Population.

Usage (YAML config)
-------------------
    opponent_pool:
      class: training.opponents.se3_population.SE3Population
      params:
        agents: 4
        generation_steps: 1000000
        generation_noise_scale: 0.01
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from training.opponents.pool import OpponentPool


class SE3Population(OpponentPool):
    """
    Population-Based Training with SE3PPOAlgorithm agents.

    Parameters match Population for YAML compatibility.
    """

    def __init__(self, num_agents: int = 1, num_workers: int = 1,
                 config: dict = None,
                 envs_per_agent: Optional[int] = None,
                 snapshot_dir: str | Path = 'models/snapshots',
                 max_snapshots: int = 50,
                 snapshot_interval: int = 50_000,
                 decay_rate: float = 0.95,
                 generation_steps: int = 1_000_000,
                 generation_noise_scale: float = 0.01,
                 **kwargs):
        OpponentPool.__init__(self, snapshot_dir, max_snapshots)
        if config is None:
            config = {}
        self._num_agents = num_agents
        self.num_workers = num_workers
        self.snapshot_interval = snapshot_interval
        self.decay_rate = decay_rate
        self._last_snapshot_step = 0

        worker_assignment = self._assign_workers(num_workers, num_agents)

        if envs_per_agent is None:
            envs_per_agent_map = {
                i: worker_assignment.count(i) for i in range(num_agents)}
        else:
            envs_per_agent_map = {i: envs_per_agent for i in range(num_agents)}

        from training.algorithms.se3_ppo import SE3PPOAlgorithm

        self._agents: List[SE3PPOAlgorithm] = []
        for i in range(num_agents):
            agent_config = {**config, 'num_envs': envs_per_agent_map[i]}
            self._agents.append(SE3PPOAlgorithm(agent_config))

        self.scores: List[List[float]] = [[] for _ in range(num_agents)]
        self._generation = 0

    @classmethod
    def default_params(cls) -> dict:
        return {
            'agents': 1,
            'max_snapshots': 50,
            'snapshot_interval': 50_000,
            'decay_rate': 0.95,
            'generation_steps': 1_000_000,
            'generation_noise_scale': 0.01,
        }

    # ── OpponentPool ABC ────────────────────────────────────────────────────

    def save_snapshot(self, algo, step: int) -> Path:
        self._last_snapshot_step = step
        return self._save_weights(algo, step)

    def sample_opponent(self) -> Optional[str]:
        snapshots = self._list_snapshots()
        if not snapshots:
            return None
        n = len(snapshots)
        weights = [self.decay_rate ** (n - 1 - i) for i in range(n)]
        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = random.choices(snapshots, weights=probs, k=1)[0]
        return str(chosen)

    def num_snapshots(self) -> int:
        return len(self._list_snapshots())

    def should_swap(self, total_step: int) -> bool:
        return total_step - self._last_snapshot_step >= self.snapshot_interval

    # ── agent management ────────────────────────────────────────────────────

    @property
    def agents(self) -> List:
        return self._agents

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def generation(self) -> int:
        return self._generation

    def add_score(self, agent_idx: int, score: float) -> None:
        if 0 <= agent_idx < self._num_agents:
            self.scores[agent_idx].append(score)

    def rank_agents(self) -> List[int]:
        means = []
        for scores in self.scores:
            means.append(np.mean(scores) if scores else float('-inf'))
        return sorted(range(self._num_agents), key=lambda i: means[i], reverse=True)

    def reset_scores(self) -> None:
        self.scores = [[] for _ in range(self._num_agents)]
        self._generation += 1

    def get_metrics(self) -> dict:
        ranking = self.rank_agents()
        return {
            'generation': self._generation,
            'num_agents': self._num_agents,
            'best_agent': ranking[0] if ranking else -1,
            'pool_size': self.num_snapshots(),
        }

    @staticmethod
    def _assign_workers(num_workers: int, num_agents: int) -> List[int]:
        """Assign workers to agents as evenly as possible."""
        base = num_workers // num_agents
        extra = num_workers % num_agents
        assignment = []
        for i in range(num_agents):
            count = base + (1 if i < extra else 0)
            assignment.extend([i] * count)
        return assignment
