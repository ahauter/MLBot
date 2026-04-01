"""
Population-Based Training
=========================
Multiple PPO agents sharing vectorized environments, unified with the
OpponentPool interface for snapshot management.

Extends OpponentPool so Population can be configured as the opponent_pool
class in YAML configs, handling both agent management and self-play
opponent snapshots through a single interface.

Usage (YAML config)
-------------------
    opponent_pool:
      class: training.opponents.population.Population
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


class Population(OpponentPool):
    """
    Population-Based Training: multiple PPO agents sharing a set of
    vectorized environments. Workers are divided evenly among agents.

    Extends OpponentPool to unify agent management and opponent snapshot
    management into a single interface. Implements snapshot save/sample
    with exponential recency-weighted sampling (same algorithm as
    HistoricalOpponentPool).

    Parameters
    ----------
    num_agents : int
        Number of PPO agents in the population.
    num_workers : int
        Total number of parallel environment workers.
    config : dict
        Shared configuration dict passed to each PPOAlgorithm.
    envs_per_agent : int, optional
        Explicit env count per agent (overrides worker_assignment inference).
    snapshot_dir : str or Path
        Directory for opponent snapshots.
    max_snapshots : int
        Maximum snapshots to retain.
    snapshot_interval : int
        Minimum steps between snapshots.
    decay_rate : float
        Exponential recency decay for snapshot sampling.
    generation_steps : int
        Steps between generation cycles (used by training loop).
    generation_noise_scale : float
        Noise added when cloning best agent to worst (used by training loop).
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

        # Assign workers to agents (used for interleaved scoring)
        self.worker_assignment = self._assign_workers(num_workers, num_agents)

        # Buffer sizing: explicit envs_per_agent (from scheduler) or infer from assignment
        if envs_per_agent is None:
            envs_per_agent_map = {
                i: self.worker_assignment.count(i) for i in range(num_agents)}
        else:
            envs_per_agent_map = {i: envs_per_agent for i in range(num_agents)}

        # Resolve algorithm class from config, defaulting to PPO
        AlgoCls = config.get('algorithm', {}).get('cls', None)
        if AlgoCls is None:
            from training.algorithms.ppo import PPOAlgorithm
            AlgoCls = PPOAlgorithm

        # Create agents, each with buffer sized for its env count
        self._agents: List = []
        for i in range(num_agents):
            agent_config = {**config, 'num_envs': envs_per_agent_map[i]}
            self._agents.append(AlgoCls(agent_config))

        # Score tracking per agent
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

    # ── OpponentPool abstract method implementations ──────────────────────

    def save_snapshot(self, algo, step: int) -> Path:
        self._last_snapshot_step = step
        return self._save_weights(algo, step)

    def sample_opponent(self) -> Optional[str]:
        """Sample a snapshot path with exponential recency weighting."""
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

    # ── agent management (overrides OpponentPool defaults) ────────────────

    @property
    def agents(self) -> List:
        return self._agents

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def generation(self) -> int:
        return self._generation

    @staticmethod
    def _assign_workers(num_workers: int, num_agents: int) -> List[int]:
        """
        Assign workers to agents as evenly as possible.

        Returns a list of length num_workers where each element is the agent index.
        Example: 8 workers, 3 agents -> [0,0,0,1,1,1,2,2]
        """
        assignment = []
        base = num_workers // num_agents
        remainder = num_workers % num_agents
        for agent_id in range(num_agents):
            count = base + (1 if agent_id < remainder else 0)
            assignment.extend([agent_id] * count)
        return assignment

    def add_score(self, agent_idx: int, score: float) -> None:
        """Record a score for an agent."""
        self.scores[agent_idx].append(score)

    def rank_agents(self) -> List[int]:
        """
        Rank agents by mean score (descending). Returns agent indices.
        Agents with no scores are ranked last.
        """
        mean_scores = []
        for i in range(self.num_agents):
            if self.scores[i]:
                mean_scores.append((i, np.mean(self.scores[i])))
            else:
                mean_scores.append((i, float('-inf')))
        mean_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in mean_scores]

    def get_metrics(self) -> dict:
        """Return population-level metrics for logging (includes pool stats)."""
        ranked = self.rank_agents()
        agent_stats = {}
        for i in range(self.num_agents):
            s = self.scores[i]
            n = len(s)
            wins = sum(1 for x in s if x > 0)
            losses = sum(1 for x in s if x < 0)
            agent_stats[f'agent_{i}/num_episodes'] = n
            agent_stats[f'agent_{i}/goals_scored'] = wins
            agent_stats[f'agent_{i}/goals_conceded'] = losses
            agent_stats[f'agent_{i}/win_rate'] = (wins - losses) / n if n > 0 else 0.0
        return {
            'generation': self.generation,
            'num_agents': self.num_agents,
            'best_agent': ranked[0] if ranked else -1,
            'pool_size': self.num_snapshots(),
            **agent_stats,
        }

    def reset_scores(self) -> None:
        """Clear all agent scores (call after a generation)."""
        self.scores = [[] for _ in range(self.num_agents)]
        self._generation += 1
