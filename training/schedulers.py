"""
Collection Schedulers
=====================
Control how agents are assigned to environment workers each step.

InterleavedScheduler — all agents collect simultaneously, envs split evenly.
                       Best when num_envs >> num_agents.

SerialScheduler     — one agent at a time uses ALL envs, then rotates.
                       Best when num_envs is small relative to num_agents
                       (larger batches per forward pass, fewer GPU kernel
                       launches per step).

Usage in YAML:
    scheduler:
      class: training.schedulers.SerialScheduler
"""
from __future__ import annotations

import math
from typing import Dict, Iterator, List

from training.abstractions import CollectionScheduler


class InterleavedScheduler(CollectionScheduler):
    """All agents collect simultaneously, envs split evenly among them."""

    def init(self, population, num_envs: int, config: dict) -> None:
        num_agents = population.num_agents
        self._assignment: Dict[int, List[int]] = {}
        base = num_envs // num_agents
        remainder = num_envs % num_agents
        idx = 0
        for i in range(num_agents):
            count = base + (1 if i < remainder else 0)
            self._assignment[i] = list(range(idx, idx + count))
            idx += count

    def envs_per_agent(self, num_envs: int, num_agents: int) -> int:
        return math.ceil(num_envs / num_agents)

    def iter_steps(self, rollout_steps: int) -> Iterator[Dict[int, List[int]]]:
        for _ in range(rollout_steps):
            yield self._assignment


class SerialScheduler(CollectionScheduler):
    """One agent at a time uses all envs. Rotates after rollout_steps."""

    def init(self, population, num_envs: int, config: dict) -> None:
        self._num_agents = population.num_agents
        self._all_workers = list(range(num_envs))

    def envs_per_agent(self, num_envs: int, num_agents: int) -> int:
        return num_envs

    def iter_steps(self, rollout_steps: int) -> Iterator[Dict[int, List[int]]]:
        for agent_idx in range(self._num_agents):
            for _ in range(rollout_steps):
                yield {agent_idx: self._all_workers}
