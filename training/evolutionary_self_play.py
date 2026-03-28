"""
Evolutionary Opponent Pool
==========================
Neuroevolution-style self-play: maintain a population of agents, run
round-robin tournaments, select survivors by cumulative reward, and
breed new agents via weight crossover with mutation.

Evolution cycle (every ``evolution_interval`` training steps):
  1. Inject the current training agent into population slot 0
  2. Round-robin tournament: every pair plays ``tournament_matches`` games
  3. Top ``num_survivors`` by cumulative reward survive
  4. Fill remaining slots by crossing two random survivors (50% weight
     swap + small Gaussian mutation noise)

Classes
-------
EvolutionaryOpponentPool(OpponentPool)
    Population-based opponent pool with tournament selection and crossover.
"""
from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from self_play import OpponentPool

_MODEL_FILE = 'model.pt'


class EvolutionaryOpponentPool(OpponentPool):
    """
    Population-based opponent pool with tournament selection and crossover.

    Maintains ``population_size`` agents. Every ``evolution_interval``
    training steps, runs a full round-robin tournament, keeps the top
    ``num_survivors``, and fills the remaining slots with offspring
    created by crossing two random survivors' weights.

    The current training agent is injected into slot 0 before each
    tournament so it competes alongside the population.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory where population agent snapshots are stored.
    algo_builder : callable
        Function that creates a fresh d3rlpy algo instance.
    population_size : int
        Number of agents in the population (default 10).
    num_survivors : int
        Number of agents kept after selection (default 3).
    tournament_matches : int
        Number of matches each pair plays in the round-robin (default 100).
    evolution_interval : int
        Training steps between evolution cycles (default 50000).
    mutation_noise : float
        Std of Gaussian noise added to crossed-over weights (default 0.01).
    t_window : int
        Frame history length for tournament match environments.
    tick_skip : int
        Physics ticks per action for tournament environments.
    max_steps : int
        Max steps per tournament episode.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        algo_builder=None,
        population_size: int = 10,
        num_survivors: int = 3,
        tournament_matches: int = 100,
        evolution_interval: int = 50_000,
        mutation_noise: float = 0.01,
        t_window: int = 8,
        tick_skip: int = 8,
        max_steps: int = 4500,
    ):
        super().__init__(snapshot_dir, algo_builder, max_snapshots=population_size)
        self.population_size = population_size
        self.num_survivors = num_survivors
        self.tournament_matches = tournament_matches
        self.evolution_interval = evolution_interval
        self.mutation_noise = mutation_noise
        self.t_window = t_window
        self.tick_skip = tick_skip
        self.max_steps = max_steps

        self._last_evolution_step = 0
        self.generation = 0
        self._initialized = False

    # ── OpponentPool interface ──────────────────────────────────────────────

    def save_snapshot(self, algo, step: int) -> Path:
        """Trigger an evolution cycle: inject trainee, tournament, breed."""
        if not self._initialized:
            self._initialize_population()

        # Inject training agent into slot 0
        slot_dir = self._agent_dir(0)
        slot_dir.mkdir(parents=True, exist_ok=True)
        algo.save_model(str(slot_dir / _MODEL_FILE))

        # Evolve
        self._evolve(step)
        return self.snapshot_dir

    def sample_opponent(self):
        """Load a random agent from the population."""
        if not self._initialized:
            self._initialize_population()

        agent_dirs = self._list_agent_dirs()
        if not agent_dirs:
            return None
        snap_dir = random.choice(agent_dirs)
        return self._load_snapshot(snap_dir)

    def latest(self):
        agent_dirs = self._list_agent_dirs()
        if not agent_dirs:
            return None
        return self._load_snapshot(agent_dirs[-1])

    def num_snapshots(self) -> int:
        return len(self._list_agent_dirs())

    def should_swap(self, total_step: int) -> bool:
        if total_step - self._last_evolution_step >= self.evolution_interval:
            self._last_evolution_step = total_step
            return True
        return False

    # ── population management ───────────────────────────────────────────────

    def _agent_dir(self, idx: int) -> Path:
        return self.snapshot_dir / f'agent_{idx:02d}'

    def _list_agent_dirs(self) -> List[Path]:
        """Return sorted list of agent directories that have a model file."""
        if not self.snapshot_dir.exists():
            return []
        return sorted(
            d for d in self.snapshot_dir.iterdir()
            if d.is_dir() and d.name.startswith('agent_') and (d / _MODEL_FILE).exists()
        )

    def _initialize_population(self) -> None:
        """Create initial population with random weights."""
        print(f'[EvolutionaryPool] Initializing population of {self.population_size} agents...')
        for i in range(self.population_size):
            agent_dir = self._agent_dir(i)
            if (agent_dir / _MODEL_FILE).exists():
                continue  # Already exists (e.g. resumed run)
            # Each agent gets fresh random weights (different seed per agent)
            torch.manual_seed(random.randint(0, 2**31))
            algo = self._algo_builder()
            agent_dir.mkdir(parents=True, exist_ok=True)
            algo.save_model(str(agent_dir / _MODEL_FILE))
            del algo
        self._initialized = True
        print(f'[EvolutionaryPool] Population initialized ({self.population_size} agents)')

    # ── tournament ──────────────────────────────────────────────────────────

    def _play_match(self, algo_a, algo_b) -> Tuple[float, float]:
        """Play one episode between two agents. Returns (reward_a, reward_b)."""
        from gym_env import BaselineGymEnv

        env = BaselineGymEnv(
            t_window=self.t_window,
            tick_skip=self.tick_skip,
            max_steps=self.max_steps,
        )
        env.set_opponent(algo_b)

        obs, _info = env.reset()
        total_reward_a = 0.0
        done = False

        while not done:
            action = algo_a.predict(obs[np.newaxis])[0]
            obs, reward, done, _, info = env.step(action)
            total_reward_a += reward

        env.close()

        # Sparse reward is symmetric: if blue gets +1, orange gets -1
        total_reward_b = -total_reward_a
        return total_reward_a, total_reward_b

    def _run_tournament(self) -> List[int]:
        """
        Round-robin tournament across all population agents.

        Returns indices of the top ``num_survivors`` agents by cumulative reward.
        """
        agent_dirs = self._list_agent_dirs()
        n = len(agent_dirs)
        rewards = np.zeros(n, dtype=np.float64)

        # Load all agents
        agents = []
        for d in agent_dirs:
            agents.append(self._load_snapshot(d))

        total_pairs = n * (n - 1) // 2
        pair_count = 0

        for i, j in combinations(range(n), 2):
            pair_count += 1
            print(
                f'[EvolutionaryPool] Tournament pair {pair_count}/{total_pairs}: '
                f'agent_{i:02d} vs agent_{j:02d}',
                end='',
                flush=True,
            )
            pair_reward_i = 0.0
            pair_reward_j = 0.0

            for m in range(self.tournament_matches):
                # Alternate who is blue/orange for fairness
                if m % 2 == 0:
                    r_a, r_b = self._play_match(agents[i], agents[j])
                    pair_reward_i += r_a
                    pair_reward_j += r_b
                else:
                    r_a, r_b = self._play_match(agents[j], agents[i])
                    pair_reward_j += r_a
                    pair_reward_i += r_b

            rewards[i] += pair_reward_i
            rewards[j] += pair_reward_j
            print(f'  -> {pair_reward_i:+.1f} / {pair_reward_j:+.1f}')

        # Clean up loaded agents
        del agents

        # Rank by cumulative reward, return top survivors
        ranked = np.argsort(rewards)[::-1]  # descending
        survivors = ranked[:self.num_survivors].tolist()

        print(f'[EvolutionaryPool] Tournament results:')
        for rank, idx in enumerate(ranked):
            marker = ' *' if idx in survivors else ''
            print(f'  #{rank+1} agent_{idx:02d}: reward={rewards[idx]:+.1f}{marker}')

        return survivors

    # ── crossover + mutation ────────────────────────────────────────────────

    def _crossover(self, parent_a_path: Path, parent_b_path: Path) -> dict:
        """
        Create a child state_dict by crossing two parents' weights.

        For each parameter tensor, ~50% of elements come from parent A
        and ~50% from parent B, plus small Gaussian mutation noise.
        """
        state_a = torch.load(str(parent_a_path), map_location='cpu')
        state_b = torch.load(str(parent_b_path), map_location='cpu')

        child_state = {}
        for key in state_a:
            tensor_a = state_a[key]
            tensor_b = state_b[key]

            if tensor_a.is_floating_point():
                mask = torch.rand_like(tensor_a) > 0.5
                child = torch.where(mask, tensor_a, tensor_b)
                # Mutation: tiny Gaussian noise
                child = child + torch.randn_like(child) * self.mutation_noise
            else:
                # Non-float tensors (e.g. int indices): just pick from parent A
                child = tensor_a.clone()

            child_state[key] = child

        return child_state

    # ── evolution cycle ─────────────────────────────────────────────────────

    def _evolve(self, step: int) -> None:
        """Run one evolution cycle: tournament, selection, crossover."""
        self.generation += 1
        print(
            f'\n[EvolutionaryPool] === Generation {self.generation} '
            f'(step {step:,}) ==='
        )

        # Run tournament
        survivor_indices = self._run_tournament()

        # Collect survivor model paths
        agent_dirs = self._list_agent_dirs()
        survivor_paths = [agent_dirs[i] / _MODEL_FILE for i in survivor_indices]

        # Save survivors to temp location, then rebuild population
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # Copy survivors
            for rank, idx in enumerate(survivor_indices):
                src = agent_dirs[idx]
                dst = tmp / f'survivor_{rank:02d}'
                shutil.copytree(str(src), str(dst))

            # Clear population directory
            for d in agent_dirs:
                shutil.rmtree(str(d), ignore_errors=True)

            # Place survivors in first slots
            for rank in range(len(survivor_indices)):
                src = tmp / f'survivor_{rank:02d}'
                dst = self._agent_dir(rank)
                shutil.copytree(str(src), str(dst))

        # Fill remaining slots with offspring
        n_offspring = self.population_size - self.num_survivors
        print(f'[EvolutionaryPool] Breeding {n_offspring} offspring...')

        for i in range(n_offspring):
            child_idx = self.num_survivors + i

            # Pick two random (distinct) survivors as parents
            parent_a_idx, parent_b_idx = random.sample(
                range(self.num_survivors), 2
            )
            parent_a_path = self._agent_dir(parent_a_idx) / _MODEL_FILE
            parent_b_path = self._agent_dir(parent_b_idx) / _MODEL_FILE

            # Crossover + mutation
            child_state = self._crossover(parent_a_path, parent_b_path)

            # Save child
            child_dir = self._agent_dir(child_idx)
            child_dir.mkdir(parents=True, exist_ok=True)
            torch.save(child_state, str(child_dir / _MODEL_FILE))

            print(
                f'  agent_{child_idx:02d} = '
                f'crossover(agent_{parent_a_idx:02d}, agent_{parent_b_idx:02d})'
            )

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'evolutionary_pool/generation': self.generation,
                    'evolutionary_pool/population_size': self.population_size,
                    'evolutionary_pool/num_survivors': self.num_survivors,
                }, step=step)
        except ImportError:
            pass

        print(f'[EvolutionaryPool] Generation {self.generation} complete\n')
