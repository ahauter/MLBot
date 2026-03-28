"""
Evolutionary Opponent Pool
==========================
Population-based self-play: maintain a pool of agents, score them by how
often they beat the training agent during normal training episodes, keep the
top survivors, and fill remaining slots with Gaussian-noise-perturbed copies
of the current training agent.

No separate tournament is run — all scoring comes from outcomes already
observed during training.

Evolution cycle (every ``evolution_interval`` training steps):
  1. Inject the current training agent into population slot 0
  2. Score each agent by wins-as-opponent accumulated since last evolution
  3. Keep top ``num_survivors`` agents (slot 0 always kept)
  4. Fill remaining slots with slot-0 weights + Gaussian noise

Classes
-------
EvolutionaryOpponentPool(OpponentPool)
    Population-based opponent pool with training-tracked selection and
    noise-only mutation.
"""
from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from self_play import OpponentPool

_MODEL_FILE = 'model.pt'


class EvolutionaryOpponentPool(OpponentPool):
    """
    Population-based opponent pool with training-tracked selection and
    noise-only mutation.

    Maintains ``population_size`` agents. Every ``evolution_interval``
    training steps, scores each agent by how many times it beat the training
    agent during normal play, keeps the top ``num_survivors`` (plus the
    current training agent), and fills the remaining slots with noisy copies
    of the current training agent.

    No separate tournament matches are played — all scoring is derived from
    episode outcomes already observed during training.

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
        The current training agent (slot 0) is always kept regardless.
    evolution_interval : int
        Training steps between evolution cycles (default 50000).
    mutation_noise : float
        Std of Gaussian noise added to offspring weights (default 0.01).
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        algo_builder=None,
        population_size: int = 10,
        num_survivors: int = 3,
        evolution_interval: int = 50_000,
        mutation_noise: float = 0.01,
    ):
        super().__init__(snapshot_dir, algo_builder, max_snapshots=population_size)
        self.population_size = population_size
        self.num_survivors = num_survivors
        self.evolution_interval = evolution_interval
        self.mutation_noise = mutation_noise

        self._last_evolution_step = 0
        self.generation = 0
        self._initialized = False

        # Per-agent outcome tracking: agent_idx -> list of goal values
        self._agent_scores: Dict[int, List[int]] = {}
        # Which agent index is currently the opponent (set in sample_opponent)
        self._current_opponent_idx: Optional[int] = None

    # ── OpponentPool interface ──────────────────────────────────────────────

    def save_snapshot(self, algo, step: int) -> Path:
        """Inject training agent into slot 0 and trigger an evolution cycle."""
        if not self._initialized:
            self._initialize_population()

        # Inject training agent into slot 0
        slot_dir = self._agent_dir(0)
        slot_dir.mkdir(parents=True, exist_ok=True)
        algo.save_model(str(slot_dir / _MODEL_FILE))

        self._evolve(step)
        return self.snapshot_dir

    def sample_opponent(self):
        """Load a random agent from the population and record its index."""
        if not self._initialized:
            self._initialize_population()

        agent_dirs = self._list_agent_dirs()
        if not agent_dirs:
            self._current_opponent_idx = None
            return None

        chosen = random.choice(agent_dirs)
        # Extract numeric index from directory name "agent_XX"
        try:
            self._current_opponent_idx = int(chosen.name.split('_')[1])
        except (IndexError, ValueError):
            self._current_opponent_idx = None

        return self._load_snapshot(chosen)

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

    def on_episode_end(self, goal: int) -> None:
        """Record the episode outcome against the current opponent agent."""
        if self._current_opponent_idx is not None:
            self._agent_scores.setdefault(self._current_opponent_idx, []).append(goal)

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
            torch.manual_seed(random.randint(0, 2**31))
            algo = self._algo_builder()
            agent_dir.mkdir(parents=True, exist_ok=True)
            algo.save_model(str(agent_dir / _MODEL_FILE))
            del algo
        self._initialized = True
        print(f'[EvolutionaryPool] Population initialized ({self.population_size} agents)')

    # ── evolution cycle ─────────────────────────────────────────────────────

    def _add_noise(self, obj):
        """Recursively add Gaussian noise to all float tensors in a nested structure."""
        import collections
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point():
                return obj + torch.randn_like(obj) * self.mutation_noise
            return obj.clone()
        if isinstance(obj, (dict, collections.OrderedDict)):
            return type(obj)((k, self._add_noise(v)) for k, v in obj.items())
        if isinstance(obj, (list, tuple)):
            result = [self._add_noise(v) for v in obj]
            return type(obj)(result)
        return obj

    def _evolve(self, step: int) -> None:
        """Score agents from training outcomes, select survivors, breed with noise."""
        self.generation += 1
        print(
            f'\n[EvolutionaryPool] === Generation {self.generation} '
            f'(step {step:,}) ==='
        )

        agent_dirs = self._list_agent_dirs()

        # Score each agent: count of episodes where it scored against the trainer
        # (goal == -1 means the opponent scored, i.e. the agent won as opponent)
        scores: Dict[int, int] = {}
        for d in agent_dirs:
            try:
                idx = int(d.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            results = self._agent_scores.get(idx, [])
            scores[idx] = sum(1 for g in results if g == -1)
            print(
                f'  agent_{idx:02d}: {scores[idx]} wins as opponent '
                f'({len(results)} games)'
            )

        # Slot 0 is always the current training agent — always keep it
        # Fill remaining survivor slots from highest-scoring opponents
        ranked = sorted(
            (i for i in scores if i != 0),
            key=lambda i: scores[i],
            reverse=True,
        )
        survivor_indices = [0] + ranked[: self.num_survivors - 1]
        print(f'[EvolutionaryPool] Survivors: {[f"agent_{i:02d}" for i in survivor_indices]}')

        # Preserve survivors via temp directory, then rebuild population
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for rank, idx in enumerate(survivor_indices):
                shutil.copytree(
                    str(self._agent_dir(idx)),
                    str(tmp_path / f's_{rank:02d}'),
                )

            # Clear existing population
            for d in agent_dirs:
                shutil.rmtree(str(d), ignore_errors=True)

            # Restore survivors to first slots
            for rank in range(len(survivor_indices)):
                shutil.copytree(
                    str(tmp_path / f's_{rank:02d}'),
                    str(self._agent_dir(rank)),
                )

        # Build 3:2:1 parent assignment across survivors by rank
        n_offspring = self.population_size - len(survivor_indices)
        base_counts = [3, 2, 1]
        parent_assignments: List[int] = []
        for rank in range(len(survivor_indices)):
            count = base_counts[rank] if rank < len(base_counts) else 1
            parent_assignments.extend([rank] * count)
        # Extras go to rank 0
        while len(parent_assignments) < n_offspring:
            parent_assignments.insert(0, 0)
        parent_assignments = parent_assignments[:n_offspring]

        print(f'[EvolutionaryPool] Breeding {n_offspring} offspring (noise std={self.mutation_noise})...')

        # Load survivor states (only those actually needed)
        survivor_states: Dict[int, object] = {}
        for rank in set(parent_assignments):
            path = self._agent_dir(rank) / _MODEL_FILE
            survivor_states[rank] = torch.load(str(path), map_location='cpu')

        for i, parent_rank in enumerate(parent_assignments):
            child_idx = len(survivor_indices) + i
            noisy_state = self._add_noise(survivor_states[parent_rank])
            child_dir = self._agent_dir(child_idx)
            child_dir.mkdir(parents=True, exist_ok=True)
            torch.save(noisy_state, str(child_dir / _MODEL_FILE))
            print(f'  agent_{child_idx:02d} = survivor_rank{parent_rank} + noise')

        # Reset tracking for next generation
        self._agent_scores.clear()
        self._current_opponent_idx = None

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'evolutionary_pool/generation': self.generation,
                    'evolutionary_pool/population_size': self.population_size,
                    'evolutionary_pool/num_survivors': len(survivor_indices),
                }, step=step)
        except ImportError:
            pass

        print(f'[EvolutionaryPool] Generation {self.generation} complete\n')
