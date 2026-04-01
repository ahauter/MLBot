"""
Training Framework Abstract Base Classes
=========================================
All ABCs for the training framework. Implement any of these to create
a custom experiment. Point your YAML config at your implementation class.

ABCs:
  Algorithm           - RL algorithm (PPO, SAC, etc.)
  MetricLogger        - Logging backend (W&B, TensorBoard, stdout, etc.)
  RewardFunction      - Reward signal (sparse, dense, learned, etc.)
  EnvironmentProvider - Environment creation and configuration
  ReplayProvider      - Expert demonstration data (Axis 2)
  FeedbackProvider    - Human feedback signal (Axis 3)
  EvaluationHook      - Evaluation protocol and convergence checking
  OpponentPool        - Self-play opponent management (re-exported from self_play.py)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import gymnasium as gym
import numpy as np


# Re-export OpponentPool from its existing home
from training.opponents.pool import OpponentPool


# ---------------------------------------------------------------------------
# ActionResult — returned by Algorithm.select_action()
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Returned by Algorithm.select_action(). Carries action + algo-specific aux data."""
    action: np.ndarray          # (batch, 8) — what goes to the env
    aux: Dict[str, Any] = field(default_factory=dict)  # algo-specific (PPO: log_prob, value)


# ---------------------------------------------------------------------------
# Algorithm — Axis 1 (Simulation)
# ---------------------------------------------------------------------------

class Algorithm(ABC):
    """
    Fully abstract RL algorithm. The training loop is 100% algorithm-agnostic:
    it only calls select_action, store_transition, should_update, update.

    PPO, SAC, or any future algorithm is an internal detail.
    """

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        """Default hyperparameters for training. YAML overrides these."""

    @classmethod
    @abstractmethod
    def default_search_space(cls) -> dict:
        """Default Optuna search ranges for tuning. YAML overrides these.

        Returns dict like:
            {'algorithm.params.lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True}}
        """

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> ActionResult:
        """Pick actions for a batch of observations."""

    @abstractmethod
    def store_transition(self, obs: np.ndarray, action_result: ActionResult,
                         reward: float, next_obs: np.ndarray, done: bool,
                         info: dict) -> None:
        """Feed one transition. Algorithm manages its own buffer internally."""

    @abstractmethod
    def should_update(self) -> bool:
        """Has enough data accumulated for a gradient update?"""

    @abstractmethod
    def update(self) -> dict:
        """Run gradient update(s), return metrics dict for logging."""

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Save full state (weights, optimizer, buffer metadata) to directory."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load full state from directory."""

    @abstractmethod
    def get_weights(self) -> dict:
        """Return encoder+policy state_dicts for opponent loading."""

    @abstractmethod
    def clone_from(self, other: 'Algorithm', noise_scale: float = 0.0) -> None:
        """Copy weights from another algorithm instance, optionally with noise."""


# ---------------------------------------------------------------------------
# MetricLogger — Logging backend
# ---------------------------------------------------------------------------

class MetricLogger(ABC):
    """
    Abstract logging backend. Swappable via YAML config.

    Implementations: WandbLogger (default), StdoutLogger, TensorBoardLogger.
    """

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def init(self, config: dict) -> None:
        """Initialize the logger with the full experiment config dict."""

    @abstractmethod
    def log(self, step: int, **metrics: Any) -> None:
        """Log scalar metrics at the given training step."""

    @abstractmethod
    def finish(self) -> None:
        """Finalize the logging session (close connections, flush buffers)."""


# ---------------------------------------------------------------------------
# RewardFunction — Axis 4 (Reward Engineering)
# ---------------------------------------------------------------------------

class RewardFunction(ABC):
    """
    Defines the reward signal. Dense, sparse, learned, or hybrid.
    Swap via YAML to compare reward engineering strategies.
    """

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def compute_reward(self, obs: np.ndarray, action: np.ndarray,
                       next_obs: np.ndarray, done: bool, info: dict) -> float:
        """Compute reward for a single transition."""

    @abstractmethod
    def on_reset(self) -> None:
        """Called at episode start. Reset any internal state."""

    def get_metrics(self) -> dict:
        """Track Axis 4 cost: number of active reward components, etc."""
        return {}


# ---------------------------------------------------------------------------
# EnvironmentProvider — Axis 1 (Simulation)
# ---------------------------------------------------------------------------

class EnvironmentProvider(ABC):
    """
    Controls how environments are created and configured.
    Swap to change simulation fidelity, domain randomization, etc.
    """

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def make_env(self, reward_fn: RewardFunction, t_window: int) -> gym.Env:
        """Create a single environment instance (called per worker)."""

    @abstractmethod
    def make_vec_env(self, num_envs: int, reward_fn: RewardFunction,
                     t_window: int):
        """Create a vectorized environment (called once per population)."""


# ---------------------------------------------------------------------------
# ReplayProvider — Axis 2 (Real-world Data)
# ---------------------------------------------------------------------------

class ReplayProvider(ABC):
    """
    Provides expert demonstration data. Offline RL, behavioral cloning,
    or replay-seeded training all go through this interface.
    """

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def load_demonstrations(self) -> Optional[list]:
        """Load expert replay data. Returns list of (obs, action, reward) tuples or None."""

    @abstractmethod
    def seed_algorithm(self, algorithm: Algorithm, demonstrations: list) -> None:
        """Pre-fill algorithm's buffer or warm-start its weights from demonstrations."""

    def on_round(self, agents: list, step: int) -> None:
        """Called each collection round.  Check for new data and load it.

        Default is a no-op.  Override in providers that watch for data
        arriving mid-training (e.g. live-play transcripts).
        """

    def get_metrics(self) -> dict:
        """Track Axis 2 cost: replays loaded, bytes processed, etc."""
        return {}


# ---------------------------------------------------------------------------
# FeedbackProvider — Axis 3 (Human Feedback)
# ---------------------------------------------------------------------------

class FeedbackProvider(ABC):
    """
    Provides reward signal from human labels/rankings (RLHF-style).
    """

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def get_feedback_reward(self, obs: np.ndarray, action: np.ndarray,
                            next_obs: np.ndarray, info: dict) -> Optional[float]:
        """Return a human-feedback-based reward signal, or None if not applicable."""

    @abstractmethod
    def should_query(self, step: int) -> bool:
        """Should the system request human feedback at this step?"""

    def get_metrics(self) -> dict:
        """Track Axis 3 cost: labels consumed, annotation time, etc."""
        return {}


# ---------------------------------------------------------------------------
# EvaluationHook — Evaluation protocol
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CollectionScheduler — Agent-environment scheduling
# ---------------------------------------------------------------------------

class CollectionScheduler(ABC):
    """
    Controls how agents are assigned to environment workers each step.

    Different hardware profiles benefit from different strategies:
      - InterleavedScheduler: all agents collect simultaneously, envs split
        evenly. Best when num_envs >> num_agents.
      - SerialScheduler: one agent at a time uses ALL envs, then rotates.
        Best when num_envs is small relative to num_agents (larger batches
        per forward pass, fewer GPU kernel launches).

    Swap via YAML config:
        scheduler:
          class: training.schedulers.SerialScheduler
    """

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def init(self, population, num_envs: int, config: dict) -> None:
        """Initialize with population and environment info."""

    @abstractmethod
    def envs_per_agent(self, num_envs: int, num_agents: int) -> int:
        """How many envs each agent's buffer should be sized for."""

    @abstractmethod
    def iter_steps(self, rollout_steps: int) -> Iterator[Dict[int, List[int]]]:
        """Yield {agent_idx: [worker_ids]} for each collection step.

        May yield more than rollout_steps total steps if agents are
        rotated serially (each agent needs its own rollout_steps).
        """

    def on_round_start(self) -> None:
        """Called at the start of each collection round. Optional hook."""
        pass


# ---------------------------------------------------------------------------
# EvaluationHook — Evaluation protocol
# ---------------------------------------------------------------------------

class EvaluationHook(ABC):
    """
    Evaluation protocol. Called periodically to assess agent performance
    and determine convergence.

    Swappable via YAML config (``evaluation.class``). Implementations
    define how episodes are run, what metrics are computed, what
    convergence means, and optionally how a human can spectate or play.
    """

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def evaluate(self, algorithm: Algorithm, step: int) -> dict:
        """Run evaluation, return metrics dict (e.g. win rates per tier)."""

    @abstractmethod
    def check_convergence(self, eval_results: dict) -> bool:
        """Should training stop based on evaluation results?"""

    def run_interactive(self, algorithm: Algorithm, step: int = 0) -> None:
        """Launch a human spectator or player session.

        Default is a no-op. Implementations should load the agent into
        the environment and run episodes with human-visible output.
        """
