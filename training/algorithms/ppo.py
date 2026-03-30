"""
PPO Algorithm + Population-Based Training
==========================================
Pure PyTorch PPO implementation with GAE, designed to plug into the
abstract Algorithm interface from abstractions.py.

Components:
  RolloutBuffer    - Stores on-policy rollouts, computes GAE
  PPOAlgorithm     - PPO-Clip with shared encoder + StochasticPolicyHead
  Population       - Multiple PPO agents sharing vectorised envs

No d3rlpy dependency.  Uses SharedTransformerEncoder (src/encoder.py)
and StochasticPolicyHead (src/policy_head.py).
"""
from __future__ import annotations

import copy
import random
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import SharedTransformerEncoder, ENTITY_TYPE_IDS_1V1, D_MODEL, N_TOKENS, TOKEN_FEATURES
from policy_head import StochasticPolicyHead
from training.abstractions import Algorithm, ActionResult
from training.opponents.pool import OpponentPool


# ---------------------------------------------------------------------------
# RolloutBuffer — stores on-policy trajectories, computes GAE
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-size buffer for on-policy rollouts with GAE computation.

    Parameters
    ----------
    capacity : int
        Max transitions per environment.
    num_envs : int
        Number of parallel environments.
    obs_dim : int
        Observation dimensionality.
    action_dim : int
        Action dimensionality.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_dim: int = 800,
        action_dim: int = 8,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.capacity = capacity
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((capacity, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
        self.dones = np.zeros((capacity, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((capacity, num_envs), dtype=np.float32)
        self.values = np.zeros((capacity, num_envs), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((capacity, num_envs), dtype=np.float32)
        self.returns = np.zeros((capacity, num_envs), dtype=np.float32)

        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Add a transition for all envs at the current position."""
        assert self.pos < self.capacity, "Buffer full — call compute_gae then reset"
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray) -> None:
        """
        Compute Generalized Advantage Estimation.

        Parameters
        ----------
        last_values : (num_envs,) array
            Value estimates for the state AFTER the last stored transition.
        """
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.pos)):
            if t == self.pos - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

    def iterate_minibatches(self, minibatch_size: int):
        """
        Yield minibatches of flattened (across envs and time) transitions.

        Yields dicts with keys: obs, actions, log_probs, advantages, returns.
        """
        total = self.pos * self.num_envs
        indices = np.random.permutation(total)

        # Flatten (time, envs) -> (total,)
        flat_obs = self.obs[:self.pos].reshape(total, self.obs_dim)
        flat_actions = self.actions[:self.pos].reshape(total, self.action_dim)
        flat_log_probs = self.log_probs[:self.pos].reshape(total)
        flat_advantages = self.advantages[:self.pos].reshape(total)
        flat_returns = self.returns[:self.pos].reshape(total)

        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = indices[start:end]
            yield {
                'obs': flat_obs[idx],
                'actions': flat_actions[idx],
                'log_probs': flat_log_probs[idx],
                'advantages': flat_advantages[idx],
                'returns': flat_returns[idx],
            }

    def iterate_minibatches_gpu(self, minibatch_size: int, device: torch.device):
        """
        Yield GPU-resident minibatch tensors. Converts numpy→GPU once per epoch,
        then slices on-device. Avoids per-minibatch numpy→torch overhead.
        """
        total = self.pos * self.num_envs

        # Single numpy→GPU transfer per array
        t_obs = torch.tensor(
            self.obs[:self.pos].reshape(total, self.obs_dim),
            dtype=torch.float32, device=device)
        t_actions = torch.tensor(
            self.actions[:self.pos].reshape(total, self.action_dim),
            dtype=torch.float32, device=device)
        t_log_probs = torch.tensor(
            self.log_probs[:self.pos].reshape(total),
            dtype=torch.float32, device=device)
        t_advantages = torch.tensor(
            self.advantages[:self.pos].reshape(total),
            dtype=torch.float32, device=device)
        t_returns = torch.tensor(
            self.returns[:self.pos].reshape(total),
            dtype=torch.float32, device=device)

        indices = torch.randperm(total, device=device)
        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = indices[start:end]
            yield {
                'obs': t_obs[idx],
                'actions': t_actions[idx],
                'log_probs': t_log_probs[idx],
                'advantages': t_advantages[idx],
                'returns': t_returns[idx],
            }

    def reset(self) -> None:
        """Reset buffer position (does not zero arrays for speed)."""
        self.pos = 0


# ---------------------------------------------------------------------------
# PPOAlgorithm — Algorithm ABC implementation
# ---------------------------------------------------------------------------

class PPOAlgorithm(Algorithm):
    """
    PPO-Clip with SharedTransformerEncoder + StochasticPolicyHead.

    Implements the Algorithm ABC from abstractions.py.
    """

    def __init__(self, config: dict):
        params = {**self.default_params(), **config.get('algorithm', {}).get('params', {})}
        self.lr = params['lr']
        self.gamma = params['gamma']
        self.gae_lambda = params['gae_lambda']
        self.clip_epsilon = params['clip_epsilon']
        self.vf_coef = params['vf_coef']
        self.ent_coef = params['ent_coef']
        self.max_grad_norm = params['max_grad_norm']
        self.rollout_steps = params['rollout_steps']
        self.ppo_epochs = params['ppo_epochs']
        self.minibatch_size = params['minibatch_size']

        self.num_envs = config.get('num_envs', 1)
        self.t_window = config.get('t_window', 8)

        _device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(_device)

        # Networks
        self.encoder = SharedTransformerEncoder(d_model=D_MODEL)
        self.policy = StochasticPolicyHead(d_model=D_MODEL)
        self.encoder.to(self.device)
        self.policy.to(self.device)

        # Optimizer over both encoder and policy
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
        )

        # Rollout buffer
        obs_dim = self.t_window * N_TOKENS * TOKEN_FEATURES
        self.buffer = RolloutBuffer(
            capacity=self.rollout_steps,
            num_envs=self.num_envs,
            obs_dim=obs_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        self._entity_ids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long, device=self.device)

        # Start in eval mode for inference; update() switches to train mode
        self.encoder.eval()
        self.policy.eval()

        # Enable TF32 tensor cores for ~2x matmul throughput on Ampere+ GPUs.
        # Minimal precision loss (10-bit mantissa vs 23-bit) — fine for RL.
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')

        # Signals when the rollout buffer is free for new collection.
        # Cleared when an update is triggered; set again after buffer.reset().
        self._buffer_ready = threading.Event()
        self._buffer_ready.set()

    @classmethod
    def default_params(cls) -> dict:
        return {
            'lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'vf_coef': 0.5,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
            'rollout_steps': 2048,
            'ppo_epochs': 4,
            'minibatch_size': 512,
        }

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            'algorithm.params.lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'algorithm.params.clip_epsilon': {'type': 'float', 'low': 0.1, 'high': 0.3},
            'algorithm.params.vf_coef': {'type': 'float', 'low': 0.1, 'high': 1.0},
            'algorithm.params.ent_coef': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
            'algorithm.params.gae_lambda': {'type': 'float', 'low': 0.9, 'high': 1.0},
            'algorithm.params.ppo_epochs': {'type': 'int', 'low': 2, 'high': 10},
            'algorithm.params.minibatch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
        }

    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        """Encode flat observations to embeddings."""
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        batch = x.shape[0]
        tokens = x.view(batch, self.t_window, N_TOKENS, TOKEN_FEATURES)
        return self.encoder(tokens, self._entity_ids)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> ActionResult:
        """Pick actions for a batch of observations (on-policy, stochastic)."""
        emb = self._encode(obs)
        action, log_prob, value, entropy = self.policy(emb)
        return ActionResult(
            action=action.cpu().numpy(),
            aux={
                'log_prob': log_prob.cpu(),
                'value': value.cpu(),
            },
        )

    def store_transition(
        self,
        obs: np.ndarray,
        action_result: ActionResult,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """Store a transition. For vectorized envs, reward/done are arrays."""
        reward_arr = np.atleast_1d(np.asarray(reward, dtype=np.float32))
        done_arr = np.atleast_1d(np.asarray(done, dtype=np.float32))
        # log_prob/value may be tensors or numpy; coerce to numpy for buffer
        lp = action_result.aux['log_prob']
        val = action_result.aux['value']
        if isinstance(lp, torch.Tensor):
            lp = lp.numpy()
        if isinstance(val, torch.Tensor):
            val = val.numpy()
        self.buffer.add(
            obs=obs,
            action=action_result.action,
            reward=reward_arr,
            done=done_arr,
            log_prob=lp,
            value=val,
        )

    def should_update(self) -> bool:
        return self.buffer.pos >= self.buffer.capacity

    def update(self) -> dict:
        """Run PPO gradient updates over the collected rollout."""
        self.encoder.train()
        self.policy.train()

        # Compute last values for GAE
        last_obs = self.buffer.obs[self.buffer.pos - 1]  # (num_envs, obs_dim)
        with torch.no_grad():
            last_emb = self._encode(last_obs)
            _, last_values = self.policy.act_deterministic(last_emb)
            last_values = last_values.cpu().numpy()

        self.buffer.compute_gae(last_values)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.iterate_minibatches_gpu(
                    self.minibatch_size, self.device):
                obs_t = batch['obs']
                actions_t = batch['actions']
                old_log_probs_t = batch['log_probs']
                advantages_t = batch['advantages']
                returns_t = batch['returns']

                # Normalize advantages
                if advantages_t.numel() > 1:
                    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

                # Forward
                b = obs_t.shape[0]
                tokens = obs_t.view(b, self.t_window, N_TOKENS, TOKEN_FEATURES)
                emb = self.encoder(tokens, self._entity_ids)
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(emb, actions_t)

                # Policy loss (PPO-Clip)
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, returns_t)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.policy.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Tracking
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = (old_log_probs_t - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_fraction += clip_fraction
                total_approx_kl += approx_kl
                n_updates += 1

        # Return to eval mode for inference
        self.encoder.eval()
        self.policy.eval()

        if n_updates == 0:
            return {}
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'clip_fraction': total_clip_fraction / n_updates,
            'approx_kl': total_approx_kl / n_updates,
        }
    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path / 'checkpoint.pt')

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(Path(path) / 'checkpoint.pt', map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.policy.load_state_dict(ckpt['policy'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def get_weights(self) -> dict:
        return {
            'encoder': {k: v.clone() for k, v in self.encoder.state_dict().items()},
            'policy': {k: v.clone() for k, v in self.policy.state_dict().items()},
        }

    def clone_from(self, other: 'PPOAlgorithm', noise_scale: float = 0.0) -> None:
        """Copy weights from another PPO agent, optionally with noise."""
        self.encoder.load_state_dict(copy.deepcopy(other.encoder.state_dict()))
        self.policy.load_state_dict(copy.deepcopy(other.policy.state_dict()))
        if noise_scale > 0:
            with torch.no_grad():
                for p in list(self.encoder.parameters()) + list(self.policy.parameters()):
                    p.add_(torch.randn_like(p) * noise_scale)
        # Reset optimizer so it doesn't carry momentum from the source
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
        )


# ---------------------------------------------------------------------------
# Population — multiple PPO agents sharing vectorized envs
# ---------------------------------------------------------------------------

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
    """

    def __init__(self, num_agents: int, num_workers: int, config: dict,
                 envs_per_agent: Optional[int] = None,
                 snapshot_dir: str | Path = 'models/snapshots',
                 max_snapshots: int = 50,
                 snapshot_interval: int = 50_000,
                 decay_rate: float = 0.95):
        OpponentPool.__init__(self, snapshot_dir, max_snapshots)
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

        # Create agents, each with buffer sized for its env count
        self._agents: List[PPOAlgorithm] = []
        for i in range(num_agents):
            agent_config = {**config, 'num_envs': envs_per_agent_map[i]}
            self._agents.append(PPOAlgorithm(agent_config))

        # Score tracking per agent
        self.scores: List[List[float]] = [[] for _ in range(num_agents)]
        self._generation = 0

    @classmethod
    def default_params(cls) -> dict:
        return {
            'max_snapshots': 50,
            'snapshot_interval': 50_000,
            'decay_rate': 0.95,
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
    def agents(self) -> List[PPOAlgorithm]:
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
