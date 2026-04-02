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
        self._inference_only = config.get('inference', False)

        _device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(_device)

        # Networks
        self.encoder = SharedTransformerEncoder(d_model=D_MODEL)
        self.policy = StochasticPolicyHead(d_model=D_MODEL)
        self.encoder.to(self.device)
        self.policy.to(self.device)

        if not self._inference_only:
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
            'minibatch_size': 2048,
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
            'algorithm.params.minibatch_size': {'type': 'categorical', 'choices': [256, 512, 1024, 2048]},
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

    def load_weights(self, weights: dict) -> None:
        self.encoder.load_state_dict(weights['encoder'])
        self.policy.load_state_dict(weights['policy'])
        self.encoder.eval()
        self.policy.eval()

    @torch.no_grad()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        emb = self._encode(obs)
        action, _ = self.policy.act_deterministic(emb)
        return action.cpu().numpy()

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


# Re-export Population for backward compatibility
from training.opponents.population import Population  # noqa: F401
