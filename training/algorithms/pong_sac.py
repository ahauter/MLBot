"""
Pong SAC Algorithm — Algorithm ABC Implementation
===================================================
CleanRL-style SAC for Pong environments, conforming to the training
framework's Algorithm interface.

Single-file: ReplayBuffer, Actor (squashed Gaussian), twin Q-critics,
automatic entropy tuning, per-step TD updates.

Usage (YAML config)
-------------------
    algorithm:
      class: training.algorithms.pong_sac.PongSACAlgorithm
      params:
        obs_dim: 4
        hidden: 256
        lr: 3e-4
"""
from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from training.abstractions import Algorithm, ActionResult


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular replay buffer with pre-allocated numpy arrays."""

    def __init__(self, obs_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def reset(self):
        """No-op. Circular replay buffer persists across updates."""
        pass

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )


# ---------------------------------------------------------------------------
# Actor — Squashed Gaussian Policy
# ---------------------------------------------------------------------------

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """Reparameterized sample with log-prob (tanh squashing)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        action = torch.tanh(u)
        log_prob = normal.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def get_deterministic(self, obs):
        """Deterministic action (mean, tanh-squashed)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


# ---------------------------------------------------------------------------
# Twin Q-Networks
# ---------------------------------------------------------------------------

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.q1_fc1 = nn.Linear(obs_dim + 1, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_out = nn.Linear(hidden, 1)
        self.q2_fc1 = nn.Linear(obs_dim + 1, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_out = nn.Linear(hidden, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


# ---------------------------------------------------------------------------
# PongSACAlgorithm — Algorithm ABC
# ---------------------------------------------------------------------------

class PongSACAlgorithm(Algorithm):
    """
    SAC for Pong environments — implements the Algorithm ABC.

    Pong uses 1D continuous actions. Internally we work with scalar actions
    and pad to 8-dim when returning ActionResult (framework convention).
    """

    def __init__(self, config: dict):
        params = config.get('algorithm', {}).get('params', {})

        # Hyperparameters
        self.obs_dim = params.get('obs_dim', 4)
        self.hidden = params.get('hidden', 256)
        self.lr = params.get('lr', 3e-4)
        self.alpha_lr = params.get('alpha_lr', 1e-3)
        self.gamma = params.get('gamma', 0.99)
        self.tau = params.get('tau', 0.005)
        self.batch_size = params.get('batch_size', 256)
        self.buffer_size = params.get('buffer_size', 100_000)
        self.random_steps = params.get('random_steps', 5000)
        self.update_every = params.get('update_every', 1)

        self._inference_only = config.get('inference', False)
        _device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(_device)

        # Networks
        self.actor = Actor(self.obs_dim, self.hidden).to(self.device)
        self.critic = SoftQNetwork(self.obs_dim, self.hidden).to(self.device)
        self.target_critic = SoftQNetwork(self.obs_dim, self.hidden).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        if not self._inference_only:
            # Optimizers
            self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
            self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

            # Entropy tuning
            self.target_entropy = -1.0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=self.alpha_lr)

            # Replay buffer
            self.buffer = ReplayBuffer(self.obs_dim, self.buffer_size)

        self.alpha = 0.0 if self._inference_only else self.log_alpha.exp().item()

        # Step counter for random exploration and update scheduling
        self._total_steps = 0
        self._steps_since_update = 0

        # Required by AsyncUpdater in train.py
        self._buffer_ready = threading.Event()
        self._buffer_ready.set()

        # Eval mode by default
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    @classmethod
    def default_params(cls) -> dict:
        return {
            'obs_dim': 4,
            'hidden': 256,
            'lr': 3e-4,
            'alpha_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 256,
            'buffer_size': 100_000,
            'random_steps': 5000,
            'update_every': 1,
        }

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            'algorithm.params.lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'algorithm.params.alpha_lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'algorithm.params.hidden': {'type': 'categorical', 'choices': [64, 128, 256]},
            'algorithm.params.batch_size': {'type': 'categorical', 'choices': [64, 128, 256]},
            'algorithm.params.tau': {'type': 'float', 'low': 0.001, 'high': 0.01},
        }

    def _pad_action(self, action_1d: np.ndarray) -> np.ndarray:
        """Pad 1D pong action to 8-dim framework convention."""
        batch = action_1d.shape[0]
        padded = np.zeros((batch, 8), dtype=np.float32)
        padded[:, 0] = action_1d.ravel()
        return padded

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> ActionResult:
        """Pick actions for a batch of observations."""
        batch = obs.shape[0] if obs.ndim > 1 else 1
        obs_2d = obs.reshape(batch, -1)
        self._total_steps += batch

        if self._total_steps <= self.random_steps:
            action_1d = np.random.uniform(-1.0, 1.0, size=(batch,))
        else:
            obs_t = torch.from_numpy(obs_2d.astype(np.float32)).to(self.device)
            action_t, _ = self.actor.sample(obs_t)
            action_1d = action_t.cpu().numpy().ravel()
            action_1d = np.clip(action_1d, -1.0, 1.0)

        return ActionResult(
            action=self._pad_action(action_1d),
            aux={},
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
        """Store transitions. Handles vectorized envs (batch dimension)."""
        obs_2d = np.atleast_2d(obs)
        next_obs_2d = np.atleast_2d(next_obs)
        reward_arr = np.atleast_1d(np.asarray(reward, dtype=np.float32))
        done_arr = np.atleast_1d(np.asarray(done, dtype=np.float32))
        actions = action_result.action  # (batch, 8)

        for i in range(obs_2d.shape[0]):
            self.buffer.add(
                obs_2d[i],
                actions[i, 0],  # extract 1D pong action
                reward_arr[i],
                next_obs_2d[i],
                done_arr[i],
            )
            self._steps_since_update += 1

    def should_update(self) -> bool:
        """True when we have enough data and enough steps since last update."""
        return (self.buffer.size >= self.random_steps
                and self._steps_since_update >= self.update_every)

    def update(self) -> dict:
        """Run SAC update. Returns metrics dict."""
        self._steps_since_update = 0

        self.actor.train()
        self.critic.train()

        b_obs, b_act, b_rew, b_next, b_done = self.buffer.sample(
            self.batch_size, self.device)

        alpha = self.log_alpha.exp().item()

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(b_next)
            q1_next, q2_next = self.target_critic(b_next, next_action)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
            q_target = b_rew + self.gamma * (1 - b_done) * q_next

        q1_pred, q2_pred = self.critic(b_obs, b_act)
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        new_action, log_prob = self.actor.sample(b_obs)
        q1_new, q2_new = self.critic(b_obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Entropy coefficient update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # Target network soft update
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.target_critic.parameters()):
                tp.data.lerp_(p.data, self.tau)

        self.actor.eval()
        self.critic.eval()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'q_mean': q1_pred.mean().item(),
            'entropy': -log_prob.mean().item(),
            'buffer_size': self.buffer.size,
        }

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'total_steps': self._total_steps,
        }, path / 'checkpoint.pt')

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(
            Path(path) / 'checkpoint.pt',
            map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.log_alpha = ckpt['log_alpha'].to(self.device).requires_grad_(True)
        self.alpha_opt.load_state_dict(ckpt['alpha_opt'])
        self._total_steps = ckpt.get('total_steps', 0)

    def get_weights(self) -> dict:
        """Return actor+critic weights for snapshot saving."""
        return {
            'encoder': {k: v.clone() for k, v in self.actor.state_dict().items()},
            'policy': {k: v.clone() for k, v in self.critic.state_dict().items()},
        }

    def load_weights(self, weights: dict) -> None:
        """Load actor+critic weights from snapshot."""
        self.actor.load_state_dict(weights['encoder'])
        self.critic.load_state_dict(weights['policy'])
        self.actor.eval()
        self.critic.eval()

    @torch.no_grad()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic inference. Returns (batch, 8) numpy array."""
        batch = obs.shape[0] if obs.ndim > 1 else 1
        obs_2d = obs.reshape(batch, -1)
        obs_t = torch.from_numpy(obs_2d[:, :self.obs_dim].astype(np.float32)).to(self.device)
        action_1d = self.actor.get_deterministic(obs_t).cpu().numpy().ravel()
        return self._pad_action(action_1d)

    def clone_from(self, other: 'PongSACAlgorithm', noise_scale: float = 0.0) -> None:
        """Copy weights from another SAC agent, optionally with noise."""
        self.actor.load_state_dict(copy.deepcopy(other.actor.state_dict()))
        self.critic.load_state_dict(copy.deepcopy(other.critic.state_dict()))
        self.target_critic.load_state_dict(copy.deepcopy(other.target_critic.state_dict()))
        if noise_scale > 0:
            with torch.no_grad():
                for p in list(self.actor.parameters()) + list(self.critic.parameters()):
                    p.add_(torch.randn_like(p) * noise_scale)
        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)
