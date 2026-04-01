"""
SAC Algorithm — Off-Policy Soft Actor-Critic
=============================================
Implements the Algorithm ABC for off-policy training from game transcripts.

Components:
  ReplayBuffer    - Circular off-policy buffer for (s, a, r, s', d) tuples
  QNetwork        - Q(s, a) critic head (takes encoder embedding + action)
  SACAlgorithm    - Full SAC with twin critics, auto-alpha, target networks

Designed for the live-play training loop: transcripts are loaded into the
replay buffer, then SAC runs gradient updates between games.
"""
from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from encoder import SharedTransformerEncoder, ENTITY_TYPE_IDS_1V1, D_MODEL, N_TOKENS, TOKEN_FEATURES
from policy_head import StochasticPolicyHead
from training.abstractions import Algorithm, ActionResult


# ---------------------------------------------------------------------------
# ReplayBuffer — circular off-policy storage
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity circular replay buffer for SAC.

    Stores flat observations (obs_dim = T_WINDOW * N_TOKENS * TOKEN_FEATURES).
    All arrays are pre-allocated for zero-copy writes.
    """

    def __init__(
        self,
        capacity: int = 500_000,
        obs_dim: int = 800,
        action_dim: int = 8,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float | np.ndarray,
        next_obs: np.ndarray,
        done: float | np.ndarray,
    ) -> None:
        """Add a single transition or a batch of transitions."""
        obs = np.atleast_2d(obs)
        action = np.atleast_2d(action)
        reward = np.atleast_1d(np.asarray(reward, dtype=np.float32))
        next_obs = np.atleast_2d(next_obs)
        done = np.atleast_1d(np.asarray(done, dtype=np.float32))

        batch = obs.shape[0]
        for i in range(batch):
            self.obs[self.pos] = obs[i]
            self.actions[self.pos] = action[i]
            self.rewards[self.pos] = reward[i]
            self.next_obs[self.pos] = next_obs[i]
            self.dones[self.pos] = done[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a contiguous batch efficiently."""
        n = obs.shape[0]
        if n == 0:
            return
        # Handle wraparound
        end = self.pos + n
        if end <= self.capacity:
            self.obs[self.pos:end] = obs
            self.actions[self.pos:end] = actions
            self.rewards[self.pos:end] = rewards
            self.next_obs[self.pos:end] = next_obs
            self.dones[self.pos:end] = dones
        else:
            first = self.capacity - self.pos
            self.obs[self.pos:] = obs[:first]
            self.actions[self.pos:] = actions[:first]
            self.rewards[self.pos:] = rewards[:first]
            self.next_obs[self.pos:] = next_obs[:first]
            self.dones[self.pos:] = dones[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.dones[:rest] = dones[first:]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        """Sample a random batch and return as tensors on device."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.tensor(self.obs[indices], dtype=torch.float32, device=device),
            'actions': torch.tensor(self.actions[indices], dtype=torch.float32, device=device),
            'rewards': torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            'next_obs': torch.tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
        }

    def reset(self) -> None:
        """No-op for off-policy buffer.

        The AsyncUpdater calls buffer.reset() after each update.
        For on-policy (PPO), that clears the rollout.  For off-policy
        (SAC), we keep all data — the circular buffer handles eviction.
        """
        pass

    def clear(self) -> None:
        """Actually clear the buffer (for testing or explicit reset)."""
        self.pos = 0
        self.size = 0


# ---------------------------------------------------------------------------
# QNetwork — critic head
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Q(s, a) = f(encoder_embedding || action).

    Takes the 64-dim encoder output concatenated with the 8-dim action.
    Two hidden layers with ReLU, outputs a scalar Q-value.
    """

    def __init__(self, d_model: int = D_MODEL, action_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) Q-values."""
        return self.net(torch.cat([embedding, action], dim=-1))


# ---------------------------------------------------------------------------
# SACAlgorithm — Algorithm ABC implementation
# ---------------------------------------------------------------------------

class SACAlgorithm(Algorithm):
    """
    Soft Actor-Critic with:
      - SharedTransformerEncoder (shared between actor and critics)
      - StochasticPolicyHead as the actor
      - Twin Q-networks as critics
      - Automatic entropy coefficient (alpha) tuning
      - Target networks with Polyak averaging

    Implements the Algorithm ABC from abstractions.py.
    """

    def __init__(self, config: dict):
        params = {**self.default_params(), **config.get('algorithm', {}).get('params', {})}
        self.lr = params['lr']
        self.critic_lr = params.get('critic_lr', params['lr'])
        self.alpha_lr = params.get('alpha_lr', params['lr'])
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.batch_size = params['batch_size']
        self.buffer_capacity = params['buffer_capacity']
        self.min_buffer_size = params['min_buffer_size']
        self.updates_per_step = params['updates_per_step']

        self.t_window = config.get('t_window', 8)
        _device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(_device)

        # ── Actor: encoder + stochastic policy ──
        self.encoder = SharedTransformerEncoder(d_model=D_MODEL)
        self.policy = StochasticPolicyHead(d_model=D_MODEL)
        self.encoder.to(self.device)
        self.policy.to(self.device)

        # ── Critics: twin Q-networks ──
        self.q1 = QNetwork(d_model=D_MODEL).to(self.device)
        self.q2 = QNetwork(d_model=D_MODEL).to(self.device)

        # ── Target networks (EMA copies) ──
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_q1.parameters():
            p.requires_grad = False
        for p in self.target_q2.parameters():
            p.requires_grad = False

        # ── Automatic entropy tuning ──
        self.target_entropy = -4.0  # -action_dim / 2
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32,
                                      device=self.device, requires_grad=True)

        # ── Optimizers ──
        self.actor_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.critic_lr,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr,
        )

        # ── Replay buffer ──
        obs_dim = self.t_window * N_TOKENS * TOKEN_FEATURES
        self.buffer = ReplayBuffer(
            capacity=self.buffer_capacity,
            obs_dim=obs_dim,
        )

        self._entity_ids = torch.tensor(
            ENTITY_TYPE_IDS_1V1, dtype=torch.long, device=self.device)

        self.encoder.eval()
        self.policy.eval()

        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')

        # Compatibility with train.py's async updater
        self._buffer_ready = threading.Event()
        self._buffer_ready.set()

    @classmethod
    def default_params(cls) -> dict:
        return {
            'lr': 3e-4,
            'critic_lr': 3e-4,
            'alpha_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 256,
            'buffer_capacity': 500_000,
            'min_buffer_size': 1000,
            'updates_per_step': 1,
        }

    @classmethod
    def default_search_space(cls) -> dict:
        return {
            'algorithm.params.lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'algorithm.params.gamma': {'type': 'float', 'low': 0.95, 'high': 0.999},
            'algorithm.params.tau': {'type': 'float', 'low': 0.001, 'high': 0.01},
            'algorithm.params.batch_size': {'type': 'categorical', 'choices': [128, 256, 512]},
        }

    # ── encoding ──────────────────────────────────────────────────────

    def _encode(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Flat obs → encoder embedding."""
        if isinstance(obs, np.ndarray):
            x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            x = obs
        batch = x.shape[0]
        tokens = x.view(batch, self.t_window, N_TOKENS, TOKEN_FEATURES)
        return self.encoder(tokens, self._entity_ids)

    def _encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Flat obs → target encoder embedding (no grad)."""
        batch = obs.shape[0]
        tokens = obs.view(batch, self.t_window, N_TOKENS, TOKEN_FEATURES)
        return self.target_encoder(tokens, self._entity_ids)

    # ── Algorithm ABC ─────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> ActionResult:
        """Sample stochastic action for a batch of observations."""
        emb = self._encode(obs)
        action, log_prob, value, entropy = self.policy(emb)
        return ActionResult(
            action=action.cpu().numpy(),
            aux={'log_prob': log_prob.cpu(), 'value': value.cpu()},
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
        """Store transition in replay buffer."""
        self.buffer.add(
            obs=obs,
            action=action_result.action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )

    def should_update(self) -> bool:
        return self.buffer.size >= self.min_buffer_size

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self) -> dict:
        """Run SAC gradient updates. Returns metrics dict."""
        self.encoder.train()
        self.policy.train()
        self.q1.train()
        self.q2.train()

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        total_alpha = 0.0
        total_q_mean = 0.0
        n_updates = 0

        for _ in range(self.updates_per_step):
            batch = self.buffer.sample(self.batch_size, self.device)
            obs = batch['obs']
            actions = batch['actions']
            rewards = batch['rewards']
            next_obs = batch['next_obs']
            dones = batch['dones']

            # ── Critic update ────────────────────────────────────────
            with torch.no_grad():
                next_emb = self._encode_target(next_obs)
                next_action, next_log_prob, _, _ = self.policy(next_emb)
                target_q1 = self.target_q1(next_emb, next_action)
                target_q2 = self.target_q2(next_emb, next_action)
                target_q = torch.min(target_q1, target_q2).squeeze(-1)
                target_q = target_q - self.alpha.detach() * next_log_prob
                target = rewards + self.gamma * (1.0 - dones) * target_q

            emb = self._encode(obs)
            q1_val = self.q1(emb.detach(), actions).squeeze(-1)
            q2_val = self.q2(emb.detach(), actions).squeeze(-1)
            critic_loss = F.mse_loss(q1_val, target) + F.mse_loss(q2_val, target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ── Actor update ─────────────────────────────────────────
            emb_actor = self._encode(obs)
            new_action, log_prob, _, _ = self.policy(emb_actor)
            q1_new = self.q1(emb_actor, new_action)
            q2_new = self.q2(emb_actor, new_action)
            q_new = torch.min(q1_new, q2_new).squeeze(-1)
            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ── Alpha update ─────────────────────────────────────────
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # ── Soft-update target networks ──────────────────────────
            self._soft_update(self.encoder, self.target_encoder)
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_alpha_loss += alpha_loss.item()
            total_alpha += self.alpha.item()
            total_q_mean += q_new.mean().item()
            n_updates += 1

        self.encoder.eval()
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()

        if n_updates == 0:
            return {}
        return {
            'critic_loss': total_critic_loss / n_updates,
            'actor_loss': total_actor_loss / n_updates,
            'alpha_loss': total_alpha_loss / n_updates,
            'alpha': total_alpha / n_updates,
            'q_mean': total_q_mean / n_updates,
            'buffer_size': self.buffer.size,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak averaging: θ_targ ← τ*θ + (1-τ)*θ_targ."""
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    # ── checkpoint management ─────────────────────────────────────────

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'buffer_pos': self.buffer.pos,
            'buffer_size': self.buffer.size,
        }, path / 'checkpoint.pt')

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(
            Path(path) / 'checkpoint.pt',
            map_location=self.device,
            weights_only=False,
        )
        self.encoder.load_state_dict(ckpt['encoder'])
        self.policy.load_state_dict(ckpt['policy'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        self.target_encoder.load_state_dict(ckpt['target_encoder'])
        self.target_q1.load_state_dict(ckpt['target_q1'])
        self.target_q2.load_state_dict(ckpt['target_q2'])
        self.log_alpha = ckpt['log_alpha'].to(self.device).requires_grad_(True)
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_optimizer'])
        # Restore alpha optimizer param group to point at new log_alpha
        self.alpha_optimizer.param_groups[0]['params'] = [self.log_alpha]

    def save_deployment_weights(self, model_dir: Path) -> None:
        """Save encoder + policy for the bot to load at inference time."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), model_dir / 'encoder.pt')
        self.policy.save(str(model_dir / 'policy.pt'))

    def get_weights(self) -> dict:
        return {
            'encoder': {k: v.clone() for k, v in self.encoder.state_dict().items()},
            'policy': {k: v.clone() for k, v in self.policy.state_dict().items()},
        }

    def clone_from(self, other: 'SACAlgorithm', noise_scale: float = 0.0) -> None:
        """Copy weights from another SAC agent, optionally with noise."""
        self.encoder.load_state_dict(copy.deepcopy(other.encoder.state_dict()))
        self.policy.load_state_dict(copy.deepcopy(other.policy.state_dict()))
        self.q1.load_state_dict(copy.deepcopy(other.q1.state_dict()))
        self.q2.load_state_dict(copy.deepcopy(other.q2.state_dict()))
        self.target_encoder.load_state_dict(copy.deepcopy(other.target_encoder.state_dict()))
        self.target_q1.load_state_dict(copy.deepcopy(other.target_q1.state_dict()))
        self.target_q2.load_state_dict(copy.deepcopy(other.target_q2.state_dict()))
        if noise_scale > 0:
            with torch.no_grad():
                for p in (list(self.encoder.parameters()) +
                          list(self.policy.parameters()) +
                          list(self.q1.parameters()) +
                          list(self.q2.parameters())):
                    p.add_(torch.randn_like(p) * noise_scale)
        # Reset optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.critic_lr,
        )
