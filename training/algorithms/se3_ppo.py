"""
SE3 PPO Algorithm
=================
PPO-Clip with SE3Encoder + StochasticSE3Policy.

The SE3Encoder holds learned field geometry (k_spatial, quaternions) as
nn.Parameters.  During updates, the encoder recomputes the coefficient
update differentiably from stored (raw_state, prev_coefficients), giving
gradient flow to the field geometry via truncated BPTT (depth 1).

Implements the Algorithm ABC from abstractions.py.
"""
from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from se3_field import SE3Encoder, SE3_OBS_DIM, COEFF_DIM
from se3_policy import StochasticSE3Policy
from training.abstractions import Algorithm, ActionResult
from training.algorithms.ppo import RolloutBuffer


class SE3PPOAlgorithm(Algorithm):
    """
    PPO-Clip with SE3Encoder (learned field geometry) + StochasticSE3Policy.

    obs_dim = 185 = raw_state (57) + prev_coefficients (128).
    The encoder's forward pass produces 128-dim policy input differentiably.
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

        _device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(_device)

        # Networks
        self.encoder = SE3Encoder()
        self.policy = StochasticSE3Policy(obs_dim=COEFF_DIM)
        self.encoder.to(self.device)
        self.policy.to(self.device)

        # Optimizer over both encoder and policy
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
        )

        # Rollout buffer — obs_dim = 185
        self.buffer = RolloutBuffer(
            capacity=self.rollout_steps,
            num_envs=self.num_envs,
            obs_dim=SE3_OBS_DIM,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Start in eval mode for inference
        self.encoder.eval()
        self.policy.eval()

        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')

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
            'algorithm.params.minibatch_size': {
                'type': 'categorical', 'choices': [256, 512, 1024, 2048]},
        }

    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        """Encode packed observations (185-dim) → 128-dim coefficients."""
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.encoder(x)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> ActionResult:
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
        reward_arr = np.atleast_1d(np.asarray(reward, dtype=np.float32))
        done_arr = np.atleast_1d(np.asarray(done, dtype=np.float32))
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
        self.encoder.train()
        self.policy.train()

        # Compute last values for GAE
        last_obs = self.buffer.obs[self.buffer.pos - 1]
        with torch.no_grad():
            last_emb = self._encode(last_obs)
            _, last_values = self.policy.act_deterministic(last_emb)
            last_values = last_values.cpu().numpy()

        self.buffer.compute_gae(last_values)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.iterate_minibatches_gpu(
                    self.minibatch_size, self.device):
                obs_t = batch['obs']             # (mb, 185)
                actions_t = batch['actions']
                old_log_probs_t = batch['log_probs']
                advantages_t = batch['advantages']
                returns_t = batch['returns']

                if advantages_t.numel() > 1:
                    advantages_t = (advantages_t - advantages_t.mean()) / (
                        advantages_t.std() + 1e-8)

                # Differentiable encoder forward
                emb = self.encoder(obs_t)        # (mb, 128)
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    emb, actions_t)

                # PPO-Clip loss
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, returns_t)

                loss = (policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.policy.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Project quaternions back to unit sphere
                self.encoder.normalise_quaternions_()

                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = (old_log_probs_t - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_fraction += clip_fraction
                total_approx_kl += approx_kl
                n_updates += 1

        self.encoder.eval()
        self.policy.eval()

        self.buffer.reset()
        self._buffer_ready.set()

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
        ckpt = torch.load(
            Path(path) / 'checkpoint.pt',
            map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.policy.load_state_dict(ckpt['policy'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def get_weights(self) -> dict:
        return {
            'encoder': {k: v.clone() for k, v in self.encoder.state_dict().items()},
            'policy': {k: v.clone() for k, v in self.policy.state_dict().items()},
        }

    def clone_from(self, other: 'SE3PPOAlgorithm', noise_scale: float = 0.0) -> None:
        self.encoder.load_state_dict(copy.deepcopy(other.encoder.state_dict()))
        self.policy.load_state_dict(copy.deepcopy(other.policy.state_dict()))
        if noise_scale > 0:
            with torch.no_grad():
                for p in list(self.encoder.parameters()) + list(self.policy.parameters()):
                    p.add_(torch.randn_like(p) * noise_scale)
            self.encoder.normalise_quaternions_()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
        )
