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

from se3_field import (SE3Encoder, SE3_OBS_DIM, RAW_STATE_DIM, COEFF_DIM,
                       N_OBJECTS, K, DT, _BALL_OFF, _EGO_OFF, _OPP_OFF)
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
        self.spectral_ent_coef_intra = params['spectral_ent_coef_intra']
        self.spectral_ent_coef_inter = params['spectral_ent_coef_inter']
        self.dream_steps = params['dream_steps']
        self.dream_entropy_high = params['dream_entropy_high']
        self.dream_entropy_low = params['dream_entropy_low']
        self.dream_ratio = params['dream_ratio']

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
            'spectral_ent_coef_intra': 0.001,
            'spectral_ent_coef_inter': 0.001,
            'dream_steps': 5,
            'dream_entropy_high': 2.0,
            'dream_entropy_low': 0.1,
            'dream_ratio': 0.25,
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
            'algorithm.params.spectral_ent_coef_intra': {
                'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'algorithm.params.spectral_ent_coef_inter': {
                'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
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

    @torch.no_grad()
    def _dream_rollout(self, seed_obs: np.ndarray) -> np.ndarray:
        """Free-run encoder from a seed observation using velocity extrapolation.

        Euler-steps positions (ball, ego, opp) using velocities from the raw
        state, then runs the encoder's LMS update on the extrapolated obs.
        Terminates when H_intra exits [dream_entropy_low, dream_entropy_high].

        Returns (n, SE3_OBS_DIM) array of imagined observations, n ∈ [0, dream_steps].
        """
        raw = seed_obs[:RAW_STATE_DIM].copy()
        coeff = seed_obs[RAW_STATE_DIM:].copy()

        # (pos_start, pos_end, vel_start, vel_end) for ball, ego, opp
        _moveable = [
            (_BALL_OFF,     _BALL_OFF + 3, _BALL_OFF + 3, _BALL_OFF + 6),
            (_EGO_OFF,      _EGO_OFF + 3,  _EGO_OFF + 3,  _EGO_OFF + 6),
            (_OPP_OFF,      _OPP_OFF + 3,  _OPP_OFF + 3,  _OPP_OFF + 6),
        ]

        dream_obs = []
        for _ in range(self.dream_steps):
            # Euler step: pos_next = pos + vel * dt
            raw_next = raw.copy()
            for ps, pe, vs, ve in _moveable:
                raw_next[ps:pe] = raw[ps:pe] + raw[vs:ve] * DT

            obs_next = np.concatenate([raw_next, coeff]).astype(np.float32)

            # LMS update on extrapolated observation
            obs_t = torch.tensor(obs_next[None], dtype=torch.float32, device=self.device)
            coeff_t = self.encoder(obs_t)   # (1, COEFF_DIM)

            # Spectral entropy check
            emb = coeff_t.reshape(1, N_OBJECTS, K, 3, 2)
            energy = emb.pow(2).sum(dim=-1)              # (1, N_OBJECTS, K, 3)
            energy_k = energy.sum(dim=-1)                # (1, N_OBJECTS, K)
            p = energy_k / (energy_k.sum(dim=-1, keepdim=True) + 1e-8)
            H_intra = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()

            if H_intra > self.dream_entropy_high or H_intra < self.dream_entropy_low:
                break

            coeff = coeff_t.cpu().numpy().ravel()
            dream_obs.append(obs_next)
            raw = raw_next

        if not dream_obs:
            return np.empty((0, SE3_OBS_DIM), dtype=np.float32)
        return np.stack(dream_obs, axis=0)

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

        # ── Next-state prediction diagnostic (no grad) ───────────────────
        # Measures how well the encoder's coefficient update tracks actual
        # state changes. Compares encoder output at t to raw state at t+1.
        buf_pos = self.buffer.pos
        prediction_mse = 0.0
        if buf_pos >= 2:
            with torch.no_grad():
                # Sample a subset to avoid OOM on large buffers
                _max_diag = min(512, (buf_pos - 1) * self.num_envs)
                _total = (buf_pos - 1) * self.num_envs
                _idx = np.random.choice(_total, _max_diag, replace=False)

                flat_obs_t = self.buffer.obs[:buf_pos - 1].reshape(_total, SE3_OBS_DIM)
                flat_obs_tp1 = self.buffer.obs[1:buf_pos].reshape(_total, SE3_OBS_DIM)
                flat_dones = self.buffer.dones[:buf_pos - 1].reshape(_total)

                obs_t = torch.tensor(flat_obs_t[_idx], dtype=torch.float32, device=self.device)
                obs_tp1 = torch.tensor(flat_obs_tp1[_idx], dtype=torch.float32, device=self.device)
                not_done = torch.tensor(1.0 - flat_dones[_idx], dtype=torch.float32, device=self.device)

                # Encoder output at t: these are the coefficients the policy sees
                coeff_t = self.encoder(obs_t)  # (N, 384)

                # Actual next-step coefficients (computed by the env's numpy mirror)
                coeff_tp1_actual = obs_tp1[:, RAW_STATE_DIM:]  # (N, 384)

                # Encoder prediction: if we fed obs_tp1's raw state + coeff_t as prev,
                # how close would it be? This measures tracking quality.
                # MSE between encoder's output and what the env actually computed next
                err = (coeff_t - coeff_tp1_actual).pow(2).mean(dim=-1)  # (N,)
                prediction_mse = (err * not_done).sum().item() / max(not_done.sum().item(), 1.0)

        # Generate dream observations from the last real observation
        dream_obs_np = self._dream_rollout(last_obs[0])
        self.encoder.train()   # _dream_rollout leaves encoder in eval; restore
        _has_dream = dream_obs_np.shape[0] > 0 and self.dream_ratio > 0.0
        dream_steps_total = dream_obs_np.shape[0]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        total_H_intra = 0.0
        total_H_inter = 0.0
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

                # Spectral entropy regularization
                # energy: (mb, N_OBJECTS, K, 3) — magnitude² per component per axis
                coeff = emb.reshape(emb.shape[0], N_OBJECTS, K, 3, 2)
                energy = coeff.pow(2).sum(dim=-1)              # (mb, N_OBJECTS, K, 3)

                # Intra-object: sum over axes → entropy over K components
                energy_k = energy.sum(dim=-1)                  # (mb, N_OBJECTS, K)
                p_intra = energy_k / (energy_k.sum(dim=-1, keepdim=True) + 1e-8)
                H_intra = -(p_intra * (p_intra + 1e-8).log()).sum(dim=-1).mean()

                # Inter-object: sum over K and axes → entropy over N_OBJECTS
                E_per_obj = energy.sum(dim=(2, 3))             # (mb, N_OBJECTS)
                p_inter = E_per_obj / (E_per_obj.sum(dim=-1, keepdim=True) + 1e-8)
                H_inter = -(p_inter * (p_inter + 1e-8).log()).sum(dim=-1).mean()

                # Dream policy loss: policy on imagined states, value as advantage
                dream_policy_loss = torch.tensor(0.0, device=self.device)
                if _has_dream:
                    n_dream = max(1, int(self.minibatch_size * self.dream_ratio))
                    n_avail = dream_obs_np.shape[0]
                    idx = np.random.choice(n_avail, min(n_dream, n_avail), replace=False)
                    dream_t = torch.tensor(
                        dream_obs_np[idx], dtype=torch.float32, device=self.device)
                    dream_emb = self.encoder(dream_t)
                    _, dream_log_prob, dream_value, _ = self.policy(dream_emb)
                    dream_adv = dream_value.detach()
                    if dream_adv.numel() > 1:
                        dream_adv = (dream_adv - dream_adv.mean()) / (
                            dream_adv.std() + 1e-8)
                    dream_policy_loss = -(dream_log_prob * dream_adv).mean()

                loss = (policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy.mean()
                        - self.spectral_ent_coef_intra * H_intra
                        - self.spectral_ent_coef_inter * H_inter
                        + self.dream_ratio * dream_policy_loss)

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
                total_H_intra += H_intra.item()
                total_H_inter += H_inter.item()
                n_updates += 1

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
            'spectral_H_intra': total_H_intra / n_updates,
            'spectral_H_inter': total_H_inter / n_updates,
            'dream_steps_mean': dream_steps_total,
            'next_state_pred_mse': prediction_mse,
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
