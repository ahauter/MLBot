"""
SE(3) Spectral Field Tests
==========================
Tests for quaternion math, SE3Encoder, SE3 policies, and SE3PPOAlgorithm.

All tests run without rlgym-sim (no live simulation needed).

Run with:
    python -m pytest training/tests/test_se3.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from se3_field import (
    euler_to_quaternion, euler_to_quaternion_batch,
    normalise_quaternion, quaternion_inner, quaternion_exponential,
    detect_contact, detect_contact_np,
    SE3Encoder, SE3_OBS_DIM, RAW_STATE_DIM, COEFF_DIM, N_OBJECTS, K,
    make_initial_coefficients, update_coefficients_np, pack_observation,
)
from se3_policy import SE3Policy, StochasticSE3Policy


# ── Quaternion tests ─────────────────────────────────────────────────────────

class TestQuaternions:

    def test_euler_to_quaternion_identity(self):
        """(0,0,0) → (1,0,0,0)."""
        q = euler_to_quaternion(0.0, 0.0, 0.0)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(q, expected, atol=1e-6, rtol=1e-6)

    def test_euler_to_quaternion_unit_norm(self):
        """Output should be unit norm for any input."""
        for _ in range(20):
            yaw = np.random.uniform(-math.pi, math.pi)
            pitch = np.random.uniform(-math.pi / 2, math.pi / 2)
            roll = np.random.uniform(-math.pi, math.pi)
            q = euler_to_quaternion(yaw, pitch, roll)
            assert abs(q.norm().item() - 1.0) < 1e-5, f"Non-unit: {q.norm()}"

    def test_euler_to_quaternion_canonical_hemisphere(self):
        """w >= 0 always (canonical hemisphere)."""
        for _ in range(20):
            yaw = np.random.uniform(-math.pi, math.pi)
            pitch = np.random.uniform(-math.pi / 2, math.pi / 2)
            roll = np.random.uniform(-math.pi, math.pi)
            q = euler_to_quaternion(yaw, pitch, roll)
            assert q[0].item() >= 0.0, f"w < 0: {q}"

    def test_euler_to_quaternion_batch(self):
        """Batch version matches scalar version."""
        yaws = np.random.uniform(-math.pi, math.pi, 5)
        pitches = np.random.uniform(-math.pi / 2, math.pi / 2, 5)
        rolls = np.random.uniform(-math.pi, math.pi, 5)
        batch = euler_to_quaternion_batch(yaws, pitches, rolls)
        assert batch.shape == (5, 4)
        for i in range(5):
            q_scalar = euler_to_quaternion(yaws[i], pitches[i], rolls[i]).numpy()
            np.testing.assert_allclose(batch[i], q_scalar, atol=1e-5)

    def test_quaternion_inner_self(self):
        """q . q = 1 for unit quaternions."""
        q = normalise_quaternion(torch.randn(K, 4))
        for i in range(K):
            result = quaternion_inner(q[i:i+1], q[i])
            assert abs(result.item() - 1.0) < 1e-5

    def test_quaternion_inner_negative(self):
        """q . (-q) = -1 for unit quaternions."""
        q = normalise_quaternion(torch.randn(4))
        result = quaternion_inner(q.unsqueeze(0), -q)
        assert abs(result.item() + 1.0) < 1e-5

    def test_normalise_quaternion(self):
        """Output has unit norm."""
        q = torch.randn(10, 4) * 5
        normed = normalise_quaternion(q)
        norms = normed.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(10), atol=1e-5, rtol=1e-5)

    def test_quaternion_exponential(self):
        """e^(0 * q_hat) = identity quaternion."""
        q_hat = torch.tensor([0.0, 0.0, 0.0, 1.0])  # pure z
        result = quaternion_exponential(0.0, q_hat)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


# ── Contact detection tests ─────────────────────────────────────────────────

class TestContactDetection:

    def test_contact_detected(self):
        """Large velocity change → contact."""
        v0 = torch.tensor([0.0, 0.0, 0.0])
        v1 = torch.tensor([100.0, 0.0, 0.0])
        assert detect_contact(v0, v1, dt=1.0/120, threshold=50.0).item()

    def test_no_contact(self):
        """Small velocity change → no contact."""
        v0 = torch.tensor([0.0, 0.0, 0.0])
        v1 = torch.tensor([0.1, 0.0, 0.0])
        assert not detect_contact(v0, v1, dt=1.0/120, threshold=50.0).item()

    def test_contact_np(self):
        """Numpy version matches torch."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([100.0, 0.0, 0.0])
        assert detect_contact_np(v0, v1, dt=1.0/120, threshold=50.0)


# ── SE3Encoder tests ─────────────────────────────────────────────────────────

class TestSE3Encoder:

    def test_forward_shape(self):
        """(batch=4, 185) → (4, 128)."""
        encoder = SE3Encoder()
        obs = torch.randn(4, SE3_OBS_DIM)
        out = encoder(obs)
        assert out.shape == (4, COEFF_DIM), f"Expected (4, {COEFF_DIM}), got {out.shape}"

    def test_forward_single(self):
        """(1, 185) → (1, 128)."""
        encoder = SE3Encoder()
        obs = torch.randn(1, SE3_OBS_DIM)
        out = encoder(obs)
        assert out.shape == (1, COEFF_DIM)

    def test_gradient_flows_to_k_spatial(self):
        """Backprop from output reaches k_spatial."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        out = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.k_spatial.grad is not None
        assert encoder.k_spatial.grad.abs().sum() > 0

    def test_gradient_flows_to_quaternions(self):
        """Backprop from output reaches quaternion params."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        out = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.quaternions.grad is not None
        assert encoder.quaternions.grad.abs().sum() > 0

    def test_gradient_flows_to_log_lr(self):
        """Backprop from output reaches log_lr."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        out = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.log_lr.grad is not None
        assert encoder.log_lr.grad.abs().sum() > 0

    def test_contact_resets_ball_coefficients(self):
        """Ball coefficients zero when contact is detected."""
        encoder = SE3Encoder()
        # Create obs where ball velocity changes drastically
        obs = torch.zeros(1, SE3_OBS_DIM)
        # Set prev_ball_vel to zero (offset 54-56)
        # Set current ball vel to something huge (offset 3-5)
        obs[0, 3] = 10.0  # ball vx huge (normalised)
        obs[0, 4] = 10.0  # ball vy huge
        # prev_ball_vel at offset 54 stays zero
        # Give some non-zero prev coefficients for ball (first K*2=16 of coeff section)
        obs[0, RAW_STATE_DIM:RAW_STATE_DIM + K * 2] = 1.0

        with torch.no_grad():
            out = encoder(obs)

        # Ball coefficients (first K*2=16) should be zero after contact reset
        ball_coeff = out[0, :K * 2]
        assert ball_coeff.abs().sum().item() < 1e-5, \
            f"Ball coeff should be ~0 after contact, got {ball_coeff}"

    def test_normalise_quaternions(self):
        """normalise_quaternions_ projects to unit sphere."""
        encoder = SE3Encoder()
        # Perturb quaternions away from unit
        with torch.no_grad():
            encoder.quaternions.data *= 3.0
        encoder.normalise_quaternions_()
        norms = encoder.quaternions.data.norm(dim=-1)
        torch.testing.assert_close(
            norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_d_model_property(self):
        encoder = SE3Encoder()
        assert encoder.d_model == COEFF_DIM

    def test_save_load_roundtrip(self, tmp_path):
        encoder = SE3Encoder()
        path = str(tmp_path / 'enc.pt')
        encoder.save(path)
        encoder2 = SE3Encoder.load_from(path)
        for (n1, p1), (n2, p2) in zip(
            encoder.named_parameters(), encoder2.named_parameters()
        ):
            torch.testing.assert_close(p1, p2, msg=f"Param {n1} mismatch")


# ── Coefficient numpy helpers tests ──────────────────────────────────────────

class TestCoefficientHelpers:

    def test_make_initial_coefficients(self):
        coeff = make_initial_coefficients()
        assert coeff.shape == (COEFF_DIM,)
        assert coeff.sum() == 0.0

    def test_update_coefficients_modifies(self):
        """Coefficients should change after update with non-zero state."""
        k = np.random.randn(N_OBJECTS, K, 3).astype(np.float32) * 0.1
        q = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)
        coeff = make_initial_coefficients()
        raw = np.random.randn(RAW_STATE_DIM).astype(np.float32) * 0.1
        # Set valid quaternions in raw state
        raw[6 + 6:6 + 10] = [1.0, 0.0, 0.0, 0.0]  # ego quat
        raw[17 + 6:17 + 10] = [1.0, 0.0, 0.0, 0.0]  # opp quat

        new_coeff = update_coefficients_np(k, q, lr, coeff, raw)
        assert new_coeff.shape == (COEFF_DIM,)
        assert not np.allclose(new_coeff, coeff), "Coefficients should change"

    def test_pack_observation(self):
        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        coeff = make_initial_coefficients()
        packed = pack_observation(raw, coeff)
        assert packed.shape == (SE3_OBS_DIM,)
        np.testing.assert_array_equal(packed[:RAW_STATE_DIM], raw)
        np.testing.assert_array_equal(packed[RAW_STATE_DIM:], coeff)


# ── SE3Policy tests ──────────────────────────────────────────────────────────

class TestSE3Policy:

    def test_forward_shapes(self):
        policy = SE3Policy()
        obs = torch.randn(4, COEFF_DIM)
        action, value = policy(obs)
        assert action.shape == (4, 8)
        assert value.shape == (4, 1)

    def test_analog_range(self):
        policy = SE3Policy()
        obs = torch.randn(100, COEFF_DIM)
        action, _ = policy(obs)
        assert action[:, :5].min() >= -1.0
        assert action[:, :5].max() <= 1.0

    def test_binary_range(self):
        policy = SE3Policy()
        obs = torch.randn(100, COEFF_DIM)
        action, _ = policy(obs)
        assert action[:, 5:].min() >= 0.0
        assert action[:, 5:].max() <= 1.0

    def test_act_numpy(self):
        policy = SE3Policy()
        obs = np.random.randn(1, COEFF_DIM).astype(np.float32)
        action, value = policy.act(obs)
        assert action.shape == (8,)
        assert isinstance(value, float)


# ── StochasticSE3Policy tests ────────────────────────────────────────────────

class TestStochasticSE3Policy:

    def test_forward_shapes(self):
        head = StochasticSE3Policy()
        obs = torch.randn(4, COEFF_DIM)
        action, log_prob, value, entropy = head(obs)
        assert action.shape == (4, 8)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert entropy.shape == (4,)

    def test_log_probs_finite(self):
        head = StochasticSE3Policy()
        obs = torch.randn(8, COEFF_DIM)
        _, log_prob, _, _ = head(obs)
        assert torch.isfinite(log_prob).all()

    def test_entropy_positive(self):
        head = StochasticSE3Policy()
        obs = torch.randn(4, COEFF_DIM)
        _, _, _, entropy = head(obs)
        assert (entropy > 0).all()

    def test_evaluate_actions_matches_forward(self):
        head = StochasticSE3Policy()
        torch.manual_seed(42)
        obs = torch.randn(4, COEFF_DIM)
        action, lp_fwd, val_fwd, ent_fwd = head(obs)
        lp_eval, val_eval, ent_eval = head.evaluate_actions(obs, action)
        torch.testing.assert_close(lp_eval, lp_fwd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(val_eval, val_fwd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ent_eval, ent_fwd, atol=1e-5, rtol=1e-5)

    def test_act_deterministic(self):
        head = StochasticSE3Policy()
        head.eval()
        obs = torch.randn(2, COEFF_DIM)
        a1, v1 = head.act_deterministic(obs)
        a2, v2 = head.act_deterministic(obs)
        torch.testing.assert_close(a1, a2)
        torch.testing.assert_close(v1, v2)

    def test_gradient_flows_through_evaluate(self):
        head = StochasticSE3Policy()
        obs = torch.randn(4, COEFF_DIM, requires_grad=True)
        analog = torch.randn(4, 5).clamp(-1, 1)
        binary = torch.bernoulli(torch.ones(4, 3) * 0.5)
        actions = torch.cat([analog, binary], dim=-1)
        lp, val, ent = head.evaluate_actions(obs, actions)
        loss = lp.sum() + val.sum()
        loss.backward()
        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0

    def test_binary_actions_are_binary(self):
        head = StochasticSE3Policy()
        obs = torch.randn(100, COEFF_DIM)
        action, _, _, _ = head(obs)
        binary = action[:, 5:]
        unique = torch.unique(binary)
        assert all(v in [0.0, 1.0] for v in unique.tolist())


# ── SE3PPOAlgorithm tests ────────────────────────────────────────────────────

class TestSE3PPOAlgorithm:

    def _make_config(self):
        return {
            'algorithm': {
                'params': {
                    'lr': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95,
                    'clip_epsilon': 0.2, 'vf_coef': 0.5, 'ent_coef': 0.01,
                    'max_grad_norm': 0.5, 'rollout_steps': 64,
                    'ppo_epochs': 2, 'minibatch_size': 32,
                }
            },
            'num_envs': 2,
        }

    def test_select_action_shapes(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        from training.abstractions import ActionResult
        algo = SE3PPOAlgorithm(self._make_config())
        obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
        result = algo.select_action(obs)
        assert isinstance(result, ActionResult)
        assert result.action.shape == (2, 8)
        assert result.aux['log_prob'].shape == (2,)
        assert result.aux['value'].shape == (2,)

    def test_store_transition_fills_buffer(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = SE3PPOAlgorithm(self._make_config())
        obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
        result = algo.select_action(obs)
        next_obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
        assert algo.buffer.pos == 0
        algo.store_transition(obs, result, np.zeros(2), next_obs, np.zeros(2), {})
        assert algo.buffer.pos == 1

    def test_should_update_triggers(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = SE3PPOAlgorithm(self._make_config())
        assert not algo.should_update()
        for _ in range(64):
            obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            result = algo.select_action(obs)
            algo.store_transition(obs, result, np.zeros(2), obs, np.zeros(2), {})
        assert algo.should_update()

    def test_update_returns_metrics(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = SE3PPOAlgorithm(self._make_config())
        for _ in range(64):
            obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            result = algo.select_action(obs)
            algo.store_transition(obs, result, np.zeros(2), obs, np.zeros(2), {})
        metrics = algo.update()
        assert isinstance(metrics, dict)
        expected = {'policy_loss', 'value_loss', 'entropy', 'clip_fraction', 'approx_kl'}
        assert expected.issubset(metrics.keys())

    def test_update_changes_weights(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = SE3PPOAlgorithm(self._make_config())
        w_before = {n: p.data.clone() for n, p in algo.encoder.named_parameters()}
        for _ in range(64):
            obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            result = algo.select_action(obs)
            rewards = np.random.randn(2).astype(np.float32)
            algo.store_transition(obs, result, rewards, obs, np.zeros(2), {})
        algo.update()
        any_changed = any(
            not torch.equal(p.data, w_before[n])
            for n, p in algo.encoder.named_parameters()
        )
        assert any_changed, "Encoder weights should change after update"

    def test_save_load_checkpoint(self, tmp_path):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        config = self._make_config()
        a = SE3PPOAlgorithm(config)
        for _ in range(64):
            obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            result = a.select_action(obs)
            a.store_transition(obs, result, np.zeros(2), obs, np.zeros(2), {})
        a.update()
        a.save_checkpoint(tmp_path / 'ckpt')

        b = SE3PPOAlgorithm(config)
        b.load_checkpoint(tmp_path / 'ckpt')

        for (na, pa), (nb, pb) in zip(
            a.encoder.named_parameters(), b.encoder.named_parameters()
        ):
            torch.testing.assert_close(pa, pb, msg=f"Encoder {na} mismatch")
        for (na, pa), (nb, pb) in zip(
            a.policy.named_parameters(), b.policy.named_parameters()
        ):
            torch.testing.assert_close(pa, pb, msg=f"Policy {na} mismatch")

    def test_get_weights(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = SE3PPOAlgorithm(self._make_config())
        weights = algo.get_weights()
        assert 'encoder' in weights
        assert 'policy' in weights
        assert len(weights['encoder']) > 0
        assert len(weights['policy']) > 0

    def test_clone_from(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        config = self._make_config()
        a = SE3PPOAlgorithm(config)
        b = SE3PPOAlgorithm(config)
        b.clone_from(a, noise_scale=0.0)
        for pa, pb in zip(a.encoder.parameters(), b.encoder.parameters()):
            torch.testing.assert_close(pa, pb)

    def test_clone_from_with_noise(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        config = self._make_config()
        a = SE3PPOAlgorithm(config)
        b = SE3PPOAlgorithm(config)
        b.clone_from(a, noise_scale=0.1)
        any_diff = any(
            not torch.equal(pa.data, pb.data)
            for pa, pb in zip(a.encoder.parameters(), b.encoder.parameters())
        )
        assert any_diff

    def test_algorithm_abc_compliance(self):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        from training.abstractions import Algorithm
        abstract_methods = {
            name for name in dir(Algorithm)
            if getattr(getattr(Algorithm, name), '__isabstractmethod__', False)
        }
        for method in abstract_methods:
            assert hasattr(SE3PPOAlgorithm, method)
            assert not getattr(getattr(SE3PPOAlgorithm, method), '__isabstractmethod__', False)


# ── SE3Population tests ──────────────────────────────────────────────────────

class TestSE3Population:

    def test_creates_se3_agents(self, tmp_path):
        from training.opponents.se3_population import SE3Population
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
        }
        pop = SE3Population(
            num_agents=2, num_workers=4, config=config,
            snapshot_dir=tmp_path / 'snaps')
        assert len(pop.agents) == 2
        for agent in pop.agents:
            assert isinstance(agent, SE3PPOAlgorithm)

    def test_worker_assignment(self):
        from training.opponents.se3_population import SE3Population
        assignment = SE3Population._assign_workers(8, 3)
        assert len(assignment) == 8
        assert assignment.count(0) == 3
        assert assignment.count(1) == 3
        assert assignment.count(2) == 2

    def test_ranking(self, tmp_path):
        from training.opponents.se3_population import SE3Population
        config = {
            'algorithm': {'params': {'rollout_steps': 8}},
            'num_envs': 1,
        }
        pop = SE3Population(
            num_agents=3, num_workers=3, config=config,
            snapshot_dir=tmp_path / 'snaps')
        pop.add_score(0, 1.0)
        pop.add_score(1, 5.0)
        pop.add_score(2, 3.0)
        ranking = pop.rank_agents()
        assert ranking == [1, 2, 0]
