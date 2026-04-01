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
    _BALL, _EGO, _OPP, _BALL_OFF, _EGO_OFF, _OPP_OFF, _PAD_OFF,
    _PREV_VEL_OFF,
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


# ── Math correctness tests ────────────────────────────────────────────────────

class TestSpectralMath:
    """Verify the spectral field update math is correct."""

    def _make_raw_state(self, seed: int = 0) -> np.ndarray:
        """Build a plausible raw state with valid unit quaternions."""
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-0.5, 0.5, RAW_STATE_DIM).astype(np.float32)
        # Ego quaternion (offset 12-15): unit norm
        eq = rng.standard_normal(4).astype(np.float32)
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = eq / np.linalg.norm(eq)
        # Opp quaternion: unit norm
        oq = rng.standard_normal(4).astype(np.float32)
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = oq / np.linalg.norm(oq)
        # Pad active flags: clamp to [0,1]
        raw[_PAD_OFF + 3::4] = np.clip(raw[_PAD_OFF + 3::4], 0.0, 1.0)
        # Zero prev_ball_vel so contact is not triggered
        raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 3] = 0.0
        raw[_BALL_OFF + 3:_BALL_OFF + 6] = 0.0
        return raw

    def test_torch_numpy_parity(self):
        """SE3Encoder.forward must exactly mirror update_coefficients_np."""
        torch.manual_seed(7)
        encoder = SE3Encoder()
        encoder.eval()

        raw = self._make_raw_state(seed=1)
        coeff0 = make_initial_coefficients()

        # Extract encoder params as numpy (match normalisation done in forward)
        with torch.no_grad():
            q = encoder.quaternions
            q_norm = (q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)).numpy()
            k_np = encoder.k_spatial.numpy()
            lr_np = np.exp(encoder.log_lr.numpy())

        # Numpy update
        coeff_np = update_coefficients_np(k_np, q_norm, lr_np, coeff0, raw)

        # PyTorch update
        obs = torch.tensor(pack_observation(raw, coeff0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            coeff_pt = encoder(obs)[0].numpy()

        np.testing.assert_allclose(
            coeff_pt, coeff_np, atol=1e-5,
            err_msg="PyTorch encoder and numpy mirror must produce identical results")

    def test_coefficient_convergence_real(self):
        """Real coefficients converge: repeated updates on a fixed state drive residual → 0."""
        rng = np.random.default_rng(42)
        k = rng.standard_normal((N_OBJECTS, K, 3)).astype(np.float32) * 0.3
        q = rng.standard_normal((N_OBJECTS, K, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)

        raw = self._make_raw_state(seed=2)
        coeff = make_initial_coefficients()

        def residual_real(coeff, obj):
            """Mean absolute real-part residual over x/y/z for one object."""
            c = coeff.reshape(N_OBJECTS, K, 3, 2)
            pos = np.zeros(3, dtype=np.float32)
            if obj == 0:    pos = raw[_BALL_OFF:_BALL_OFF + 3]
            elif obj == 1:  pos = raw[_EGO_OFF:_EGO_OFF + 3]
            elif obj == 2:  pos = raw[_EGO_OFF:_EGO_OFF + 3]
            elif obj == 3:  pos = raw[_OPP_OFF:_OPP_OFF + 3]
            elif obj == 5:  pos = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            elif obj == 6:  pos = np.array([0.0,  1.0, 0.0], dtype=np.float32)
            phase = k[obj] @ pos
            s_cos = np.cos(phase)
            if obj == 1:
                ori = raw[_EGO_OFF + 6:_EGO_OFF + 10]
            elif obj == 3:
                ori = raw[_OPP_OFF + 6:_OPP_OFF + 10]
            else:
                ori = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            orient = (q[obj] * ori).sum(axis=-1)          # (K,)
            pred = (s_cos * orient) @ c[obj, :, :, 0]    # (K,)@(K,3) → (3,)
            return np.abs(pos - pred).mean()

        # Measure initial residual for ball (obj 0)
        res_before = residual_real(coeff, obj=0)

        for _ in range(200):
            coeff = update_coefficients_np(k, q, lr, coeff, raw)

        res_after = residual_real(coeff, obj=0)
        assert res_after < res_before, (
            f"Real residual should decrease: {res_before:.4f} → {res_after:.4f}")
        assert res_after < 0.05, f"Real residual should be small after 200 steps: {res_after:.4f}"

    def test_coefficient_convergence_imaginary(self):
        """Imaginary coefficients converge independently of real part."""
        rng = np.random.default_rng(99)
        k = rng.standard_normal((N_OBJECTS, K, 3)).astype(np.float32) * 0.3
        q = rng.standard_normal((N_OBJECTS, K, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)

        raw = self._make_raw_state(seed=3)
        coeff = make_initial_coefficients()

        def residual_imag(coeff):
            """Mean absolute imaginary-part residual over x/y/z for ball."""
            c = coeff.reshape(N_OBJECTS, K, 3, 2)
            pos = raw[_BALL_OFF:_BALL_OFF + 3]
            phase = k[_BALL] @ pos
            s_sin = np.sin(phase)
            ori = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            orient = (q[_BALL] * ori).sum(axis=-1)        # (K,)
            pred = (s_sin * orient) @ c[_BALL, :, :, 1]  # (K,)@(K,3) → (3,)
            return np.abs(pos - pred).mean()

        res_before = residual_imag(coeff)

        for _ in range(200):
            coeff = update_coefficients_np(k, q, lr, coeff, raw)

        res_after = residual_imag(coeff)
        assert res_after < res_before or res_before < 1e-6, (
            f"Imaginary residual should decrease: {res_before:.4f} → {res_after:.4f}")

    def test_contact_reset_only_affects_ball(self):
        """Contact resets ball coefficients but leaves all other objects untouched."""
        encoder = SE3Encoder()
        encoder.eval()

        # Give all coefficients a non-zero prev value
        obs = torch.zeros(1, SE3_OBS_DIM)
        obs[0, RAW_STATE_DIM:] = 1.0  # all prev coefficients = 1

        # Trigger contact: large velocity change on ball
        obs[0, _BALL_OFF + 3] = 10.0   # ball vx (current)
        # prev_ball_vel at _PREV_VEL_OFF stays 0 → huge Δv/dt → contact

        with torch.no_grad():
            out = encoder(obs)

        coeff = out[0].reshape(N_OBJECTS, K, 3, 2)

        # Ball must be zeroed
        assert coeff[_BALL].abs().sum().item() < 1e-5, \
            f"Ball should be reset on contact, got {coeff[_BALL].abs().sum():.6f}"

        # All other objects must have non-zero coefficients
        for obj in range(1, N_OBJECTS):
            assert coeff[obj].abs().sum().item() > 0, \
                f"Object {obj} should NOT be reset on contact"

    def test_zero_state_zero_coefficients_stable(self):
        """All-zero state with zero coefficients should not produce NaN or Inf."""
        encoder = SE3Encoder()
        encoder.eval()
        obs = torch.zeros(1, SE3_OBS_DIM)
        with torch.no_grad():
            out = encoder(obs)
        assert torch.isfinite(out).all(), "Forward pass on zero state must be finite"

    def test_parity_with_nonzero_prev_coefficients(self):
        """Parity holds when starting from non-zero coefficients (not just episode start)."""
        torch.manual_seed(13)
        encoder = SE3Encoder()
        encoder.eval()

        rng = np.random.default_rng(5)
        raw = self._make_raw_state(seed=5)
        coeff0 = rng.uniform(-0.5, 0.5, COEFF_DIM).astype(np.float32)

        with torch.no_grad():
            q = encoder.quaternions
            q_norm = (q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)).numpy()
            k_np = encoder.k_spatial.numpy()
            lr_np = np.exp(encoder.log_lr.numpy())

        coeff_np = update_coefficients_np(k_np, q_norm, lr_np, coeff0, raw)

        obs = torch.tensor(pack_observation(raw, coeff0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            coeff_pt = encoder(obs)[0].numpy()

        np.testing.assert_allclose(
            coeff_pt, coeff_np, atol=1e-5,
            err_msg="Parity must hold for non-zero initial coefficients")


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

    def test_spectral_entropy_metrics_logged(self):
        """update() must return spectral_H_intra and spectral_H_inter as finite floats."""
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = SE3PPOAlgorithm(self._make_config())
        for _ in range(64):
            obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            result = algo.select_action(obs)
            algo.store_transition(obs, result, np.zeros(2), obs, np.zeros(2), {})
        metrics = algo.update()
        assert 'spectral_H_intra' in metrics, "spectral_H_intra missing from metrics"
        assert 'spectral_H_inter' in metrics, "spectral_H_inter missing from metrics"
        assert math.isfinite(metrics['spectral_H_intra']), "spectral_H_intra is not finite"
        assert math.isfinite(metrics['spectral_H_inter']), "spectral_H_inter is not finite"
        assert metrics['spectral_H_intra'] >= 0.0, "Entropy must be non-negative"
        assert metrics['spectral_H_inter'] >= 0.0, "Entropy must be non-negative"

    def test_spectral_entropy_gradients(self):
        """Spectral entropy term must actually flow gradient to k_spatial."""
        from training.algorithms.se3_ppo import SE3PPOAlgorithm

        def run_update(coef_intra):
            cfg = {**self._make_config()}
            cfg['algorithm']['params']['spectral_ent_coef_intra'] = coef_intra
            cfg['algorithm']['params']['spectral_ent_coef_inter'] = 0.0
            torch.manual_seed(0)
            np.random.seed(0)
            algo = SE3PPOAlgorithm(cfg)
            for _ in range(64):
                obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
                result = algo.select_action(obs)
                rewards = np.ones(2, dtype=np.float32)
                algo.store_transition(obs, result, rewards, obs, np.zeros(2), {})
            algo.update()
            return algo.encoder.k_spatial.data.clone()

        k_no_reg = run_update(coef_intra=0.0)
        k_with_reg = run_update(coef_intra=0.1)
        assert not torch.equal(k_no_reg, k_with_reg), \
            "k_spatial should differ when spectral_ent_coef_intra is non-zero"

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


# ── Dream rollout tests ──────────────────────────────────────────────────────

class TestDreamRollout:

    def _make_algo(self, **dream_kwargs):
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        cfg = {
            'algorithm': {
                'params': {
                    'lr': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95,
                    'clip_epsilon': 0.2, 'vf_coef': 0.5, 'ent_coef': 0.01,
                    'max_grad_norm': 0.5, 'rollout_steps': 64,
                    'ppo_epochs': 2, 'minibatch_size': 32,
                    **dream_kwargs,
                }
            },
            'num_envs': 2,
        }
        return SE3PPOAlgorithm(cfg)

    def test_dream_obs_shape(self):
        """Each dream step produces an obs with shape (SE3_OBS_DIM,)."""
        algo = self._make_algo(dream_steps=3, dream_entropy_high=100.0, dream_entropy_low=-1.0)
        seed = np.random.randn(SE3_OBS_DIM).astype(np.float32)
        dream = algo._dream_rollout(seed)
        assert dream.ndim == 2
        assert dream.shape[1] == SE3_OBS_DIM
        assert 0 <= dream.shape[0] <= 3

    def test_dream_terminates_on_entropy_explosion(self):
        """dream_entropy_high below any achievable entropy → 0 dream steps."""
        algo = self._make_algo(dream_steps=10, dream_entropy_high=-1.0, dream_entropy_low=-2.0)
        seed = np.random.randn(SE3_OBS_DIM).astype(np.float32)
        dream = algo._dream_rollout(seed)
        assert dream.shape[0] == 0, "Should terminate immediately: entropy_high < 0"

    def test_dream_terminates_on_entropy_collapse(self):
        """dream_entropy_low above any achievable entropy → 0 dream steps."""
        algo = self._make_algo(dream_steps=10, dream_entropy_high=100.0, dream_entropy_low=99.0)
        seed = np.random.randn(SE3_OBS_DIM).astype(np.float32)
        dream = algo._dream_rollout(seed)
        assert dream.shape[0] == 0, "Should terminate immediately: entropy_low=99.0"

    def test_dream_positions_extrapolated(self):
        """Ball position in dream obs changes by vel * dt per step."""
        from se3_field import DT
        algo = self._make_algo(dream_steps=1, dream_entropy_high=100.0, dream_entropy_low=-1.0)
        seed = np.zeros(SE3_OBS_DIM, dtype=np.float32)
        seed[_BALL_OFF + 3] = 1.0   # ball vx
        seed[_BALL_OFF + 4] = 2.0   # ball vy
        seed[_BALL_OFF + 5] = 3.0   # ball vz
        dream = algo._dream_rollout(seed)
        if dream.shape[0] > 0:
            expected = np.array([1.0 * DT, 2.0 * DT, 3.0 * DT], dtype=np.float32)
            np.testing.assert_allclose(
                dream[0, _BALL_OFF:_BALL_OFF + 3], expected, atol=1e-5)

    def test_dream_zero_vel_stable_positions(self):
        """Zero velocity → ball position unchanged across all dream steps."""
        algo = self._make_algo(dream_steps=5, dream_entropy_high=100.0, dream_entropy_low=-1.0)
        seed = np.zeros(SE3_OBS_DIM, dtype=np.float32)
        dream = algo._dream_rollout(seed)
        if dream.shape[0] > 1:
            np.testing.assert_allclose(
                dream[0, _BALL_OFF:_BALL_OFF + 3],
                dream[-1, _BALL_OFF:_BALL_OFF + 3],
                atol=1e-5,
                err_msg="Ball position should not drift with zero velocity")

    def test_dream_steps_logged_in_update(self):
        """update() returns dream_steps_mean in metrics."""
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = self._make_algo(dream_steps=3, dream_entropy_high=100.0, dream_entropy_low=-1.0)
        for _ in range(64):
            obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            result = algo.select_action(obs)
            algo.store_transition(obs, result, np.zeros(2), obs, np.zeros(2), {})
        metrics = algo.update()
        assert 'dream_steps_mean' in metrics
        assert isinstance(metrics['dream_steps_mean'], (int, float))


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


# ── Coefficient clipping tests ──────────────────────────────────────────────

class TestCoefficientClipping:

    def test_torch_clipping(self):
        """SE3Encoder output stays within [-COEFF_CLIP, COEFF_CLIP]."""
        from se3_field import COEFF_CLIP
        encoder = SE3Encoder()
        # Feed large prev-coefficients to trigger clipping
        obs = torch.randn(2, SE3_OBS_DIM)
        obs[:, RAW_STATE_DIM:] = 50.0  # way above clip bound
        with torch.no_grad():
            out = encoder(obs)
        assert out.max().item() <= COEFF_CLIP + 1e-6
        assert out.min().item() >= -COEFF_CLIP - 1e-6

    def test_numpy_clipping(self):
        """update_coefficients_np output stays within [-COEFF_CLIP, COEFF_CLIP]."""
        from se3_field import COEFF_CLIP
        k_sp = np.random.randn(N_OBJECTS, K, 3).astype(np.float32)
        quats = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)
        # Large prev coefficients
        prev_coeff = np.full(COEFF_DIM, 50.0, dtype=np.float32)
        raw = np.random.randn(RAW_STATE_DIM).astype(np.float32) * 0.5
        out = update_coefficients_np(k_sp, quats, lr, prev_coeff, raw)
        assert out.max() <= COEFF_CLIP + 1e-6
        assert out.min() >= -COEFF_CLIP - 1e-6

    def test_clipping_preserves_gradients(self):
        """Gradients flow through clamp for in-range values."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM, requires_grad=False)
        out = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.k_spatial.grad is not None
        assert encoder.k_spatial.grad.abs().sum().item() > 0


# ── LayerNorm tests ─────────────────────────────────────────────────────────

class TestLayerNorm:

    def test_se3_policy_has_layernorm(self):
        """SE3Policy should have LayerNorm as first layer."""
        policy = SE3Policy(obs_dim=COEFF_DIM)
        first_layer = policy.net[0]
        assert isinstance(first_layer, torch.nn.LayerNorm), \
            f"Expected LayerNorm, got {type(first_layer)}"

    def test_stochastic_policy_has_layernorm(self):
        """StochasticSE3Policy should have LayerNorm."""
        policy = StochasticSE3Policy(obs_dim=COEFF_DIM)
        assert hasattr(policy, 'layer_norm')
        assert isinstance(policy.layer_norm, torch.nn.LayerNorm)
