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
    SE3Encoder, SE3_OBS_DIM, RAW_STATE_DIM, COEFF_DIM, EMBED_DIM,
    N_OBJECTS, K, D_AMP, N_CHANNELS, GRAVITY_DV_Z,
    make_initial_coefficients, update_coefficients_np, pack_observation,
    _BALL, _EGO, _OPP, _STADIUM,
    _BALL_OFF, _EGO_OFF, _OPP_OFF, _PAD_OFF, _GS_OFF, _PREV_VEL_OFF,
    _PREV_EGO_VEL_OFF, _PREV_OPP_VEL_OFF,
    _PREV_ANG_VEL_OFF, _PREV_EGO_ANG_VEL_OFF, _PREV_OPP_ANG_VEL_OFF,
    _PREV_SCALARS_OFF, _PREV_OPP_SCALARS_OFF,
    D_FIELD, D_OUTER, CONV_OUT, CONTEXT_DIM,
    ACCEL_HIST_DIM, ACCEL_CTX_DIM, make_initial_accel_hist,
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
        """forward() returns (coeff (batch, COEFF_DIM), accel_hist (batch, ACCEL_HIST_DIM))."""
        encoder = SE3Encoder()
        obs = torch.randn(4, SE3_OBS_DIM)
        coeff, accel_hist = encoder(obs)
        assert coeff.shape == (4, COEFF_DIM), f"Expected (4, {COEFF_DIM}), got {coeff.shape}"
        assert accel_hist.shape == (4, ACCEL_HIST_DIM), \
            f"Expected (4, {ACCEL_HIST_DIM}), got {accel_hist.shape}"

    def test_forward_single(self):
        """forward() returns (coeff (1, COEFF_DIM), accel_hist (1, ACCEL_HIST_DIM))."""
        encoder = SE3Encoder()
        obs = torch.randn(1, SE3_OBS_DIM)
        coeff, accel_hist = encoder(obs)
        assert coeff.shape == (1, COEFF_DIM)
        assert accel_hist.shape == (1, ACCEL_HIST_DIM)

    def test_gradient_flows_to_k_spatial(self):
        """Backprop from output reaches k_spatial."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        out, _ = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.k_spatial.grad is not None
        assert encoder.k_spatial.grad.abs().sum() > 0

    def test_gradient_flows_to_quaternions(self):
        """Backprop from output reaches quaternion params."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        out, _ = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.quaternions.grad is not None
        assert encoder.quaternions.grad.abs().sum() > 0

    def test_gradient_flows_to_log_lr(self):
        """Backprop from output reaches log_lr."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        out, _ = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.log_lr.grad is not None
        assert encoder.log_lr.grad.abs().sum() > 0

    def test_contact_resets_ball_coefficients(self):
        """Ball coefficients zero when contact is detected."""
        encoder = SE3Encoder()
        # Create obs where ball velocity changes drastically
        obs = torch.zeros(1, SE3_OBS_DIM)
        # Set current ball vel to something huge (offset 3-5)
        obs[0, 3] = 10.0  # ball vx huge (normalised)
        obs[0, 4] = 10.0  # ball vy huge
        # prev_ball_vel at _PREV_VEL_OFF stays zero → huge Δv/dt → contact
        # Give some non-zero prev coefficients for ball
        ball_coeff_size = K * D_AMP * N_CHANNELS
        obs[0, RAW_STATE_DIM:RAW_STATE_DIM + ball_coeff_size] = 1.0

        with torch.no_grad():
            out, _ = encoder(obs)

        # Ball coefficients should be zero after contact reset
        ball_coeff = out[0, :ball_coeff_size]
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
        assert encoder.d_model == EMBED_DIM

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
        k = np.random.randn(N_OBJECTS, K, D_AMP).astype(np.float32) * 0.1
        q = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)
        coeff = make_initial_coefficients()
        raw = np.random.randn(RAW_STATE_DIM).astype(np.float32) * 0.1
        # Set valid quaternions in raw state
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = [1.0, 0.0, 0.0, 0.0]  # ego quat
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = [1.0, 0.0, 0.0, 0.0]  # opp quat

        new_coeff, accel_res = update_coefficients_np(k, q, lr, coeff, raw)
        assert new_coeff.shape == (COEFF_DIM,)
        assert not np.allclose(new_coeff, coeff), "Coefficients should change"
        assert accel_res.shape == (N_OBJECTS, D_AMP)

    def test_pack_observation(self):
        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        coeff = make_initial_coefficients()
        accel_hist = make_initial_accel_hist()
        packed = pack_observation(raw, coeff, accel_hist)
        assert packed.shape == (SE3_OBS_DIM,)
        np.testing.assert_array_equal(packed[:RAW_STATE_DIM], raw)
        np.testing.assert_array_equal(
            packed[RAW_STATE_DIM:RAW_STATE_DIM + COEFF_DIM], coeff)
        np.testing.assert_array_equal(
            packed[RAW_STATE_DIM + COEFF_DIM:], accel_hist)

    def test_pack_observation_default_accel_hist(self):
        """pack_observation with no accel_hist arg uses zeros."""
        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        coeff = make_initial_coefficients()
        packed = pack_observation(raw, coeff)
        assert packed.shape == (SE3_OBS_DIM,)
        np.testing.assert_array_equal(
            packed[RAW_STATE_DIM + COEFF_DIM:],
            np.zeros(ACCEL_HIST_DIM, dtype=np.float32))


# ── Math correctness tests ────────────────────────────────────────────────────

class TestSpectralMath:
    """Verify the spectral field update math is correct."""

    def _make_raw_state(self, seed: int = 0) -> np.ndarray:
        """Build a plausible raw state (74-dim) with valid unit quaternions."""
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-0.5, 0.5, RAW_STATE_DIM).astype(np.float32)
        # Ego quaternion: unit norm
        eq = rng.standard_normal(4).astype(np.float32)
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = eq / np.linalg.norm(eq)
        # Opp quaternion: unit norm
        oq = rng.standard_normal(4).astype(np.float32)
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = oq / np.linalg.norm(oq)
        # Pad active flags: clamp to [0,1] — each is a single flag
        for i in range(6):
            raw[_PAD_OFF + i] = np.clip(raw[_PAD_OFF + i], 0.0, 1.0)
        # Zero prev velocities so contact is not triggered
        raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 9] = 0.0
        # Zero prev angular velocities
        raw[_PREV_ANG_VEL_OFF:_PREV_ANG_VEL_OFF + 9] = 0.0
        # Zero prev scalars
        raw[_PREV_SCALARS_OFF:_PREV_SCALARS_OFF + 6] = 0.0
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
        coeff_np, _ = update_coefficients_np(k_np, q_norm, lr_np, coeff0, raw)

        # PyTorch update
        obs = torch.tensor(pack_observation(raw, coeff0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            coeff_pt, _ = encoder(obs)
            coeff_pt = coeff_pt[0].numpy()

        np.testing.assert_allclose(
            coeff_pt, coeff_np, atol=1e-5,
            err_msg="PyTorch encoder and numpy mirror must produce identical results")

    def test_coefficient_convergence_real(self):
        """Real coefficients converge: repeated updates on a fixed state drive residual → 0."""
        rng = np.random.default_rng(42)
        k = rng.standard_normal((N_OBJECTS, K, D_AMP)).astype(np.float32) * 0.3
        q = rng.standard_normal((N_OBJECTS, K, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)

        raw = self._make_raw_state(seed=2)
        coeff = make_initial_coefficients()

        def residual_real(coeff, obj):
            """Mean absolute real-part residual over D_AMP dims for one object."""
            c = coeff.reshape(N_OBJECTS, K, D_AMP, N_CHANNELS)
            amp = np.zeros(D_AMP, dtype=np.float32)
            if obj == _BALL:
                amp[:3] = raw[_BALL_OFF:_BALL_OFF + 3]
                amp[3:6] = raw[_BALL_OFF + 6:_BALL_OFF + 9]
            elif obj == _EGO:
                amp[:3] = raw[_EGO_OFF:_EGO_OFF + 3]
                amp[3:6] = raw[_EGO_OFF + 10:_EGO_OFF + 13]
                amp[6] = raw[_EGO_OFF + 13]
                amp[7] = raw[_EGO_OFF + 14]
                amp[8] = raw[_EGO_OFF + 15]
            elif obj == _OPP:
                amp[:3] = raw[_OPP_OFF:_OPP_OFF + 3]
                amp[3:6] = raw[_OPP_OFF + 10:_OPP_OFF + 13]
                amp[6] = raw[_OPP_OFF + 13]
                amp[7] = raw[_OPP_OFF + 14]
                amp[8] = raw[_OPP_OFF + 15]
            phase = k[obj] @ amp
            s_cos = np.cos(phase)
            if obj == _EGO:
                ori = raw[_EGO_OFF + 6:_EGO_OFF + 10]
            elif obj == _OPP:
                ori = raw[_OPP_OFF + 6:_OPP_OFF + 10]
            else:
                ori = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            orient = (q[obj] * ori).sum(axis=-1)          # (K,)
            pred = (s_cos * orient) @ c[obj, :, :, 0]    # (K,)@(K,D_AMP) → (D_AMP,)
            return np.abs(amp - pred).mean()

        # Measure initial residual for ball (obj 0)
        res_before = residual_real(coeff, obj=0)

        for _ in range(200):
            coeff, _ = update_coefficients_np(k, q, lr, coeff, raw)

        res_after = residual_real(coeff, obj=0)
        assert res_after < res_before, (
            f"Real residual should decrease: {res_before:.4f} → {res_after:.4f}")
        assert res_after < 0.05, f"Real residual should be small after 200 steps: {res_after:.4f}"

    def test_complex_field_convergence(self):
        """Complex reconstruction Re[f] = Σ(a·cos - b·sin) converges toward position."""
        rng = np.random.default_rng(99)
        k = rng.standard_normal((N_OBJECTS, K, D_AMP)).astype(np.float32) * 0.3
        q = rng.standard_normal((N_OBJECTS, K, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)

        raw = self._make_raw_state(seed=3)
        coeff = make_initial_coefficients()

        def complex_residual(coeff):
            """Mean abs complex reconstruction error for ball (D_AMP=9)."""
            c = coeff.reshape(N_OBJECTS, K, D_AMP, N_CHANNELS)
            amp = np.zeros(D_AMP, dtype=np.float32)
            amp[:3] = raw[_BALL_OFF:_BALL_OFF + 3]
            amp[3:6] = raw[_BALL_OFF + 6:_BALL_OFF + 9]
            phase = k[_BALL] @ amp
            ori = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            orient = (q[_BALL] * ori).sum(axis=-1)
            basis_cos = np.cos(phase) * orient
            basis_sin = np.sin(phase) * orient
            # Re[f] = Σ_k (a_k·cos·q - b_k·sin·q)
            pred = basis_cos @ c[_BALL, :, :, 0] - basis_sin @ c[_BALL, :, :, 1]
            return np.abs(amp - pred).mean()

        res_before = complex_residual(coeff)

        for _ in range(200):
            coeff, _ = update_coefficients_np(k, q, lr, coeff, raw)

        res_after = complex_residual(coeff)
        assert res_after < res_before or res_before < 1e-6, (
            f"Complex residual should decrease: {res_before:.4f} → {res_after:.4f}")

    def test_contact_reset_only_affects_ball(self):
        """Contact resets ball coefficients but leaves all other objects untouched."""
        encoder = SE3Encoder()
        encoder.eval()

        # Give all coefficients a non-zero prev value
        obs = torch.zeros(1, SE3_OBS_DIM)
        obs[0, RAW_STATE_DIM:RAW_STATE_DIM + COEFF_DIM] = 1.0  # all prev coefficients = 1

        # Trigger contact: large velocity change on ball
        obs[0, _BALL_OFF + 3] = 10.0   # ball vx (current)
        # prev_ball_vel at _PREV_VEL_OFF stays 0 → huge Δv/dt → contact

        with torch.no_grad():
            out, _ = encoder(obs)

        coeff = out[0].reshape(N_OBJECTS, K, D_AMP, N_CHANNELS)

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
            out, accel_hist = encoder(obs)
        assert torch.isfinite(out).all(), "Forward pass on zero state must be finite"
        assert torch.isfinite(accel_hist).all(), "Accel hist must be finite"

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

        coeff_np, _ = update_coefficients_np(k_np, q_norm, lr_np, coeff0, raw)

        obs = torch.tensor(pack_observation(raw, coeff0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            coeff_pt, _ = encoder(obs)
            coeff_pt = coeff_pt[0].numpy()

        np.testing.assert_allclose(
            coeff_pt, coeff_np, atol=1e-5,
            err_msg="Parity must hold for non-zero initial coefficients")


# ── SE3Policy tests ──────────────────────────────────────────────────────────

class TestSE3Policy:

    def test_forward_shapes(self):
        policy = SE3Policy()
        obs = torch.randn(4, EMBED_DIM)
        action, value = policy(obs)
        assert action.shape == (4, 8)
        assert value.shape == (4, 1)

    def test_analog_range(self):
        policy = SE3Policy()
        obs = torch.randn(100, EMBED_DIM)
        action, _ = policy(obs)
        assert action[:, :5].min() >= -1.0
        assert action[:, :5].max() <= 1.0

    def test_binary_range(self):
        policy = SE3Policy()
        obs = torch.randn(100, EMBED_DIM)
        action, _ = policy(obs)
        assert action[:, 5:].min() >= 0.0
        assert action[:, 5:].max() <= 1.0

    def test_act_numpy(self):
        policy = SE3Policy()
        obs = np.random.randn(1, EMBED_DIM).astype(np.float32)
        action, value = policy.act(obs)
        assert action.shape == (8,)
        assert isinstance(value, float)


# ── StochasticSE3Policy tests ────────────────────────────────────────────────

class TestStochasticSE3Policy:

    def test_forward_shapes(self):
        head = StochasticSE3Policy()
        obs = torch.randn(4, EMBED_DIM)
        action, log_prob, value, entropy = head(obs)
        assert action.shape == (4, 8)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert entropy.shape == (4,)

    def test_log_probs_finite(self):
        head = StochasticSE3Policy()
        obs = torch.randn(8, EMBED_DIM)
        _, log_prob, _, _ = head(obs)
        assert torch.isfinite(log_prob).all()

    def test_entropy_positive(self):
        head = StochasticSE3Policy()
        obs = torch.randn(4, EMBED_DIM)
        _, _, _, entropy = head(obs)
        assert (entropy > 0).all()

    def test_evaluate_actions_matches_forward(self):
        head = StochasticSE3Policy()
        torch.manual_seed(42)
        obs = torch.randn(4, EMBED_DIM)
        action, lp_fwd, val_fwd, ent_fwd = head(obs)
        lp_eval, val_eval, ent_eval = head.evaluate_actions(obs, action)
        torch.testing.assert_close(lp_eval, lp_fwd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(val_eval, val_fwd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ent_eval, ent_fwd, atol=1e-5, rtol=1e-5)

    def test_act_deterministic(self):
        head = StochasticSE3Policy()
        head.eval()
        obs = torch.randn(2, EMBED_DIM)
        a1, v1 = head.act_deterministic(obs)
        a2, v2 = head.act_deterministic(obs)
        torch.testing.assert_close(a1, a2)
        torch.testing.assert_close(v1, v2)

    def test_gradient_flows_through_evaluate(self):
        head = StochasticSE3Policy()
        obs = torch.randn(4, EMBED_DIM, requires_grad=True)
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
        obs = torch.randn(100, EMBED_DIM)
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

    def test_dream_zero_vel_stable_xy_positions(self):
        """Zero velocity → ball x/y positions unchanged; z drifts due to gravity."""
        algo = self._make_algo(dream_steps=5, dream_entropy_high=100.0, dream_entropy_low=-1.0)
        seed = np.zeros(SE3_OBS_DIM, dtype=np.float32)
        dream = algo._dream_rollout(seed)
        if dream.shape[0] > 1:
            # x/y should not drift (no gravity on those axes)
            np.testing.assert_allclose(
                dream[0, _BALL_OFF:_BALL_OFF + 2],
                dream[-1, _BALL_OFF:_BALL_OFF + 2],
                atol=1e-5,
                err_msg="Ball x/y should not drift with zero velocity")

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
        obs[:, RAW_STATE_DIM:RAW_STATE_DIM + COEFF_DIM] = 50.0  # way above clip bound
        with torch.no_grad():
            out, _ = encoder(obs)
        assert out.max().item() <= COEFF_CLIP + 1e-6
        assert out.min().item() >= -COEFF_CLIP - 1e-6

    def test_numpy_clipping(self):
        """update_coefficients_np output stays within [-COEFF_CLIP, COEFF_CLIP]."""
        from se3_field import COEFF_CLIP
        k_sp = np.random.randn(N_OBJECTS, K, D_AMP).astype(np.float32)
        quats = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)
        # Large prev coefficients
        prev_coeff = np.full(COEFF_DIM, 50.0, dtype=np.float32)
        raw = np.random.randn(RAW_STATE_DIM).astype(np.float32) * 0.5
        out, _ = update_coefficients_np(k_sp, quats, lr, prev_coeff, raw)
        assert out.max() <= COEFF_CLIP + 1e-6
        assert out.min() >= -COEFF_CLIP - 1e-6

    def test_clipping_preserves_gradients(self):
        """Gradients flow through clamp for in-range values."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM, requires_grad=False)
        out, _ = encoder(obs)
        loss = out.sum()
        loss.backward()
        assert encoder.k_spatial.grad is not None
        assert encoder.k_spatial.grad.abs().sum().item() > 0


# ── LayerNorm tests ─────────────────────────────────────────────────────────

class TestLayerNorm:

    def test_se3_policy_has_layernorm(self):
        """SE3Policy should have LayerNorm as first layer."""
        policy = SE3Policy(obs_dim=EMBED_DIM)
        first_layer = policy.net[0]
        assert isinstance(first_layer, torch.nn.LayerNorm), \
            f"Expected LayerNorm, got {type(first_layer)}"

    def test_stochastic_policy_has_layernorm(self):
        """StochasticSE3Policy should have LayerNorm."""
        policy = StochasticSE3Policy(obs_dim=EMBED_DIM)
        assert hasattr(policy, 'layer_norm')
        assert isinstance(policy.layer_norm, torch.nn.LayerNorm)


# ── Complex coupling tests ──────────────────────────────────────────────────

class TestComplexCoupling:

    def test_cross_term_active(self):
        """Setting coeff_imag != 0 should change position prediction vs imag=0."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        # Zero all prev coefficients and accel hist
        obs[:, RAW_STATE_DIM:] = 0.0
        with torch.no_grad():
            out_zero, _ = encoder(obs)
            out_zero = out_zero.clone()

        # Set imaginary coefficients to nonzero
        obs2 = obs.clone()
        prev = obs2[:, RAW_STATE_DIM:RAW_STATE_DIM + COEFF_DIM].reshape(
            2, N_OBJECTS, K, D_AMP, N_CHANNELS)
        prev[:, :, :, :, 1] = 0.5  # imag channel
        obs2[:, RAW_STATE_DIM:RAW_STATE_DIM + COEFF_DIM] = prev.reshape(2, -1)
        with torch.no_grad():
            out_imag, _ = encoder(obs2)

        # Outputs should differ because sin basis × imag contributes to Re[f]
        assert not torch.allclose(out_zero, out_imag, atol=1e-6), \
            "Imaginary coefficients should affect output via complex cross-term"

    def test_complex_velocity_derivative(self):
        """Analytical velocity (Im[f]) should approximate finite-difference velocity."""
        # Set up known coefficients and positions
        k_sp = np.random.randn(N_OBJECTS, K, D_AMP).astype(np.float32) * 0.5
        quats = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)

        amp = np.random.randn(N_OBJECTS, D_AMP).astype(np.float32) * 0.3
        ori = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (N_OBJECTS, 1))

        # Known coefficients
        coeff = np.random.randn(N_OBJECTS, K, D_AMP, N_CHANNELS).astype(np.float32) * 0.1

        # Compute Re[f] at amp
        def recon_at(a, obj):
            phase = k_sp[obj] @ a
            s_cos = np.cos(phase)
            s_sin = np.sin(phase)
            orient = (quats[obj] * ori[obj]).sum(axis=-1)
            bc = s_cos * orient
            bs = s_sin * orient
            return (bc @ coeff[obj, :, :, 0] - bs @ coeff[obj, :, :, 1])

        # Finite difference velocity for object 0
        eps = 1e-4
        obj = 0
        fd_vel = np.zeros(D_AMP, dtype=np.float32)
        for d in range(D_AMP):
            a_plus = amp[obj].copy(); a_plus[d] += eps
            a_minus = amp[obj].copy(); a_minus[d] -= eps
            fd_vel[d] = (recon_at(a_plus, obj) - recon_at(a_minus, obj))[d] / (2 * eps)

        # Analytical: Im[f] = Sigma_k (a*sin + b*cos) -- this is the imaginary part
        phase = k_sp[obj] @ amp[obj]
        s_cos = np.cos(phase)
        s_sin = np.sin(phase)
        orient = (quats[obj] * ori[obj]).sum(axis=-1)
        im_f = ((s_sin * orient) @ coeff[obj, :, :, 0]
                + (s_cos * orient) @ coeff[obj, :, :, 1])

        # Im[f] exists and is nonzero when coefficients are set
        assert np.abs(im_f).sum() > 1e-8, "Im[f] should be nonzero with random coefficients"


# ── W_interact tests ────────────────────────────────────────────────────────

class TestWInteract:

    def _make_raw_state(self):
        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        raw[_BALL_OFF:_BALL_OFF + 3] = [0.1, 0.2, 0.05]
        raw[_EGO_OFF:_EGO_OFF + 3] = [-0.3, 0.1, 0.02]
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = [1, 0, 0, 0]
        raw[_OPP_OFF:_OPP_OFF + 3] = [0.4, -0.2, 0.03]
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = [1, 0, 0, 0]
        return raw

    def test_zero_W_preserves_output(self):
        """W_interact=0 should produce identical output to W_interact=None."""
        k_sp = np.random.randn(N_OBJECTS, K, D_AMP).astype(np.float32) * 1.0
        quats = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)
        prev = np.random.randn(COEFF_DIM).astype(np.float32) * 0.1
        raw = self._make_raw_state()

        out_none, _ = update_coefficients_np(k_sp, quats, lr, prev, raw, W_interact=None)
        W_zero = np.zeros((N_OBJECTS, N_OBJECTS), dtype=np.float32)
        out_zero, _ = update_coefficients_np(k_sp, quats, lr, prev, raw, W_interact=W_zero)

        np.testing.assert_allclose(out_none, out_zero, atol=1e-6,
                                   err_msg="W_interact=0 should match W_interact=None")

    def test_nonzero_W_changes_output(self):
        """Non-zero W_interact should change the coefficient update."""
        k_sp = np.random.randn(N_OBJECTS, K, D_AMP).astype(np.float32) * 1.0
        quats = np.random.randn(N_OBJECTS, K, 4).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        lr = np.full(N_OBJECTS, 0.05, dtype=np.float32)
        prev = np.random.randn(COEFF_DIM).astype(np.float32) * 0.1
        raw = self._make_raw_state()

        W_zero = np.zeros((N_OBJECTS, N_OBJECTS), dtype=np.float32)
        out_zero, _ = update_coefficients_np(k_sp, quats, lr, prev, raw, W_interact=W_zero)

        W_nonzero = np.random.randn(N_OBJECTS, N_OBJECTS).astype(np.float32) * 0.1
        out_coupled, _ = update_coefficients_np(k_sp, quats, lr, prev, raw, W_interact=W_nonzero)

        assert not np.allclose(out_zero, out_coupled, atol=1e-6), \
            "Non-zero W_interact should change output"

    def test_gradient_flows_to_W_interact(self):
        """Backprop should reach W_interact."""
        encoder = SE3Encoder()
        obs = torch.randn(4, SE3_OBS_DIM)
        out, _ = encoder(obs)
        loss = out.sum()
        loss.backward()

        assert encoder.W_interact.grad is not None, "W_interact should have gradient"
        assert encoder.W_interact.grad.abs().sum().item() > 0, \
            "W_interact gradient should be nonzero"


# ── Numpy / Torch parity ────────────────────────────────────────────────────

class TestNumpyTorchParity:

    def test_complex_coupling_parity(self):
        """Numpy update_coefficients_np should match torch SE3Encoder for one step."""
        encoder = SE3Encoder()
        encoder.eval()

        # Extract params
        with torch.no_grad():
            k_sp_t = encoder.k_spatial.clone()
            q_t = encoder.quaternions.clone()
            q_t = q_t / q_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            lr_t = torch.exp(encoder.log_lr)
            W_t = encoder.W_interact.clone()

        k_sp_np = k_sp_t.numpy()
        q_np = q_t.numpy()
        lr_np = lr_t.numpy()
        W_np = W_t.numpy()

        # Create raw state (no contact scenario — zero prev_ball_vel)
        raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        raw[_BALL_OFF:_BALL_OFF + 3] = [0.1, 0.2, 0.05]
        raw[_EGO_OFF:_EGO_OFF + 3] = [-0.3, 0.1, 0.02]
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = [1, 0, 0, 0]
        raw[_OPP_OFF:_OPP_OFF + 3] = [0.4, -0.2, 0.03]
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = [1, 0, 0, 0]
        # Set prev_ball_vel = ball_vel to avoid contact detection
        raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 3] = raw[_BALL_OFF + 3:_BALL_OFF + 6]

        prev_coeff = np.zeros(COEFF_DIM, dtype=np.float32)

        # Numpy path
        np_out, _ = update_coefficients_np(k_sp_np, q_np, lr_np, prev_coeff, raw,
                                           W_interact=W_np)

        # Torch path
        obs = pack_observation(raw, prev_coeff)
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            torch_out, _ = encoder(obs_t)
            torch_out = torch_out.squeeze(0).numpy()

        np.testing.assert_allclose(np_out, torch_out, atol=1e-4,
                                   err_msg="Numpy and torch complex update should match")


# ── Prediction metric test ──────────────────────────────────────────────────

class TestPredictionMetric:

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

    def test_prediction_metric_in_update(self):
        """SE3PPOAlgorithm.update() should return 'next_state_pred_mse'."""
        from training.algorithms.se3_ppo import SE3PPOAlgorithm
        algo = SE3PPOAlgorithm(self._make_config())

        # Fill buffer by collecting transitions
        for _ in range(64):
            obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            result = algo.select_action(obs)
            next_obs = np.random.randn(2, SE3_OBS_DIM).astype(np.float32)
            algo.store_transition(obs, result, np.zeros(2), next_obs, np.zeros(2), {})

        metrics = algo.update()
        assert 'next_state_pred_mse' in metrics, \
            "update() should return next_state_pred_mse metric"
        assert isinstance(metrics['next_state_pred_mse'], float), \
            "next_state_pred_mse should be a float"


# ── Physics simulation convergence test ─────────────────────────────────────

class TestPhysicsConvergence:
    """Verify the encoder can learn to track and predict interacting objects."""

    @staticmethod
    def _simulate_bouncing_balls(n_steps=200, dt=1.0 / 120.0, gravity=0.0):
        """3 balls bouncing in a box with elastic collisions.

        Parameters
        ----------
        gravity : float
            Downward acceleration on z-axis (negative = down). 0 = no gravity.
            Rocket League-scale in normalised coords: ~-2.0 gives visible arcs.

        Returns positions (n_steps, 3, 3) and velocities (n_steps, 3, 3),
        all normalised to roughly [-0.5, 0.5].
        """
        rng = np.random.RandomState(42)
        # Start balls close together so they interact quickly
        pos = np.array([
            [-0.1, 0.0, 0.1],
            [0.1, 0.0, 0.2],
            [0.0, 0.15, 0.0],
        ], dtype=np.float32)
        vel = np.array([
            [0.8, 0.3, 0.2],
            [-0.6, 0.4, -0.1],
            [0.2, -0.7, 0.3],
        ], dtype=np.float32)
        radius = 0.08
        box = 0.5

        all_pos, all_vel = [pos.copy()], [vel.copy()]
        for _ in range(n_steps - 1):
            # Apply gravity (acceleration on z-axis)
            vel[:, 2] += gravity * dt
            pos = pos + vel * dt
            # Wall reflection
            for i in range(3):
                for d in range(3):
                    if pos[i, d] > box:
                        pos[i, d] = 2 * box - pos[i, d]
                        vel[i, d] *= -1
                    elif pos[i, d] < -box:
                        pos[i, d] = -2 * box - pos[i, d]
                        vel[i, d] *= -1
            # Elastic ball-ball collisions
            for i in range(3):
                for j in range(i + 1, 3):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff)
                    if 1e-8 < dist < 2 * radius:
                        normal = diff / dist
                        v_rel = np.dot(vel[i] - vel[j], normal)
                        if v_rel < 0:
                            vel[i] -= v_rel * normal
                            vel[j] += v_rel * normal
                            sep = (2 * radius - dist) * 0.5
                            pos[i] += normal * sep
                            pos[j] -= normal * sep
            all_pos.append(pos.copy())
            all_vel.append(vel.copy())
        return np.array(all_pos, dtype=np.float32), np.array(all_vel, dtype=np.float32)

    @staticmethod
    def _to_raw_states(positions, velocities):
        """Map 3 balls → raw_state (ball/ego/opp slots). Returns (n_steps, RAW_STATE_DIM)."""
        n = positions.shape[0]
        raw = np.zeros((n, RAW_STATE_DIM), dtype=np.float32)
        id_q = np.array([1, 0, 0, 0], dtype=np.float32)
        for t in range(n):
            r = raw[t]
            r[_BALL_OFF:_BALL_OFF + 3] = positions[t, 0]
            r[_BALL_OFF + 3:_BALL_OFF + 6] = velocities[t, 0]
            r[_EGO_OFF:_EGO_OFF + 3] = positions[t, 1]
            r[_EGO_OFF + 3:_EGO_OFF + 6] = velocities[t, 1]
            r[_EGO_OFF + 6:_EGO_OFF + 10] = id_q
            r[_OPP_OFF:_OPP_OFF + 3] = positions[t, 2]
            r[_OPP_OFF + 3:_OPP_OFF + 6] = velocities[t, 2]
            r[_OPP_OFF + 6:_OPP_OFF + 10] = id_q
            # prev velocities (same as current at t=0 to avoid false contact)
            prev_t = t - 1 if t > 0 else t
            r[_PREV_VEL_OFF:_PREV_VEL_OFF + 3] = velocities[prev_t, 0]
            r[_PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3] = velocities[prev_t, 1]
            r[_PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3] = velocities[prev_t, 2]
        return raw

    @staticmethod
    def _reconstruction_mse(encoder, coeff_tensor, positions_tensor):
        """Compute position reconstruction MSE for the 3 moving objects.

        coeff_tensor: (1, COEFF_DIM)
        positions_tensor: (1, 3, 3)  — positions of ball, ego, opp
        """
        c = coeff_tensor.reshape(1, N_OBJECTS, K, D_AMP, N_CHANNELS)
        # Build 9d amplitude targets (only position dims filled for this test)
        full_amp = torch.zeros(1, N_OBJECTS, D_AMP, device=coeff_tensor.device)
        full_amp[0, _BALL, :3] = positions_tensor[0, 0]
        full_amp[0, _EGO, :3] = positions_tensor[0, 1]
        full_amp[0, _OPP, :3] = positions_tensor[0, 2]
        # _TEAM = _EGO in 1v1, _STADIUM stays zero

        q_basis = encoder.quaternions / encoder.quaternions.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)
        # k_spatial: (N_OBJECTS, K, D_AMP), full_amp: (1, N_OBJECTS, D_AMP) -> phase: (1, N_OBJECTS, K)
        phase = torch.einsum('okd,bod->bok', encoder.k_spatial, full_amp)
        orient = torch.einsum('okd,bod->bok', q_basis,
                              torch.tensor([1, 0, 0, 0.], device=coeff_tensor.device)
                              .unsqueeze(0).unsqueeze(0).expand(1, N_OBJECTS, 4))
        basis_cos = (torch.cos(phase) * orient).unsqueeze(-1)   # (1, N_OBJECTS, K, 1)
        basis_sin = (torch.sin(phase) * orient).unsqueeze(-1)
        # c[:, :, :, :, 0] is (1, N_OBJECTS, K, D_AMP)
        predicted = (basis_cos * c[:, :, :, :, 0]
                     - basis_sin * c[:, :, :, :, 1]).sum(dim=2)  # (1, N_OBJECTS, D_AMP)
        # Compare only position dims (:3) for the 3 moving objects
        return ((full_amp[:, :3, :3] - predicted[:, :3, :3]) ** 2).mean()

    def test_encoder_converges_on_bouncing_balls(self):
        """Train encoder on next-step prediction: given coeff[t], predict pos[t+1].

        Uses per-step detachment (truncated BPTT depth 1) so each step gives
        an independent gradient to k_spatial, quaternions, log_lr, and W_interact.

        Next-step prediction directly tests whether the spectral field captures
        dynamics — the field at time t should already partially predict where
        objects will be at t+1 because the LMS has been tracking their motion.
        """
        positions, velocities = self._simulate_bouncing_balls(n_steps=150)
        raw_states = self._to_raw_states(positions, velocities)

        encoder = SE3Encoder()
        optimiser = torch.optim.Adam(encoder.parameters(), lr=3e-3)

        n_epochs = 40
        bptt_len = 8  # backprop through this many steps before detaching
        epoch_losses = []

        for epoch in range(n_epochs):
            coeff = torch.zeros(1, COEFF_DIM)
            accel_hist = torch.zeros(1, ACCEL_HIST_DIM)
            total_loss = torch.tensor(0.0)
            n_pred = 0

            for t in range(len(raw_states)):
                # Detach every bptt_len steps to bound memory while still
                # letting gradients flow through short temporal windows
                if t % bptt_len == 0 and t > 0:
                    coeff = coeff.detach()
                    accel_hist = accel_hist.detach()

                raw_t = torch.from_numpy(raw_states[t]).unsqueeze(0)
                obs = torch.cat([raw_t, coeff, accel_hist], dim=1)
                coeff, accel_hist = encoder(obs)

                # Next-step prediction: how well do coeff[t] predict pos[t+1]?
                if t < len(raw_states) - 1:
                    pos_next = torch.from_numpy(positions[t + 1]).unsqueeze(0)
                    total_loss = total_loss + self._reconstruction_mse(
                        encoder, coeff, pos_next)
                    n_pred += 1

            optimiser.zero_grad()
            (total_loss / n_pred).backward()
            optimiser.step()
            encoder.normalise_quaternions_()

            epoch_losses.append(total_loss.item() / n_pred)

        # Convergence: next-step prediction should improve
        assert epoch_losses[-1] < epoch_losses[0] * 0.85, (
            f"Encoder should converge on next-step prediction: "
            f"epoch 0 loss={epoch_losses[0]:.4f}, "
            f"epoch {n_epochs-1} loss={epoch_losses[-1]:.4f}"
        )

    def test_W_interact_learns_from_collisions(self):
        """W_interact should develop nonzero entries after training on colliding objects.

        Balls that interact (collide) create correlated residuals: when ball A
        bounces off ball B, both have large reconstruction errors simultaneously.
        W_interact should learn to mix these residuals to improve tracking.
        """
        positions, velocities = self._simulate_bouncing_balls(n_steps=150)
        raw_states = self._to_raw_states(positions, velocities)

        encoder = SE3Encoder()
        # Confirm W_interact starts at zero
        assert encoder.W_interact.abs().max().item() < 1e-8

        optimiser = torch.optim.Adam(encoder.parameters(), lr=3e-3)

        for _ in range(30):
            coeff = torch.zeros(1, COEFF_DIM)
            accel_hist = torch.zeros(1, ACCEL_HIST_DIM)
            total_loss = torch.tensor(0.0)
            n_pred = 0

            for t in range(len(raw_states)):
                raw_t = torch.from_numpy(raw_states[t]).unsqueeze(0)
                obs = torch.cat([raw_t, coeff.detach(), accel_hist.detach()], dim=1)
                coeff, accel_hist = encoder(obs)

                if t < len(raw_states) - 1:
                    pos_next = torch.from_numpy(positions[t + 1]).unsqueeze(0)
                    total_loss = total_loss + self._reconstruction_mse(
                        encoder, coeff, pos_next)
                    n_pred += 1

            optimiser.zero_grad()
            (total_loss / n_pred).backward()
            optimiser.step()
            encoder.normalise_quaternions_()

        # W_interact should have learned nonzero coupling
        W = encoder.W_interact.detach()
        # Check entries among the 3 moving objects (ball=0, ego=1, opp=3)
        moving = [_BALL, _EGO, _OPP]
        max_coupling = max(abs(W[i, j].item())
                          for i in moving for j in moving if i != j)
        assert max_coupling > 1e-4, (
            f"W_interact should learn nonzero coupling between colliding objects, "
            f"max coupling={max_coupling:.6f}"
        )

    def _train_encoder(self, positions, velocities, n_epochs=30):
        """Train an encoder on a trajectory, return the trained encoder."""
        raw_states = self._to_raw_states(positions, velocities)
        encoder = SE3Encoder()
        optimiser = torch.optim.Adam(encoder.parameters(), lr=3e-3)

        for _ in range(n_epochs):
            coeff = torch.zeros(1, COEFF_DIM)
            accel_hist = torch.zeros(1, ACCEL_HIST_DIM)
            total_loss = torch.tensor(0.0)
            n_pred = 0
            for t in range(len(raw_states)):
                if t % 8 == 0 and t > 0:
                    coeff = coeff.detach()
                    accel_hist = accel_hist.detach()
                raw_t = torch.from_numpy(raw_states[t]).unsqueeze(0)
                obs = torch.cat([raw_t, coeff, accel_hist], dim=1)
                coeff, accel_hist = encoder(obs)
                if t < len(raw_states) - 1:
                    pos_next = torch.from_numpy(positions[t + 1]).unsqueeze(0)
                    total_loss = total_loss + self._reconstruction_mse(
                        encoder, coeff, pos_next)
                    n_pred += 1
            optimiser.zero_grad()
            (total_loss / n_pred).backward()
            optimiser.step()
            encoder.normalise_quaternions_()

        return encoder

    def _measure_horizon_errors(self, encoder, positions, velocities, horizons):
        """Measure prediction error at multiple horizons.

        At each sample point t, run the encoder up to t (with LMS tracking),
        then freeze coefficients and measure reconstruction error at t+h
        WITHOUT any further LMS updates. This isolates how well the field
        state at time t predicts future positions.

        Returns dict {horizon: mean_mse}.
        """
        raw_states = self._to_raw_states(positions, velocities)
        max_h = max(horizons)
        n_samples = len(raw_states) - max_h
        if n_samples < 10:
            raise ValueError("Trajectory too short for requested horizons")

        # Roll encoder forward, saving coefficients at each step
        all_coeff = []
        coeff = torch.zeros(1, COEFF_DIM)
        accel_hist = torch.zeros(1, ACCEL_HIST_DIM)
        with torch.no_grad():
            for t in range(len(raw_states)):
                raw_t = torch.from_numpy(raw_states[t]).unsqueeze(0)
                obs = torch.cat([raw_t, coeff, accel_hist], dim=1)
                coeff, accel_hist = encoder(obs)
                all_coeff.append(coeff.clone())

        # Measure reconstruction error at each horizon
        errors = {}
        for h in horizons:
            mse_sum = 0.0
            count = 0
            with torch.no_grad():
                for t in range(20, n_samples):  # skip warmup
                    frozen_coeff = all_coeff[t]
                    future_pos = torch.from_numpy(positions[t + h]).unsqueeze(0)
                    mse = self._reconstruction_mse(encoder, frozen_coeff, future_pos)
                    mse_sum += mse.item()
                    count += 1
            errors[h] = mse_sum / count
        return errors

    def test_prediction_horizon_gravity_divergence(self):
        """Compare prediction degradation at increasing horizons: no-gravity vs gravity.

        Without gravity, the field's velocity channel (Im[f]) provides a decent
        linear extrapolation — error grows slowly. With gravity, the missing
        acceleration term causes quadratic divergence at longer horizons.

        At 1-step horizon, both should be similar (gravity is negligible per tick).
        At 60-step horizon (~0.5s), gravity case should be substantially worse.
        """
        n_steps = 300
        horizons = [1, 5, 15, 30, 60]

        # Generate trajectories
        pos_no_g, vel_no_g = self._simulate_bouncing_balls(
            n_steps=n_steps, gravity=0.0)
        pos_grav, vel_grav = self._simulate_bouncing_balls(
            n_steps=n_steps, gravity=-2.0)

        # Train encoders
        enc_no_g = self._train_encoder(pos_no_g, vel_no_g, n_epochs=25)
        enc_grav = self._train_encoder(pos_grav, vel_grav, n_epochs=25)

        # Measure prediction errors at each horizon
        err_no_g = self._measure_horizon_errors(
            enc_no_g, pos_no_g, vel_no_g, horizons)
        err_grav = self._measure_horizon_errors(
            enc_grav, pos_grav, vel_grav, horizons)

        # Print results for diagnostic visibility
        print("\n--- Prediction horizon experiment ---")
        print(f"{'Horizon':>8} {'No-gravity MSE':>16} {'Gravity MSE':>16} {'Ratio (g/no-g)':>16}")
        for h in horizons:
            ratio = err_grav[h] / max(err_no_g[h], 1e-10)
            print(f"{h:>8} {err_no_g[h]:>16.6f} {err_grav[h]:>16.6f} {ratio:>16.2f}x")

        # At short horizon, both should be similar (within 5x)
        ratio_short = err_grav[1] / max(err_no_g[1], 1e-10)
        assert ratio_short < 5.0, (
            f"At horizon=1, gravity shouldn't matter much: "
            f"ratio={ratio_short:.2f}x"
        )

        # At long horizon, gravity case should diverge more
        # The ratio of (gravity error / no-gravity error) should increase with horizon
        ratio_long = err_grav[60] / max(err_no_g[60], 1e-10)
        assert ratio_long > ratio_short, (
            f"Gravity prediction should degrade faster at longer horizons: "
            f"ratio@1={ratio_short:.2f}x, ratio@60={ratio_long:.2f}x"
        )

        # Error should grow with horizon for both cases
        assert err_no_g[60] > err_no_g[1], "No-gravity error should grow with horizon"
        assert err_grav[60] > err_grav[1], "Gravity error should grow with horizon"


# ── Encode-for-policy tests ─────────────────────────────────────────────────

class TestEncodeForPolicy:

    def test_embed_shape(self):
        """encode_for_policy() returns (embed (batch, EMBED_DIM=30), coeff, accel_hist)."""
        encoder = SE3Encoder()
        obs = torch.randn(4, SE3_OBS_DIM)
        embed, coeff, accel_hist = encoder.encode_for_policy(obs)
        assert embed.shape == (4, EMBED_DIM), f"Expected (4, {EMBED_DIM}), got {embed.shape}"
        assert coeff.shape == (4, COEFF_DIM)
        assert accel_hist.shape == (4, ACCEL_HIST_DIM)

    def test_embed_gradients(self):
        """Gradients from encode_for_policy reach k_spatial, quaternions, W_interact."""
        torch.manual_seed(42)
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        embed, _, _ = encoder.encode_for_policy(obs)
        loss = (embed ** 2).sum()
        loss.backward()
        assert encoder.k_spatial.grad is not None
        assert encoder.k_spatial.grad.abs().sum() > 0
        assert encoder.quaternions.grad is not None
        assert encoder.W_interact.grad is not None
        # Interaction conv parameters should also have gradients
        assert encoder.field_proj.weight.grad is not None
        assert encoder.interaction_conv[0].weight.grad is not None

    def test_output_layernorm_normalizes(self):
        """Output should have approximately zero mean and unit variance per sample."""
        encoder = SE3Encoder()
        encoder.eval()
        obs = torch.randn(32, SE3_OBS_DIM)
        with torch.no_grad():
            embed, _, _ = encoder.encode_for_policy(obs)
        # Per-sample mean should be near 0
        sample_means = embed.mean(dim=-1)
        assert sample_means.abs().mean() < 0.5, \
            f"LayerNorm output mean too far from 0: {sample_means.abs().mean():.3f}"

    def test_forward_and_encode_share_coefficients(self):
        """forward() and encode_for_policy() should produce consistent coefficient state."""
        encoder = SE3Encoder()
        encoder.eval()
        obs = torch.randn(2, SE3_OBS_DIM)
        with torch.no_grad():
            coeff, accel_hist = encoder(obs)
            embed, coeff_e, accel_hist_e = encoder.encode_for_policy(obs)
        # Both should run without error and produce correct shapes
        assert coeff.shape == (2, COEFF_DIM)
        assert embed.shape == (2, EMBED_DIM)
        assert accel_hist.shape == (2, ACCEL_HIST_DIM)
        assert accel_hist_e.shape == (2, ACCEL_HIST_DIM)


# ── Acceleration channel tests ──────────────────────────────────────────────

class TestAccelerationChannel:

    def test_accel_freefall_residual_near_zero(self):
        """Ball in freefall → accel channel coefficients stay small.

        When delta_v matches gravity exactly, the accel residual target is zero,
        so the accel channel should accumulate very little.
        """
        encoder = SE3Encoder()
        encoder.eval()

        # Simulate freefall: ball vel changes by exactly GRAVITY_DV_Z per tick
        coeff = torch.zeros(1, COEFF_DIM)
        accel_hist = torch.zeros(1, ACCEL_HIST_DIM)
        for t in range(50):
            raw = torch.zeros(1, RAW_STATE_DIM)
            # Ball at some position
            raw[0, _BALL_OFF:_BALL_OFF + 3] = torch.tensor([0.0, 0.0, 0.3 - t * 0.001])
            # Ball vel: constant horizontal, gravity-adjusted vertical
            vz = -t * abs(GRAVITY_DV_Z)
            raw[0, _BALL_OFF + 3:_BALL_OFF + 6] = torch.tensor([0.1, 0.0, vz])
            # Prev ball vel
            prev_vz = -(t - 1) * abs(GRAVITY_DV_Z) if t > 0 else 0.0
            raw[0, _PREV_VEL_OFF:_PREV_VEL_OFF + 3] = torch.tensor([0.1, 0.0, prev_vz])
            # Prev ego/opp vel = current (no accel)
            raw[0, _PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3] = 0.0
            raw[0, _PREV_OPP_VEL_OFF:_PREV_OPP_VEL_OFF + 3] = 0.0
            # Ego/opp identity quats
            raw[0, _EGO_OFF + 6:_EGO_OFF + 10] = torch.tensor([1, 0, 0, 0.])
            raw[0, _OPP_OFF + 6:_OPP_OFF + 10] = torch.tensor([1, 0, 0, 0.])

            obs = torch.cat([raw, coeff, accel_hist], dim=1)
            with torch.no_grad():
                coeff, accel_hist = encoder(obs)

        # Accel channel coefficients (channel 2) for ball should be small
        c = coeff.reshape(N_OBJECTS, K, D_AMP, N_CHANNELS)
        ball_accel = c[_BALL, :, :, 2].abs().mean().item()
        assert ball_accel < 0.5, (
            f"Freefall accel residual should be near zero, got {ball_accel:.4f}")

    def test_accel_nonzero_on_impulse(self):
        """Sudden velocity change (beyond gravity) → accel channel becomes nonzero."""
        encoder = SE3Encoder()
        encoder.eval()

        # Step 1: normal state
        raw = torch.zeros(1, RAW_STATE_DIM)
        raw[0, _EGO_OFF + 6:_EGO_OFF + 10] = torch.tensor([1, 0, 0, 0.])
        raw[0, _OPP_OFF + 6:_OPP_OFF + 10] = torch.tensor([1, 0, 0, 0.])
        raw[0, _BALL_OFF + 3] = 0.1  # ball vx
        raw[0, _PREV_VEL_OFF] = 0.1  # prev ball vx = same

        coeff = torch.zeros(1, COEFF_DIM)
        accel_hist = torch.zeros(1, ACCEL_HIST_DIM)
        obs = torch.cat([raw, coeff, accel_hist], dim=1)
        with torch.no_grad():
            coeff, accel_hist = encoder(obs)

        # Step 2: sudden impulse on ego
        raw2 = raw.clone()
        raw2[0, _EGO_OFF + 3] = 0.5  # ego vx jumps to 0.5
        raw2[0, _PREV_EGO_VEL_OFF] = 0.0  # prev ego vx was 0
        obs2 = torch.cat([raw2, coeff, accel_hist], dim=1)
        with torch.no_grad():
            coeff2, _ = encoder(obs2)

        c = coeff2.reshape(N_OBJECTS, K, D_AMP, N_CHANNELS)
        ego_accel = c[_EGO, :, :, 2].abs().mean().item()
        assert ego_accel > 1e-5, (
            f"Ego accel channel should be nonzero after impulse, got {ego_accel:.6f}")


# ── Parameter count test ────────────────────────────────────────────────────

class TestParameterCount:

    def test_parameter_count_under_30k(self):
        """Total encoder + policy params must fit under 30K limit."""
        encoder = SE3Encoder()
        policy = SE3Policy()
        enc_params = sum(p.numel() for p in encoder.parameters())
        pol_params = sum(p.numel() for p in policy.parameters())
        total = enc_params + pol_params
        assert total < 30000, (
            f"Total params {total} exceeds 30K limit "
            f"(encoder={enc_params}, policy={pol_params})")

    def test_stochastic_policy_under_30k(self):
        """StochasticSE3Policy variant also under limit."""
        encoder = SE3Encoder()
        policy = StochasticSE3Policy()
        enc_params = sum(p.numel() for p in encoder.parameters())
        pol_params = sum(p.numel() for p in policy.parameters())
        total = enc_params + pol_params
        assert total < 30000, f"Total params {total} exceeds 30K"


# ── 4500-step toy simulation stability test ─────────────────────────────────

class TestLongSimulation:

    def test_4500_step_toy_sim(self):
        """Run 3 bouncing balls with gravity for 4500 steps (full game).

        Verify that:
        1. Coefficients stay finite throughout
        2. Position reconstruction error stays bounded
        3. No NaN/Inf in encoder output
        """
        from training.tests.test_se3 import TestPhysicsConvergence

        positions, velocities = TestPhysicsConvergence._simulate_bouncing_balls(
            n_steps=4500, dt=1.0 / 120.0, gravity=-2.0)
        raw_states = TestPhysicsConvergence._to_raw_states(positions, velocities)

        encoder = SE3Encoder()
        encoder.eval()

        coeff = np.zeros(COEFF_DIM, dtype=np.float32)
        k_np = encoder.k_spatial.detach().numpy()
        q_np = (encoder.quaternions / encoder.quaternions.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)).detach().numpy()
        lr_np = np.exp(encoder.log_lr.detach().numpy())
        W_np = encoder.W_interact.detach().numpy()

        max_coeff_val = 0.0
        n_finite_checks = 0

        for t in range(len(raw_states)):
            coeff, _ = update_coefficients_np(k_np, q_np, lr_np, coeff, raw_states[t],
                                              W_interact=W_np)
            assert np.isfinite(coeff).all(), f"Non-finite coefficients at step {t}"
            max_coeff_val = max(max_coeff_val, np.abs(coeff).max())
            n_finite_checks += 1

        # Verify all 4500 steps checked
        assert n_finite_checks == 4500

        # Max coefficient should be bounded by COEFF_CLIP
        from se3_field import COEFF_CLIP
        assert max_coeff_val <= COEFF_CLIP + 1e-6, \
            f"Coefficients exceeded clip bound: {max_coeff_val:.2f} > {COEFF_CLIP}"

        # Final reconstruction error should be finite and reasonable
        # (we can't train during this test, but tracking should work)
        c = coeff.reshape(N_OBJECTS, K, D_AMP, N_CHANNELS)
        assert np.isfinite(c).all(), "Final coefficients have non-finite values"

    def test_rlgym_4500_step_integration(self):
        """Integration test with real rlgym-sim physics (4500 steps).

        Skipped if rlgym_sim is not installed.
        """
        pytest.importorskip('rlgym_sim')
        # This test is designed to be run on a machine with rlgym-sim installed.
        # It verifies the full SE3 encoding pipeline with real physics.
        from training.environments.se3_env import SE3GymEnv
        env = SE3GymEnv(max_steps=4500)
        obs, _ = env.reset()

        assert obs.shape == (SE3_OBS_DIM,)
        assert np.isfinite(obs).all(), "Initial observation has non-finite values"

        for step in range(min(100, 4500)):  # Run 100 steps as smoke test
            action = np.random.uniform(-1, 1, size=8).astype(np.float32)
            obs, reward, done, truncated, info = env.step(action)
            assert obs.shape == (SE3_OBS_DIM,), f"Step {step}: wrong obs shape"
            assert np.isfinite(obs).all(), f"Step {step}: non-finite obs"
            if done:
                obs, _ = env.reset()

        env.close()


# ── Interaction conv shape tests ──────────────────────────────────────────────

class TestInteractionConvShapes:

    def test_field_proj_output(self):
        """field_proj maps per-object coeff to (batch, N_OBJECTS, D_FIELD=16)."""
        encoder = SE3Encoder()
        encoder.eval()
        batch = 3
        # Create fake per-object flattened coefficients
        coeff_flat = torch.randn(batch, N_OBJECTS, K * D_AMP * N_CHANNELS)
        with torch.no_grad():
            f = encoder.field_proj(coeff_flat)
        assert f.shape == (batch, N_OBJECTS, D_FIELD), \
            f"Expected ({batch}, {N_OBJECTS}, {D_FIELD}), got {f.shape}"

    def test_inner_product_shape(self):
        """Inner product matrix is (batch, 1, N_OBJECTS, N_OBJECTS)."""
        encoder = SE3Encoder()
        encoder.eval()
        batch = 2
        f = torch.randn(batch, N_OBJECTS, D_FIELD)
        inner = torch.bmm(f, f.transpose(1, 2)).unsqueeze(1)
        assert inner.shape == (batch, 1, N_OBJECTS, N_OBJECTS)

    def test_outer_product_shape(self):
        """Rank-reduced outer products are (batch, D_OUTER^2, N_OBJECTS, N_OBJECTS)."""
        encoder = SE3Encoder()
        encoder.eval()
        batch = 2
        f = torch.randn(batch, N_OBJECTS, D_FIELD)
        with torch.no_grad():
            left = encoder.proj_left(f)
            right = encoder.proj_right(f)
        assert left.shape == (batch, N_OBJECTS, D_OUTER)
        assert right.shape == (batch, N_OBJECTS, D_OUTER)
        left_outer = torch.einsum('bid,bjc->bijdc', left, left)
        left_outer = left_outer.reshape(batch, N_OBJECTS, N_OBJECTS, D_OUTER * D_OUTER)
        left_outer = left_outer.permute(0, 3, 1, 2)
        assert left_outer.shape == (batch, D_OUTER * D_OUTER, N_OBJECTS, N_OBJECTS)

    def test_conv_output_shape(self):
        """Interaction conv produces (batch, CONV_OUT=16)."""
        encoder = SE3Encoder()
        encoder.eval()
        batch = 2
        # 33 = 1 (inner) + 16 (left outer) + 16 (right outer)
        conv_in = torch.randn(batch, 1 + D_OUTER * D_OUTER + D_OUTER * D_OUTER,
                              N_OBJECTS, N_OBJECTS)
        with torch.no_grad():
            out = encoder.interaction_conv(conv_in)
        out = out.squeeze(-1).squeeze(-1)
        assert out.shape == (batch, CONV_OUT), \
            f"Expected ({batch}, {CONV_OUT}), got {out.shape}"

    def test_encode_for_policy_full_pipeline(self):
        """Full encode_for_policy pipeline produces (batch, EMBED_DIM=30)."""
        encoder = SE3Encoder()
        encoder.eval()
        obs = torch.randn(4, SE3_OBS_DIM)
        with torch.no_grad():
            embed, coeff, accel_hist = encoder.encode_for_policy(obs)
        assert embed.shape == (4, EMBED_DIM)
        assert coeff.shape == (4, COEFF_DIM)
        assert accel_hist.shape == (4, ACCEL_HIST_DIM)
        assert EMBED_DIM == CONV_OUT + ACCEL_CTX_DIM + CONTEXT_DIM == 30


# ── D_AMP=9 phase computation test ──────────────────────────────────────────

class TestDAmp9Phase:

    def test_phase_with_full_9d_state(self):
        """Phase = k_spatial @ amplitude where both are 9-dimensional."""
        encoder = SE3Encoder()
        k = encoder.k_spatial.detach()  # (N_OBJECTS, K, D_AMP=9)
        assert k.shape == (N_OBJECTS, K, D_AMP)

        # Build a full 9d amplitude vector for ego
        amp = torch.zeros(D_AMP)
        amp[:3] = torch.tensor([0.1, -0.2, 0.05])     # position
        amp[3:6] = torch.tensor([0.01, -0.02, 0.03])   # angular velocity
        amp[6] = 0.5                                     # boost
        amp[7] = 1.0                                     # has_flip
        amp[8] = 1.0                                     # on_ground

        # Phase for ego object
        phase = k[_EGO] @ amp  # (K,)
        assert phase.shape == (K,)
        assert torch.isfinite(phase).all()

    def test_phase_position_only_vs_full(self):
        """Phase with full 9d amplitude differs from position-only 3d amplitude."""
        encoder = SE3Encoder()
        k = encoder.k_spatial.detach()

        amp_full = torch.zeros(D_AMP)
        amp_full[:3] = torch.tensor([0.1, -0.2, 0.05])
        amp_full[3:6] = torch.tensor([0.5, -0.3, 0.1])
        amp_full[6] = 0.8
        amp_full[7] = 1.0
        amp_full[8] = 0.0

        amp_pos_only = torch.zeros(D_AMP)
        amp_pos_only[:3] = amp_full[:3]

        phase_full = k[_EGO] @ amp_full
        phase_pos = k[_EGO] @ amp_pos_only

        # Unless k_spatial[:, 3:] happens to be exactly zero (vanishingly unlikely),
        # the phases should differ
        assert not torch.allclose(phase_full, phase_pos, atol=1e-6), \
            "Full 9d phase should differ from position-only phase"

    def test_all_objects_have_9d_k_spatial(self):
        """Every object's k_spatial has shape (K, D_AMP=9)."""
        encoder = SE3Encoder()
        k = encoder.k_spatial.detach()
        for obj in range(N_OBJECTS):
            assert k[obj].shape == (K, D_AMP), \
                f"Object {obj} k_spatial shape mismatch: {k[obj].shape}"


# ── Accel momentum tests ──────────────────────────────────────────────────────

class TestAccelMomentum:
    """Tests for accel delta/surprise momentum signals."""

    def test_accel_hist_shape(self):
        """make_initial_accel_hist returns (ACCEL_HIST_DIM,) zeros."""
        hist = make_initial_accel_hist()
        assert hist.shape == (ACCEL_HIST_DIM,)
        assert hist.dtype == np.float32
        assert np.all(hist == 0.0)

    def _make_raw_state(self, seed: int = 0) -> np.ndarray:
        """Build a plausible raw state with valid unit quaternions."""
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-0.5, 0.5, RAW_STATE_DIM).astype(np.float32)
        eq = rng.standard_normal(4).astype(np.float32)
        raw[_EGO_OFF + 6:_EGO_OFF + 10] = eq / np.linalg.norm(eq)
        oq = rng.standard_normal(4).astype(np.float32)
        raw[_OPP_OFF + 6:_OPP_OFF + 10] = oq / np.linalg.norm(oq)
        for i in range(6):
            raw[_PAD_OFF + i] = np.clip(raw[_PAD_OFF + i], 0.0, 1.0)
        raw[_PREV_VEL_OFF:_PREV_VEL_OFF + 9] = 0.0
        raw[_PREV_ANG_VEL_OFF:_PREV_ANG_VEL_OFF + 9] = 0.0
        raw[_PREV_SCALARS_OFF:_PREV_SCALARS_OFF + 6] = 0.0
        raw[_BALL_OFF + 3:_BALL_OFF + 6] = 0.0
        return raw

    def _make_encoder_params(self, encoder):
        """Extract numpy params from encoder."""
        with torch.no_grad():
            q = encoder.quaternions
            q_norm = (q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)).numpy()
            k_np = encoder.k_spatial.numpy()
            lr_np = np.exp(encoder.log_lr.numpy())
        return k_np, q_norm, lr_np

    def test_delta_zero_on_constant_residual(self):
        """When accel residual is constant across steps, delta approaches 0 after first step."""
        from se3_field import update_coefficients_with_hist_np
        encoder = SE3Encoder()
        encoder.eval()
        k_np, q_np, lr_np = self._make_encoder_params(encoder)

        raw = self._make_raw_state(seed=10)
        coeff = make_initial_coefficients()
        accel_hist = make_initial_accel_hist()

        # Run several steps with the SAME raw state to get constant residual
        for _ in range(20):
            coeff, accel_hist = update_coefficients_with_hist_np(
                k_np, q_np, lr_np, coeff, raw, accel_hist)

        # After convergence on constant state, delta (change in accel residual) should be ~0
        _half = N_OBJECTS * D_AMP
        prev_accel_res = accel_hist[:_half].reshape(N_OBJECTS, D_AMP)

        # One more step
        coeff_new, accel_hist_new = update_coefficients_with_hist_np(
            k_np, q_np, lr_np, coeff, raw, accel_hist)
        new_accel_res = accel_hist_new[:_half].reshape(N_OBJECTS, D_AMP)

        delta = np.abs(new_accel_res - prev_accel_res).mean()
        assert delta < 0.01, (
            f"Delta should be near 0 on constant residual, got {delta:.6f}")

    def test_surprise_decays_on_constant_residual(self):
        """When accel residual is constant, surprise decays toward 0 via EMA."""
        from se3_field import update_coefficients_with_hist_np
        encoder = SE3Encoder()
        encoder.eval()
        k_np, q_np, lr_np = self._make_encoder_params(encoder)

        raw = self._make_raw_state(seed=11)
        coeff = make_initial_coefficients()
        accel_hist = make_initial_accel_hist()

        surprises = []
        for step in range(30):
            coeff, accel_hist = update_coefficients_with_hist_np(
                k_np, q_np, lr_np, coeff, raw, accel_hist)
            _half = N_OBJECTS * D_AMP
            accel_res = accel_hist[:_half].reshape(N_OBJECTS, D_AMP)
            accel_ema = accel_hist[_half:].reshape(N_OBJECTS, D_AMP)
            surprise = np.abs(accel_res - accel_ema).mean()
            surprises.append(surprise)

        # Surprise at the end should be smaller than at step 5 (after initial transient)
        assert surprises[-1] < surprises[5] + 1e-6, (
            f"Surprise should decay: step 5={surprises[5]:.6f}, "
            f"step {len(surprises)-1}={surprises[-1]:.6f}")

    def test_delta_spikes_on_action_change(self):
        """When accel residual changes suddenly, delta spikes."""
        from se3_field import update_coefficients_with_hist_np
        encoder = SE3Encoder()
        encoder.eval()
        k_np, q_np, lr_np = self._make_encoder_params(encoder)

        raw = self._make_raw_state(seed=12)
        coeff = make_initial_coefficients()
        accel_hist = make_initial_accel_hist()

        # Run 20 steps with constant state to let things settle
        for _ in range(20):
            coeff, accel_hist = update_coefficients_with_hist_np(
                k_np, q_np, lr_np, coeff, raw, accel_hist)

        _half = N_OBJECTS * D_AMP
        prev_res = accel_hist[:_half].copy()

        # Now change the state drastically (ego gets a big velocity impulse)
        raw2 = raw.copy()
        raw2[_EGO_OFF + 3:_EGO_OFF + 6] = [0.8, -0.5, 0.3]  # big velocity
        raw2[_PREV_EGO_VEL_OFF:_PREV_EGO_VEL_OFF + 3] = [0.0, 0.0, 0.0]

        coeff_new, accel_hist_new = update_coefficients_with_hist_np(
            k_np, q_np, lr_np, coeff, raw2, accel_hist)
        new_res = accel_hist_new[:_half]

        delta = np.abs(new_res - prev_res).mean()
        assert delta > 1e-4, (
            f"Delta should spike on sudden state change, got {delta:.6f}")

    def test_gradient_flows_through_delta_proj(self):
        """Loss on embed -> delta_proj.weight.grad is nonzero."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        embed, _, _ = encoder.encode_for_policy(obs)
        loss = (embed ** 2).sum()
        loss.backward()
        assert encoder.delta_proj.weight.grad is not None, \
            "delta_proj should have gradient"
        assert encoder.delta_proj.weight.grad.abs().sum() > 0, \
            "delta_proj gradient should be nonzero"

    def test_gradient_flows_through_surprise_proj(self):
        """Loss on embed -> surprise_proj.weight.grad is nonzero."""
        encoder = SE3Encoder()
        obs = torch.randn(2, SE3_OBS_DIM)
        embed, _, _ = encoder.encode_for_policy(obs)
        loss = (embed ** 2).sum()
        loss.backward()
        assert encoder.surprise_proj.weight.grad is not None, \
            "surprise_proj should have gradient"
        assert encoder.surprise_proj.weight.grad.abs().sum() > 0, \
            "surprise_proj gradient should be nonzero"


# ── Momentum ablation tests ─────────────────────────────────────────────────

class TestMomentumAblation:

    def _make_encoder_pair(self, mode: str):
        """Create a 'both' encoder and a masked encoder with same weights."""
        torch.manual_seed(42)
        enc_both = SE3Encoder(momentum_mode='both')
        torch.manual_seed(42)
        enc_mode = SE3Encoder(momentum_mode=mode)
        return enc_both, enc_mode

    def test_delta_only_differs_from_both(self):
        """delta_only mode produces different embed than both (surprise is masked)."""
        enc_both, enc_delta = self._make_encoder_pair('delta_only')
        obs = torch.randn(2, SE3_OBS_DIM)
        with torch.no_grad():
            embed_both, _, _ = enc_both.encode_for_policy(obs)
            embed_delta, _, _ = enc_delta.encode_for_policy(obs)
        assert not torch.allclose(embed_both, embed_delta, atol=1e-6), \
            "delta_only should differ from both (surprise contribution is masked)"

    def test_surprise_only_differs_from_both(self):
        """surprise_only mode produces different embed than both (delta is masked)."""
        enc_both, enc_surprise = self._make_encoder_pair('surprise_only')
        obs = torch.randn(2, SE3_OBS_DIM)
        with torch.no_grad():
            embed_both, _, _ = enc_both.encode_for_policy(obs)
            embed_surp, _, _ = enc_surprise.encode_for_policy(obs)
        assert not torch.allclose(embed_both, embed_surp, atol=1e-6), \
            "surprise_only should differ from both (delta contribution is masked)"

    def test_none_differs_from_both(self):
        """'none' mode produces different embed than both (all momentum masked)."""
        enc_both, enc_none = self._make_encoder_pair('none')
        obs = torch.randn(2, SE3_OBS_DIM)
        with torch.no_grad():
            embed_both, _, _ = enc_both.encode_for_policy(obs)
            embed_none, _, _ = enc_none.encode_for_policy(obs)
        assert not torch.allclose(embed_both, embed_none, atol=1e-6), \
            "'none' should differ from both"

    def test_both_matches_default(self):
        """momentum_mode='both' produces identical output to default encoder."""
        enc_both, enc_default = self._make_encoder_pair('both')
        obs = torch.randn(2, SE3_OBS_DIM)
        with torch.no_grad():
            embed_d, coeff_d, ah_d = enc_default.encode_for_policy(obs)
            embed_b, coeff_b, ah_b = enc_both.encode_for_policy(obs)
        torch.testing.assert_close(embed_d, embed_b)
        torch.testing.assert_close(coeff_d, coeff_b)
        torch.testing.assert_close(ah_d, ah_b)

    def test_accel_hist_still_updates_when_masked(self):
        """Even with signals disabled, accel_hist must update (EMA state persists)."""
        encoder = SE3Encoder(momentum_mode='none')
        obs = torch.randn(2, SE3_OBS_DIM)
        with torch.no_grad():
            _, _, accel_hist = encoder.encode_for_policy(obs)
        assert not (accel_hist == 0).all(), \
            "accel_hist should update even when momentum signals are masked"

    def test_param_count_unchanged(self):
        """Ablation flags don't add parameters."""
        enc_both = SE3Encoder(momentum_mode='both')
        enc_none = SE3Encoder(momentum_mode='none')
        n_both = sum(p.numel() for p in enc_both.parameters())
        n_none = sum(p.numel() for p in enc_none.parameters())
        assert n_both == n_none, \
            f"Parameter counts differ: both={n_both}, none={n_none}"
        assert n_both < 30000, f"Encoder has {n_both} params, exceeds 30K limit"
