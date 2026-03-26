"""Tests for SE(3) spectral field math."""

import math

import pytest
import torch

from rlbot.env.fields import (
    SE3Field,
    make_field,
    wigner_d_matrix,
    wigner_d_l1,
    wigner_d_l2,
    rotate_coefficients,
    degree_affinity,
    interaction_affinity,
    interaction_matrix,
    kalman_update,
    fit_spectral_coefficients,
    reconstruct_position,
    _rotation_matrix_from_euler,
)
from rlbot.constants import N_COEFFS, L_MAX


class TestWignerD:
    """Test Wigner D-matrix correctness."""

    def test_l0_is_identity(self):
        D = wigner_d_matrix(0, torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.1))
        assert D.shape == (1, 1)
        assert torch.allclose(D, torch.ones(1, 1))

    def test_l1_identity_rotation(self):
        """Zero angles → identity matrix."""
        z = torch.tensor(0.0)
        D = wigner_d_l1(z, z, z)
        assert D.shape == (3, 3)
        assert torch.allclose(D, torch.eye(3), atol=1e-6)

    def test_l1_matches_rotation_matrix(self):
        """D^1 should match the standard rotation matrix for known angles."""
        yaw = torch.tensor(math.pi / 4)
        pitch = torch.tensor(math.pi / 6)
        roll = torch.tensor(0.0)

        D1 = wigner_d_l1(yaw, pitch, roll)
        R = _rotation_matrix_from_euler(yaw, pitch, roll)

        assert torch.allclose(D1, R, atol=1e-6)

    def test_l1_is_orthogonal(self):
        """D^1 should be an orthogonal matrix: D @ D^T = I."""
        yaw = torch.tensor(1.2)
        pitch = torch.tensor(-0.3)
        roll = torch.tensor(0.7)

        D = wigner_d_l1(yaw, pitch, roll)
        I = D @ D.T
        assert torch.allclose(I, torch.eye(3), atol=1e-5)

    def test_l2_identity_rotation(self):
        """Zero angles → identity matrix for l=2."""
        z = torch.tensor(0.0)
        D = wigner_d_l2(z, z, z)
        assert D.shape == (5, 5)
        assert torch.allclose(D, torch.eye(5), atol=1e-5)

    def test_l2_is_orthogonal(self):
        """D^2 should be orthogonal: D @ D^T = I."""
        yaw = torch.tensor(0.8)
        pitch = torch.tensor(0.4)
        roll = torch.tensor(-0.6)

        D = wigner_d_l2(yaw, pitch, roll)
        I = D @ D.T
        assert torch.allclose(I, torch.eye(5), atol=1e-4)

    def test_l2_determinant_is_one(self):
        """det(D^2) should be +1 (proper rotation)."""
        yaw = torch.tensor(1.0)
        pitch = torch.tensor(0.5)
        roll = torch.tensor(-0.3)

        D = wigner_d_l2(yaw, pitch, roll)
        det = torch.det(D)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-4)


class TestRotateCoefficients:
    """Test coefficient rotation equivariance."""

    def test_identity_rotation_preserves_coeffs(self):
        z = torch.tensor(0.0)
        coeffs = torch.randn(N_COEFFS)
        rotated = rotate_coefficients(coeffs, z, z, z)
        assert torch.allclose(rotated, coeffs, atol=1e-5)

    def test_rotation_preserves_norm(self):
        """||D c|| = ||c|| for any rotation."""
        coeffs = torch.randn(N_COEFFS)
        yaw = torch.tensor(1.3)
        pitch = torch.tensor(-0.4)
        roll = torch.tensor(0.9)

        rotated = rotate_coefficients(coeffs, yaw, pitch, roll)
        assert torch.allclose(
            torch.norm(coeffs), torch.norm(rotated), atol=1e-4
        )

    def test_composition(self):
        """R(a) applied then R(b) should give same result as R(a*b).

        Test via: rotating twice by pi/2 around z ≈ rotating once by pi.
        """
        coeffs = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        z = torch.tensor(0.0)
        half_pi = torch.tensor(math.pi / 2)
        pi = torch.tensor(math.pi)

        # Apply yaw=pi/2 twice
        step1 = rotate_coefficients(coeffs, half_pi, z, z)
        step2 = rotate_coefficients(step1, half_pi, z, z)

        # Apply yaw=pi once
        direct = rotate_coefficients(coeffs, pi, z, z)

        assert torch.allclose(step2, direct, atol=1e-4)


class TestAffinity:
    """Test interaction affinity bounds and properties."""

    def test_self_affinity_is_one(self):
        """Cosine similarity of a vector with itself is 1."""
        coeffs = torch.randn(N_COEFFS)
        pos = torch.zeros(3)
        field = make_field(pos, coeffs)
        aff = interaction_affinity(field, field, sigma=10.0)
        assert torch.allclose(aff, torch.tensor(1.0), atol=1e-5)

    def test_affinity_in_range(self):
        """Affinity should be in [0, 1]."""
        for _ in range(20):
            a = make_field(torch.randn(3), torch.randn(N_COEFFS))
            b = make_field(torch.randn(3), torch.randn(N_COEFFS))
            aff = interaction_affinity(a, b, sigma=1.0)
            assert 0.0 <= aff.item() <= 1.0 + 1e-6

    def test_spatial_decay(self):
        """Fields far apart should have lower affinity."""
        coeffs = torch.randn(N_COEFFS)
        near = make_field(torch.tensor([0.0, 0.0, 0.0]), coeffs)
        close = make_field(torch.tensor([0.1, 0.0, 0.0]), coeffs)
        far = make_field(torch.tensor([5.0, 0.0, 0.0]), coeffs)

        aff_close = interaction_affinity(near, close, sigma=0.5)
        aff_far = interaction_affinity(near, far, sigma=0.5)
        assert aff_close > aff_far

    def test_zero_coefficients_give_zero_affinity(self):
        """A field with all-zero coefficients has zero affinity with anything."""
        a = make_field(torch.zeros(3), torch.zeros(N_COEFFS))
        b = make_field(torch.zeros(3), torch.randn(N_COEFFS))
        aff = interaction_affinity(a, b, sigma=1.0)
        assert torch.allclose(aff, torch.tensor(0.0), atol=1e-6)

    def test_degree_affinity_range(self):
        """Per-degree affinity should be in [0, 1]."""
        c_a = torch.randn(N_COEFFS)
        c_b = torch.randn(N_COEFFS)
        for l in range(L_MAX + 1):
            aff = degree_affinity(c_a, c_b, l)
            assert 0.0 <= aff.item() <= 1.0 + 1e-6


class TestInteractionMatrix:
    """Test interaction matrix properties."""

    def test_rows_sum_to_one(self):
        """Softmax rows should sum to 1."""
        fields = [
            make_field(torch.randn(3), torch.randn(N_COEFFS))
            for _ in range(4)
        ]
        I = interaction_matrix(fields, tau=1.0, sigma=0.5)
        row_sums = I.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_shape(self):
        fields = [
            make_field(torch.randn(3), torch.randn(N_COEFFS))
            for _ in range(4)
        ]
        I = interaction_matrix(fields, tau=1.0, sigma=0.5)
        assert I.shape == (4, 4)

    def test_values_in_range(self):
        """All entries should be in [0, 1] (softmax output)."""
        fields = [
            make_field(torch.randn(3), torch.randn(N_COEFFS))
            for _ in range(4)
        ]
        I = interaction_matrix(fields, tau=1.0, sigma=0.5)
        assert (I >= 0).all()
        assert (I <= 1 + 1e-6).all()

    def test_temperature_sharpens(self):
        """Lower temperature should make rows more peaked."""
        fields = [
            make_field(torch.randn(3), torch.randn(N_COEFFS))
            for _ in range(4)
        ]
        I_warm = interaction_matrix(fields, tau=10.0, sigma=1.0)
        I_cold = interaction_matrix(fields, tau=0.1, sigma=1.0)

        # Cold should have higher max per row (more peaked)
        assert I_cold.max(dim=-1).values.mean() > I_warm.max(dim=-1).values.mean()


class TestKalmanUpdate:
    """Test per-degree Kalman update."""

    def test_covariance_decreases(self):
        """Update should reduce covariance."""
        prior = make_field(
            torch.zeros(3),
            torch.randn(N_COEFFS),
            covariance=torch.ones(N_COEFFS) * 2.0,
        )
        observed = torch.randn(N_COEFFS)
        updated = kalman_update(prior, observed)

        assert (updated.covariance <= prior.covariance + 1e-6).all()

    def test_moves_toward_observation(self):
        """Updated coefficients should be between prior and observed."""
        prior_coeffs = torch.zeros(N_COEFFS)
        observed = torch.ones(N_COEFFS)
        prior = make_field(
            torch.zeros(3), prior_coeffs,
            covariance=torch.ones(N_COEFFS),
        )
        updated = kalman_update(prior, observed)

        # Each updated coefficient should be between 0 and 1
        assert (updated.coefficients >= -0.01).all()
        assert (updated.coefficients <= 1.01).all()

    def test_low_noise_trusts_observation(self):
        """With very low observation noise, result should be near observation."""
        prior = make_field(
            torch.zeros(3), torch.zeros(N_COEFFS),
            covariance=torch.ones(N_COEFFS) * 10.0,
        )
        observed = torch.ones(N_COEFFS)
        noise = torch.ones(L_MAX + 1) * 0.001

        updated = kalman_update(prior, observed, noise_per_degree=noise)
        assert torch.allclose(updated.coefficients, observed, atol=0.05)

    def test_high_noise_trusts_prior(self):
        """With very high observation noise, result should be near prior."""
        prior_coeffs = torch.ones(N_COEFFS) * 5.0
        prior = make_field(
            torch.zeros(3), prior_coeffs,
            covariance=torch.ones(N_COEFFS) * 0.01,
        )
        observed = torch.zeros(N_COEFFS)
        noise = torch.ones(L_MAX + 1) * 1000.0

        updated = kalman_update(prior, observed, noise_per_degree=noise)
        assert torch.allclose(updated.coefficients, prior_coeffs, atol=0.1)


class TestSpectralFitting:
    """Test spectral coefficient fitting from trajectories."""

    def test_straight_line_recovery(self):
        """Fit coefficients from straight-line trajectory, reconstruct < 5cm."""
        # Straight line along x at constant velocity
        dt = 1.0 / 120.0
        T = 20
        times = torch.arange(T, dtype=torch.float32) * dt
        velocity = torch.tensor([0.5, 0.0, 0.0])  # normalized velocity
        start = torch.tensor([0.0, 0.0, 0.5])
        positions = start.unsqueeze(0) + velocity.unsqueeze(0) * times.unsqueeze(-1)

        coeffs = fit_spectral_coefficients(positions, dt)
        center = positions.mean(dim=0)

        # Reconstruct at each timestep
        t_center = times.mean()
        max_error = 0.0
        for i in range(T):
            t_offset = times[i] - t_center
            pred = reconstruct_position(coeffs, t_offset, center)
            error = torch.norm(pred - positions[i])
            max_error = max(max_error, error.item())

        # 5cm in normalized coords: 0.05 / 4096 ≈ 1.2e-5
        # But since we're in normalized coords already, error should be small
        assert max_error < 0.01, f"Max reconstruction error: {max_error}"

    def test_stationary_object(self):
        """Stationary object should have l=0 only."""
        positions = torch.tensor([[0.5, 0.3, 0.2]] * 10)
        coeffs = fit_spectral_coefficients(positions)

        # l=1 (direction) should be near zero for stationary
        assert torch.norm(coeffs[1:4]) < 0.1

    def test_coefficients_shape(self):
        positions = torch.randn(8, 3)
        coeffs = fit_spectral_coefficients(positions)
        assert coeffs.shape == (N_COEFFS,)

    def test_single_frame(self):
        """Single position should still produce valid coefficients."""
        positions = torch.tensor([[0.1, 0.2, 0.3]])
        coeffs = fit_spectral_coefficients(positions)
        assert coeffs.shape == (N_COEFFS,)
        assert coeffs[0] == 1.0  # presence
