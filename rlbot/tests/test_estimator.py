"""Tests for the token-to-field estimator."""

import math

import pytest
import torch

from rlbot.constants import N_COEFFS, N_ENTITIES
from rlbot.env.estimator import (
    tokens_to_fields,
    tokens_window_to_fields,
    _ball_field,
    _car_field,
    _boost_density_field,
)


def _make_dummy_tokens() -> torch.Tensor:
    """Create a realistic 10×10 token array."""
    tokens = torch.zeros(10, 10)

    # Token 0: Ball at center, moving along +y
    tokens[0] = torch.tensor([0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Token 1: Own car at (-0.5, -0.8, 0), yaw=0, boost=0.5
    tokens[1] = torch.tensor([-0.5, -0.8, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

    # Token 2: Opponent car at (0.5, 0.8, 0), yaw=pi
    tokens[2] = torch.tensor([0.5, 0.8, 0.0, -0.3, 0.0, 0.0, math.pi, 0.0, 0.0, 0.0])

    # Tokens 3-8: Big boost pads (3 active, 3 inactive)
    tokens[3] = torch.tensor([-0.875, 0.0, 0.036, 1.0, 0, 0, 0, 0, 0, 0])
    tokens[4] = torch.tensor([0.875, 0.0, 0.036, 1.0, 0, 0, 0, 0, 0, 0])
    tokens[5] = torch.tensor([-0.75, 0.8, 0.036, 0.0, 0, 0, 0, 0, 0, 0])
    tokens[6] = torch.tensor([0.75, 0.8, 0.036, 0.0, 0, 0, 0, 0, 0, 0])
    tokens[7] = torch.tensor([-0.75, -0.8, 0.036, 1.0, 0, 0, 0, 0, 0, 0])
    tokens[8] = torch.tensor([0.75, -0.8, 0.036, 0.0, 0, 0, 0, 0, 0, 0])

    # Token 9: Game state
    tokens[9] = torch.tensor([0.1, 0.5, 0.0, 0, 0, 0, 0, 0, 0, 0])

    return tokens


class TestBallField:
    def test_position_matches_token(self):
        token = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        field = _ball_field(token)
        assert torch.allclose(field.position, token[:3])

    def test_l1_is_velocity_direction(self):
        """Ball moving along +y should have l=1 ≈ (0, 1, 0)."""
        token = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        field = _ball_field(token)
        assert torch.allclose(field.coefficients[1:4],
                              torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)

    def test_stationary_ball(self):
        """Stationary ball: l=0 should be small, l=1 should be zero."""
        token = torch.zeros(10)
        field = _ball_field(token)
        assert torch.norm(field.coefficients[1:4]) < 1e-5

    def test_coefficients_shape(self):
        token = torch.randn(10)
        field = _ball_field(token)
        assert field.coefficients.shape == (N_COEFFS,)
        assert field.covariance.shape == (N_COEFFS,)


class TestCarField:
    def test_position_matches_token(self):
        token = torch.tensor([0.3, -0.5, 0.1, 0, 0, 0, 0.0, 0.0, 0.0, 0.5])
        field = _car_field(token)
        assert torch.allclose(field.position, token[:3])

    def test_yaw_zero_forward_is_x(self):
        """Car with yaw=0, pitch=0, roll=0 should face +x.

        The l=1 coefficients should be close to (1, 0, 0) after rotation.
        """
        token = torch.tensor([0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.5])
        field = _car_field(token)
        # l=1 should be the forward direction = (1, 0, 0) for yaw=0
        l1 = field.coefficients[1:4]
        # The canonical forward is along x, rotated by identity
        assert l1[0] > 0.5  # x component should be dominant

    def test_yaw_pi_faces_minus_x(self):
        """Car with yaw=pi should face -x."""
        token = torch.tensor([0.0, 0.0, 0.0, 0, 0, 0, math.pi, 0.0, 0.0, 0.5])
        field = _car_field(token)
        l1 = field.coefficients[1:4]
        assert l1[0] < -0.5  # x component should be negative

    def test_boost_in_l0(self):
        """Boost value should be encoded in l=0 coefficient."""
        token_high = torch.tensor([0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.8])
        token_low = torch.tensor([0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.1])
        field_high = _car_field(token_high)
        field_low = _car_field(token_low)
        assert field_high.coefficients[0] > field_low.coefficients[0]


class TestBoostDensityField:
    def test_no_active_pads(self):
        """All pads inactive → near-zero l=0 coefficient."""
        pads = torch.zeros(6, 10)
        pads[:, :3] = torch.randn(6, 3)  # random positions
        pads[:, 3] = 0.0  # all inactive
        field = _boost_density_field(pads)
        assert field.coefficients[0] < 1e-5

    def test_all_active_full_energy(self):
        """All pads active → l=0 coefficient near 1.0."""
        pads = torch.zeros(6, 10)
        pads[:, :3] = torch.randn(6, 3)
        pads[:, 3] = 1.0  # all active
        field = _boost_density_field(pads)
        assert torch.allclose(field.coefficients[0], torch.tensor(1.0), atol=0.01)

    def test_centroid_is_weighted(self):
        """With one active pad, centroid should be at that pad's position."""
        pads = torch.zeros(6, 10)
        pads[0, :3] = torch.tensor([0.5, 0.3, 0.1])
        pads[0, 3] = 1.0  # only first pad active
        field = _boost_density_field(pads)
        assert torch.allclose(field.position, torch.tensor([0.5, 0.3, 0.1]), atol=1e-5)

    def test_directional_bias(self):
        """Pads on one side should create directional bias in l=1."""
        pads = torch.zeros(6, 10)
        # Two active pads on the +x side
        pads[0, :3] = torch.tensor([0.8, 0.0, 0.0])
        pads[0, 3] = 1.0
        pads[1, :3] = torch.tensor([0.9, 0.0, 0.0])
        pads[1, 3] = 1.0
        field = _boost_density_field(pads)
        # l=1 x-component should be positive (boost is to the right)
        assert field.coefficients[1] > 0

    def test_coefficients_shape(self):
        pads = torch.zeros(6, 10)
        pads[:, 3] = 1.0
        pads[:, :3] = torch.randn(6, 3)
        field = _boost_density_field(pads)
        assert field.coefficients.shape == (N_COEFFS,)


class TestTokensToFields:
    def test_returns_four_fields(self):
        tokens = _make_dummy_tokens()
        fields, game_state = tokens_to_fields(tokens)
        assert len(fields) == N_ENTITIES

    def test_game_state_shape(self):
        tokens = _make_dummy_tokens()
        _, game_state = tokens_to_fields(tokens)
        assert game_state.shape == (3,)

    def test_field_positions_match_tokens(self):
        tokens = _make_dummy_tokens()
        fields, _ = tokens_to_fields(tokens)
        # Ball position
        assert torch.allclose(fields[0].position, tokens[0, :3])
        # Own car position
        assert torch.allclose(fields[1].position, tokens[1, :3])
        # Opp car position
        assert torch.allclose(fields[2].position, tokens[2, :3])


class TestTokensWindowToFields:
    def test_single_frame_matches(self):
        """Single-frame window should give same result as tokens_to_fields."""
        tokens = _make_dummy_tokens()
        fields_single, gs_single = tokens_to_fields(tokens)
        fields_window, gs_window = tokens_window_to_fields(tokens.unsqueeze(0))

        assert len(fields_window) == N_ENTITIES
        assert torch.allclose(gs_single, gs_window)

    def test_multi_frame_shape(self):
        """Multi-frame window should produce valid fields."""
        T = 4
        window = torch.stack([_make_dummy_tokens() for _ in range(T)])
        fields, game_state = tokens_window_to_fields(window)
        assert len(fields) == N_ENTITIES
        assert game_state.shape == (3,)
        for f in fields:
            assert f.coefficients.shape == (N_COEFFS,)
