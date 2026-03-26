"""Tests for scene assembly."""

import time

import pytest
import torch

from rlbot.constants import N_ENTITIES
from rlbot.env.scene import assemble_scene, assemble_scene_windowed, OBS_DIM
from rlbot.env.stadium import wall_distance_features


def _make_dummy_tokens() -> torch.Tensor:
    """Create realistic tokens for testing."""
    tokens = torch.zeros(10, 10)
    tokens[0] = torch.tensor([0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    tokens[1] = torch.tensor([-0.5, -0.8, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    tokens[2] = torch.tensor([0.5, 0.8, 0.0, -0.3, 0.0, 0.0, 3.14, 0.0, 0.0, 0.0])
    tokens[3] = torch.tensor([-0.875, 0.0, 0.036, 1.0, 0, 0, 0, 0, 0, 0])
    tokens[4] = torch.tensor([0.875, 0.0, 0.036, 1.0, 0, 0, 0, 0, 0, 0])
    tokens[5] = torch.tensor([-0.75, 0.8, 0.036, 0.0, 0, 0, 0, 0, 0, 0])
    tokens[6] = torch.tensor([0.75, 0.8, 0.036, 0.0, 0, 0, 0, 0, 0, 0])
    tokens[7] = torch.tensor([-0.75, -0.8, 0.036, 1.0, 0, 0, 0, 0, 0, 0])
    tokens[8] = torch.tensor([0.75, -0.8, 0.036, 0.0, 0, 0, 0, 0, 0, 0])
    tokens[9] = torch.tensor([0.1, 0.5, 0.0, 0, 0, 0, 0, 0, 0, 0])
    return tokens


class TestAssembleScene:
    def test_output_shape(self):
        tokens = _make_dummy_tokens()
        obs = assemble_scene(tokens)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_no_nan_or_inf(self):
        tokens = _make_dummy_tokens()
        obs = assemble_scene(tokens)
        assert torch.isfinite(obs).all(), "Output contains NaN or Inf"

    def test_different_inputs_give_different_outputs(self):
        tokens1 = _make_dummy_tokens()
        tokens2 = _make_dummy_tokens()
        tokens2[0, 3] = 0.9  # change ball velocity

        obs1 = assemble_scene(tokens1)
        obs2 = assemble_scene(tokens2)
        assert not torch.allclose(obs1, obs2)

    def test_game_state_in_output(self):
        """Game state should appear at the end of the observation vector."""
        tokens = _make_dummy_tokens()
        obs = assemble_scene(tokens)
        # Last 3 elements should be game state [0.1, 0.5, 0.0]
        assert torch.allclose(obs[-3:], torch.tensor([0.1, 0.5, 0.0]), atol=1e-5)


class TestAssembleSceneWindowed:
    def test_output_shape(self):
        window = torch.stack([_make_dummy_tokens() for _ in range(4)])
        obs = assemble_scene_windowed(window)
        assert obs.shape == (OBS_DIM,)

    def test_no_nan_or_inf(self):
        window = torch.stack([_make_dummy_tokens() for _ in range(4)])
        obs = assemble_scene_windowed(window)
        assert torch.isfinite(obs).all()


class TestWallDistanceFeatures:
    def test_center_of_field(self):
        """Center of field should have equal side wall distances."""
        pos = torch.tensor([0.0, 0.0, 0.5])
        feats = wall_distance_features(pos)
        assert feats.shape == (6,)
        # Side wall distance at x=0 should be 1.0
        assert torch.allclose(feats[0], torch.tensor(1.0), atol=1e-5)
        # End wall distance at y=0 should be 1.0
        assert torch.allclose(feats[1], torch.tensor(1.0), atol=1e-5)

    def test_near_wall(self):
        """Position near side wall should have small side distance."""
        pos = torch.tensor([0.95, 0.0, 0.5])
        feats = wall_distance_features(pos)
        assert feats[0] < 0.1  # close to side wall

    def test_features_positive(self):
        """All features should be non-negative for positions inside the arena."""
        pos = torch.tensor([0.3, -0.4, 0.5])
        feats = wall_distance_features(pos)
        assert (feats >= -1e-6).all()

    def test_goal_distance_asymmetric(self):
        """Position near blue goal should be closer to blue than orange."""
        pos = torch.tensor([0.0, -0.9, 0.0])
        feats = wall_distance_features(pos)
        d_own = feats[4]   # distance to blue goal
        d_opp = feats[5]   # distance to orange goal
        assert d_own < d_opp


class TestLatency:
    def test_scene_assembly_under_4ms(self):
        """assemble_scene should complete in under 4ms on CPU."""
        tokens = _make_dummy_tokens()

        # Warmup
        for _ in range(5):
            assemble_scene(tokens)

        # Time 100 iterations
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            assemble_scene(tokens)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_iters) * 1000
        assert avg_ms < 4.0, f"Average scene assembly time: {avg_ms:.2f}ms (limit: 4ms)"
