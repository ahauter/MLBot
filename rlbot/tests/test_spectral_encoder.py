"""Tests for the learned spectral encoder."""

import torch

from rlbot.constants import N_COEFFS, N_ENTITIES
from rlbot.env.spectral_encoder import (
    SpectralEncoder,
    EntityEncoder,
    BoostEncoder,
    OBS_DIM,
)


def _make_dummy_tokens() -> torch.Tensor:
    """Create realistic (10, 10) tokens."""
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


class TestEntityEncoder:
    def test_output_shapes(self):
        enc = EntityEncoder(token_dim=10, hidden_dim=32)
        token = torch.randn(10)
        field = enc(token)
        assert field.position.shape == (3,)
        assert field.coefficients.shape == (N_COEFFS,)
        assert field.covariance.shape == (N_COEFFS,)

    def test_covariance_positive(self):
        """Covariance should always be positive (softplus)."""
        enc = EntityEncoder(token_dim=10, hidden_dim=32)
        token = torch.randn(10)
        field = enc(token)
        assert (field.covariance > 0).all()

    def test_batched_output(self):
        enc = EntityEncoder(token_dim=10, hidden_dim=32)
        tokens = torch.randn(8, 10)
        field = enc(tokens)
        assert field.position.shape == (8, 3)
        assert field.coefficients.shape == (8, N_COEFFS)

    def test_gradients_flow(self):
        enc = EntityEncoder(token_dim=10, hidden_dim=32)
        token = torch.randn(10, requires_grad=True)
        field = enc(token)
        loss = field.coefficients.sum()
        loss.backward()
        assert token.grad is not None
        assert token.grad.abs().sum() > 0


class TestBoostEncoder:
    def test_output_shapes(self):
        enc = BoostEncoder(token_dim=10, hidden_dim=32)
        pads = torch.randn(6, 10)
        field = enc(pads)
        assert field.position.shape == (3,)
        assert field.coefficients.shape == (N_COEFFS,)
        assert field.covariance.shape == (N_COEFFS,)

    def test_batched_output(self):
        enc = BoostEncoder(token_dim=10, hidden_dim=32)
        pads = torch.randn(4, 6, 10)
        field = enc(pads)
        assert field.position.shape == (4, 3)
        assert field.coefficients.shape == (4, N_COEFFS)

    def test_gradients_flow(self):
        enc = BoostEncoder(token_dim=10, hidden_dim=32)
        pads = torch.randn(6, 10, requires_grad=True)
        field = enc(pads)
        loss = field.coefficients.sum()
        loss.backward()
        assert pads.grad is not None


class TestSpectralEncoder:
    def test_single_frame_output_shape(self):
        enc = SpectralEncoder(token_dim=10, hidden_dim=32)
        tokens = _make_dummy_tokens()
        obs = enc(tokens)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_batched_output_shape(self):
        enc = SpectralEncoder(token_dim=10, hidden_dim=32)
        tokens = torch.stack([_make_dummy_tokens() for _ in range(4)])
        obs = enc(tokens)
        assert obs.shape == (4, OBS_DIM)

    def test_no_nan_or_inf(self):
        enc = SpectralEncoder(token_dim=10, hidden_dim=32)
        tokens = _make_dummy_tokens()
        obs = enc(tokens)
        assert torch.isfinite(obs).all(), "Output contains NaN or Inf"

    def test_different_inputs_give_different_outputs(self):
        enc = SpectralEncoder(token_dim=10, hidden_dim=32)
        tokens1 = _make_dummy_tokens()
        tokens2 = _make_dummy_tokens()
        tokens2[0, 3] = 0.9
        obs1 = enc(tokens1)
        obs2 = enc(tokens2)
        assert not torch.allclose(obs1, obs2)

    def test_gradients_flow_end_to_end(self):
        """Gradients should flow from output back through to input tokens."""
        enc = SpectralEncoder(token_dim=10, hidden_dim=32)
        tokens = _make_dummy_tokens().requires_grad_(True)
        obs = enc(tokens)
        loss = obs.sum()
        loss.backward()
        assert tokens.grad is not None
        assert tokens.grad.abs().sum() > 0

    def test_batched_gradients(self):
        """Gradients should work in batched mode."""
        enc = SpectralEncoder(token_dim=10, hidden_dim=32)
        tokens = torch.stack([_make_dummy_tokens() for _ in range(4)])
        tokens.requires_grad_(True)
        obs = enc(tokens)
        loss = obs.sum()
        loss.backward()
        assert tokens.grad is not None

    def test_obs_dim_matches_scene(self):
        """OBS_DIM should match the hand-crafted scene.py OBS_DIM."""
        from rlbot.env.scene import OBS_DIM as SCENE_OBS_DIM
        assert OBS_DIM == SCENE_OBS_DIM


class TestEncoderFactory:
    def test_create_encoder(self):
        """Factory should create a working encoder."""
        from rlbot.training.spectral_encoder_factory import SpectralEncoderFactory
        factory = SpectralEncoderFactory(encoder_hidden=32, policy_hidden=64, policy_layers=1)
        encoder = factory.create(observation_shape=(100,))
        x = torch.randn(2, 100)
        out = encoder(x)
        assert out.shape == (2, 64)

    def test_create_encoder_with_action(self):
        """Factory should create a working encoder-with-action."""
        from rlbot.training.spectral_encoder_factory import SpectralEncoderFactory
        factory = SpectralEncoderFactory(encoder_hidden=32, policy_hidden=64, policy_layers=1)
        encoder = factory.create_with_action(observation_shape=(100,), action_size=8)
        x = torch.randn(2, 100)
        a = torch.randn(2, 8)
        out = encoder(x, a)
        assert out.shape == (2, 64)

    def test_end_to_end_gradients(self):
        """Full pipeline should support gradient computation."""
        from rlbot.training.spectral_encoder_factory import SpectralEncoderFactory
        factory = SpectralEncoderFactory(encoder_hidden=32, policy_hidden=64, policy_layers=1)
        encoder = factory.create(observation_shape=(100,))
        x = torch.randn(2, 100, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
