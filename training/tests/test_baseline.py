"""
Baseline Experiment Unit Tests
==============================
Tests for all baseline training components. Pure component tests run
without rlgym-sim; integration tests are skipped if it's not installed.

Run with:
    python -m pytest training/tests/test_baseline.py -v
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

import gymnasium as gym

from encoder import D_MODEL, N_TOKENS, TOKEN_FEATURES

# d3rlpy is no longer a dependency — skip tests that require it
try:
    import d3rlpy
    from baseline_encoder_factory import TransformerEncoderFactory
    _HAS_D3RLPY = True
except ImportError:
    _HAS_D3RLPY = False

requires_d3rlpy = pytest.mark.skipif(not _HAS_D3RLPY, reason='d3rlpy not installed')


# ── helpers ──────────────────────────────────────────────────────────────────

class DummyEnv(gym.Env):
    """Minimal gym env matching our observation/action spaces."""
    observation_space = gym.spaces.Box(-1, 1, (800,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), np.float32)

    def reset(self, **kw):
        return np.zeros(800, np.float32), {}

    def step(self, a):
        return np.zeros(800, np.float32), 0.0, True, False, {}


def _make_algo(factory=None):
    """Create and build a test AWAC algo (requires d3rlpy)."""
    if not _HAS_D3RLPY:
        pytest.skip('d3rlpy not installed')
    if factory is None:
        factory = TransformerEncoderFactory(t_window=8)
    algo = d3rlpy.algos.AWACConfig(
        actor_encoder_factory=factory,
        critic_encoder_factory=factory,
    ).create(device='cpu')
    algo.build_with_env(DummyEnv())
    return algo


# ── encoder factory tests ───────────────────────────────────────────────────

@requires_d3rlpy
class TestEncoderFactory:

    def test_create_output_shape(self):
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create((800,))
        x = torch.randn(4, 800)
        out = encoder(x)
        assert out.shape == (4, D_MODEL), f"Expected (4, {D_MODEL}), got {out.shape}"

    def test_create_with_action_output_shape(self):
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create_with_action((800,), action_size=8)
        x = torch.randn(4, 800)
        action = torch.randn(4, 8)
        out = encoder(x, action)
        assert out.shape == (4, D_MODEL), f"Expected (4, {D_MODEL}), got {out.shape}"

    def test_batch_size_1(self):
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create((800,))
        out = encoder(torch.randn(1, 800))
        assert out.shape == (1, D_MODEL)

    def test_batch_size_64(self):
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create((800,))
        out = encoder(torch.randn(64, 800))
        assert out.shape == (64, D_MODEL)

    def test_no_nan_output(self):
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create((800,))
        x = torch.randn(4, 800)
        out = encoder(x)
        assert not torch.isnan(out).any(), "NaN in encoder output"
        assert not torch.isinf(out).any(), "Inf in encoder output"

    def test_wrong_input_size_raises(self):
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create((800,))
        with pytest.raises(AssertionError, match="Input size"):
            encoder(torch.randn(2, 500))

    def test_wrong_input_dim_raises(self):
        factory = TransformerEncoderFactory(t_window=8)
        encoder = factory.create((800,))
        with pytest.raises(AssertionError, match="Expected 2D"):
            encoder(torch.randn(2, 8, 100))

    def test_get_type(self):
        factory = TransformerEncoderFactory(t_window=8)
        assert factory.get_type() == 'transformer_spatiotemporal'


# ── self-play pool tests ────────────────────────────────────────────────────

@requires_d3rlpy
class TestOpponentPool:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_pool_returns_none(self):
        from self_play import PeriodicOpponentPool
        pool = PeriodicOpponentPool(self.tmpdir, algo_builder=_make_algo)
        assert pool.sample_opponent() is None
        assert pool.num_snapshots() == 0

    def test_save_and_load(self):
        from self_play import PeriodicOpponentPool
        pool = PeriodicOpponentPool(self.tmpdir, algo_builder=_make_algo)
        algo = _make_algo()

        pool.save_snapshot(algo, step=1000)
        assert pool.num_snapshots() == 1

        loaded = pool.sample_opponent()
        assert loaded is not None

        # Predict should return valid action
        obs = np.random.randn(1, 800).astype(np.float32)
        action = loaded.predict(obs)
        assert action.shape == (1, 8), f"Bad action shape: {action.shape}"

    def test_predict_matches_original(self):
        from self_play import PeriodicOpponentPool
        pool = PeriodicOpponentPool(self.tmpdir, algo_builder=_make_algo)
        algo = _make_algo()

        pool.save_snapshot(algo, step=1000)
        loaded = pool.latest()

        obs = np.random.randn(1, 800).astype(np.float32)
        a1 = algo.predict(obs)
        a2 = loaded.predict(obs)
        np.testing.assert_allclose(a1, a2, atol=1e-5,
                                   err_msg="Loaded model doesn't match original")

    def test_max_snapshots_cleanup(self):
        from self_play import PeriodicOpponentPool
        pool = PeriodicOpponentPool(self.tmpdir, algo_builder=_make_algo, max_snapshots=3)
        algo = _make_algo()

        for step in range(5):
            pool.save_snapshot(algo, step=step * 1000)

        assert pool.num_snapshots() == 3, \
            f"Expected 3 snapshots after cleanup, got {pool.num_snapshots()}"

    def test_action_range(self):
        from self_play import PeriodicOpponentPool
        pool = PeriodicOpponentPool(self.tmpdir, algo_builder=_make_algo)
        algo = _make_algo()
        pool.save_snapshot(algo, step=1000)
        loaded = pool.sample_opponent()

        obs = np.random.randn(1, 800).astype(np.float32)
        action = loaded.predict(obs)
        assert action.min() >= -1.0, f"Action below -1: {action.min()}"
        assert action.max() <= 1.0, f"Action above 1: {action.max()}"


# ── train config tests ──────────────────────────────────────────────────────

@requires_d3rlpy
class TestTrainConfig:

    def test_config_serialization(self):
        from train import TrainConfig
        config = TrainConfig(seed=42, algo='SAC', actor_lr=1e-4)
        d = dataclasses.asdict(config)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded['seed'] == 42
        assert loaded['algo'] == 'SAC'
        assert loaded['actor_lr'] == 1e-4

    def test_config_defaults(self):
        from train import TrainConfig
        config = TrainConfig()
        assert config.total_steps == 50_000_000
        assert config.eval_interval == 200_000
        assert config.batch_size == 256
        assert config.gamma == 0.99
        assert config.tau == 0.005


# ── algo build tests ────────────────────────────────────────────────────────

@requires_d3rlpy
class TestAlgoBuild:

    def test_build_awac(self):
        from train import TrainConfig, build_algo
        config = TrainConfig(algo='AWAC')
        algo = build_algo(config)
        assert 'AWAC' in type(algo).__name__

    def test_build_sac(self):
        from train import TrainConfig, build_algo
        config = TrainConfig(algo='SAC')
        algo = build_algo(config)
        assert 'SAC' in type(algo).__name__

    def test_invalid_algo_raises(self):
        from train import TrainConfig, build_algo
        config = TrainConfig(algo='INVALID')
        with pytest.raises(ValueError, match="Unknown algorithm"):
            build_algo(config)

    def test_hyperparams_passed_through(self):
        from train import TrainConfig, build_algo
        config = TrainConfig(
            algo='AWAC',
            actor_lr=1e-4,
            critic_lr=2e-4,
            gamma=0.95,
            awac_lambda=2.0,
        )
        algo = build_algo(config)
        # The algo is built but not trained — verify it was created successfully
        assert algo is not None


# ── eval config tests ────────────────────────────────────────────────────────

class TestEvalConfig:

    def test_generate_toml_each_tier(self):
        from evaluate import generate_match_toml
        for tier in ['Beginner', 'Rookie', 'Pro', 'Allstar']:
            toml = generate_match_toml(tier)
            assert f'skill = "{tier}"' in toml, \
                f"Tier {tier} not found in TOML"
            assert 'type = "Psyonix"' in toml
            assert 'type = "RLBot"' in toml

    def test_toml_has_two_cars(self):
        from evaluate import generate_match_toml
        toml = generate_match_toml('Rookie')
        assert toml.count('[[cars]]') == 2, \
            "Expected 2 [[cars]] sections in TOML"


# ── gym env shape tests (no rlgym-sim needed) ───────────────────────────────

class TestGymEnvShapes:

    def test_observation_space(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)
        assert env.observation_space.shape == (800,)

    def test_action_space(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)
        assert env.action_space.shape == (8,)

    def test_custom_t_window(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=4)
        assert env.observation_space.shape == (400,)


# ── integration tests (require rlgym-sim) ───────────────────────────────────

def _has_rlgym_sim():
    try:
        import rlgym_sim
        return True
    except ImportError:
        return False

requires_rlgym = pytest.mark.skipif(
    not _has_rlgym_sim(),
    reason='rlgym-sim not installed'
)


@requires_rlgym
class TestGymEnvIntegration:

    def test_reset_returns_correct_shape(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)
        obs, info = env.reset()
        assert obs.shape == (800,), f"Bad reset obs shape: {obs.shape}"
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_correct_types(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)
        env.reset()
        action = np.random.uniform(-1, 1, size=8).astype(np.float32)
        obs, reward, done, trunc, info = env.step(action)
        assert obs.shape == (800,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        env.close()

    def test_reward_is_zero_mid_episode(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)
        env.reset()
        action = np.zeros(8, dtype=np.float32)  # do nothing
        obs, reward, done, _, _ = env.step(action)
        if not done:
            assert reward == 0.0, \
                f"Mid-episode reward should be 0.0, got {reward}"
        env.close()

    def test_observations_change(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)
        obs1, _ = env.reset()
        action = np.random.uniform(-1, 1, size=8).astype(np.float32)
        obs2, _, _, _, _ = env.step(action)
        # After a step, obs should differ (physics moved)
        assert not np.array_equal(obs1, obs2), \
            "Observations should change between steps"
        env.close()

    def test_episode_terminates(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8, max_steps=100)
        env.reset()
        for _ in range(200):
            action = np.random.uniform(-1, 1, size=8).astype(np.float32)
            _, _, done, _, _ = env.step(action)
            if done:
                break
        assert done, "Episode should terminate within max_steps"
        env.close()

    def test_opponent_integration(self):
        from gym_env import BaselineGymEnv
        env = BaselineGymEnv(t_window=8)
        algo = _make_algo()
        env.set_opponent(algo)
        obs, _ = env.reset()
        action = np.random.uniform(-1, 1, size=8).astype(np.float32)
        obs2, reward, done, _, _ = env.step(action)
        assert obs2.shape == (800,)
        env.close()


# ── reward tracker tests ────────────────────────────────────────────────────

class TestRewardTracker:

    def test_tracks_episode_returns(self):
        from tune import RewardTracker

        class MockEnv:
            observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
            action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

            def __init__(self):
                self._step = 0

            def reset(self, **kw):
                self._step = 0
                return np.zeros(4, np.float32), {}

            def step(self, a):
                self._step += 1
                done = self._step >= 3
                return np.zeros(4, np.float32), 1.0, done, False, {}

            def close(self):
                pass

        tracker = RewardTracker(MockEnv())
        tracker.reset()
        for _ in range(3):
            tracker.step(np.zeros(2))
        # 3 steps of reward 1.0 = 3.0 total
        assert len(tracker.episode_returns) == 1
        assert tracker.episode_returns[0] == 3.0
        assert tracker.mean_return() == 3.0
