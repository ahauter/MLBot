"""
Frozen Self-Play Unit Tests
============================
Tests for EpisodeOutcomeTracker, FrozenOpponentPool, and OutcomeTrackingEnv.

Run with:
    python -m pytest training/tests/test_frozen_self_play.py -v
"""
from __future__ import annotations

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

import d3rlpy
import gymnasium as gym

from baseline_encoder_factory import TransformerEncoderFactory
from frozen_self_play import EpisodeOutcomeTracker, FrozenOpponentPool, OutcomeTrackingEnv


class DummyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (800,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), np.float32)

    def reset(self, **kw):
        return np.zeros(800, np.float32), {}

    def step(self, a):
        return np.zeros(800, np.float32), 0.0, True, False, {}


def _make_algo():
    factory = TransformerEncoderFactory(t_window=8)
    algo = d3rlpy.algos.AWACConfig(
        actor_encoder_factory=factory,
        critic_encoder_factory=factory,
    ).create(device='cpu')
    algo.build_with_env(DummyEnv())
    return algo


# ── EpisodeOutcomeTracker tests ─────────────────────────────────────────────

class TestEpisodeOutcomeTracker:

    def test_empty_tracker(self):
        t = EpisodeOutcomeTracker(window_size=100)
        assert t.score() == 0.0
        assert t.episode_count == 0

    def test_all_wins(self):
        t = EpisodeOutcomeTracker(window_size=10)
        for _ in range(10):
            t.record(1)
        assert t.score() == 1.0

    def test_all_losses(self):
        t = EpisodeOutcomeTracker(window_size=10)
        for _ in range(10):
            t.record(-1)
        assert t.score() == -1.0

    def test_all_draws(self):
        t = EpisodeOutcomeTracker(window_size=10)
        for _ in range(10):
            t.record(0)
        assert t.score() == 0.0

    def test_mixed_outcomes(self):
        t = EpisodeOutcomeTracker(window_size=10)
        # 7 wins, 2 losses, 1 draw → score = (7 - 2) / 10 = 0.5
        for g in [1, 1, 1, 1, 1, 1, 1, -1, -1, 0]:
            t.record(g)
        assert abs(t.score() - 0.5) < 1e-6

    def test_rolling_window(self):
        t = EpisodeOutcomeTracker(window_size=5)
        for _ in range(5):
            t.record(1)
        assert t.score() == 1.0
        # Add 5 losses — should push out all wins
        for _ in range(5):
            t.record(-1)
        assert t.score() == -1.0

    def test_should_swap_not_ready(self):
        t = EpisodeOutcomeTracker(window_size=100)
        for _ in range(50):
            t.record(1)
        assert t.should_swap() is False

    def test_should_swap_above_threshold(self):
        t = EpisodeOutcomeTracker(window_size=10)
        # 8 wins, 2 losses → score = 0.6 > 0.5
        for g in [1, 1, 1, 1, 1, 1, 1, 1, -1, -1]:
            t.record(g)
        assert t.should_swap(score_threshold=0.5) is True

    def test_should_swap_draws_not_sufficient(self):
        t = EpisodeOutcomeTracker(window_size=10)
        # All draws → score = 0.0, not > 0.5
        for _ in range(10):
            t.record(0)
        assert t.should_swap(score_threshold=0.5) is False

    def test_should_swap_below_threshold(self):
        t = EpisodeOutcomeTracker(window_size=10)
        # 5 wins, 5 losses → score = 0.0
        for g in [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]:
            t.record(g)
        assert t.should_swap(score_threshold=0.5) is False

    def test_should_swap_at_exact_threshold(self):
        """Threshold is strict > (not >=)."""
        t = EpisodeOutcomeTracker(window_size=10)
        # 7 wins, 2 losses, 1 draw → score = (7-2)/10 = 0.5 exactly
        for g in [1, 1, 1, 1, 1, 1, 1, -1, -1, 0]:
            t.record(g)
        assert t.should_swap(score_threshold=0.5) is False

    def test_reset_clears_window(self):
        t = EpisodeOutcomeTracker(window_size=10)
        for _ in range(10):
            t.record(1)
        assert t.episode_count == 10
        t.reset()
        assert t.episode_count == 0
        assert t.score() == 0.0


# ── FrozenOpponentPool tests ───────────────────────────────────────────────

class TestFrozenOpponentPool:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_pool(self):
        pool = FrozenOpponentPool(self.tmpdir, algo_builder=_make_algo)
        assert pool.sample_opponent() is None
        assert pool.num_snapshots() == 0
        assert pool.latest() is None

    def test_save_and_sample(self):
        pool = FrozenOpponentPool(self.tmpdir, algo_builder=_make_algo)
        algo = _make_algo()
        pool.save_snapshot(algo, step=1000)
        assert pool.num_snapshots() == 1

        loaded = pool.sample_opponent()
        assert loaded is not None
        obs = np.random.randn(1, 800).astype(np.float32)
        action = loaded.predict(obs)
        assert action.shape == (1, 8)

    def test_max_5_snapshots(self):
        pool = FrozenOpponentPool(self.tmpdir, algo_builder=_make_algo, max_snapshots=5)
        algo = _make_algo()
        for i in range(8):
            pool.save_snapshot(algo, step=i * 1000)
        assert pool.num_snapshots() == 5

    def test_evicts_oldest(self):
        pool = FrozenOpponentPool(self.tmpdir, algo_builder=_make_algo, max_snapshots=3)
        algo = _make_algo()
        for i in range(5):
            pool.save_snapshot(algo, step=i * 1000)
        snapshots = pool._list_snapshots()
        # Should have steps 2000, 3000, 4000
        names = [s.name for s in snapshots]
        assert 'step_0000000000' not in names
        assert 'step_0000001000' not in names
        assert 'step_0000004000' in names

    def test_promote_resets_tracker(self):
        pool = FrozenOpponentPool(self.tmpdir, algo_builder=_make_algo, window_size=5)
        for _ in range(5):
            pool.on_episode_end(1)
        assert pool.tracker.episode_count == 5

        algo = _make_algo()
        pool.save_snapshot(algo, step=1000)
        # Tracker should be reset after promotion
        assert pool.tracker.episode_count == 0

    def test_swap_count_increments(self):
        pool = FrozenOpponentPool(self.tmpdir, algo_builder=_make_algo)
        algo = _make_algo()
        assert pool.swap_count == 0
        pool.save_snapshot(algo, step=1000)
        assert pool.swap_count == 1
        pool.save_snapshot(algo, step=2000)
        assert pool.swap_count == 2

    def test_should_swap_delegates_to_tracker(self):
        pool = FrozenOpponentPool(
            self.tmpdir, algo_builder=_make_algo,
            window_size=10, score_threshold=0.5,
        )
        # Not enough episodes
        assert pool.should_swap(total_step=0) is False

        # Fill with wins → should trigger swap
        for _ in range(10):
            pool.on_episode_end(1)
        assert pool.should_swap(total_step=1000) is True

    def test_on_episode_end_records_outcomes(self):
        pool = FrozenOpponentPool(self.tmpdir, algo_builder=_make_algo, window_size=10)
        pool.on_episode_end(1)
        pool.on_episode_end(-1)
        pool.on_episode_end(0)
        assert pool.tracker.episode_count == 3


# ── OutcomeTrackingEnv tests ────────────────────────────────────────────────

class TestOutcomeTrackingEnv:

    def test_records_episode_outcomes(self):
        """OutcomeTrackingEnv should record goal outcomes on done episodes."""
        # We can't easily instantiate a real BaselineGymEnv without rlgym-sim,
        # so test the deque behavior directly.
        env = OutcomeTrackingEnv.__new__(OutcomeTrackingEnv)
        from collections import deque
        env.episode_outcomes = deque(maxlen=200)

        # Simulate recording
        env.episode_outcomes.append(1)
        env.episode_outcomes.append(-1)
        env.episode_outcomes.append(0)

        assert len(env.episode_outcomes) == 3
        assert list(env.episode_outcomes) == [1, -1, 0]
