"""
Evaluation Pipeline Integration Tests
=======================================
Tests the full eval pipeline (config → checkpoint → worker → results → convergence)
using DummyEnv so no rlgym-sim is needed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO))

from encoder import SharedTransformerEncoder, D_MODEL
from policy_head import StochasticPolicyHead
from training.evaluation.eval_config import EvalConfig
from training.evaluation.eval_worker import run_eval_worker, _episode_seed, _load_env_class
from training.evaluation.sim_eval import SimEvaluationHook


DUMMY_ENV = 'training.environments.dummy_env.DummyEnv'


def _make_config(tmpdir: str, **overrides) -> dict:
    """Build a minimal config dict for testing."""
    cfg = {
        'eval_interval': 1000,
        't_window': 8,
        'env_class': DUMMY_ENV,
        'evaluation': {
            'params': {
                'episodes_per_tier': 3,
                'episode_timeout_steps': 100,
                'checkpoint_dir': tmpdir,
                'skill_target_tier': 'Rookie',
                'skill_target_win_rate': 0.0,  # low threshold for testing
                'tier_opponents': {'Rookie': None},
            },
        },
    }
    cfg.update(overrides)
    return cfg


def _save_dummy_checkpoint(path: str, step: int = 5000) -> str:
    """Save a random-weights checkpoint and return its path."""
    encoder = SharedTransformerEncoder(d_model=D_MODEL)
    policy = StochasticPolicyHead(d_model=D_MODEL)
    ckpt_path = Path(path) / 'eval_checkpoint.pt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'policy': policy.state_dict(),
        'step': step,
        'run_id': 'test_run',
        'seed': 42,
        'axis_costs': {'1_env_steps': step},
        'intervention': 'baseline',
        'obs_dim': 800,
    }, str(ckpt_path))
    return str(ckpt_path)


class _FakeAlgo:
    """Minimal stand-in for Algorithm with encoder + policy."""

    def __init__(self):
        self.encoder = SharedTransformerEncoder(d_model=D_MODEL)
        self.policy = StochasticPolicyHead(d_model=D_MODEL)


def _make_dummy_vecenv(config: dict):
    """Create a small SubprocVecEnv with DummyEnv for testing."""
    from train import SubprocVecEnv
    return SubprocVecEnv(
        num_envs=2,
        t_window=config.get('t_window', 8),
        reward_type='sparse',
        env_class=DUMMY_ENV,
    )


# ── EvalConfig tests ──────────────────────────────────────────────────────


class TestEvalConfig:

    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.eval_interval == 50_000
        assert cfg.env_class is None
        assert 'Rookie' in cfg.tier_opponents

    def test_from_config_reads_env_class(self):
        cfg = EvalConfig.from_config({
            'env_class': DUMMY_ENV,
            't_window': 4,
        })
        assert cfg.env_class == DUMMY_ENV
        assert cfg.t_window == 4

    def test_from_config_reads_eval_params(self):
        cfg = EvalConfig.from_config({
            'evaluation': {
                'params': {
                    'episodes_per_tier': 50,
                    'episode_timeout_steps': 200,
                    'skill_target_win_rate': 0.75,
                },
            },
        })
        assert cfg.episodes_per_tier == 50
        assert cfg.episode_timeout_steps == 200
        assert cfg.skill_target_win_rate == 0.75

    def test_tier_order(self):
        assert EvalConfig.TIER_ORDER == ['Beginner', 'Rookie', 'Pro', 'Allstar']


# ── eval_worker tests ─────────────────────────────────────────────────────


class TestEvalWorker:

    def test_episode_seed_deterministic(self):
        s1 = _episode_seed(100, 'Rookie', 0)
        s2 = _episode_seed(100, 'Rookie', 0)
        assert s1 == s2

    def test_episode_seed_varies(self):
        s1 = _episode_seed(100, 'Rookie', 0)
        s2 = _episode_seed(100, 'Rookie', 1)
        s3 = _episode_seed(100, 'Beginner', 0)
        assert s1 != s2
        assert s1 != s3

    def test_load_env_class_dummy(self):
        from training.environments.dummy_env import DummyEnv
        cls = _load_env_class(DUMMY_ENV)
        assert cls is DummyEnv

    def test_load_env_class_none_fallback(self):
        # None should attempt to load BaselineGymEnv — may fail without
        # rlgym-sim, which is fine; we just test the import path works
        try:
            cls = _load_env_class(None)
            assert cls.__name__ == 'BaselineGymEnv'
        except ImportError:
            pytest.skip('rlgym-sim not installed')

    def test_worker_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            ckpt_path = _save_dummy_checkpoint(
                str(Path(tmpdir) / 'step_0000005000'))
            result_path = str(Path(tmpdir) / 'eval_results.json')
            error_log = str(Path(tmpdir) / 'eval_errors.log')

            cfg = EvalConfig.from_config(config)
            run_eval_worker(ckpt_path, result_path, error_log, asdict(cfg))

            # Results file must exist
            assert Path(result_path).exists()

            results = json.loads(Path(result_path).read_text())
            assert results['checkpoint_step'] == 5000
            assert 'eval_wall_time' in results
            assert 'tiers' in results
            assert 'Rookie' in results['tiers']

            rookie = results['tiers']['Rookie']
            assert 'win_rate' in rookie
            assert 'loss_rate' in rookie
            assert 'timeout_rate' in rookie
            assert 'mean_score' in rookie
            assert rookie['episodes'] == 3

            # Rates should sum to ~1.0
            total = rookie['win_rate'] + rookie['loss_rate'] + rookie['timeout_rate']
            assert abs(total - 1.0) < 1e-6


# ── SimEvaluationHook tests ───────────────────────────────────────────────


class TestSimEvaluationHook:

    def test_default_params(self):
        params = SimEvaluationHook.default_params()
        assert 'episodes_per_tier' in params
        assert 'env_class' in params

    def test_should_evaluate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = SimEvaluationHook(_make_config(tmpdir))
            assert not hook.should_evaluate(0)
            assert hook.should_evaluate(1000)
            assert hook.should_evaluate(5000)

    def test_check_convergence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = SimEvaluationHook(_make_config(
                tmpdir,
                evaluation={'params': {
                    'skill_target_win_rate': 0.60,
                    'skill_target_tier': 'Rookie',
                    'checkpoint_dir': tmpdir,
                    'tier_opponents': {'Rookie': None},
                }},
            ))
            assert not hook.check_convergence(
                {'tiers': {'Rookie': {'win_rate': 0.5}}})
            assert hook.check_convergence(
                {'tiers': {'Rookie': {'win_rate': 0.6}}})
            assert hook.check_convergence(
                {'tiers': {'Rookie': {'win_rate': 0.8}}})

    def test_evaluate_sync(self):
        """Synchronous evaluate() — full in-process run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            hook = SimEvaluationHook(config)
            algo = _FakeAlgo()

            results = hook.evaluate(algo, step=1000)

            assert not results.get('error')
            assert results['checkpoint_step'] == 1000
            assert 'Rookie' in results['tiers']

    def test_format_metrics(self):
        results = {
            'tiers': {
                'Rookie': {
                    'win_rate': 0.5,
                    'mean_score': 0.1,
                    'loss_rate': 0.3,
                    'timeout_rate': 0.2,
                    'episodes': 100,
                },
            },
            'axis_costs': {'1_env_steps': 50000},
            'eval_wall_time': 42.0,
            'convergence_reached': False,
        }
        metrics = SimEvaluationHook.format_metrics(results)
        assert metrics['eval/Rookie/win_rate'] == 0.5
        assert metrics['eval/Rookie/mean_score'] == 0.1
        assert 'eval/Rookie/episodes' not in metrics  # excluded
        assert metrics['axis/1_env_steps'] == 50000
        assert metrics['eval/wall_time'] == 42.0
        assert metrics['eval/convergence_reached'] == 0

    def test_format_metrics_error(self):
        assert SimEvaluationHook.format_metrics({'error': True}) == {}

    def test_convergence_file_written(self):
        """When win_rate >= target, convergence.json should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use 0.0 threshold so convergence is guaranteed
            config = _make_config(tmpdir)
            hook = SimEvaluationHook(config)
            algo = _FakeAlgo()

            results = hook.evaluate(algo, step=2000)
            conv_path = Path(tmpdir) / 'convergence.json'

            if results.get('convergence_reached'):
                assert conv_path.exists()
                conv = json.loads(conv_path.read_text())
                assert conv['step'] == 2000
                assert conv['tier'] == 'Rookie'

    def test_evaluate_inline(self):
        """Inline evaluation using SubprocVecEnv workers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            hook = SimEvaluationHook(config)
            algo = _FakeAlgo()

            envs = _make_dummy_vecenv(config)
            try:
                results = hook.evaluate_inline(
                    algo, envs, step=3000, device='cpu')

                assert results['checkpoint_step'] == 3000
                assert 'eval_wall_time' in results
                assert 'tiers' in results
                assert 'Rookie' in results['tiers']

                rookie = results['tiers']['Rookie']
                assert 'win_rate' in rookie
                assert 'loss_rate' in rookie
                assert 'timeout_rate' in rookie
                assert 'mean_score' in rookie
                assert rookie['episodes'] == 3

                # Rates should sum to ~1.0
                total = (rookie['win_rate'] + rookie['loss_rate']
                         + rookie['timeout_rate'])
                assert abs(total - 1.0) < 1e-6

                # Should have updated _last_eval_step
                assert hook._last_eval_step == 3000
            finally:
                envs.close()

    def test_evaluate_inline_convergence(self):
        """Inline eval with 0.0 threshold should report convergence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(tmpdir)
            hook = SimEvaluationHook(config)
            algo = _FakeAlgo()

            envs = _make_dummy_vecenv(config)
            try:
                results = hook.evaluate_inline(
                    algo, envs, step=5000, device='cpu')
                # threshold is 0.0, so any result converges
                assert hook.check_convergence(results)
            finally:
                envs.close()
