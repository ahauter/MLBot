"""
Tests for the experiment infrastructure:
  - ExperimentConfig YAML round-trip
  - AxisTracker accumulation and serialisation
  - to_train_configs() output shape for sweeps and single conditions
  - replay_dataset load_replays_into_buffer with mock buffer
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))


# ── AxisTracker tests ─────────────────────────────────────────────────────────

class TestAxisTracker:
    def test_initial_values(self):
        from axis_tracker import AxisTracker
        t = AxisTracker()
        assert t.sim_steps == 0
        assert t.replays_loaded == 0
        assert t.labels_consumed == 0
        assert t.reward_components == 0
        assert t.pretrain_gpu_hours == 0.0

    def test_record_sim_steps(self):
        from axis_tracker import AxisTracker
        t = AxisTracker()
        t.record_sim_steps(100)
        t.record_sim_steps(50)
        assert t.sim_steps == 150

    def test_record_replays(self):
        from axis_tracker import AxisTracker
        t = AxisTracker()
        t.record_replays(10)
        assert t.replays_loaded == 10

    def test_set_reward_components(self):
        from axis_tracker import AxisTracker
        t = AxisTracker()
        t.set_reward_components(5)
        assert t.reward_components == 5
        t.set_reward_components(3)
        assert t.reward_components == 3  # set, not accumulated

    def test_as_dict(self):
        from axis_tracker import AxisTracker
        t = AxisTracker()
        t.record_sim_steps(1000)
        t.record_replays(5)
        t.set_reward_components(3)
        d = t.as_dict()
        assert d['resources_consumed/axis1_sim_steps'] == 1000
        assert d['resources_consumed/axis2_replays_loaded'] == 5
        assert d['resources_consumed/axis4_reward_components'] == 3

    def test_as_dict_custom_prefix(self):
        from axis_tracker import AxisTracker
        t = AxisTracker()
        d = t.as_dict(prefix='test/')
        assert 'test/axis1_sim_steps' in d

    def test_log_with_none_wandb(self):
        from axis_tracker import AxisTracker
        t = AxisTracker()
        t.log(None, step=0)  # should not raise


# ── ExperimentConfig tests ────────────────────────────────────────────────────

class TestExperimentConfig:
    def _make_config(self):
        from experiment_config import ExperimentConfig, AxisBudget
        return ExperimentConfig(
            name='test_experiment',
            description='A test',
            intervention='none',
            base_config={'algo': 'AWAC', 'reward_type': 'sparse', 'total_steps': 1000},
            budget=AxisBudget(sim_steps=1000),
            seeds=[0, 1],
            wandb_tags=['test'],
        )

    def test_yaml_roundtrip(self):
        from experiment_config import ExperimentConfig
        cfg = self._make_config()
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            cfg.to_yaml(f.name)
            loaded = ExperimentConfig.from_yaml(f.name)
        assert loaded.name == cfg.name
        assert loaded.intervention == cfg.intervention
        assert loaded.seeds == cfg.seeds
        assert loaded.budget.sim_steps == cfg.budget.sim_steps

    def test_to_train_configs_single_condition(self):
        cfg = self._make_config()
        configs = cfg.to_train_configs()
        # 1 condition × 2 seeds = 2 configs
        assert len(configs) == 2
        cond_name, seed, cfg_dict = configs[0]
        assert cond_name == 'baseline'
        assert seed == 0
        assert cfg_dict['algo'] == 'AWAC'
        assert cfg_dict['wandb_group'] == 'test_experiment/baseline'
        assert '_budget' in cfg_dict

    def test_to_train_configs_sweep(self):
        from experiment_config import ExperimentConfig, AxisBudget, SweepPoint
        cfg = ExperimentConfig(
            name='sweep_test',
            description='A sweep',
            intervention='dense_reward',
            base_config={'algo': 'AWAC', 'reward_type': 'dense'},
            budget=AxisBudget(sim_steps=5000),
            seeds=[0, 1, 2],
            sweep=[
                SweepPoint(name='low', overrides={'total_steps': 1000}),
                SweepPoint(name='high', overrides={'total_steps': 5000}),
            ],
            wandb_tags=['sweep'],
        )
        configs = cfg.to_train_configs()
        # 2 conditions × 3 seeds = 6 configs
        assert len(configs) == 6
        cond_names = [c[0] for c in configs]
        assert cond_names.count('low') == 3
        assert cond_names.count('high') == 3

    def test_sweep_budget_overrides(self):
        from experiment_config import ExperimentConfig, AxisBudget, SweepPoint
        cfg = ExperimentConfig(
            name='budget_test',
            description='Test budget overrides',
            intervention='test',
            base_config={'algo': 'AWAC'},
            budget=AxisBudget(sim_steps=1000, num_replays=0),
            seeds=[0],
            sweep=[
                SweepPoint(
                    name='with_data',
                    overrides={'replay_seed_dir': '/some/path'},
                    budget_overrides={'num_replays': 500},
                ),
            ],
        )
        configs = cfg.to_train_configs()
        assert len(configs) == 1
        _, _, cfg_dict = configs[0]
        assert cfg_dict['_budget']['num_replays'] == 500

    def test_wandb_tags_include_intervention(self):
        cfg = self._make_config()
        configs = cfg.to_train_configs()
        _, _, cfg_dict = configs[0]
        assert 'none' in cfg_dict['wandb_tags']

    def test_from_yaml_real_configs(self):
        """Verify all experiment YAML files parse without error."""
        from experiment_config import ExperimentConfig
        exp_dir = Path(__file__).parent.parent / 'experiments'
        yaml_files = list(exp_dir.glob('*.yaml'))
        assert len(yaml_files) > 0, "No experiment YAML files found"
        for yf in yaml_files:
            cfg = ExperimentConfig.from_yaml(yf)
            configs = cfg.to_train_configs()
            assert len(configs) > 0, f"No configs generated from {yf.name}"


# ── replay_dataset tests ─────────────────────────────────────────────────────

class _MockBuffer:
    """Minimal mock of d3rlpy's replay buffer interface."""
    def __init__(self):
        self.transitions = []
        self.episodes = []
        self._current_ep = []

    def append(self, obs, action, reward):
        self._current_ep.append((obs.copy(), action.copy(), reward))

    def clip_episode(self, terminal: bool):
        self.episodes.append((list(self._current_ep), terminal))
        self._current_ep = []

    @property
    def transition_count(self):
        return sum(len(ep) for ep, _ in self.episodes)


class TestReplayDataset:
    def _make_test_npz(self, tmpdir: Path, n_frames: int = 20, has_goal: bool = True):
        """Create a synthetic .npz replay file."""
        N_TOKENS, TOKEN_FEATURES = 10, 10
        tokens = np.random.randn(n_frames, 2, N_TOKENS, TOKEN_FEATURES).astype(np.float32)
        actions = np.random.uniform(-1, 1, (n_frames, 2, 8)).astype(np.float32)
        rewards = np.zeros((n_frames, 2), dtype=np.float32)
        dones = np.zeros((n_frames, 2), dtype=bool)

        if has_goal:
            # Goal at frame 15 for player 0
            goal_frame = min(15, n_frames - 1)
            rewards[goal_frame, 0] = 1.0
            rewards[goal_frame, 1] = -1.0
            dones[goal_frame, :] = True

        path = tmpdir / 'test_replay.npz'
        np.savez(path, tokens=tokens, actions=actions, rewards=rewards, dones=dones)
        return path

    def test_load_into_mock_buffer(self, tmp_path):
        from replay_dataset import load_replays_into_buffer
        self._make_test_npz(tmp_path)
        buf = _MockBuffer()
        n_eps = load_replays_into_buffer(tmp_path, buf, t_window=8)
        assert n_eps > 0
        assert buf.transition_count > 0

    def test_frame_stacking_shape(self, tmp_path):
        from replay_dataset import load_replays_into_buffer
        self._make_test_npz(tmp_path, n_frames=20)
        buf = _MockBuffer()
        load_replays_into_buffer(tmp_path, buf, t_window=8)
        # Each observation should be flat (T_WINDOW * N * F,) = (800,)
        for ep_transitions, _ in buf.episodes:
            for obs, action, reward in ep_transitions:
                assert obs.shape == (8 * 10 * 10,), f"Bad obs shape: {obs.shape}"
                assert action.shape == (8,)

    def test_skip_missing_actions(self, tmp_path):
        from replay_dataset import load_replays_into_buffer
        # Create npz without actions key
        np.savez(tmp_path / 'bad.npz', tokens=np.zeros((5, 2, 10, 10)))
        buf = _MockBuffer()
        n_eps = load_replays_into_buffer(tmp_path, buf, t_window=8)
        assert n_eps == 0

    def test_empty_dir(self, tmp_path):
        from replay_dataset import load_replays_into_buffer
        buf = _MockBuffer()
        n_eps = load_replays_into_buffer(tmp_path, buf, t_window=8)
        assert n_eps == 0
