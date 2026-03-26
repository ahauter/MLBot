"""Tests for spectral dataset loading."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rlbot.training.spectral_dataset import load_spectral_dataset, FLAT_OBS_DIM


def _make_synthetic_replay(T: int = 50, seed: int = 42) -> dict[str, np.ndarray]:
    """Create a synthetic replay .npz with realistic token structure."""
    rng = np.random.default_rng(seed)

    tokens = np.zeros((T, 2, 10, 10), dtype=np.float32)

    for player in range(2):
        # Ball (token 0): position + velocity
        tokens[:, player, 0, :3] = rng.uniform(-0.5, 0.5, (T, 3))
        tokens[:, player, 0, 3:6] = rng.uniform(-0.3, 0.3, (T, 3))

        # Own car (token 1): position + velocity + euler + boost
        tokens[:, player, 1, :3] = rng.uniform(-0.5, 0.5, (T, 3))
        tokens[:, player, 1, 3:6] = rng.uniform(-0.3, 0.3, (T, 3))
        tokens[:, player, 1, 6:9] = rng.uniform(-1.0, 1.0, (T, 3))
        tokens[:, player, 1, 9] = rng.uniform(0, 1, T)

        # Opponent car (token 2)
        tokens[:, player, 2, :3] = rng.uniform(-0.5, 0.5, (T, 3))
        tokens[:, player, 2, 3:6] = rng.uniform(-0.3, 0.3, (T, 3))
        tokens[:, player, 2, 6:9] = rng.uniform(-1.0, 1.0, (T, 3))

        # Boost pads (tokens 3-8): fixed positions, random active states
        pad_positions = [
            [-0.875, 0.0, 0.036],
            [0.875, 0.0, 0.036],
            [-0.75, 0.8, 0.036],
            [0.75, 0.8, 0.036],
            [-0.75, -0.8, 0.036],
            [0.75, -0.8, 0.036],
        ]
        for i, pos in enumerate(pad_positions):
            tokens[:, player, 3 + i, :3] = pos
            tokens[:, player, 3 + i, 3] = rng.choice([0.0, 1.0], T)

        # Game state (token 9)
        tokens[:, player, 9, 0] = 0.0   # score_diff
        tokens[:, player, 9, 1] = 0.5   # time_rem
        tokens[:, player, 9, 2] = 0.0   # overtime

    actions = rng.uniform(-1, 1, (T, 2, 8)).astype(np.float32)
    rewards = np.zeros((T, 2), dtype=np.float32)
    dones = np.zeros((T, 2), dtype=bool)

    # Place done flags to create episodes
    mid = T // 2
    dones[mid, :] = True
    rewards[mid, 0] = 1.0
    rewards[mid, 1] = -1.0
    dones[T - 1, :] = True
    rewards[T - 1, 0] = -1.0
    rewards[T - 1, 1] = 1.0

    return {
        "tokens": tokens,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def _save_replay(dir_path: Path, data: dict, name: str = "replay_001.npz") -> Path:
    """Save a replay dict as .npz."""
    path = dir_path / name
    np.savez(path, **data)
    return path


class TestLoadSpectralDataset:
    def test_basic_loading(self, tmp_path):
        """Should load replays and produce correct shapes."""
        data = _make_synthetic_replay()
        _save_replay(tmp_path, data)

        dataset = load_spectral_dataset(tmp_path)
        # 2 players x 2 episodes = 4 episodes
        assert dataset.size() >= 2

    def test_observation_shape(self, tmp_path):
        """Each observation should be 100-dim (flat tokens)."""
        data = _make_synthetic_replay(T=20)
        _save_replay(tmp_path, data)

        dataset = load_spectral_dataset(tmp_path)
        for ep in dataset.episodes:
            assert ep.observations.shape[1] == FLAT_OBS_DIM

    def test_observations_finite(self, tmp_path):
        """All observations should be finite (no NaN/Inf)."""
        data = _make_synthetic_replay()
        _save_replay(tmp_path, data)

        dataset = load_spectral_dataset(tmp_path)
        for ep in dataset.episodes:
            obs = ep.observations
            assert np.isfinite(obs).all(), "Observations contain NaN or Inf"

    def test_actions_passthrough(self, tmp_path):
        """Actions should be passed through unchanged."""
        data = _make_synthetic_replay(T=20)
        _save_replay(tmp_path, data)

        dataset = load_spectral_dataset(tmp_path)
        for ep in dataset.episodes:
            assert ep.actions.shape[1] == 8

    def test_episode_segmentation(self, tmp_path):
        """Should create separate episodes at done boundaries."""
        data = _make_synthetic_replay(T=30)
        # Override dones for precise control
        data["dones"][:] = False
        data["dones"][9, :] = True
        data["dones"][19, :] = True
        data["dones"][29, :] = True
        _save_replay(tmp_path, data)

        dataset = load_spectral_dataset(tmp_path)
        # 3 episodes x 2 players = 6 episodes
        assert dataset.size() == 6

    def test_multiple_files(self, tmp_path):
        """Should load from multiple .npz files."""
        data1 = _make_synthetic_replay(T=20, seed=1)
        data2 = _make_synthetic_replay(T=20, seed=2)
        _save_replay(tmp_path, data1, "replay_001.npz")
        _save_replay(tmp_path, data2, "replay_002.npz")

        dataset = load_spectral_dataset(tmp_path)
        # 2 files x 2 players x 2 episodes = 8 episodes
        assert dataset.size() == 8

    def test_skips_short_episodes(self, tmp_path):
        """Episodes shorter than min_episode_len should be skipped."""
        data = _make_synthetic_replay(T=20)
        data["dones"][:] = False
        data["dones"][0, :] = True   # 1-frame episode (too short)
        data["dones"][10, :] = True  # 10-frame episode
        data["dones"][19, :] = True  # 9-frame episode
        _save_replay(tmp_path, data)

        dataset = load_spectral_dataset(tmp_path, min_episode_len=3)
        # Should get 2 episodes per player (10-frame and 9-frame), skip 1-frame
        assert dataset.size() == 4

    def test_empty_dir_raises(self, tmp_path):
        """Empty directory should raise ValueError."""
        with pytest.raises(ValueError, match="No valid episodes"):
            load_spectral_dataset(tmp_path)

    def test_skips_files_without_actions(self, tmp_path):
        """Files missing 'actions' key should be skipped."""
        # Save one valid file
        data = _make_synthetic_replay(T=20)
        _save_replay(tmp_path, data, "good.npz")

        # Save one without actions
        bad_data = {"tokens": data["tokens"], "rewards": data["rewards"], "dones": data["dones"]}
        np.savez(tmp_path / "bad.npz", **bad_data)

        dataset = load_spectral_dataset(tmp_path)
        assert dataset.size() >= 2  # only from good.npz
