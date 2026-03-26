"""
spectral_dataset.py
===================
Loads parsed replay .npz files and converts token observations to SE(3)
spectral field observations (105-dim) for offline RL training.

Each .npz contains:
    tokens  : (T, 2, 10, 10)  float32 — pre-normalized game state tokens
    actions : (T, 2, 8)       float32 — continuous+binary actions
    rewards : (T, 2)          float32 — sparse: +1 goal, -1 concede
    dones   : (T, 2)          bool    — episode boundaries

The spectral conversion uses assemble_scene() to produce a 105-dim
observation vector per frame, encoding SE(3) field coefficients,
covariances, wall features, interaction matrix, and game state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from d3rlpy.dataset import MDPDataset

from rlbot.env.scene import assemble_scene, OBS_DIM


def load_spectral_dataset(
    replay_dir: str | Path,
    min_episode_len: int = 2,
) -> MDPDataset:
    """Load replay .npz files and return a d3rlpy MDPDataset with spectral obs.

    Parameters
    ----------
    replay_dir : path to directory containing .npz replay files
    min_episode_len : skip episodes shorter than this

    Returns
    -------
    MDPDataset with obs=(N, 105), actions=(N, 8), rewards=(N,), terminals=(N,)
    """
    replay_dir = Path(replay_dir)
    npz_files = sorted(replay_dir.glob("*.npz"))

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_terminals: list[np.ndarray] = []

    total_eps = 0
    skipped = 0

    for path in npz_files:
        data = np.load(path)

        if "actions" not in data:
            skipped += 1
            continue

        tokens = data["tokens"]    # (T, 2, 10, 10)
        actions = data["actions"]  # (T, 2, 8)
        rewards = data["rewards"]  # (T, 2)
        dones = data["dones"]      # (T, 2)
        T = len(tokens)

        for player in (0, 1):
            ep_start = 0
            for t in range(T):
                if dones[t, player]:
                    ep_len = t - ep_start + 1
                    if ep_len >= min_episode_len:
                        _append_episode(
                            tokens[:, player], actions[:, player],
                            rewards[:, player], ep_start, t + 1,
                            all_obs, all_actions, all_rewards, all_terminals,
                        )
                        total_eps += 1
                    ep_start = t + 1

            # Final episode (no trailing done=True)
            if ep_start < T and (T - ep_start) >= min_episode_len:
                _append_episode(
                    tokens[:, player], actions[:, player],
                    rewards[:, player], ep_start, T,
                    all_obs, all_actions, all_rewards, all_terminals,
                    force_terminal_last=True,
                )
                total_eps += 1

    if total_eps == 0:
        raise ValueError(
            f"No valid episodes found in {replay_dir} "
            f"({len(npz_files)} files, {skipped} skipped)"
        )

    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards_arr = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    print(
        f"Loaded {total_eps} episodes, {len(obs)} transitions "
        f"from {len(npz_files) - skipped} files"
        + (f" ({skipped} skipped)" if skipped else "")
    )

    return MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards_arr,
        terminals=terminals,
    )


def _append_episode(
    tokens_player: np.ndarray,   # (T, 10, 10) for one player
    actions_player: np.ndarray,  # (T, 8) for one player
    rewards_player: np.ndarray,  # (T,) for one player
    start: int,
    end: int,
    all_obs: list[np.ndarray],
    all_actions: list[np.ndarray],
    all_rewards: list[np.ndarray],
    all_terminals: list[np.ndarray],
    force_terminal_last: bool = False,
) -> None:
    """Convert a slice of tokens to spectral obs and append to accumulators."""
    ep_len = end - start
    obs = np.empty((ep_len, OBS_DIM), dtype=np.float32)
    acts = actions_player[start:end].astype(np.float32)
    rews = rewards_player[start:end].astype(np.float32)
    terms = np.zeros(ep_len, dtype=np.float32)

    for i, t in enumerate(range(start, end)):
        tok = torch.from_numpy(tokens_player[t].astype(np.float32))
        obs[i] = assemble_scene(tok).numpy()

    # Mark terminal
    terms[-1] = 1.0 if force_terminal_last else 1.0

    all_obs.append(obs)
    all_actions.append(acts)
    all_rewards.append(rews)
    all_terminals.append(terms)
