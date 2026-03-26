"""
spectral_dataset.py
===================
Loads parsed replay .npz files into a d3rlpy MDPDataset.

Observations are raw flat token vectors (100-dim = 10 tokens × 10 features).
The learned SpectralEncoder inside the d3rlpy encoder handles the conversion
to SE(3) spectral fields during training (end-to-end, differentiable).

Each .npz contains:
    tokens  : (T, 2, 10, 10)  float32 — pre-normalized game state tokens
    actions : (T, 2, 8)       float32 — continuous+binary actions
    rewards : (T, 2)          float32 — sparse: +1 goal, -1 concede
    dones   : (T, 2)          bool    — episode boundaries
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from d3rlpy.dataset import MDPDataset

# Flat observation = 10 tokens × 10 features
_N_TOKENS = 10
_TOKEN_FEATURES = 10
FLAT_OBS_DIM = _N_TOKENS * _TOKEN_FEATURES  # 100


def load_spectral_dataset(
    replay_dir: str | Path,
    min_episode_len: int = 2,
) -> MDPDataset:
    """Load replay .npz files and return a d3rlpy MDPDataset with flat token obs.

    Observations are raw flat tokens (100-dim). The learned encoder converts
    these to spectral fields during training.

    Parameters
    ----------
    replay_dir : path to directory containing .npz replay files
    min_episode_len : skip episodes shorter than this

    Returns
    -------
    MDPDataset with obs=(N, 100), actions=(N, 8), rewards=(N,), terminals=(N,)
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
    actions_arr = np.concatenate(all_actions, axis=0)
    rewards_arr = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    print(
        f"Loaded {total_eps} episodes, {len(obs)} transitions "
        f"from {len(npz_files) - skipped} files"
        + (f" ({skipped} skipped)" if skipped else "")
    )

    return MDPDataset(
        observations=obs,
        actions=actions_arr,
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
    """Flatten a slice of tokens and append to accumulators."""
    ep_len = end - start

    # Flatten tokens: (ep_len, 10, 10) → (ep_len, 100)
    obs = tokens_player[start:end].reshape(ep_len, -1).astype(np.float32)
    acts = actions_player[start:end].astype(np.float32)
    rews = rewards_player[start:end].astype(np.float32)
    terms = np.zeros(ep_len, dtype=np.float32)
    terms[-1] = 1.0

    all_obs.append(obs)
    all_actions.append(acts)
    all_rewards.append(rews)
    all_terminals.append(terms)
