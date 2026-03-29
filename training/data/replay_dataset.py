"""
replay_dataset.py
=================
Loads parsed replay .npz files into a d3rlpy replay buffer for offline RL
or online buffer seeding.

Each .npz is segmented into episodes using the ``dones`` array (split on True).
Observations are frame-stacked from the raw (T, N, F) tokens to match the
flat (T_WINDOW * N * F,) format used by BaselineGymEnv.

Files missing the 'actions' key (old format) are silently skipped — re-parse
them with collect_replays.py --no-resume to regenerate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_replays_into_buffer(
    parsed_dir: str | Path,
    buffer,
    t_window: int = 8,
    min_episode_len: int = 2,
) -> int:
    """
    Load all replay npz files from ``parsed_dir`` into a d3rlpy replay buffer.

    Segments each replay into episodes by splitting on done=True per player,
    then appends transitions using d3rlpy's ``buffer.append()`` /
    ``buffer.clip_episode()`` interface.

    Parameters
    ----------
    parsed_dir      : directory containing .npz files from replay_sampler.py
    buffer          : d3rlpy FIFOReplayBuffer (supports append/clip_episode)
    t_window        : number of frames to stack (must match training env)
    min_episode_len : skip episodes shorter than this

    Returns
    -------
    Total number of episodes added.
    """
    parsed_dir = Path(parsed_dir)
    npz_files = sorted(parsed_dir.glob('*.npz'))

    total_eps = 0
    total_files = 0
    skipped = 0

    for path in npz_files:
        data = np.load(path)

        if 'actions' not in data:
            skipped += 1
            continue

        tokens = data['tokens']    # (T, 2, N, F)
        actions = data['actions']  # (T, 2, 8)
        rewards = data['rewards']  # (T, 2)
        dones = data['dones']      # (T, 2) bool
        T = len(tokens)

        # Each player perspective is an independent trajectory
        for player in (0, 1):
            ep_start = 0
            for t in range(T):
                if dones[t, player]:
                    ep_len = t - ep_start + 1
                    if ep_len >= min_episode_len:
                        _flush_episode(
                            buffer, tokens, actions, rewards, dones,
                            player, ep_start, t + 1, t_window,
                        )
                        total_eps += 1
                    ep_start = t + 1

            # Flush the final episode (frames after the last done signal)
            if ep_start < T and (T - ep_start) >= min_episode_len:
                _flush_episode(
                    buffer, tokens, actions, rewards, dones,
                    player, ep_start, T, t_window,
                )
                total_eps += 1

        total_files += 1

    print(
        f'Loaded {total_eps} episodes from {total_files} replay files'
        + (f' ({skipped} skipped — missing actions, re-parse to fix)' if skipped else '')
    )
    return total_eps


def _flush_episode(
    buffer, tokens, actions, rewards, dones,
    player: int, start: int, end: int, t_window: int,
) -> None:
    """
    Frame-stack and flush one episode segment to the d3rlpy buffer.

    Each transition's observation is a flat vector matching BaselineGymEnv's
    (T_WINDOW * N * F,) format.
    """
    N, F = tokens.shape[2], tokens.shape[3]

    for i in range(start, end):
        # Build frame-stacked observation: take t_window frames ending at i
        frames = []
        for t_idx in range(i - t_window + 1, i + 1):
            if t_idx < start:
                frames.append(tokens[start, player])  # pad with first frame
            else:
                frames.append(tokens[t_idx, player])
        stacked = np.stack(frames, axis=0)  # (T_WINDOW, N, F)
        obs_flat = stacked.ravel().astype(np.float32)

        action = actions[i, player].astype(np.float32)
        reward = float(rewards[i, player])

        buffer.append(obs_flat, action, reward)

    # Clip episode — True if it ended naturally (done), False if truncated
    is_terminal = bool(dones[end - 1, player]) if end > 0 else False
    buffer.clip_episode(is_terminal)
