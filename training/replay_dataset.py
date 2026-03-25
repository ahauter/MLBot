"""
replay_dataset.py
=================
Loads parsed replay .npz files into the SequenceReplayBuffer for offline RL.

Each .npz is segmented into episodes using the `dones` array (split on True).
Each episode is added as a trajectory of (obs, action, reward, done) tuples,
matching the format expected by SequenceReplayBuffer.add_episode().

Files missing the 'actions' key (old format) are silently skipped — re-parse
them with collect_replays.py --no-resume to regenerate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_buffer import SequenceReplayBuffer


def load_replays_into_buffer(
    parsed_dir: str | Path,
    buffer: SequenceReplayBuffer,
    min_episode_len: int = 2,
) -> int:
    """
    Load all replay npz files from `parsed_dir` into `buffer`.

    Segments each replay into episodes by splitting on done=True per player,
    then calls buffer.add_episode() for each valid episode.

    Parameters
    ----------
    parsed_dir      : directory containing .npz files from replay_sampler.py
    buffer          : SequenceReplayBuffer to fill
    min_episode_len : skip episodes shorter than this (e.g. stray frames)

    Returns
    -------
    Total number of episodes added.
    """
    parsed_dir = Path(parsed_dir)
    npz_files  = sorted(parsed_dir.glob('*.npz'))

    total_eps   = 0
    total_files = 0
    skipped     = 0

    for path in npz_files:
        data = np.load(path)

        if 'actions' not in data:
            skipped += 1
            continue

        tokens  = data['tokens']   # (T, 2, N, F)
        actions = data['actions']  # (T, 2, 8)
        rewards = data['rewards']  # (T, 2)
        dones   = data['dones']    # (T, 2) bool
        T = len(tokens)

        # Each player perspective is an independent trajectory
        for player in (0, 1):
            ep_start = 0
            for t in range(T):
                if dones[t, player]:
                    ep_len = t - ep_start + 1
                    if ep_len >= min_episode_len:
                        trajectory = [
                            (
                                tokens[i, player],          # (N, F) float32
                                actions[i, player],         # (8,)   float32
                                float(rewards[i, player]),  # scalar
                                bool(dones[i, player]),     # done
                            )
                            for i in range(ep_start, t + 1)
                        ]
                        buffer.add_episode(trajectory)
                        total_eps += 1
                    ep_start = t + 1

        total_files += 1

    print(
        f'Loaded {total_eps} episodes from {total_files} replay files'
        + (f' ({skipped} skipped — missing actions, re-parse to fix)' if skipped else '')
    )
    return total_eps
