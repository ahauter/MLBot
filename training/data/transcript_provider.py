"""
Transcript Replay Provider
==========================
ReplayProvider implementation that loads live-play game transcripts
into a SAC (or any off-policy) algorithm's replay buffer.

Transcripts are .npz files written by src/transcriber.py, containing
per-tick tokens, actions, rewards, and dones from one bot's perspective.

This provider:
  1. On startup: loads all existing transcripts in the watch directory
  2. Each collection round: checks for newly arrived transcripts and
     injects them into the algorithm's buffer

Usage (in YAML config):
    replay:
      class: training.data.transcript_provider.TranscriptReplayProvider
      params:
        transcript_dir: training/transcripts
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))

from encoder import N_TOKENS, TOKEN_FEATURES, T_WINDOW
from training.abstractions import Algorithm, ReplayProvider


# ── frame-stacking helper ─────────────────────────────────────────────────

def frame_stack_tokens(
    tokens: np.ndarray,
    t_window: int = T_WINDOW,
) -> np.ndarray:
    """Convert (T, N_TOKENS, TOKEN_FEATURES) → (T, obs_dim) flat observations.

    Applies the same sliding-window as src/bot.py: for tick t, obs is the
    concatenation of tokens[max(0, t-t_window+1) : t+1].  First frames are
    padded by repeating frame 0.
    """
    T = tokens.shape[0]
    obs_dim = t_window * N_TOKENS * TOKEN_FEATURES

    # Prepend (t_window - 1) copies of first frame for padding
    pad = np.tile(tokens[0:1], (t_window - 1, 1, 1))
    padded = np.concatenate([pad, tokens], axis=0)

    obs = np.zeros((T, obs_dim), dtype=np.float32)
    for t in range(T):
        obs[t] = padded[t:t + t_window].ravel()
    return obs


def transcript_to_transitions(
    tokens: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    t_window: int = T_WINDOW,
) -> Optional[dict]:
    """Convert a single transcript to (obs, action, reward, next_obs, done).

    Returns None if the transcript is too short.
    """
    T = tokens.shape[0]
    if T < 2:
        return None

    obs_all = frame_stack_tokens(tokens, t_window)

    return {
        'obs': obs_all[:-1],
        'actions': actions[:-1],
        'rewards': rewards[:-1],
        'next_obs': obs_all[1:],
        'dones': dones[:-1].astype(np.float32),
    }


# ── TranscriptReplayProvider ──────────────────────────────────────────────

class TranscriptReplayProvider(ReplayProvider):
    """Loads live-play transcripts into an off-policy algorithm's buffer.

    Works with any Algorithm whose buffer has an ``add_batch()`` method
    (e.g. SACAlgorithm).

    Parameters (set via YAML ``params:``)
    ----------
    transcript_dir : str
        Directory to watch for .npz transcript files.
    t_window : int
        Frame-stacking window.  Must match the encoder / bot setting.
    """

    def __init__(self, **kwargs):
        self.transcript_dir = Path(kwargs.get(
            'transcript_dir', 'training/transcripts'))
        self.t_window = int(kwargs.get('t_window', T_WINDOW))
        self.processed_dir = self.transcript_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Track what we've loaded to avoid double-loading
        self._loaded_files: set[str] = set()
        self._total_transitions: int = 0
        self._total_transcripts: int = 0
        # Keep reference to agents for on_round injection
        self._agents: list = []

    @classmethod
    def default_params(cls) -> dict:
        return {
            'transcript_dir': 'training/transcripts',
            't_window': T_WINDOW,
        }

    def load_demonstrations(self) -> Optional[list]:
        """Scan transcript_dir for .npz files.  Returns list of file paths."""
        files = sorted(self.transcript_dir.glob('*.npz'))
        if not files:
            print('[transcript-provider] No transcripts found yet.  '
                  f'Watching {self.transcript_dir}/')
            return None
        print(f'[transcript-provider] Found {len(files)} transcript(s) at startup.')
        return [str(f) for f in files]

    def seed_algorithm(self, algorithm: Algorithm, demonstrations: list) -> None:
        """Load all transcripts into the algorithm's replay buffer."""
        if demonstrations is None:
            return
        for path_str in demonstrations:
            self._load_one(Path(path_str), algorithm)

    def on_round(self, agents: list, step: int) -> None:
        """Check for new transcripts each collection round and load them."""
        self._agents = agents
        files = sorted(self.transcript_dir.glob('*.npz'))
        new_files = [f for f in files if f.name not in self._loaded_files]
        if not new_files:
            return
        for f in new_files:
            for agent in agents:
                self._load_one(f, agent)

    def _load_one(self, path: Path, algorithm: Algorithm) -> int:
        """Load a single transcript into the algorithm's buffer.

        Returns number of transitions added.
        """
        if path.name in self._loaded_files:
            return 0

        try:
            data = np.load(path)
            tokens = data['tokens']
            actions = data['actions']
            rewards = data['rewards']
            dones = data['dones']
        except Exception as e:
            print(f'[transcript-provider] Failed to load {path.name}: {e}')
            return 0

        transitions = transcript_to_transitions(
            tokens, actions, rewards, dones, self.t_window)
        if transitions is None:
            print(f'[transcript-provider] Skipping {path.name} (too short)')
            self._loaded_files.add(path.name)
            return 0

        n = transitions['obs'].shape[0]

        # Add to the algorithm's replay buffer
        if hasattr(algorithm, 'buffer') and hasattr(algorithm.buffer, 'add_batch'):
            algorithm.buffer.add_batch(**transitions)
        else:
            # Fallback: add one at a time via store_transition
            from training.abstractions import ActionResult
            for i in range(n):
                ar = ActionResult(action=transitions['actions'][i:i+1])
                algorithm.store_transition(
                    transitions['obs'][i:i+1],
                    ar,
                    transitions['rewards'][i],
                    transitions['next_obs'][i:i+1],
                    transitions['dones'][i],
                    {},
                )

        self._loaded_files.add(path.name)
        self._total_transitions += n
        self._total_transcripts += 1

        # Move to processed
        dest = self.processed_dir / path.name
        try:
            path.rename(dest)
        except OSError:
            pass  # file may still be written to

        n_goals = int(np.sum(rewards != 0))
        print(f'[transcript-provider] Loaded {path.name}: '
              f'{n} transitions ({n_goals} goals)')
        return n

    def get_metrics(self) -> dict:
        return {
            'transcripts_loaded': self._total_transcripts,
            'transcript_transitions': self._total_transitions,
        }
