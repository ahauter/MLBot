#!/usr/bin/env python3
"""
Live-Play SAC Training Loop
============================
Watches for game transcript .npz files written by MLBot and HumanProxyBot,
loads them into a SAC replay buffer, runs gradient updates, and saves
updated weights for the bot to load on next game launch.

Usage
-----
    # Start the training loop (watches for transcripts):
    python training/live_play_train.py

    # With a checkpoint to resume from:
    python training/live_play_train.py --checkpoint models/sac

    # Custom settings:
    python training/live_play_train.py --updates-per-game 2000 --batch-size 512

Flow
----
    1. Create/load SACAlgorithm
    2. Watch training/transcripts/ for unprocessed .npz files
    3. Load transcript → frame-stack → add to replay buffer (both perspectives)
    4. Run N gradient updates
    5. Save encoder.pt + policy.pt to models/ (bot reloads on next launch)
    6. Move processed transcripts to training/transcripts/processed/
    7. Loop
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO))

from encoder import N_TOKENS, TOKEN_FEATURES, T_WINDOW


# ── Transcript loading & frame-stacking ───────────────────────────────────

def load_transcript(path: Path) -> dict:
    """Load a .npz transcript file.

    Returns dict with keys: tokens, actions, rewards, dones.
    """
    data = np.load(path)
    return {
        'tokens': data['tokens'],     # (T, N_TOKENS, TOKEN_FEATURES)
        'actions': data['actions'],    # (T, 8)
        'rewards': data['rewards'],    # (T,)
        'dones': data['dones'],        # (T,)
    }


def frame_stack_tokens(
    tokens: np.ndarray,
    t_window: int = T_WINDOW,
) -> np.ndarray:
    """Convert (T, N_TOKENS, TOKEN_FEATURES) tokens to (T, obs_dim) flat observations.

    Applies the same sliding-window logic as src/bot.py:77-83:
    for tick t, obs = flatten(tokens[max(0, t-t_window+1) : t+1]).
    First t_window-1 ticks are padded by repeating the first frame.

    Returns (T, t_window * N_TOKENS * TOKEN_FEATURES) float32 array.
    """
    T = tokens.shape[0]
    obs_dim = t_window * N_TOKENS * TOKEN_FEATURES
    obs = np.zeros((T, obs_dim), dtype=np.float32)

    # Build padded sequence: prepend (t_window-1) copies of first frame
    pad = np.tile(tokens[0:1], (t_window - 1, 1, 1))  # (t_window-1, N, F)
    padded = np.concatenate([pad, tokens], axis=0)       # (T + t_window - 1, N, F)

    for t in range(T):
        window = padded[t:t + t_window]  # (t_window, N, F)
        obs[t] = window.ravel()

    return obs


def transcript_to_transitions(transcript: dict, t_window: int = T_WINDOW) -> dict:
    """Convert a transcript to SAC-compatible (obs, action, reward, next_obs, done) arrays.

    Returns dict with numpy arrays ready for ReplayBuffer.add_batch().
    """
    tokens = transcript['tokens']
    actions = transcript['actions']
    rewards = transcript['rewards']
    dones = transcript['dones']

    T = tokens.shape[0]
    if T < 2:
        return None

    obs_all = frame_stack_tokens(tokens, t_window)  # (T, obs_dim)

    # Transitions are (obs_t, action_t, reward_t, obs_{t+1}, done_t)
    return {
        'obs': obs_all[:-1],          # (T-1, obs_dim)
        'actions': actions[:-1],      # (T-1, 8)
        'rewards': rewards[:-1],      # (T-1,)
        'next_obs': obs_all[1:],      # (T-1, obs_dim)
        'dones': dones[:-1].astype(np.float32),  # (T-1,)
    }


# ── Main training loop ───────────────────────────────────────────────────

def find_unprocessed(watch_dir: Path) -> list[Path]:
    """Find all .npz files in watch_dir (not in processed/ subfolder)."""
    return sorted(watch_dir.glob('*.npz'))


def process_transcript(
    path: Path,
    algorithm,
    t_window: int,
    processed_dir: Path,
) -> int:
    """Load a transcript, add to buffer, move to processed. Returns transitions added."""
    transcript = load_transcript(path)
    transitions = transcript_to_transitions(transcript, t_window)
    if transitions is None:
        print(f'  [skip] {path.name} — too short')
        path.rename(processed_dir / path.name)
        return 0

    n = transitions['obs'].shape[0]
    algorithm.buffer.add_batch(
        obs=transitions['obs'],
        actions=transitions['actions'],
        rewards=transitions['rewards'],
        next_obs=transitions['next_obs'],
        dones=transitions['dones'],
    )
    path.rename(processed_dir / path.name)
    return n


def train_loop(
    checkpoint_dir: Path,
    watch_dir: Path,
    model_dir: Path,
    updates_per_game: int,
    batch_size: int,
    t_window: int,
    poll_interval: float = 5.0,
):
    """Main training loop. Watches for transcripts, trains SAC, saves weights."""
    from training.algorithms.sac import SACAlgorithm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[live-train] device={device}, t_window={t_window}')
    print(f'[live-train] Watching {watch_dir} for transcripts...')

    # Create SAC algorithm
    config = {
        'device': device,
        't_window': t_window,
        'algorithm': {'params': {'batch_size': batch_size}},
    }
    algorithm = SACAlgorithm(config)

    # Load checkpoint if exists
    ckpt_file = checkpoint_dir / 'checkpoint.pt'
    if ckpt_file.exists():
        algorithm.load_checkpoint(checkpoint_dir)
        print(f'[live-train] Loaded checkpoint from {checkpoint_dir}')
    else:
        print('[live-train] No checkpoint found, starting fresh.')

    # Ensure directories exist
    watch_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = watch_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    games_processed = 0
    total_updates = 0
    recent_losses = deque(maxlen=50)

    print('[live-train] Ready. Play a game against the bot, then check back here.')
    print()

    try:
        while True:
            # Find unprocessed transcripts
            files = find_unprocessed(watch_dir)
            if not files:
                time.sleep(poll_interval)
                continue

            # Process all available transcripts
            total_transitions = 0
            for f in files:
                n = process_transcript(f, algorithm, t_window, processed_dir)
                total_transitions += n
                print(f'  Loaded {f.name}: {n} transitions')

            games_processed += len(files)
            print(f'\n[live-train] Game {games_processed}: '
                  f'{total_transitions} new transitions, '
                  f'buffer size {algorithm.buffer.size:,}')

            # Run gradient updates if buffer is large enough
            if not algorithm.should_update():
                print(f'  Buffer too small ({algorithm.buffer.size} < '
                      f'{algorithm.min_buffer_size}), waiting for more data.')
                continue

            # Scale updates to transcript size (at least updates_per_game)
            n_updates = max(updates_per_game, total_transitions // batch_size)
            algorithm.updates_per_step = n_updates

            print(f'  Running {n_updates} SAC updates...')
            t0 = time.time()
            metrics = algorithm.update()
            elapsed = time.time() - t0
            total_updates += n_updates

            if metrics:
                recent_losses.append(metrics.get('critic_loss', 0.0))
                print(f'  Done in {elapsed:.1f}s '
                      f'({n_updates / elapsed:.0f} updates/sec)')
                print(f'  critic_loss={metrics["critic_loss"]:.4f}  '
                      f'actor_loss={metrics["actor_loss"]:.4f}  '
                      f'alpha={metrics["alpha"]:.4f}  '
                      f'q_mean={metrics["q_mean"]:.4f}')

            # Save deployment weights (bot loads these on next launch)
            algorithm.save_deployment_weights(model_dir)
            print(f'  Saved weights → {model_dir}/encoder.pt, policy.pt')

            # Save full checkpoint for resume
            algorithm.save_checkpoint(checkpoint_dir)
            print(f'  Saved checkpoint → {checkpoint_dir}/')

            print(f'\n  Total: {games_processed} games, {total_updates:,} updates, '
                  f'buffer {algorithm.buffer.size:,}')
            if recent_losses:
                print(f'  Recent mean critic loss: {np.mean(recent_losses):.4f}')
            print()
            print('[live-train] Waiting for next game...')

    except KeyboardInterrupt:
        print('\n[live-train] Interrupted. Saving checkpoint...')
        algorithm.save_checkpoint(checkpoint_dir)
        algorithm.save_deployment_weights(model_dir)
        print(f'[live-train] Saved. {games_processed} games, {total_updates:,} total updates.')


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Live-play SAC training: watch for game transcripts and train')
    parser.add_argument(
        '--checkpoint', default='models/sac',
        help='Directory for SAC checkpoints (default: models/sac)')
    parser.add_argument(
        '--watch-dir', default='training/transcripts',
        help='Directory to watch for .npz transcripts (default: training/transcripts)')
    parser.add_argument(
        '--model-dir', default='models',
        help='Directory to save deployment weights (default: models)')
    parser.add_argument(
        '--updates-per-game', type=int, default=1000,
        help='Minimum gradient updates per game (default: 1000)')
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='SAC minibatch size (default: 256)')
    parser.add_argument(
        '--t-window', type=int, default=T_WINDOW,
        help=f'Frame stacking window (default: {T_WINDOW})')
    parser.add_argument(
        '--poll-interval', type=float, default=5.0,
        help='Seconds between checking for new transcripts (default: 5)')
    args = parser.parse_args()

    train_loop(
        checkpoint_dir=Path(args.checkpoint),
        watch_dir=Path(args.watch_dir),
        model_dir=Path(args.model_dir),
        updates_per_game=args.updates_per_game,
        batch_size=args.batch_size,
        t_window=args.t_window,
        poll_interval=args.poll_interval,
    )


if __name__ == '__main__':
    main()
