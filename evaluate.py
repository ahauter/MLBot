#!/usr/bin/env python3
"""
Evaluate a Snapshot or Checkpoint
==================================
Top-level CLI for running the evaluation pipeline on an arbitrary
snapshot/checkpoint, or launching a human-play session against it.

Usage
-----
    # Evaluate a training checkpoint (encoder + policy in a directory):
    python evaluate.py --snapshot models/snapshots/step_0000050000

    # Evaluate an eval checkpoint file directly:
    python evaluate.py --checkpoint checkpoints/step_0000050000/eval_checkpoint.pt

    # Evaluate with fewer episodes (quick check):
    python evaluate.py --snapshot models/snapshots/step_0000050000 --episodes 10

    # Single tier:
    python evaluate.py --snapshot models/snapshots/step_0000050000 --tier Rookie

    # Human play against a snapshot:
    python evaluate.py --snapshot models/snapshots/step_0000050000 --human
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO))

from encoder import (
    SharedTransformerEncoder,
    D_MODEL,
    N_TOKENS,
    TOKEN_FEATURES,
    ENTITY_TYPE_IDS_1V1,
)
from policy_head import StochasticPolicyHead
from training.evaluation.eval_config import EvalConfig
from training.evaluation.eval_worker import _eval_tier


def load_from_snapshot(snap_path: str) -> dict:
    """Load encoder + policy from a snapshot directory (encoder.pt + policy.pt)."""
    snap = Path(snap_path)
    encoder_path = snap / 'encoder.pt'
    policy_path = snap / 'policy.pt'
    if not encoder_path.exists() or not policy_path.exists():
        raise FileNotFoundError(
            f'Snapshot directory {snap} must contain encoder.pt and policy.pt')
    return {
        'encoder': torch.load(str(encoder_path), map_location='cpu', weights_only=True),
        'policy': torch.load(str(policy_path), map_location='cpu', weights_only=True),
    }


def load_from_checkpoint(ckpt_path: str) -> dict:
    """Load encoder + policy from a checkpoint .pt file."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'encoder' in ckpt and 'policy' in ckpt:
        return {'encoder': ckpt['encoder'], 'policy': ckpt['policy']}
    raise KeyError(
        f'Checkpoint {ckpt_path} must contain "encoder" and "policy" keys. '
        f'Found: {list(ckpt.keys())}')


def build_models(weights: dict):
    """Instantiate encoder + policy and load weights."""
    encoder = SharedTransformerEncoder(d_model=D_MODEL)
    policy = StochasticPolicyHead(d_model=D_MODEL)
    encoder.load_state_dict(weights['encoder'])
    policy.load_state_dict(weights['policy'])
    encoder.eval()
    policy.eval()
    return encoder, policy


def run_human_play(encoder, policy, entity_ids, t_window: int):
    """Launch a human-controlled session where a human plays against the loaded model."""
    from training.environments.baseline_env import BaselineGymEnv

    env = BaselineGymEnv(t_window=t_window, max_steps=4500, reward_type='sparse')
    # Load the model as the opponent so the human "plays" by providing actions
    # For now, we run the model as blue and print observations for human inspection
    print('\n=== Human Play Mode ===')
    print('The loaded model plays as blue. Watch it play against a random opponent.')
    print('Press Ctrl+C to stop.\n')

    try:
        episode = 0
        while True:
            episode += 1
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                with torch.no_grad():
                    x = torch.tensor(obs[np.newaxis], dtype=torch.float32)
                    tokens = x.view(1, t_window, N_TOKENS, TOKEN_FEATURES)
                    emb = encoder(tokens, entity_ids)
                    action, _ = policy.act_deterministic(emb)
                action_np = action[0].cpu().numpy().astype(np.float32)
                obs, reward, done, _, info = env.step(action_np)
                steps += 1

            goal = info.get('goal', 0)
            outcome = {1: 'SCORED', -1: 'CONCEDED', 0: 'TIMEOUT'}[goal]
            print(f'  Episode {episode}: {outcome} after {steps} steps')
    except KeyboardInterrupt:
        print(f'\nStopped after {episode} episodes.')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a model snapshot or checkpoint against bot tiers.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--snapshot', type=str,
        help='Path to snapshot directory (contains encoder.pt + policy.pt)')
    group.add_argument(
        '--checkpoint', type=str,
        help='Path to checkpoint .pt file (contains encoder + policy keys)')

    parser.add_argument(
        '--tier', type=str, default=None,
        choices=EvalConfig.TIER_ORDER,
        help='Evaluate against a single tier (default: all tiers)')
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Override episodes per tier (default: 100)')
    parser.add_argument(
        '--timeout-steps', type=int, default=3000,
        help='Max env steps per episode (default: 3000)')
    parser.add_argument(
        '--t-window', type=int, default=8,
        help='Frame stacking window (default: 8)')
    parser.add_argument(
        '--opponent', type=str, default=None,
        help='Path to opponent snapshot directory (default: random opponent)')
    parser.add_argument(
        '--human', action='store_true',
        help='Watch the model play episodes (human observation mode)')
    parser.add_argument(
        '--json-output', type=str, default=None,
        help='Write results to a JSON file')

    args = parser.parse_args()

    # Load weights
    if args.snapshot:
        print(f'Loading snapshot from {args.snapshot}')
        weights = load_from_snapshot(args.snapshot)
    else:
        print(f'Loading checkpoint from {args.checkpoint}')
        weights = load_from_checkpoint(args.checkpoint)

    encoder, policy = build_models(weights)
    entity_ids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long)

    if args.human:
        run_human_play(encoder, policy, entity_ids, args.t_window)
        return

    # Determine tiers to evaluate
    episodes = args.episodes or 100
    if args.tier:
        tiers = {args.tier: args.opponent}
    else:
        tiers = {t: args.opponent for t in EvalConfig.TIER_ORDER}

    # Run evaluation
    print(f'\nEvaluating ({episodes} episodes per tier, '
          f'timeout={args.timeout_steps} steps):')
    t0 = time.monotonic()
    all_results = {}

    for tier, opp_path in tiers.items():
        opp_label = opp_path if opp_path else 'random'
        print(f'\n  vs {tier} (opponent: {opp_label})...')
        result = _eval_tier(
            tier=tier,
            opponent_path=opp_path,
            encoder=encoder,
            policy=policy,
            entity_ids=entity_ids,
            step=0,
            episodes=episodes,
            timeout_steps=args.timeout_steps,
            t_window=args.t_window,
        )
        all_results[tier] = result
        print(f'    win_rate={result["win_rate"]:.1%}  '
              f'loss_rate={result["loss_rate"]:.1%}  '
              f'timeout_rate={result["timeout_rate"]:.1%}  '
              f'mean_score={result["mean_score"]:.3f}')

    wall_time = time.monotonic() - t0

    # Summary
    print(f'\n{"="*50}')
    print(f'{"Tier":>10}  {"Win%":>6}  {"Loss%":>6}  {"Draw%":>6}  {"Mean":>6}')
    print(f'{"-"*10}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*6}')
    for tier, r in all_results.items():
        marker = ' ***' if tier == 'Rookie' else ''
        print(f'{tier:>10}  {r["win_rate"]:>5.1%}  {r["loss_rate"]:>5.1%}  '
              f'{r["timeout_rate"]:>5.1%}  {r["mean_score"]:>+5.3f}{marker}')
    print(f'\nCompleted in {wall_time:.1f}s')

    # JSON output
    if args.json_output:
        output = {
            'eval_wall_time': round(wall_time, 2),
            'tiers': all_results,
        }
        Path(args.json_output).write_text(json.dumps(output, indent=2))
        print(f'Results written to {args.json_output}')


if __name__ == '__main__':
    main()
