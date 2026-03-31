#!/usr/bin/env python3
"""
Evaluate a Snapshot or Checkpoint
==================================
Top-level CLI for running the evaluation pipeline on an arbitrary
snapshot/checkpoint, or launching a human spectator session.

All evaluation logic is routed through the EvaluationHook abstraction,
so swapping the eval hook class (via --eval-class or YAML config) changes
the entire evaluation protocol without modifying this file.

Usage
-----
    # Evaluate a training snapshot:
    python evaluate.py --snapshot models/snapshots/step_0000050000

    # Evaluate an eval checkpoint file:
    python evaluate.py --checkpoint checkpoints/step_0000050000/eval_checkpoint.pt

    # Quick check with fewer episodes:
    python evaluate.py --snapshot models/snapshots/step_0000050000 --episodes 10

    # Single tier:
    python evaluate.py --snapshot models/snapshots/step_0000050000 --tier Rookie

    # Use DummyEnv (no rlgym-sim needed):
    python evaluate.py --snapshot path --env-class training.environments.dummy_env.DummyEnv

    # Watch the model play (spectator mode):
    python evaluate.py --snapshot models/snapshots/step_0000050000 --human
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
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


def _load_class(dotted_path: str):
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), class_name)


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


class _WeightsHolder:
    """Lightweight stand-in for Algorithm so we can call hook.run_interactive()."""

    def __init__(self, encoder, policy):
        self.encoder = encoder
        self.policy = policy


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
        '--env-class', type=str, default=None,
        help='Dotted path to env class (default: BaselineGymEnv)')
    parser.add_argument(
        '--eval-class', type=str, default=None,
        help='Dotted path to EvaluationHook class (default: SimEvaluationHook)')
    parser.add_argument(
        '--human', action='store_true',
        help='Watch the model play episodes (spectator mode)')
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

    encoder = SharedTransformerEncoder(d_model=D_MODEL)
    policy = StochasticPolicyHead(d_model=D_MODEL)
    encoder.load_state_dict(weights['encoder'])
    policy.load_state_dict(weights['policy'])
    encoder.eval()
    policy.eval()

    # Build a config dict matching the YAML convention
    config = {
        't_window': args.t_window,
        'env_class': args.env_class,
        'eval_interval': 1,  # enable eval hook
        'evaluation': {
            'params': {
                'episodes_per_tier': args.episodes or 100,
                'episode_timeout_steps': args.timeout_steps,
            },
        },
    }

    # Resolve eval hook class
    if args.eval_class:
        EvalCls = _load_class(args.eval_class)
    else:
        from training.evaluation.sim_eval import SimEvaluationHook
        EvalCls = SimEvaluationHook

    hook = EvalCls(config)

    # Interactive / spectator mode
    if args.human:
        algo = _WeightsHolder(encoder, policy)
        hook.run_interactive(algo)
        return

    # Run evaluation via _eval_tier (direct, not through subprocess)
    entity_ids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long)

    if args.tier:
        tiers = {args.tier: args.opponent}
    else:
        tiers = {t: args.opponent for t in EvalConfig.TIER_ORDER}

    episodes = args.episodes or 100
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
            env_class=args.env_class,
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
