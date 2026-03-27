#!/usr/bin/env python3
"""
Simulation-Based Evaluation Protocol
=====================================
Evaluates a trained agent by running episodes in rlgym-sim against
reference opponents. No RLBot runtime required.

Evaluation tiers:
    Random   — opponent plays random actions (sanity check)
    Snapshot — opponent is a frozen self-play snapshot (improvement check)

Usage
-----
    # Evaluate a saved model against random:
    python training/evaluate.py --model-dir models/baseline/seed_0

    # Specific tier and episode count:
    python training/evaluate.py --model-dir models/baseline/seed_0 --tier Random --episodes 20
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))


# ── evaluation result ────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Structured result from evaluating against a single opponent tier."""
    tier: str
    n_episodes: int
    wins: int
    losses: int
    draws: int
    mean_return: float

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.n_episodes, 1)

    @property
    def loss_rate(self) -> float:
        return self.losses / max(self.n_episodes, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.n_episodes, 1)


# ── default tiers ────────────────────────────────────────────────────────────

SIM_EVAL_TIERS = {
    'Random': 50,
    'Snapshot': 50,
}


# ── core evaluation function ─────────────────────────────────────────────────

def run_sim_evaluation(
    algo,
    opponent=None,
    n_episodes: int = 50,
    t_window: int = 8,
    tier_name: str = 'Random',
) -> EvalResult:
    """
    Run N episodes in rlgym-sim and return win/loss/draw stats.

    Parameters
    ----------
    algo : d3rlpy algo
        Agent to evaluate (must have .predict()).
    opponent : d3rlpy algo or None
        Opponent model. None means random actions.
    n_episodes : int
        Number of episodes to play.
    t_window : int
        Frame-stacking window size.
    tier_name : str
        Label for the tier (used in EvalResult).

    Returns
    -------
    EvalResult
    """
    from gym_env import BaselineGymEnv

    env = BaselineGymEnv(t_window=t_window)
    env.set_opponent(opponent)

    wins = 0
    losses = 0
    draws = 0
    total_return = 0.0

    try:
        for _ in range(n_episodes):
            obs, _info = env.reset()
            episode_return = 0.0
            done = False

            while not done:
                action = algo.predict(obs[np.newaxis])[0]
                obs, reward, done, _truncated, _info = env.step(action)
                episode_return += reward

            if episode_return > 0.5:
                wins += 1
            elif episode_return < -0.5:
                losses += 1
            else:
                draws += 1
            total_return += episode_return
    finally:
        env.close()

    return EvalResult(
        tier=tier_name,
        n_episodes=n_episodes,
        wins=wins,
        losses=losses,
        draws=draws,
        mean_return=total_return / max(n_episodes, 1),
    )


# ── multi-tier evaluation ────────────────────────────────────────────────────

def run_evaluation(
    algo,
    snapshot_opponent=None,
    tiers: Optional[Dict[str, int]] = None,
    t_window: int = 8,
) -> Dict[str, EvalResult]:
    """
    Run evaluation against all configured tiers.

    Parameters
    ----------
    algo : d3rlpy algo
        Agent to evaluate.
    snapshot_opponent : d3rlpy algo or None
        Frozen snapshot for the Snapshot tier. If None, Snapshot tier is skipped.
    tiers : dict, optional
        Override tier → episode count mapping. Defaults to SIM_EVAL_TIERS.
    t_window : int
        Frame-stacking window size.

    Returns
    -------
    dict[str, EvalResult]
    """
    if tiers is None:
        tiers = SIM_EVAL_TIERS

    results = {}

    for tier_name, n_episodes in tiers.items():
        if tier_name == 'Snapshot' and snapshot_opponent is None:
            continue

        opponent = None if tier_name == 'Random' else snapshot_opponent

        print(f'  Evaluating vs {tier_name} ({n_episodes} episodes)...')
        result = run_sim_evaluation(
            algo=algo,
            opponent=opponent,
            n_episodes=n_episodes,
            t_window=t_window,
            tier_name=tier_name,
        )
        results[tier_name] = result
        print(f'    {tier_name}: WR={result.win_rate:.1%} '
              f'({result.wins}W/{result.losses}L/{result.draws}D)')

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained agent in rlgym-sim.')
    parser.add_argument('--model-dir', default='models/baseline/seed_0',
                        help='Directory containing the d3rlpy model checkpoint')
    parser.add_argument('--tier', default=None, choices=['Random', 'Snapshot'],
                        help='Evaluate against a single tier')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override episode count')
    parser.add_argument('--t-window', type=int, default=8)
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f'Model directory not found: {model_path}', file=sys.stderr)
        sys.exit(1)

    import d3rlpy
    from baseline_encoder_factory import TransformerEncoderFactory
    from gym_env import BaselineGymEnv

    encoder_factory = TransformerEncoderFactory(t_window=args.t_window)
    algo = d3rlpy.algos.AWACConfig(
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
    ).create(device='cpu')

    dummy_env = BaselineGymEnv(t_window=args.t_window)
    algo.build_with_env(dummy_env)
    dummy_env.close()

    # Try to load model weights
    model_file = model_path / 'final_model.d3'
    if model_file.exists():
        algo.load_model(str(model_file))
    else:
        # Try legacy path
        legacy = model_path / 'd3rlpy_model.d3'
        if legacy.exists():
            algo.load_model(str(legacy))
        else:
            print(f'Warning: No model file found in {model_path}, '
                  f'evaluating with random weights', file=sys.stderr)

    # Build tier config
    if args.tier:
        tiers = {args.tier: args.episodes or SIM_EVAL_TIERS.get(args.tier, 50)}
    else:
        tiers = SIM_EVAL_TIERS.copy()
        if args.episodes:
            tiers = {k: args.episodes for k in tiers}

    print(f'Evaluating model from {args.model_dir}:')
    results = run_evaluation(
        algo=algo,
        snapshot_opponent=None,  # CLI mode: no snapshot available
        tiers=tiers,
        t_window=args.t_window,
    )

    print('\nResults:')
    for tier_name, result in results.items():
        print(f'  {tier_name:>10}: WR={result.win_rate:.1%}  '
              f'({result.wins}W/{result.losses}L/{result.draws}D)  '
              f'mean_return={result.mean_return:.3f}')


if __name__ == '__main__':
    main()
