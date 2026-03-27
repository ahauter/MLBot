#!/usr/bin/env python3
"""
Psyonix Bot Evaluation Protocol
================================
Runs the trained agent against all four Psyonix bot tiers in RLBot v5
live matches and reports win rates.

Evaluation schedule (per checkpoint):
    50  episodes vs Beginner  — sanity check
    100 episodes vs Rookie    — primary convergence criterion
    50  episodes vs Pro       — ceiling check
    50  episodes vs Allstar   — upper ceiling check

Usage
-----
    # Evaluate a saved model:
    python training/evaluate.py --model-dir models/baseline/seed_0

    # Single tier:
    python training/evaluate.py --model-dir models/baseline/seed_0 --tier Rookie --episodes 10

    # Just generate match configs (dry run):
    python training/evaluate.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

_REPO = Path(__file__).parent.parent


# ── evaluation tiers ────────────��────────────────────────���───────────────────

EVAL_TIERS = {
    'Beginner': 50,
    'Rookie':   100,   # primary target
    'Pro':      50,
    'Allstar':  50,
}


# ── match.toml generation ────────────���──────────────────────────────────────

def generate_match_toml(
    tier: str,
    bot_toml_path: str = 'src/bot.toml',
) -> str:
    """
    Generate a match.toml for evaluation against a Psyonix bot tier.

    RLBot v5 format: Psyonix bots are configured inline with type + skill.
    """
    return f"""# Auto-generated evaluation config — {tier}
[match]
game_mode = "Soccer"
game_map = "Mannfield"
instant_start = true
enable_rendering = false
enable_state_setting = true

[[cars]]
team = 0
type = "RLBot"
config = "{bot_toml_path}"

[[cars]]
team = 1
type = "Psyonix"
skill = "{tier}"
"""


# ── match runner ─────���───────────────────────────────────────────────────────

def _parse_match_result(output: str) -> Optional[str]:
    """
    Parse RLBot v5 match output to determine win/loss/draw.

    Looks for final score lines in stdout. RLBot v5 logs match results
    as "Match ended: <blue_score> - <orange_score>" or similar patterns.
    Our bot is always team 0 (blue).

    Returns 'win', 'loss', 'draw', or None if parsing fails.
    """
    # Pattern: "Match ended" or "Final score" with two numbers
    # Try several patterns that RLBot v5 may output
    patterns = [
        # "Match ended: 3 - 1" or "Final score: 3 - 1"
        r'(?:Match ended|Final score|Game ended|Score)[:\s]+(\d+)\s*[-–]\s*(\d+)',
        # "Blue: 3, Orange: 1" style
        r'Blue[:\s]+(\d+).*?Orange[:\s]+(\d+)',
        # Generic "N - M" near end of output (last 20 lines)
        r'(\d+)\s*-\s*(\d+)',
    ]

    # Search from end of output (most likely to contain final score)
    lines = output.strip().split('\n')
    tail = '\n'.join(lines[-30:]) if len(lines) > 30 else output

    for pattern in patterns:
        matches = list(re.finditer(pattern, tail, re.IGNORECASE))
        if matches:
            # Use last match (closest to end of output)
            m = matches[-1]
            blue_score = int(m.group(1))
            orange_score = int(m.group(2))

            if blue_score > orange_score:
                return 'win'
            elif orange_score > blue_score:
                return 'loss'
            else:
                return 'draw'

    return None


def run_single_match(
    match_toml_path: str,
    timeout: int = 600,
) -> Optional[str]:
    """
    Launch a single RLBot v5 match and return the outcome.

    Returns 'win', 'loss', 'draw', or None if the match failed.
    """
    try:
        result = subprocess.run(
            ['rlbot', 'run', '--config', match_toml_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f'  RLBot exited with code {result.returncode}',
                  file=sys.stderr)
            if result.stderr:
                print(f'  stderr: {result.stderr[:200]}', file=sys.stderr)
            return None

        # Parse match result from stdout
        outcome = _parse_match_result(result.stdout)
        if outcome is None and result.stderr:
            # Some versions log to stderr
            outcome = _parse_match_result(result.stderr)

        if outcome is None:
            print(f'  Could not parse match result from RLBot output',
                  file=sys.stderr)
        return outcome

    except FileNotFoundError:
        # rlbot CLI not installed
        return None
    except subprocess.TimeoutExpired:
        print(f'  Match timed out after {timeout}s', file=sys.stderr)
        return None


def run_evaluation(
    model_dir: str,
    tiers: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Run evaluation against Psyonix tiers and return win rates.

    Parameters
    ----------
    model_dir : str
        Directory containing the model checkpoint to evaluate.
    tiers : dict, optional
        Override tier → episode count mapping. Defaults to EVAL_TIERS.

    Returns
    -------
    dict[str, float] — tier name → win rate (0.0 to 1.0)
    """
    if tiers is None:
        tiers = EVAL_TIERS

    results = {}

    with tempfile.TemporaryDirectory(prefix='rlbot_eval_') as tmpdir:
        for tier, n_episodes in tiers.items():
            # Generate match config
            toml_content = generate_match_toml(tier)
            toml_path = Path(tmpdir) / f'eval_{tier.lower()}.toml'
            toml_path.write_text(toml_content)

            wins = 0
            played = 0

            print(f'  Evaluating vs {tier} ({n_episodes} episodes)...')
            for ep in range(n_episodes):
                outcome = run_single_match(str(toml_path))
                if outcome is None:
                    # RLBot not available — skip live evaluation
                    print(f'    RLBot not available, skipping live eval for {tier}')
                    results[tier] = 0.0
                    break
                played += 1
                if outcome == 'win':
                    wins += 1
            else:
                results[tier] = wins / max(played, 1)
                print(f'    {tier}: {results[tier]:.1%} '
                      f'({wins}/{played})')

    return results


# ── static eval config generation ────────────────────��───────────────────────

def generate_eval_configs(output_dir: str = 'eval_configs') -> None:
    """Generate static match.toml files for each Psyonix tier."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tier in EVAL_TIERS:
        toml_path = out / f'vs_{tier.lower()}.toml'
        toml_path.write_text(generate_match_toml(tier))
        print(f'  Generated {toml_path}')


# ��─ CLI ───────────────────────────��───────────────────────────���──────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate against Psyonix bots.')
    parser.add_argument('--model-dir', default='models/baseline/seed_0')
    parser.add_argument('--tier', default=None,
                        choices=['Beginner', 'Rookie', 'Pro', 'Allstar'],
                        help='Evaluate against a single tier')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override episode count for --tier')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate eval configs without running matches')
    args = parser.parse_args()

    if args.dry_run:
        generate_eval_configs()
        return

    if args.tier:
        tiers = {args.tier: args.episodes or EVAL_TIERS[args.tier]}
    else:
        tiers = EVAL_TIERS

    print(f'Evaluating model from {args.model_dir}:')
    results = run_evaluation(args.model_dir, tiers)

    print('\nResults:')
    for tier, wr in results.items():
        marker = ' *** PRIMARY' if tier == 'Rookie' else ''
        print(f'  {tier:>10}: {wr:.1%}{marker}')


if __name__ == '__main__':
    main()
