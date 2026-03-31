"""
Evaluation Worker — Subprocess Entry Point
============================================
Runs the full evaluation suite for a single checkpoint in an isolated process.

Called by SimEvaluationHook.spawn_eval() via multiprocessing.Process.
Loads the checkpoint, creates environments, runs episodes, writes results
to a JSON file. Never touches W&B — the training process handles logging.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_REPO = Path(__file__).parent.parent.parent
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


def _episode_seed(step: int, tier: str, episode_idx: int) -> int:
    """Deterministic seed from (step, tier, episode_index)."""
    return hash((step, tier, episode_idx)) % (2**31)


def _run_episode(
    env,
    encoder: SharedTransformerEncoder,
    policy: StochasticPolicyHead,
    entity_ids: torch.Tensor,
    t_window: int,
    seed: int,
) -> int:
    """Run a single episode with deterministic policy. Returns +1, -1, or 0."""
    obs, _info = env.reset(seed=seed)

    done = False
    score = 0
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs[np.newaxis], dtype=torch.float32)
            tokens = x.view(1, t_window, N_TOKENS, TOKEN_FEATURES)
            emb = encoder(tokens, entity_ids)
            action, _ = policy.act_deterministic(emb)
        action_np = action[0].cpu().numpy().astype(np.float32)

        obs, _reward, done, _trunc, info = env.step(action_np)
        if done:
            score = info.get('goal', 0)

    return score


def _eval_tier(
    tier: str,
    opponent_path: Optional[str],
    encoder: SharedTransformerEncoder,
    policy: StochasticPolicyHead,
    entity_ids: torch.Tensor,
    step: int,
    episodes: int,
    timeout_steps: int,
    t_window: int,
) -> Dict:
    """Evaluate against a single tier. Returns metrics dict."""
    from training.environments.baseline_env import BaselineGymEnv

    env = BaselineGymEnv(
        t_window=t_window,
        max_steps=timeout_steps,
        reward_type='sparse',
    )
    if opponent_path is not None:
        env.load_ppo_opponent(opponent_path)

    scores: List[int] = []
    for ep_idx in range(episodes):
        seed = _episode_seed(step, tier, ep_idx)
        score = _run_episode(env, encoder, policy, entity_ids, t_window, seed)
        scores.append(score)

    env.close()

    wins = sum(1 for s in scores if s == 1)
    losses = sum(1 for s in scores if s == -1)
    timeouts = sum(1 for s in scores if s == 0)
    n = len(scores)

    return {
        'win_rate': wins / n if n else 0.0,
        'loss_rate': losses / n if n else 0.0,
        'timeout_rate': timeouts / n if n else 0.0,
        'mean_score': sum(scores) / n if n else 0.0,
        'episodes': n,
    }


def run_eval_worker(
    checkpoint_path: str,
    result_path: str,
    error_log_path: str,
    eval_config_dict: dict,
) -> None:
    """
    Main entry point for the eval subprocess.

    Parameters
    ----------
    checkpoint_path : str
        Path to the eval checkpoint .pt file.
    result_path : str
        Where to write the JSON results file.
    error_log_path : str
        Where to append error tracebacks on failure.
    eval_config_dict : dict
        Serialized EvalConfig fields.
    """
    try:
        _run(checkpoint_path, result_path, eval_config_dict)
    except Exception:
        tb = traceback.format_exc()
        print(f'[eval_worker] FAILED on {checkpoint_path}:\n{tb}',
              file=sys.stderr)
        try:
            with open(error_log_path, 'a') as f:
                f.write(f'\n--- {datetime.now(timezone.utc).isoformat()} ---\n')
                f.write(f'checkpoint: {checkpoint_path}\n')
                f.write(tb)
                f.write('\n')
        except Exception:
            pass  # can't even log the error — nothing more to do


def _run(checkpoint_path: str, result_path: str, eval_config_dict: dict) -> None:
    cfg = EvalConfig(**eval_config_dict)
    t0 = time.monotonic()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    step = ckpt['step']

    # Build models on CPU
    encoder = SharedTransformerEncoder(d_model=D_MODEL)
    policy = StochasticPolicyHead(d_model=D_MODEL)
    encoder.load_state_dict(ckpt['encoder'])
    policy.load_state_dict(ckpt['policy'])
    encoder.eval()
    policy.eval()
    entity_ids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long)

    # Run all tiers
    tier_results = {}
    for tier in EvalConfig.TIER_ORDER:
        if tier not in cfg.tier_opponents:
            continue
        opponent_path = cfg.tier_opponents[tier]
        print(f'[eval_worker] step={step} evaluating vs {tier} '
              f'({cfg.episodes_per_tier} episodes)...')
        tier_results[tier] = _eval_tier(
            tier=tier,
            opponent_path=opponent_path,
            encoder=encoder,
            policy=policy,
            entity_ids=entity_ids,
            step=step,
            episodes=cfg.episodes_per_tier,
            timeout_steps=cfg.episode_timeout_steps,
            t_window=cfg.t_window,
        )
        wr = tier_results[tier]['win_rate']
        print(f'[eval_worker]   {tier}: win_rate={wr:.2%}')

    wall_time = time.monotonic() - t0

    # Check convergence
    target_tier = cfg.skill_target_tier
    target_wr = tier_results.get(target_tier, {}).get('win_rate', 0.0)
    converged = target_wr >= cfg.skill_target_win_rate

    # Build results
    results = {
        'checkpoint_step': step,
        'eval_wall_time': round(wall_time, 2),
        'seed': ckpt.get('seed', 0),
        'intervention': ckpt.get('intervention', ''),
        'convergence_reached': converged,
        'axis_costs': ckpt.get('axis_costs', {}),
        'tiers': tier_results,
    }

    # Write results atomically
    tmp_path = result_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, result_path)

    # Write convergence file if threshold met
    if converged:
        conv_path = Path(cfg.checkpoint_dir) / 'convergence.json'
        conv_data = {
            'step': step,
            'tier': target_tier,
            'win_rate': target_wr,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        conv_tmp = str(conv_path) + '.tmp'
        with open(conv_tmp, 'w') as f:
            json.dump(conv_data, f, indent=2)
        os.replace(conv_tmp, str(conv_path))
        print(f'[eval_worker] CONVERGENCE at step {step}: '
              f'{target_tier} win_rate={target_wr:.2%}')

    print(f'[eval_worker] step={step} done in {wall_time:.1f}s')
