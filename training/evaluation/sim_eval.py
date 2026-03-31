"""
Simulation-Based Evaluation Hook
==================================
Implements the EvaluationHook ABC for inline evaluation.

Runs evaluation episodes on the existing SubprocVecEnv workers between
collect_and_train() calls. The next collect_and_train() unconditionally
resets all envs, so eval state never leaks into training data.

Configured via YAML::

    evaluation:
      class: training.evaluation.sim_eval.SimEvaluationHook
      params:
        episodes_per_tier: 100
        episode_timeout_steps: 3000
        checkpoint_dir: checkpoints
        skill_target_tier: Rookie
        skill_target_win_rate: 0.60
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from training.abstractions import Algorithm, EvaluationHook
from training.evaluation.eval_config import EvalConfig


class SimEvaluationHook(EvaluationHook):
    """
    Inline evaluation using existing env workers.

    Usage from training loop::

        hook = SimEvaluationHook(config)

        # Between collect_and_train() calls:
        if hook.should_evaluate(total_steps):
            results = hook.evaluate_inline(algorithm, envs, total_steps, device)
            logger.log(total_steps, **hook.format_metrics(results))
    """

    @classmethod
    def default_params(cls) -> dict:
        return asdict(EvalConfig())

    def __init__(self, config: dict):
        self.cfg = EvalConfig.from_config(config)
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._last_eval_step = 0

    # ------------------------------------------------------------------
    # Training-loop API
    # ------------------------------------------------------------------

    def should_evaluate(self, step: int) -> bool:
        """True when enough steps have elapsed since the last eval."""
        return step - self._last_eval_step >= self.cfg.eval_interval

    def save_eval_checkpoint(
        self,
        algorithm: Algorithm,
        step: int,
        run_id: str = '',
        axis_costs: Optional[Dict] = None,
        intervention: str = '',
    ) -> Path:
        """Write an eval-only checkpoint (encoder + policy, no optimizer)."""
        ckpt_dir = Path(self.cfg.checkpoint_dir) / f'step_{step:010d}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / 'eval_checkpoint.pt'

        torch.save({
            'encoder': {k: v.cpu().clone()
                        for k, v in algorithm.encoder.state_dict().items()},
            'policy': {k: v.cpu().clone()
                       for k, v in algorithm.policy.state_dict().items()},
            'step': step,
            'run_id': run_id,
            'seed': self.cfg.t_window,  # propagate for reproducibility
            'axis_costs': axis_costs or {},
            'intervention': intervention,
            'obs_dim': self.cfg.t_window * 10 * 10,  # T * N_TOKENS * TOKEN_FEATURES
        }, ckpt_path)

        return ckpt_path

    def evaluate_inline(
        self,
        algorithm: Algorithm,
        envs,
        step: int,
        device: str = 'cpu',
    ) -> dict:
        """Run eval episodes inline on existing SubprocVecEnv workers.

        This is BLOCKING — runs between collect_and_train() calls.
        The next collect_and_train() will call envs.reset_all(), so
        eval state never contaminates training data.

        Parameters
        ----------
        algorithm : Algorithm
            Agent with .encoder and .policy attributes.
        envs : SubprocVecEnv
            The same env workers used for training.
        step : int
            Current training step (for logging/seeding).
        device : str
            Torch device for inference.

        Returns
        -------
        dict
            Results with 'tiers', 'eval_wall_time', 'convergence_reached', etc.
        """
        from encoder import N_TOKENS, TOKEN_FEATURES, ENTITY_TYPE_IDS_1V1

        t0 = time.monotonic()

        encoder = algorithm.encoder
        policy = algorithm.policy
        encoder.eval()
        policy.eval()
        entity_ids = torch.tensor(
            ENTITY_TYPE_IDS_1V1, dtype=torch.long, device=device)

        # Clear opponent snapshots — eval uses random opponents only
        envs.set_opponent_snapshot(None)

        tier_results = {}
        for tier in EvalConfig.TIER_ORDER:
            if tier not in self.cfg.tier_opponents:
                continue
            tier_results[tier] = self._eval_tier_inline(
                envs, encoder, policy, entity_ids, step, tier, device)

        wall_time = time.monotonic() - t0

        # Check convergence
        target_tier = self.cfg.skill_target_tier
        target_wr = tier_results.get(target_tier, {}).get('win_rate', 0.0)
        converged = target_wr >= self.cfg.skill_target_win_rate

        self._last_eval_step = step

        results = {
            'checkpoint_step': step,
            'eval_wall_time': round(wall_time, 2),
            'convergence_reached': converged,
            'axis_costs': {},
            'tiers': tier_results,
        }

        print(f'[eval] step={step} done in {wall_time:.1f}s '
              f'(converged={converged})')
        for tier, data in tier_results.items():
            print(f'[eval]   {tier}: win_rate={data["win_rate"]:.2%}')

        return results

    def _eval_tier_inline(
        self,
        envs,
        encoder,
        policy,
        entity_ids: torch.Tensor,
        step: int,
        tier: str,
        device: str,
    ) -> Dict:
        """Evaluate against a single tier using vectorized envs.

        Runs episodes_per_tier episodes across all env workers in parallel.
        Agent inference is batched on GPU; opponents are random (no snapshot).
        """
        from encoder import N_TOKENS, TOKEN_FEATURES

        num_envs = envs.num_envs
        episodes_needed = self.cfg.episodes_per_tier
        t_window = self.cfg.t_window
        timeout = self.cfg.episode_timeout_steps

        scores = []
        env_steps = np.zeros(num_envs, dtype=np.int64)

        # Reset all envs to start eval
        obs = envs.reset_all()  # (num_envs, obs_dim)

        while len(scores) < episodes_needed:
            # Batched GPU inference — deterministic policy
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device)
                tokens = x.view(x.shape[0], t_window, N_TOKENS, TOKEN_FEATURES)
                emb = encoder(tokens, entity_ids)
                actions, _ = policy.act_deterministic(emb)
            actions_np = actions.cpu().numpy().astype(np.float32)

            # Step all envs (random opponents — no snapshot loaded)
            next_obs, rewards, dones, infos = envs.step(actions_np)
            env_steps += 1

            # Check for completed episodes
            for i in range(num_envs):
                done = bool(dones[i]) or env_steps[i] >= timeout
                if done:
                    goal = infos[i].get('goal', 0)
                    scores.append(goal)
                    env_steps[i] = 0
                    if len(scores) >= episodes_needed:
                        break

            obs = next_obs

            # If any envs finished, they auto-reset via env.step() returning
            # a fresh obs (gymnasium convention). env_steps tracks timeout.

        # Trim to exact count
        scores = scores[:episodes_needed]
        n = len(scores)
        wins = sum(1 for s in scores if s == 1)
        losses = sum(1 for s in scores if s == -1)
        timeouts = sum(1 for s in scores if s == 0)

        return {
            'win_rate': wins / n if n else 0.0,
            'loss_rate': losses / n if n else 0.0,
            'timeout_rate': timeouts / n if n else 0.0,
            'mean_score': sum(scores) / n if n else 0.0,
            'episodes': n,
        }

    @staticmethod
    def format_metrics(results: dict) -> dict:
        """Flatten eval results dict into W&B-ready metric keys."""
        if results.get('error'):
            return {}

        metrics: Dict[str, float] = {}

        for tier_name, tier_data in results.get('tiers', {}).items():
            for key, value in tier_data.items():
                if key != 'episodes':
                    metrics[f'eval/{tier_name}/{key}'] = value

        for axis_key, axis_val in results.get('axis_costs', {}).items():
            metrics[f'axis/{axis_key}'] = axis_val

        metrics['eval/wall_time'] = results.get('eval_wall_time', 0.0)
        metrics['eval/convergence_reached'] = int(
            results.get('convergence_reached', False))

        return metrics

    # ------------------------------------------------------------------
    # EvaluationHook ABC
    # ------------------------------------------------------------------

    def evaluate(self, algorithm: Algorithm, step: int) -> dict:
        """
        Synchronous evaluation (ABC contract).

        Runs the eval worker in-process via checkpoint. Used by the CLI
        (evaluate.py) when no SubprocVecEnv is available.
        """
        ckpt_path = self.save_eval_checkpoint(algorithm, step)
        result_path = str(ckpt_path.parent / 'eval_results.json')
        error_log = str(Path(self.cfg.checkpoint_dir) / 'eval_errors.log')

        from training.evaluation.eval_worker import run_eval_worker
        run_eval_worker(str(ckpt_path), result_path, error_log, asdict(self.cfg))

        rp = Path(result_path)
        if rp.exists():
            return json.loads(rp.read_text())
        return {'error': True}

    def check_convergence(self, eval_results: dict) -> bool:
        """Check whether the skill target was met."""
        tier = self.cfg.skill_target_tier
        tier_data = eval_results.get('tiers', {}).get(tier, {})
        return tier_data.get('win_rate', 0.0) >= self.cfg.skill_target_win_rate

    def run_interactive(self, algorithm: Algorithm, step: int = 0) -> None:
        """Launch a spectator session: watch the agent play episodes.

        Loads the agent's encoder + policy and runs episodes in the
        configured environment, printing per-episode outcomes.
        Press Ctrl+C to stop.
        """
        from encoder import (
            SharedTransformerEncoder, D_MODEL, N_TOKENS,
            TOKEN_FEATURES, ENTITY_TYPE_IDS_1V1,
        )
        from policy_head import StochasticPolicyHead
        from training.evaluation.eval_worker import _load_env_class

        # Build models on CPU from algorithm weights
        encoder = SharedTransformerEncoder(d_model=D_MODEL)
        policy = StochasticPolicyHead(d_model=D_MODEL)
        encoder.load_state_dict(algorithm.encoder.state_dict())
        policy.load_state_dict(algorithm.policy.state_dict())
        encoder.eval()
        policy.eval()
        entity_ids = torch.tensor(ENTITY_TYPE_IDS_1V1, dtype=torch.long)

        EnvCls = _load_env_class(self.cfg.env_class)
        env = EnvCls(
            t_window=self.cfg.t_window,
            max_steps=self.cfg.episode_timeout_steps,
            reward_type='sparse',
        )

        print('\n=== Interactive Mode ===')
        print(f'Environment: {EnvCls.__module__}.{EnvCls.__name__}')
        print('The loaded model plays as blue. Press Ctrl+C to stop.\n')

        episode = 0
        try:
            while True:
                episode += 1
                obs, _ = env.reset()
                done = False
                steps = 0
                while not done:
                    with torch.no_grad():
                        x = torch.tensor(obs[np.newaxis], dtype=torch.float32)
                        tokens = x.view(
                            1, self.cfg.t_window, N_TOKENS, TOKEN_FEATURES)
                        emb = encoder(tokens, entity_ids)
                        action, _ = policy.act_deterministic(emb)
                    action_np = action[0].cpu().numpy().astype(np.float32)
                    obs, _reward, done, _, info = env.step(action_np)
                    steps += 1

                goal = info.get('goal', 0)
                outcome = {1: 'SCORED', -1: 'CONCEDED', 0: 'TIMEOUT'}[goal]
                print(f'  Episode {episode}: {outcome} after {steps} steps')
        except KeyboardInterrupt:
            print(f'\nStopped after {episode} episodes.')
        finally:
            env.close()
