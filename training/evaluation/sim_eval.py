"""
Simulation-Based Evaluation Hook
==================================
Implements the EvaluationHook ABC for non-blocking evaluation.

Training-side coordinator: saves eval checkpoints, spawns worker processes,
collects results from JSON files. All W&B logging stays in the training process.
"""
from __future__ import annotations

import json
import multiprocessing
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from training.abstractions import Algorithm, EvaluationHook
from training.evaluation.eval_config import EvalConfig


class SimEvaluationHook(EvaluationHook):
    """
    Non-blocking evaluation via subprocess workers.

    Usage from training loop::

        hook = SimEvaluationHook(config)

        # In the training loop:
        if hook.should_evaluate(total_steps):
            hook.spawn_eval(algorithm, total_steps, run_id=wandb_run_id)

        # At each log interval:
        for step, results in hook.collect_results():
            logger.log(step, **hook.format_metrics(results))
    """

    @classmethod
    def default_params(cls) -> dict:
        return asdict(EvalConfig())

    def __init__(self, config: dict):
        self.cfg = EvalConfig.from_config(config)
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._active: List[Tuple[multiprocessing.Process, int, str]] = []
        self._last_eval_step = 0

        # Use 'spawn' to avoid fork issues with rlgym-sim / PyTorch
        self._mp_ctx = multiprocessing.get_context('spawn')

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

    def spawn_eval(
        self,
        algorithm: Algorithm,
        step: int,
        run_id: str = '',
        axis_costs: Optional[Dict] = None,
        intervention: str = '',
    ) -> None:
        """Save checkpoint and launch eval worker. Non-blocking."""
        ckpt_path = self.save_eval_checkpoint(
            algorithm, step, run_id, axis_costs, intervention)
        result_path = str(ckpt_path.parent / 'eval_results.json')
        error_log = str(Path(self.cfg.checkpoint_dir) / 'eval_errors.log')

        from training.evaluation.eval_worker import run_eval_worker

        p = self._mp_ctx.Process(
            target=run_eval_worker,
            args=(str(ckpt_path), result_path, error_log, asdict(self.cfg)),
            daemon=False,
        )
        p.start()
        self._active.append((p, step, result_path))
        self._last_eval_step = step
        print(f'[eval] Spawned eval worker for step {step} (pid={p.pid})')

    def collect_results(self) -> List[Tuple[int, dict]]:
        """
        Check for completed eval processes. Non-blocking.

        Returns list of (step, results_dict) for processes that finished
        since the last call. Cleans up terminated processes.
        """
        completed = []
        still_running = []

        for proc, step, result_path in self._active:
            if not proc.is_alive():
                rp = Path(result_path)
                if rp.exists():
                    results = json.loads(rp.read_text())
                    completed.append((step, results))
                else:
                    completed.append((step, {
                        'error': True,
                        'exit_code': proc.exitcode,
                        'checkpoint_step': step,
                    }))
                proc.join(timeout=5)
            else:
                still_running.append((proc, step, result_path))

        self._active = still_running
        return completed

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

    def cleanup(self, timeout: float = 30.0) -> None:
        """Join any remaining eval processes. Call in the finally block."""
        for proc, step, _ in self._active:
            if proc.is_alive():
                print(f'[eval] Waiting for eval worker step={step} '
                      f'(pid={proc.pid})...')
                proc.join(timeout=timeout)
                if proc.is_alive():
                    print(f'[eval] Eval worker step={step} did not finish, '
                          f'terminating.')
                    proc.terminate()
        self._active.clear()

    # ------------------------------------------------------------------
    # EvaluationHook ABC — synchronous fallback
    # ------------------------------------------------------------------

    def evaluate(self, algorithm: Algorithm, step: int) -> dict:
        """
        Synchronous evaluation (ABC contract).

        Runs the eval worker in-process. Prefer spawn_eval() + collect_results()
        for non-blocking usage from the training loop.
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
