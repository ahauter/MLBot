"""
Experiment Logger
=================
Thin wrapper around Weights & Biases with a graceful stdout fallback.

Usage
-----
    from logger import ExperimentLogger
    from train_config import TrainConfig

    config = TrainConfig()
    logger = ExperimentLogger(config)          # W&B if available, else stdout
    logger.log(episode, total_loss=0.5, ...)
    logger.finish()

W&B is enabled when:
  - The `wandb` package is installed, AND
  - either WANDB_API_KEY is set or the user is already logged in locally.

Pass ``enabled=False`` or set ``WANDB_DISABLED=true`` to force stdout mode.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from collections import deque
from typing import Any, Optional


class ExperimentLogger:
    """
    Logs training metrics to W&B (if available) with a stdout fallback.

    Parameters
    ----------
    config : TrainConfig
        Full training configuration (logged as W&B run hyperparameters).
    enabled : bool
        Set False to force stdout-only mode regardless of W&B availability.
    group : str | None
        Optional W&B run group (e.g. 'optuna-study').
    """

    def __init__(self, config, *, enabled: bool = True, group: Optional[str] = None) -> None:
        self._wandb = None
        self._enabled = enabled

        if enabled:
            self._wandb = self._try_init_wandb(config, group)

        if self._wandb is None:
            print('[logger] W&B not active — logging to stdout only.')

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, step: int, **metrics: Any) -> None:
        """Log a dict of scalar metrics at the given training step."""
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        else:
            # Compact stdout line every 100 steps; always log when called
            parts = '  '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                              for k, v in metrics.items())
            print(f'[ep {step:05d}] {parts}')

    def finish(self) -> None:
        """Mark the W&B run as finished."""
        if self._wandb is not None:
            self._wandb.finish()

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _try_init_wandb(config, group: Optional[str]):
        """Attempt to initialise W&B. Returns the wandb module or None."""
        if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
            return None
        try:
            import wandb  # type: ignore
        except ImportError:
            return None

        # Check credentials via wandb's own mechanism — handles env vars,
        # ~/.netrc, ~/_netrc (Windows), and wandb settings files correctly.
        api_key = os.environ.get('WANDB_API_KEY', '')
        if not api_key:
            try:
                # Returns True if already authenticated (no prompts, no network)
                logged_in = wandb.login(anonymous='never', relogin=False)
            except Exception:
                logged_in = False
            if not logged_in:
                print('[logger] No W&B credentials found — '
                      'falling back to stdout. Run `wandb login` to enable W&B.',
                      file=sys.stderr)
                return None

        try:
            cfg_dict = dataclasses.asdict(config)
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                group=group or config.wandb_group,
                config=cfg_dict,
                resume='allow',
            )
            return wandb
        except Exception as exc:  # noqa: BLE001
            print(f'[logger] W&B init failed ({exc}) — falling back to stdout.',
                  file=sys.stderr)
            return None
