"""W&B MetricLogger with stdout fallback."""
from __future__ import annotations

import os
import sys
from typing import Any

from training.abstractions import MetricLogger


class WandbLogger(MetricLogger):
    """W&B logging with stdout fallback. Default logger."""

    @classmethod
    def default_params(cls) -> dict:
        return {'project': 'rlbot-baseline', 'enabled': True, 'group': None}

    def __init__(self) -> None:
        self._wandb = None

    def init(self, config: dict) -> None:
        logger_section = config.get('logger', {})
        params = {**self.default_params(), **logger_section.get('params', {})}

        # Also merge top-level 'wandb' key if present (YAML shorthand)
        wandb_section = config.get('wandb', {})
        if wandb_section:
            params.update(wandb_section)

        if not params.get('enabled', True):
            print('[logger] W&B disabled by config — logging to stdout only.')
            return

        self._wandb = self._try_init_wandb(config, params)
        if self._wandb is None:
            print('[logger] W&B not active — logging to stdout only.')

    def log(self, step: int, **metrics: Any) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        else:
            parts = '  '.join(
                f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                for k, v in metrics.items()
            )
            print(f'[step {step:07d}] {parts}')

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

    @staticmethod
    def _try_init_wandb(config: dict, params: dict):
        """Attempt to initialise W&B. Returns the wandb module or None."""
        if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
            return None
        try:
            import wandb  # type: ignore
        except ImportError:
            return None

        api_key = os.environ.get('WANDB_API_KEY', '')
        if not api_key:
            try:
                logged_in = wandb.login(anonymous='never', relogin=False)
            except Exception:
                logged_in = False
            if not logged_in:
                print(
                    '[logger] No W&B credentials found — '
                    'falling back to stdout. Run `wandb login` to enable W&B.',
                    file=sys.stderr,
                )
                return None

        try:
            project = params.get('project', 'rlbot-baseline')
            group = params.get('group', None)
            run_name = params.get('run_name', None)

            wandb_config = {
                k: v for k, v in config.items()
                if k not in ('algorithm', 'opponent_pool', 'environment',
                             'reward', 'logger') or not isinstance(v, dict)
            }

            wandb.init(
                project=project,
                name=run_name,
                group=group,
                config=wandb_config,
                resume='allow',
            )
            return wandb
        except Exception as exc:  # noqa: BLE001
            print(
                f'[logger] W&B init failed ({exc}) — falling back to stdout.',
                file=sys.stderr,
            )
            return None
