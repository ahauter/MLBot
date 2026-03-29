"""
Experiment Loggers
==================
Three MetricLogger implementations (W&B, stdout, TensorBoard) plus the
MetricsRegistry for decoupled, main-thread-safe custom metric collection.

Usage
-----
    from logger import WandbLogger, StdoutLogger, TensorBoardLogger, MetricsRegistry

    logger = WandbLogger()
    logger.init(config)               # config is a dict (from YAML)
    logger.log(step, total_loss=0.5)
    logger.finish()

    # Registry pattern — components register metric providers,
    # main thread collects them all at each logging point:
    registry = MetricsRegistry()
    registry.register('frozen_self_play', pool.get_metrics)
    registry.register('reward', reward_fn.get_metrics)
    metrics = registry.collect()       # {'frozen_self_play/swap_count': 3, ...}

W&B is enabled when:
  - The ``wandb`` package is installed, AND
  - either WANDB_API_KEY is set or the user is already logged in locally.

Pass ``enabled: false`` in config['wandb'] or set ``WANDB_DISABLED=true``
to force stdout mode.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

from abstractions import MetricLogger


# ── W&B Logger ───────────────────────────────────────────────────────────────

class WandbLogger(MetricLogger):
    """W&B logging with stdout fallback. Default logger."""

    @classmethod
    def default_params(cls) -> dict:
        return {'project': 'rlbot-baseline', 'enabled': True, 'group': None}

    def __init__(self) -> None:
        self._wandb = None

    def init(self, config: dict) -> None:
        # Extract params from config — support both nested logger.params and
        # top-level wandb key for convenience
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

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _try_init_wandb(config: dict, params: dict):
        """Attempt to initialise W&B. Returns the wandb module or None."""
        if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
            return None
        try:
            import wandb  # type: ignore
        except ImportError:
            return None

        # Check credentials
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

            # Build a flat config dict for W&B (exclude non-serializable keys)
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


# ── Stdout Logger ────────────────────────────────────────────────────────────

class StdoutLogger(MetricLogger):
    """CLI-only logging. No external dependencies."""

    @classmethod
    def default_params(cls) -> dict:
        return {}

    def init(self, config: dict) -> None:
        pass

    def log(self, step: int, **metrics: Any) -> None:
        parts = '  '.join(
            f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
            for k, v in metrics.items()
        )
        print(f'[step {step:07d}] {parts}')

    def finish(self) -> None:
        pass


# ── TensorBoard Logger ──────────────────────────────────────────────────────

class TensorBoardLogger(MetricLogger):
    """TensorBoard logging."""

    @classmethod
    def default_params(cls) -> dict:
        return {'log_dir': 'runs/'}

    def __init__(self) -> None:
        self._writer = None

    def init(self, config: dict) -> None:
        from torch.utils.tensorboard import SummaryWriter

        logger_section = config.get('logger', {})
        params = {**self.default_params(), **logger_section.get('params', {})}
        self._writer = SummaryWriter(log_dir=params.get('log_dir', 'runs/'))

    def log(self, step: int, **metrics: Any) -> None:
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, step)

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.close()


# ── Metrics Registry (unchanged) ────────────────────────────────────────────

class MetricsRegistry:
    """
    Collects custom metrics from registered providers for main-thread logging.

    Components (opponent pools, reward functions, encoders, etc.) register a
    callable that returns a flat dict of metric names to scalar values. The
    main training loop calls ``collect()`` at each logging point, which
    invokes every provider and returns a single merged dict with namespaced
    keys (``"{namespace}/{key}"``).

    This keeps wandb.log() calls exclusively in the main thread, avoiding
    step-count race conditions from background threads or subprocesses.

    Usage
    -----
        registry = MetricsRegistry()
        registry.register('frozen_self_play', pool.get_metrics)
        registry.register('reward', reward_fn.get_metrics)

        # In the main-thread logging block:
        metrics = registry.collect()
        wandb.log(metrics, step=total_step)

    Provider contract
    -----------------
    Each provider callable must:
      - Accept no arguments: ``def get_metrics() -> dict[str, float | int]``
      - Return a flat dict of metric names to scalar values
      - Be safe to call from the main thread (read-only access to shared state)
    """

    def __init__(self) -> None:
        self._providers: OrderedDict[str, Callable[[], Dict[str, Any]]] = OrderedDict()

    def register(self, namespace: str, provider: Callable[[], Dict[str, Any]]) -> None:
        """Register a metrics provider under the given namespace.

        Parameters
        ----------
        namespace : str
            Prefix for all keys returned by this provider (e.g. 'frozen_self_play').
        provider : callable
            No-arg callable returning ``{metric_name: scalar_value}``.
        """
        self._providers[namespace] = provider

    def unregister(self, namespace: str) -> None:
        """Remove a previously registered provider."""
        self._providers.pop(namespace, None)

    def collect(self) -> Dict[str, Any]:
        """Call all providers and return a merged, namespaced metrics dict.

        Keys are formatted as ``"{namespace}/{metric_name}"``.
        If a provider raises an exception it is silently skipped and a warning
        is printed to stderr.
        """
        merged: Dict[str, Any] = {}
        for namespace, provider in self._providers.items():
            try:
                raw = provider()
            except Exception as exc:
                print(f'[metrics] Provider {namespace!r} failed: {exc}',
                      file=sys.stderr)
                continue
            for key, value in raw.items():
                merged[f'{namespace}/{key}'] = value
        return merged

    @property
    def namespaces(self) -> list[str]:
        """Return list of registered namespace names."""
        return list(self._providers.keys())
