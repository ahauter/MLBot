"""Stdout-only MetricLogger. Zero external dependencies."""
from __future__ import annotations

from typing import Any

from training.abstractions import MetricLogger


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
