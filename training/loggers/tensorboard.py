"""TensorBoard MetricLogger."""
from __future__ import annotations

from typing import Any

from training.abstractions import MetricLogger


class TensorBoardLogger(MetricLogger):
    """TensorBoard logging. Requires tensorboard package."""

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
