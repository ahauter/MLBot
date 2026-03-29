from training.loggers.wandb import WandbLogger
from training.loggers.stdout import StdoutLogger
from training.loggers.tensorboard import TensorBoardLogger
from training.loggers.registry import MetricsRegistry

__all__ = ['WandbLogger', 'StdoutLogger', 'TensorBoardLogger', 'MetricsRegistry']
