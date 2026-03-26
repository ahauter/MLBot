"""
train_spectral.py
=================
Offline RL training using SE(3) spectral field observations.

Loads processed replay .npz files, converts tokens to 105-dim spectral
observations, and trains an AWAC policy using d3rlpy.

Usage
-----
    python -m rlbot.training.train_spectral /path/to/replay/dir
    python -m rlbot.training.train_spectral /path/to/replay/dir --n-steps 100000 --batch-size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

from d3rlpy.algos import AWACConfig

from rlbot.training.spectral_dataset import load_spectral_dataset
from rlbot.training.spectral_encoder_factory import SpectralEncoderFactory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AWAC policy from replays using spectral field observations"
    )
    parser.add_argument("replay_dir", type=str, help="Directory containing .npz replay files")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=500_000)
    parser.add_argument("--n-steps-per-epoch", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=1.0, help="AWAC advantage temperature")
    parser.add_argument("--n-critics", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden layer width")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of MLP hidden layers")
    parser.add_argument("--output-dir", type=str, default="spectral_checkpoints")
    parser.add_argument("--experiment-name", type=str, default="spectral_awac")
    args = parser.parse_args()

    # Load and convert replay data
    print(f"Loading replays from {args.replay_dir}...")
    dataset = load_spectral_dataset(args.replay_dir)

    # Configure encoder
    factory = SpectralEncoderFactory(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )

    # Configure AWAC
    config = AWACConfig(
        batch_size=args.batch_size,
        gamma=args.gamma,
        actor_learning_rate=args.lr,
        critic_learning_rate=args.lr,
        actor_encoder_factory=factory,
        critic_encoder_factory=factory,
        lam=args.lam,
        n_critics=args.n_critics,
    )
    algo = config.create()

    # Train offline
    print(f"Training for {args.n_steps} steps (batch_size={args.batch_size})...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        experiment_name=args.experiment_name,
        show_progress=True,
    )

    # Save final model
    model_path = output_dir / "spectral_awac_final.d3"
    algo.save(str(model_path))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
