"""
SE3 Spectral Field Pretraining
===============================
Optimises the SE3Encoder's learned field geometry (k_spatial, quaternions,
log_lr, W_interact) and optionally interaction conv parameters on real
trajectory data to minimise reconstruction residual.

This is an Axis 5 (pre-training compute) intervention: the hypothesis is that
pre-optimised spectral geometry reduces the simulation steps needed during RL.

The pretraining uses truncated BPTT: for each trajectory window, coefficients
are unrolled step-by-step through the encoder's _update_coefficients() method.
At each step the reconstruction error (predicted vs actual amplitude targets)
is accumulated. Gradients flow through a single step (detach between steps)
to keep memory bounded.

Usage:
    python training/pretrain_se3.py --data-dir training/data/parsed/ --epochs 50

    # Load pretrained encoder into RL training:
    python training/algorithms/se3_ppo.py --pretrained-encoder models/pretrained_se3/encoder.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'training'))

from se3_field import (
    SE3Encoder, SE3_OBS_DIM, RAW_STATE_DIM, COEFF_DIM, ACCEL_HIST_DIM,
    D_AMP, N_OBJECTS, K, N_CHANNELS,
    make_initial_coefficients, make_initial_accel_hist, pack_observation,
    _BALL_OFF, _EGO_OFF, _OPP_OFF,
)
from trajectory_dataset import TrajectoryDataset


def build_amplitude_targets(raw: torch.Tensor) -> torch.Tensor:
    """Extract ground-truth 9d amplitude targets from raw state.

    raw: (batch, RAW_STATE_DIM)
    returns: (batch, N_OBJECTS, D_AMP)
    """
    batch = raw.shape[0]
    device = raw.device
    targets = torch.zeros(batch, N_OBJECTS, D_AMP, device=device)

    # Ball: pos(3) + ang_vel(3) + zeros(3)
    targets[:, 0, :3] = raw[:, _BALL_OFF:_BALL_OFF + 3]
    targets[:, 0, 3:6] = raw[:, _BALL_OFF + 6:_BALL_OFF + 9]

    # Ego: pos(3) + ang_vel(3) + boost + has_flip + on_ground
    targets[:, 1, :3] = raw[:, _EGO_OFF:_EGO_OFF + 3]
    targets[:, 1, 3:6] = raw[:, _EGO_OFF + 10:_EGO_OFF + 13]
    targets[:, 1, 6] = raw[:, _EGO_OFF + 13]
    targets[:, 1, 7] = raw[:, _EGO_OFF + 14]
    targets[:, 1, 8] = raw[:, _EGO_OFF + 15]

    # Team = ego
    targets[:, 2] = targets[:, 1]

    # Opp
    targets[:, 3, :3] = raw[:, _OPP_OFF:_OPP_OFF + 3]
    targets[:, 3, 3:6] = raw[:, _OPP_OFF + 10:_OPP_OFF + 13]
    targets[:, 3, 6] = raw[:, _OPP_OFF + 13]
    targets[:, 3, 7] = raw[:, _OPP_OFF + 14]
    targets[:, 3, 8] = raw[:, _OPP_OFF + 15]

    # Stadium = zeros (already initialised)
    return targets


def reconstruct_amplitudes(
    coeff_5d: torch.Tensor,
    basis_cos: torch.Tensor,
    basis_sin: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct amplitude predictions from coefficients and basis functions.

    coeff_5d: (batch, N_OBJECTS, K, D_AMP, N_CHANNELS)
    basis_cos/sin: (batch, N_OBJECTS, K) — scalar basis per component
    returns: (batch, N_OBJECTS, D_AMP)
    """
    ca = coeff_5d[:, :, :, :, 0]   # (batch, N_OBJECTS, K, D_AMP)
    cb = coeff_5d[:, :, :, :, 1]
    bc = basis_cos.unsqueeze(-1)    # (batch, N_OBJECTS, K, 1)
    bs = basis_sin.unsqueeze(-1)
    return (bc * ca - bs * cb).sum(dim=2)  # (batch, N_OBJECTS, D_AMP)


def pretrain(args):
    device = torch.device(args.device)

    # Load dataset
    print(f'Loading trajectories from {args.data_dir}...')
    dataset = TrajectoryDataset(
        args.data_dir,
        window_len=args.window_len,
        stride=args.stride,
        min_episode_len=args.min_episode_len,
    )
    print(f'  {len(dataset)} windows ({args.window_len} steps each)')

    if len(dataset) == 0:
        print('No trajectory data found. Exiting.')
        return

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'))

    # Encoder
    encoder = SE3Encoder(momentum_mode=args.momentum_mode).to(device)
    encoder.train()

    # Optimizer — field geometry + optionally conv params
    if args.pretrain_conv:
        params = list(encoder.parameters())
    else:
        params = [encoder.k_spatial, encoder.quaternions, encoder.log_lr, encoder.W_interact]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Loss weights per amplitude dimension
    loss_weights = torch.ones(D_AMP, device=device)
    loss_weights[:3] = 1.0    # position
    loss_weights[3:6] = 0.5   # angular velocity
    loss_weights[6] = 0.2     # boost
    loss_weights[7] = 0.2     # has_flip
    loss_weights[8] = 0.2     # on_ground
    loss_weights = loss_weights / loss_weights.sum() * D_AMP  # normalise

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B logging
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project='rlbot-baseline',
                name=f'pretrain-se3-seed{args.seed}',
                config=vars(args),
            )
        except ImportError:
            print('wandb not installed, logging to stdout only')

    best_loss = float('inf')
    torch.manual_seed(args.seed)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_pos_loss = 0.0
        epoch_ang_loss = 0.0
        epoch_scalar_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_windows in loader:
            # batch_windows: (batch, window_len, RAW_STATE_DIM)
            batch_windows = batch_windows.to(device)
            B, W, _ = batch_windows.shape

            # Initialise coefficients and accel history to zero
            coefficients = torch.zeros(B, COEFF_DIM, device=device)
            accel_hist = torch.zeros(B, ACCEL_HIST_DIM, device=device)

            total_loss = torch.tensor(0.0, device=device)
            total_pos = 0.0
            total_ang = 0.0
            total_scalar = 0.0

            for t in range(W):
                raw_t = batch_windows[:, t, :]
                packed = torch.cat([raw_t, coefficients, accel_hist], dim=-1)

                # Differentiable encoder forward
                (new_coeff, basis_cos, basis_sin, _, coeff_5d,
                 _, _, new_accel_hist, _) = encoder._update_coefficients(packed)

                # Reconstruction loss
                targets = build_amplitude_targets(raw_t)
                predicted = reconstruct_amplitudes(coeff_5d, basis_cos, basis_sin)

                # Weighted MSE per amplitude dimension
                err = (predicted - targets).pow(2)   # (B, N_OBJECTS, D_AMP)
                weighted_err = err * loss_weights.unsqueeze(0).unsqueeze(0)
                loss_t = weighted_err.mean()
                total_loss = total_loss + loss_t

                # Per-category losses for logging
                with torch.no_grad():
                    total_pos += err[:, :, :3].mean().item()
                    total_ang += err[:, :, 3:6].mean().item()
                    total_scalar += err[:, :, 6:].mean().item()

                # Detach between steps (truncated BPTT depth 1)
                coefficients = new_coeff.detach()
                accel_hist = new_accel_hist.detach()

            avg_loss = total_loss / W
            optimizer.zero_grad()
            avg_loss.backward()
            nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            encoder.normalise_quaternions_()

            epoch_loss += avg_loss.item()
            epoch_pos_loss += total_pos / W
            epoch_ang_loss += total_ang / W
            epoch_scalar_loss += total_scalar / W
            n_batches += 1

        # Epoch summary
        dt = time.time() - t0
        avg = epoch_loss / max(n_batches, 1)
        avg_pos = epoch_pos_loss / max(n_batches, 1)
        avg_ang = epoch_ang_loss / max(n_batches, 1)
        avg_scl = epoch_scalar_loss / max(n_batches, 1)

        print(f'Epoch {epoch + 1:3d}/{args.epochs} | '
              f'loss={avg:.6f} pos={avg_pos:.6f} ang={avg_ang:.6f} scl={avg_scl:.6f} | '
              f'{dt:.1f}s')

        if wandb_run is not None:
            wandb_run.log({
                'pretrain/loss': avg,
                'pretrain/pos_loss': avg_pos,
                'pretrain/ang_vel_loss': avg_ang,
                'pretrain/scalar_loss': avg_scl,
                'pretrain/epoch': epoch + 1,
            })

        # Save best
        if avg < best_loss:
            best_loss = avg
            torch.save(encoder.state_dict(), output_dir / 'encoder.pt')
            print(f'  → saved best (loss={best_loss:.6f})')

    # Save final
    torch.save(encoder.state_dict(), output_dir / 'encoder_final.pt')
    print(f'\nPretraining complete. Best loss: {best_loss:.6f}')
    print(f'Saved to {output_dir}')

    if wandb_run is not None:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description='SE3 Spectral Field Pretraining')
    parser.add_argument('--data-dir', type=str, default='training/data/parsed/',
                        help='Directory with parsed .npz replay files')
    parser.add_argument('--window-len', type=int, default=64,
                        help='Trajectory window length for truncated BPTT')
    parser.add_argument('--stride', type=int, default=32,
                        help='Stride between windows')
    parser.add_argument('--min-episode-len', type=int, default=100,
                        help='Skip episodes shorter than this')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='models/pretrained_se3/')
    parser.add_argument('--momentum-mode', type=str, default='correction',
                        choices=['additive', 'correction', 'both'],
                        help='SE3Encoder momentum mode (default: correction)')
    parser.add_argument('--pretrain-conv', action='store_true',
                        help='Also pretrain interaction conv parameters')
    parser.add_argument('--wandb', action='store_true',
                        help='Log to Weights & Biases')
    args = parser.parse_args()
    pretrain(args)


if __name__ == '__main__':
    main()
