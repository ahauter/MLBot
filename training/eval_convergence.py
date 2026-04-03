"""
SE3 Spectral Convergence Evaluation
====================================
Evaluate how well the SE3 spectral field's LMS filter tracks real Rocket League
trajectories from parsed replay data.

Loads replay .npz files in chunks, runs the real SE3Encoder._update_coefficients
on GPU in batches, and reports per-object / per-dimension reconstruction residuals.

Usage:
    # Random-init encoder on replay data
    python training/eval_convergence.py --data-dir training/data/parsed/

    # Pretrained encoder
    python training/eval_convergence.py --data-dir training/data/parsed/ \
        --encoder models/pretrained_se3/encoder.pt

    # Compare pretrained vs random init
    python training/eval_convergence.py --data-dir training/data/parsed/ \
        --encoder models/pretrained_se3/encoder.pt --compare-random

    # Save convergence plot
    python training/eval_convergence.py --data-dir training/data/parsed/ \
        --plot convergence.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Generator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "training"))

from se3_field import (
    SE3Encoder,
    ACCEL_HIST_DIM,
    COEFF_DIM,
    EMBED_DIM,
    D_AMP,
    N_OBJECTS,
    OBJECTS,
    _TEAM,
)
from trajectory_dataset import TrajectoryDataset
from pretrain_se3 import build_amplitude_targets, reconstruct_amplitudes

# Amplitude dimension labels for reporting
AMP_DIM_NAMES = [
    "pos_x", "pos_y", "pos_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "boost", "has_flip", "on_ground",
]


# ── data loading ──────────────────────────────────────────────────────────────


def stream_dataset_chunks(
    data_dir: str,
    window_len: int,
    stride: int,
    files_per_chunk: int = 10,
) -> Generator[TrajectoryDataset, None, None]:
    """Yield TrajectoryDataset instances one chunk of .npz files at a time.

    Avoids loading all replay files into memory simultaneously.
    """
    paths = sorted(Path(data_dir).glob("*.npz"))
    for i in range(0, len(paths), files_per_chunk):
        yield TrajectoryDataset(
            data_dir,
            window_len=window_len,
            stride=stride,
            npz_files=paths[i : i + files_per_chunk],
        )


# ── core evaluation ───────────────────────────────────────────────────────────


def evaluate_convergence(
    encoder: SE3Encoder,
    data_dir: str,
    window_len: int,
    stride: int,
    max_windows: int,
    batch_size: int,
    files_per_chunk: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run SE3Encoder._update_coefficients on trajectory windows and collect residuals.

    Returns dict:
        residual_curve: (W, N_OBJECTS, D_AMP) — mean MSE per step
        final_residual: (N_OBJECTS, D_AMP)    — residual at last step
        convergence_step: (N_OBJECTS,)        — first step where object MSE < 0.01
        steady_state: (N_OBJECTS, D_AMP)      — mean over last 25% of window
        object_summary: (N_OBJECTS,)          — final residual per object
        dim_summary: (D_AMP,)                 — final residual per dimension
        delta_norm_curve: (W, N_OBJECTS)      — mean accel delta norm per step
        surprise_norm_curve: (W, N_OBJECTS)   — mean accel surprise norm per step
    """
    encoder.eval().to(device)

    residual_sum = torch.zeros(window_len, N_OBJECTS, D_AMP, device=device)
    delta_norm_sum = torch.zeros(window_len, N_OBJECTS, device=device)
    surprise_norm_sum = torch.zeros(window_len, N_OBJECTS, device=device)
    embed_norm_sum = torch.zeros(window_len, device=device)
    embed_var_sum = torch.zeros(window_len, device=device)
    n_windows = 0
    t0 = time.time()

    for chunk_ds in stream_dataset_chunks(data_dir, window_len, stride, files_per_chunk):
        if n_windows >= max_windows:
            break
        loader = DataLoader(chunk_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        for batch in loader:
            if n_windows >= max_windows:
                break
            batch = batch.to(device)                # (B, W, RAW_STATE_DIM)
            B = batch.shape[0]
            coefficients = torch.zeros(B, COEFF_DIM, device=device)
            accel_hist = torch.zeros(B, ACCEL_HIST_DIM, device=device)
            batch_start = n_windows

            for t in range(window_len):
                t_step = time.time()
                raw_t = batch[:, t, :]              # (B, RAW_STATE_DIM)
                packed = torch.cat([raw_t, coefficients, accel_hist], dim=-1)

                with torch.no_grad():
                    (new_coeff, basis_cos, basis_sin, _, coeff_5d,
                     accel_delta, accel_surprise, new_accel_hist, _) = \
                        encoder._update_coefficients(packed)

                targets   = build_amplitude_targets(raw_t)
                predicted = reconstruct_amplitudes(coeff_5d, basis_cos, basis_sin)
                step_residual = (targets - predicted).pow(2)    # (B, N_OBJECTS, D_AMP)
                residual_sum[t] += step_residual.mean(dim=0)

                # Track momentum signal norms
                delta_norm_sum[t] += accel_delta.norm(dim=-1).mean(dim=0)      # (N_OBJECTS,)
                surprise_norm_sum[t] += accel_surprise.norm(dim=-1).mean(dim=0)

                # Policy embedding metrics (exercises correction path)
                embed, _, _ = encoder.encode_for_policy(packed)
                embed_norm_sum[t] += embed.norm(dim=-1).mean()          # scalar
                embed_var_sum[t] += embed.var(dim=-1).mean()            # scalar

                coefficients = new_coeff.detach()
                accel_hist = new_accel_hist.detach()

                step_ms = (time.time() - t_step) * 1000
                print(
                    f"  windows {batch_start + 1}\u2013{batch_start + B}/{max_windows}"
                    f"  step {t + 1}/{window_len}"
                    f"  mse={step_residual.mean().item():.6f}"
                    f"  ({step_ms:.2f}ms)"
                )

            n_windows += B
            elapsed = time.time() - t0
            print(f"  [{n_windows}/{max_windows}] windows processed ({elapsed:.1f}s)")

    if n_windows == 0:
        raise ValueError("No trajectory windows found \u2014 check --data-dir.")

    residual_curve = (residual_sum / n_windows).cpu().numpy().astype(np.float32)
    final_residual = residual_curve[-1]
    steady_start = int(window_len * 0.75)
    steady_state = residual_curve[steady_start:].mean(axis=0)

    object_curve = residual_curve.mean(axis=-1)     # (W, N_OBJECTS)
    convergence_step = np.full(N_OBJECTS, window_len, dtype=np.int32)
    for obj in range(N_OBJECTS):
        below = np.where(object_curve[:, obj] < 0.01)[0]
        if len(below) > 0:
            convergence_step[obj] = below[0]

    # Momentum signal curves
    delta_norm_curve = (delta_norm_sum / max(n_windows, 1)).cpu().numpy().astype(np.float32)
    surprise_norm_curve = (surprise_norm_sum / max(n_windows, 1)).cpu().numpy().astype(np.float32)
    embed_norm_curve = (embed_norm_sum / max(n_windows, 1)).cpu().numpy().astype(np.float32)
    embed_var_curve = (embed_var_sum / max(n_windows, 1)).cpu().numpy().astype(np.float32)

    return {
        "residual_curve": residual_curve,
        "final_residual": final_residual,
        "convergence_step": convergence_step,
        "steady_state": steady_state,
        "object_summary": final_residual.mean(axis=-1),
        "dim_summary": final_residual.mean(axis=0),
        "delta_norm_curve": delta_norm_curve,       # (W, N_OBJECTS)
        "surprise_norm_curve": surprise_norm_curve,  # (W, N_OBJECTS)
        "embed_norm_curve": embed_norm_curve,        # (W,)
        "embed_var_curve": embed_var_curve,          # (W,)
        "n_windows": n_windows,
        "window_len": window_len,
    }


# ── reporting ─────────────────────────────────────────────────────────────────


def print_report(results: Dict[str, np.ndarray], label: str = "",
                 momentum_mode: str = "correction") -> None:
    """Print a formatted convergence report."""
    W = results["window_len"]
    n = results["n_windows"]
    header = "SE3 Spectral Convergence Report"
    if label:
        header += f" ({label})"

    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    steady_start = int(W * 0.75)
    print(f"Momentum mode: {momentum_mode}")
    print(f"Windows evaluated: {n}")
    print(f"Window length: {W} steps")

    print(f"\nPer-Object Final Residual (MSE):")
    for obj in range(N_OBJECTS):
        name = OBJECTS[obj]
        mse = results["object_summary"][obj]
        step = results["convergence_step"][obj]
        conv_str = f"converged at step {step}" if step < W else "did not converge"
        extra = " (= ego)" if obj == _TEAM else ""
        print(f"  {name:<12s} {mse:.6f}  ({conv_str}){extra}")

    print(f"\nPer-Dimension Final Residual (MSE, avg over objects):")
    for d in range(D_AMP):
        print(f"  {AMP_DIM_NAMES[d]:<12s} {results['dim_summary'][d]:.6f}")

    print(f"\nSteady-State Residual (last 25% of window):")
    ss = results["steady_state"]
    for obj in range(N_OBJECTS):
        name = OBJECTS[obj]
        print(f"  {name:<12s} {ss[obj].mean():.6f}")

    # Action momentum signals
    if "delta_norm_curve" in results:
        print(f"\nAction Momentum (mean norm, last 25% of window):")
        delta_ss = results["delta_norm_curve"][steady_start:].mean(axis=0)
        surprise_ss = results["surprise_norm_curve"][steady_start:].mean(axis=0)
        print(f"  {'object':<12s} {'delta':>8s} {'surprise':>10s}")
        for obj in range(N_OBJECTS):
            name = OBJECTS[obj]
            print(f"  {name:<12s} {delta_ss[obj]:8.6f} {surprise_ss[obj]:10.6f}")

    # Policy embedding stats
    if "embed_norm_curve" in results:
        embed_norm = results["embed_norm_curve"]
        embed_var = results["embed_var_curve"]
        print(f"\nPolicy Embedding (EMBED_DIM={EMBED_DIM}):")
        print(f"  Mean L2 norm (final step):  {embed_norm[-1]:.6f}")
        print(f"  Mean L2 norm (steady-state): {embed_norm[steady_start:].mean():.6f}")
        print(f"  Std of norm across window:  {embed_norm.std():.6f}")
        print(f"  Mean variance (final step):  {embed_var[-1]:.6f}")
        print(f"  Mean variance (steady-state):{embed_var[steady_start:].mean():.6f}")

    overall = results["final_residual"].mean()
    n_converged = (results["convergence_step"] < W).sum()
    print(
        f"\nOverall: {overall:.6f} MSE"
        f" | Converged: {n_converged}/{N_OBJECTS} objects within {W} steps"
    )
    print(f"{'=' * 60}\n")


def save_plot(
    results: Dict[str, np.ndarray],
    path: str,
    compare_results: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save convergence curve plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed \u2014 skipping plot")
        return

    curve = results["residual_curve"]   # (W, N_OBJECTS, D_AMP)
    W = curve.shape[0]
    steps = np.arange(W)
    has_embed = "embed_norm_curve" in results

    ncols = 3 if has_embed else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

    ax = axes[0]
    for obj in range(N_OBJECTS):
        obj_mse = curve[:, obj, :].mean(axis=-1)
        style = "--" if obj == _TEAM else "-"
        ax.plot(steps, obj_mse, style, label=OBJECTS[obj], linewidth=1.5)

    if compare_results is not None:
        comp_curve = compare_results["residual_curve"]
        for obj in range(N_OBJECTS):
            if obj == _TEAM:
                continue
            obj_mse = comp_curve[:, obj, :].mean(axis=-1)
            ax.plot(steps, obj_mse, ":", alpha=0.5, label=f"{OBJECTS[obj]} (random)")

    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Object Convergence")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    groups = [("position", slice(0, 3)), ("ang_vel", slice(3, 6)), ("scalars", slice(6, 9))]
    for name, sl in groups:
        dim_mse = curve[:, :, sl].mean(axis=(1, 2))
        ax.plot(steps, dim_mse, label=name, linewidth=1.5)

    if compare_results is not None:
        comp_curve = compare_results["residual_curve"]
        for name, sl in groups:
            dim_mse = comp_curve[:, :, sl].mean(axis=(1, 2))
            ax.plot(steps, dim_mse, ":", alpha=0.5, label=f"{name} (random)")

    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Dimension Convergence")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if has_embed:
        ax = axes[2]
        ax.plot(steps, results["embed_norm_curve"], label="L2 norm", linewidth=1.5)
        ax.plot(steps, results["embed_var_curve"], label="variance", linewidth=1.5)
        if compare_results is not None and "embed_norm_curve" in compare_results:
            ax.plot(steps, compare_results["embed_norm_curve"], ":",
                    alpha=0.5, label="L2 norm (random)")
            ax.plot(steps, compare_results["embed_var_curve"], ":",
                    alpha=0.5, label="variance (random)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title("Policy Embedding")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SE3 spectral field convergence on replay data"
    )
    parser.add_argument(
        "--data-dir", type=str, default="training/data/parsed/",
        help="Directory with parsed .npz replay files",
    )
    parser.add_argument(
        "--encoder", type=str, default=None,
        help="Path to pretrained SE3Encoder state_dict (.pt)",
    )
    parser.add_argument(
        "--compare-random", action="store_true",
        help="Also evaluate a random-init encoder for comparison",
    )
    parser.add_argument(
        "--max-windows", type=int, default=200,
        help="Max trajectory windows to evaluate",
    )
    parser.add_argument(
        "--window-len", type=int, default=64,
        help="Trajectory window length",
    )
    parser.add_argument(
        "--stride", type=int, default=32,
        help="Stride between windows",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Windows per GPU batch",
    )
    parser.add_argument(
        "--files-per-chunk", type=int, default=10,
        help="Number of .npz files to load at a time",
    )
    parser.add_argument(
        "--momentum-mode", type=str, default="correction",
        choices=["additive", "correction", "both"],
        help="SE3Encoder momentum mode (default: correction)",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Save convergence plot to this path (e.g. convergence.png)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = SE3Encoder(momentum_mode=args.momentum_mode)
    if args.encoder:
        print(f"Loading encoder from {args.encoder}...")
        state = torch.load(args.encoder, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state)

    label = Path(args.encoder).stem if args.encoder else "random init"
    print(f"Hello i am done loading replays, i am now starting {label}...")

    eval_kwargs = dict(
        data_dir=args.data_dir,
        window_len=args.window_len,
        stride=args.stride,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
        files_per_chunk=args.files_per_chunk,
        device=device,
    )

    print(f"\nEvaluating ({label})...")
    results = evaluate_convergence(encoder, **eval_kwargs)
    print_report(results, label=label, momentum_mode=args.momentum_mode)

    compare_results = None
    if args.compare_random and args.encoder:
        print("Evaluating (random init)...")
        random_encoder = SE3Encoder(momentum_mode=args.momentum_mode)
        compare_results = evaluate_convergence(random_encoder, **eval_kwargs)
        print_report(compare_results, label="random init",
                     momentum_mode=args.momentum_mode)

    if args.plot:
        save_plot(results, args.plot, compare_results=compare_results)


if __name__ == "__main__":
    main()
