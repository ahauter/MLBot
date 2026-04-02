"""
SE3 Spectral Convergence Evaluation
====================================
Evaluate how well the SE3 spectral field's LMS filter tracks real Rocket League
trajectories from parsed replay data.

Loads replay .npz files via TrajectoryDataset, runs the numpy coefficient update
loop, and reports per-object / per-dimension reconstruction residuals.

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
from typing import Dict, Optional, Tuple

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "training"))

from se3_field import (
    SE3Encoder,
    COEFF_DIM,
    D_AMP,
    K,
    N_CHANNELS,
    N_OBJECTS,
    OBJECTS,
    RAW_STATE_DIM,
    _BALL,
    _BALL_OFF,
    _EGO,
    _EGO_OFF,
    _OPP,
    _OPP_OFF,
    _STADIUM,
    _TEAM,
    make_initial_coefficients,
    update_coefficients_np,
)
from trajectory_dataset import TrajectoryDataset

# Amplitude dimension labels for reporting
AMP_DIM_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "boost",
    "has_flip",
    "on_ground",
]


# ── helpers ──────────────────────────────────────────────────────────────────


def extract_encoder_params(encoder: SE3Encoder) -> Dict[str, np.ndarray]:
    """Extract SE3Encoder's learned parameters as numpy arrays."""
    import torch

    with torch.no_grad():
        q = encoder.quaternions / encoder.quaternions.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        return {
            "k_spatial": encoder.k_spatial.cpu().numpy(),
            "quaternions": q.cpu().numpy(),
            "lr": np.exp(encoder.log_lr.cpu().numpy()),
            "W_interact": encoder.W_interact.cpu().numpy(),
        }


def numpy_amplitude_targets(raw_state: np.ndarray) -> np.ndarray:
    """Extract (N_OBJECTS, D_AMP) amplitude targets from a single raw state.

    Mirrors pretrain_se3.build_amplitude_targets but in pure numpy.
    """
    targets = np.zeros((N_OBJECTS, D_AMP), dtype=np.float32)

    # Ball: pos(3) + ang_vel(3) + zeros(3)
    targets[_BALL, :3] = raw_state[_BALL_OFF : _BALL_OFF + 3]
    targets[_BALL, 3:6] = raw_state[_BALL_OFF + 6 : _BALL_OFF + 9]

    # Ego: pos(3) + ang_vel(3) + boost + has_flip + on_ground
    targets[_EGO, :3] = raw_state[_EGO_OFF : _EGO_OFF + 3]
    targets[_EGO, 3:6] = raw_state[_EGO_OFF + 10 : _EGO_OFF + 13]
    targets[_EGO, 6] = raw_state[_EGO_OFF + 13]
    targets[_EGO, 7] = raw_state[_EGO_OFF + 14]
    targets[_EGO, 8] = raw_state[_EGO_OFF + 15]

    # Team = ego in 1v1
    targets[_TEAM] = targets[_EGO]

    # Opponent
    targets[_OPP, :3] = raw_state[_OPP_OFF : _OPP_OFF + 3]
    targets[_OPP, 3:6] = raw_state[_OPP_OFF + 10 : _OPP_OFF + 13]
    targets[_OPP, 6] = raw_state[_OPP_OFF + 13]
    targets[_OPP, 7] = raw_state[_OPP_OFF + 14]
    targets[_OPP, 8] = raw_state[_OPP_OFF + 15]

    # Stadium: all zeros (already initialized)
    return targets


def numpy_reconstruct(
    coeff: np.ndarray,
    k_spatial: np.ndarray,
    quaternions: np.ndarray,
    raw_state: np.ndarray,
) -> np.ndarray:
    """Reconstruct amplitude predictions from coefficients.

    Returns (N_OBJECTS, D_AMP) predicted amplitudes via
    Re[f] = sum_k (a_k * cos(phase_k) * orient_k - b_k * sin(phase_k) * orient_k)
    """
    c = coeff.reshape(N_OBJECTS, K, D_AMP, N_CHANNELS)
    targets = numpy_amplitude_targets(raw_state)

    id_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    orientations = np.tile(id_q, (N_OBJECTS, 1))
    orientations[_EGO] = raw_state[_EGO_OFF + 6 : _EGO_OFF + 10]
    orientations[_TEAM] = orientations[_EGO]
    orientations[_OPP] = raw_state[_OPP_OFF + 6 : _OPP_OFF + 10]

    predicted = np.zeros((N_OBJECTS, D_AMP), dtype=np.float32)
    for obj in range(N_OBJECTS):
        phase = k_spatial[obj] @ targets[obj]  # (K,)
        orient = (quaternions[obj] * orientations[obj]).sum(axis=-1)  # (K,)
        bc = (np.cos(phase) * orient)[:, np.newaxis]  # (K, 1)
        bs = (np.sin(phase) * orient)[:, np.newaxis]
        predicted[obj] = (bc * c[obj, :, :, 0] - bs * c[obj, :, :, 1]).sum(axis=0)

    return predicted


# ── core evaluation ──────────────────────────────────────────────────────────


def evaluate_convergence(
    params: Dict[str, np.ndarray],
    dataset: TrajectoryDataset,
    max_windows: int = 200,
) -> Dict[str, np.ndarray]:
    """Run LMS update on trajectory windows and collect per-step residuals.

    Returns dict:
        residual_curve: (W, N_OBJECTS, D_AMP) — mean MSE per step
        final_residual: (N_OBJECTS, D_AMP)    — residual at last step
        convergence_step: (N_OBJECTS,)        — first step where object MSE < 0.01
        steady_state: (N_OBJECTS, D_AMP)      — mean over last 25% of window
        object_summary: (N_OBJECTS,)          — final residual per object
        dim_summary: (D_AMP,)                 — final residual per dimension
    """
    k_np = params["k_spatial"]
    q_np = params["quaternions"]
    lr_np = params["lr"]
    W_np = params["W_interact"]

    n_windows = min(max_windows, len(dataset))
    if n_windows == 0:
        raise ValueError("Dataset is empty — no trajectory windows to evaluate.")

    # Get window length from first item
    first_window = dataset[0].numpy()
    W = first_window.shape[0]

    # Accumulate residuals: (W, N_OBJECTS, D_AMP)
    residual_sum = np.zeros((W, N_OBJECTS, D_AMP), dtype=np.float64)

    t0 = time.time()
    for wi in range(n_windows):
        window = dataset[wi].numpy()  # (W, RAW_STATE_DIM)
        coeff = make_initial_coefficients()

        for t in range(W):
            raw_t = window[t]
            coeff = update_coefficients_np(k_np, q_np, lr_np, coeff, raw_t, W_interact=W_np)

            # Compute reconstruction and residual
            targets = numpy_amplitude_targets(raw_t)
            predicted = numpy_reconstruct(coeff, k_np, q_np, raw_t)
            residual_sum[t] += (targets - predicted) ** 2

        if (wi + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{wi + 1}/{n_windows}] windows processed ({elapsed:.1f}s)")

    # Average over windows
    residual_curve = (residual_sum / n_windows).astype(np.float32)
    final_residual = residual_curve[-1]
    steady_start = int(W * 0.75)
    steady_state = residual_curve[steady_start:].mean(axis=0)

    # Convergence step: first step where per-object mean MSE < 0.01
    object_curve = residual_curve.mean(axis=-1)  # (W, N_OBJECTS)
    convergence_step = np.full(N_OBJECTS, W, dtype=np.int32)
    for obj in range(N_OBJECTS):
        below = np.where(object_curve[:, obj] < 0.01)[0]
        if len(below) > 0:
            convergence_step[obj] = below[0]

    return {
        "residual_curve": residual_curve,
        "final_residual": final_residual,
        "convergence_step": convergence_step,
        "steady_state": steady_state,
        "object_summary": final_residual.mean(axis=-1),
        "dim_summary": final_residual.mean(axis=0),
        "n_windows": n_windows,
        "window_len": W,
    }


# ── reporting ────────────────────────────────────────────────────────────────


def print_report(results: Dict[str, np.ndarray], label: str = "") -> None:
    """Print a formatted convergence report."""
    W = results["window_len"]
    n = results["n_windows"]
    header = f"SE3 Spectral Convergence Report"
    if label:
        header += f" ({label})"

    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"Windows evaluated: {n}")
    print(f"Window length: {W} steps")

    # Per-object final residual
    print(f"\nPer-Object Final Residual (MSE):")
    for obj in range(N_OBJECTS):
        name = OBJECTS[obj]
        mse = results["object_summary"][obj]
        step = results["convergence_step"][obj]
        conv_str = f"converged at step {step}" if step < W else "did not converge"
        extra = ""
        if obj == _TEAM:
            extra = " (= ego)"
        print(f"  {name:<12s} {mse:.6f}  ({conv_str}){extra}")

    # Per-dimension final residual
    print(f"\nPer-Dimension Final Residual (MSE, avg over objects):")
    for d in range(D_AMP):
        print(f"  {AMP_DIM_NAMES[d]:<12s} {results['dim_summary'][d]:.6f}")

    # Steady-state
    print(f"\nSteady-State Residual (last 25% of window):")
    ss = results["steady_state"]
    for obj in range(N_OBJECTS):
        name = OBJECTS[obj]
        print(f"  {name:<12s} {ss[obj].mean():.6f}")

    overall = results["final_residual"].mean()
    n_converged = (results["convergence_step"] < W).sum()
    print(f"\nOverall: {overall:.6f} MSE | Converged: {n_converged}/{N_OBJECTS} objects within {W} steps")
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
        print("matplotlib not installed — skipping plot")
        return

    curve = results["residual_curve"]  # (W, N_OBJECTS, D_AMP)
    W = curve.shape[0]
    steps = np.arange(W)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-object convergence
    ax = axes[0]
    for obj in range(N_OBJECTS):
        obj_mse = curve[:, obj, :].mean(axis=-1)
        style = "-" if obj != _TEAM else "--"
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

    # Right: per-dimension convergence (averaged over objects)
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

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SE3 spectral field convergence on replay data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training/data/parsed/",
        help="Directory with parsed .npz replay files",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Path to pretrained encoder state_dict (.pt)",
    )
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="Also evaluate a random-init encoder for comparison",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200,
        help="Max trajectory windows to evaluate",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=64,
        help="Trajectory window length",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride between windows",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Save convergence plot to this path (e.g. convergence.png)",
    )
    args = parser.parse_args()

    import torch

    # Load dataset
    print(f"Loading trajectories from {args.data_dir}...")
    dataset = TrajectoryDataset(
        args.data_dir,
        window_len=args.window_len,
        stride=args.stride,
    )
    print(f"  {len(dataset)} windows available")

    if len(dataset) == 0:
        print("No trajectory data found. Make sure --data-dir points to parsed .npz files.")
        print("  To generate: python training/data/collect_replays.py --help")
        sys.exit(1)

    # Build encoder and extract params
    encoder = SE3Encoder()
    if args.encoder:
        print(f"Loading encoder from {args.encoder}...")
        state = torch.load(args.encoder, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state)
    params = extract_encoder_params(encoder)

    label = Path(args.encoder).stem if args.encoder else "random init"
    print(f"\nEvaluating ({label})...")
    results = evaluate_convergence(params, dataset, max_windows=args.max_windows)
    print_report(results, label=label)

    compare_results = None
    if args.compare_random and args.encoder:
        print("Evaluating (random init)...")
        random_encoder = SE3Encoder()
        random_params = extract_encoder_params(random_encoder)
        compare_results = evaluate_convergence(
            random_params, dataset, max_windows=args.max_windows
        )
        print_report(compare_results, label="random init")

    if args.plot:
        save_plot(results, args.plot, compare_results=compare_results)


if __name__ == "__main__":
    main()
