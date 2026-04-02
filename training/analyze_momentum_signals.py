"""
Momentum Signal Analysis: Delta vs Surprise
=============================================
Compare the two action momentum representations on real replay data to
determine if they are redundant or complementary.

- accel_delta:    simple difference of accel residuals (spikes on action changes)
- accel_surprise: residual vs EMA prediction (smoothed change detection)

Six metrics: per-dim breakdown, correlation, information content,
temporal autocorrelation, action-change SNR, and redundancy.

Usage:
    python training/analyze_momentum_signals.py --data-dir training/data/parsed/
    python training/analyze_momentum_signals.py --data-dir training/data/parsed/ \
        --plot momentum_analysis.png --output-json results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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
    D_AMP,
    N_OBJECTS,
    OBJECTS,
    _TEAM,
)
from eval_convergence import stream_dataset_chunks, AMP_DIM_NAMES


# ── collection ───────────────────────────────────────────────────────────────


def collect_momentum_signals(
    encoder: SE3Encoder,
    data_dir: str,
    window_len: int,
    stride: int,
    max_windows: int,
    batch_size: int,
    files_per_chunk: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Collect raw accel_delta, accel_surprise, and accel_residual tensors.

    Returns dict with arrays shaped (n_windows, window_len, N_OBJECTS, D_AMP).
    """
    encoder.eval().to(device)

    delta_list = []
    surprise_list = []
    accel_res_list = []
    n_windows = 0
    t0 = time.time()

    for chunk_ds in stream_dataset_chunks(data_dir, window_len, stride, files_per_chunk):
        if n_windows >= max_windows:
            break
        loader = DataLoader(chunk_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        for batch in loader:
            if n_windows >= max_windows:
                break
            batch = batch.to(device)
            B = batch.shape[0]
            coefficients = torch.zeros(B, COEFF_DIM, device=device)
            accel_hist = torch.zeros(B, ACCEL_HIST_DIM, device=device)

            window_delta = []
            window_surprise = []
            window_res = []

            for t in range(window_len):
                raw_t = batch[:, t, :]
                packed = torch.cat([raw_t, coefficients, accel_hist], dim=-1)

                with torch.no_grad():
                    (new_coeff, _, _, _, _,
                     accel_delta, accel_surprise, new_accel_hist) = \
                        encoder._update_coefficients(packed)

                # Extract accel_residual from the new_accel_hist (first half)
                _half = N_OBJECTS * D_AMP
                accel_res = new_accel_hist[:, :_half].reshape(B, N_OBJECTS, D_AMP)

                window_delta.append(accel_delta.cpu().numpy())
                window_surprise.append(accel_surprise.cpu().numpy())
                window_res.append(accel_res.cpu().numpy())

                coefficients = new_coeff.detach()
                accel_hist = new_accel_hist.detach()

            # Stack: (B, W, N_OBJECTS, D_AMP)
            delta_list.append(np.stack(window_delta, axis=1))
            surprise_list.append(np.stack(window_surprise, axis=1))
            accel_res_list.append(np.stack(window_res, axis=1))

            n_windows += B
            elapsed = time.time() - t0
            print(f"  [{n_windows}/{max_windows}] windows collected ({elapsed:.1f}s)")

    if n_windows == 0:
        raise ValueError("No trajectory windows found — check --data-dir.")

    # Concatenate all batches: (total_windows, W, N_OBJECTS, D_AMP)
    delta_all = np.concatenate(delta_list, axis=0)[:max_windows]
    surprise_all = np.concatenate(surprise_list, axis=0)[:max_windows]
    accel_res_all = np.concatenate(accel_res_list, axis=0)[:max_windows]

    print(f"Collected {delta_all.shape[0]} windows in {time.time() - t0:.1f}s")
    return {
        "delta": delta_all,
        "surprise": surprise_all,
        "accel_residual": accel_res_all,
    }


# ── metrics ──────────────────────────────────────────────────────────────────


def compute_per_dim_breakdown(
    delta: np.ndarray, surprise: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Mean absolute value per dimension, per object, for both signals.

    delta/surprise: (n_windows, W, N_OBJECTS, D_AMP)
    Returns: delta_mag, surprise_mag — each (N_OBJECTS, D_AMP)
    """
    # Skip step 0 (delta is meaningless at cold start)
    delta_mag = np.abs(delta[:, 1:]).mean(axis=(0, 1))
    surprise_mag = np.abs(surprise[:, 1:]).mean(axis=(0, 1))
    return {"delta_mag": delta_mag, "surprise_mag": surprise_mag}


def compute_correlation(
    delta: np.ndarray, surprise: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pearson correlation between delta and surprise per object, per dim.

    Returns: corr_matrix (N_OBJECTS, D_AMP), corr_summary (N_OBJECTS,)
    """
    # Flatten windows and time: (n_samples, N_OBJECTS, D_AMP)
    d = delta[:, 1:].reshape(-1, N_OBJECTS, D_AMP)
    s = surprise[:, 1:].reshape(-1, N_OBJECTS, D_AMP)

    corr_matrix = np.zeros((N_OBJECTS, D_AMP), dtype=np.float64)
    for obj in range(N_OBJECTS):
        for dim in range(D_AMP):
            dv = d[:, obj, dim]
            sv = s[:, obj, dim]
            if dv.std() < 1e-10 or sv.std() < 1e-10:
                corr_matrix[obj, dim] = 0.0
            else:
                corr_matrix[obj, dim] = np.corrcoef(dv, sv)[0, 1]

    corr_summary = corr_matrix.mean(axis=-1)
    return {"corr_matrix": corr_matrix, "corr_summary": corr_summary}


def compute_information_content(
    delta: np.ndarray, surprise: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Variance of each signal over time — higher = more informative.

    Returns: delta_var, surprise_var (N_OBJECTS, D_AMP), ratio = surprise/delta.
    """
    d = delta[:, 1:].reshape(-1, N_OBJECTS, D_AMP)
    s = surprise[:, 1:].reshape(-1, N_OBJECTS, D_AMP)

    delta_var = d.var(axis=0)
    surprise_var = s.var(axis=0)
    ratio = surprise_var / np.maximum(delta_var, 1e-12)
    return {"delta_var": delta_var, "surprise_var": surprise_var, "ratio": ratio}


def compute_temporal_autocorrelation(
    delta: np.ndarray, surprise: np.ndarray, max_lag: int = 20,
) -> Dict[str, np.ndarray]:
    """Autocorrelation at lags 1..max_lag, averaged across dims and windows.

    Returns: delta_autocorr, surprise_autocorr (N_OBJECTS, max_lag),
             delta_decorr_lag, surprise_decorr_lag (N_OBJECTS,).
    """
    def _autocorr(signal: np.ndarray, max_lag: int) -> np.ndarray:
        """signal: (n_windows, W, N_OBJECTS, D_AMP) → (N_OBJECTS, max_lag)"""
        # Average across dims and windows
        # Per window: compute autocorrelation per object
        n_win, W, n_obj, n_dim = signal.shape
        result = np.zeros((n_obj, max_lag), dtype=np.float64)
        count = 0
        for wi in range(min(n_win, 200)):  # cap for speed
            for dim in range(n_dim):
                for obj in range(n_obj):
                    x = signal[wi, 1:, obj, dim]  # skip step 0
                    x = x - x.mean()
                    var = x.var()
                    if var < 1e-12:
                        continue
                    for lag in range(max_lag):
                        if lag + 1 >= len(x):
                            break
                        c = np.mean(x[:len(x) - lag - 1] * x[lag + 1:])
                        result[obj, lag] += c / var
                count += 1
        if count > 0:
            result /= (count / n_obj)  # normalize per object
        return result

    delta_ac = _autocorr(delta, max_lag)
    surprise_ac = _autocorr(surprise, max_lag)

    # Decorrelation lag: first lag where autocorrelation < 0.5
    def _decorr_lag(ac: np.ndarray) -> np.ndarray:
        n_obj, ml = ac.shape
        lags = np.full(n_obj, ml, dtype=np.int32)
        for obj in range(n_obj):
            below = np.where(ac[obj] < 0.5)[0]
            if len(below) > 0:
                lags[obj] = below[0] + 1  # +1 because lag indexing starts at 1
        return lags

    return {
        "delta_autocorr": delta_ac,
        "surprise_autocorr": surprise_ac,
        "delta_decorr_lag": _decorr_lag(delta_ac),
        "surprise_decorr_lag": _decorr_lag(surprise_ac),
    }


def compute_action_change_snr(
    delta: np.ndarray, surprise: np.ndarray,
    accel_residual: np.ndarray, threshold: float = 0.1,
) -> Dict[str, np.ndarray]:
    """SNR: mean signal at action-change points / mean at non-change points.

    Action change = step where ||accel_res[t] - accel_res[t-1]|| > threshold.
    Returns: delta_snr, surprise_snr (N_OBJECTS,).
    """
    # Compute change magnitude from accel_residual
    # accel_res: (n_windows, W, N_OBJECTS, D_AMP)
    change = np.linalg.norm(
        accel_residual[:, 2:] - accel_residual[:, 1:-1],
        axis=-1)  # (n_windows, W-2, N_OBJECTS)

    delta_norms = np.linalg.norm(delta[:, 2:], axis=-1)       # (n_windows, W-2, N_OBJECTS)
    surprise_norms = np.linalg.norm(surprise[:, 2:], axis=-1)

    delta_snr = np.zeros(N_OBJECTS, dtype=np.float64)
    surprise_snr = np.zeros(N_OBJECTS, dtype=np.float64)

    for obj in range(N_OBJECTS):
        c = change[:, :, obj].ravel()
        d = delta_norms[:, :, obj].ravel()
        s = surprise_norms[:, :, obj].ravel()

        is_change = c > threshold
        is_stable = ~is_change

        if is_change.sum() < 10 or is_stable.sum() < 10:
            delta_snr[obj] = float('nan')
            surprise_snr[obj] = float('nan')
            continue

        delta_snr[obj] = d[is_change].mean() / max(d[is_stable].mean(), 1e-12)
        surprise_snr[obj] = s[is_change].mean() / max(s[is_stable].mean(), 1e-12)

    return {"delta_snr": delta_snr, "surprise_snr": surprise_snr}


def compute_redundancy(corr_matrix: np.ndarray) -> Dict[str, float]:
    """Fraction of (object, dim) pairs where |correlation| > 0.95."""
    n_total = corr_matrix.size
    n_redundant = (np.abs(corr_matrix) > 0.95).sum()
    frac = float(n_redundant) / max(n_total, 1)
    return {
        "n_redundant": int(n_redundant),
        "n_total": n_total,
        "redundancy_fraction": frac,
    }


# ── reporting ────────────────────────────────────────────────────────────────


def print_report(metrics: Dict) -> None:
    """Print formatted comparison report."""
    print(f"\n{'=' * 70}")
    print("Momentum Signal Analysis: Delta vs Surprise")
    print(f"{'=' * 70}")

    # 1. Per-dimension breakdown
    bd = metrics["per_dim_breakdown"]
    print(f"\n1. Per-Dimension Mean Magnitude:")
    print(f"  {'dim':<12s} {'delta':>10s} {'surprise':>10s} {'ratio(s/d)':>12s}")
    for d in range(D_AMP):
        dm = bd["delta_mag"][:, d].mean()  # average across objects
        sm = bd["surprise_mag"][:, d].mean()
        r = sm / max(dm, 1e-12)
        print(f"  {AMP_DIM_NAMES[d]:<12s} {dm:10.6f} {sm:10.6f} {r:12.2f}x")

    # 2. Correlation
    corr = metrics["correlation"]
    print(f"\n2. Correlation (delta vs surprise):")
    print(f"  {'object':<12s} {'mean r':>8s}  per-dim: {' '.join(f'{n[:5]:>6s}' for n in AMP_DIM_NAMES)}")
    for obj in range(N_OBJECTS):
        name = OBJECTS[obj]
        r_mean = corr["corr_summary"][obj]
        dims = "  ".join(f"{corr['corr_matrix'][obj, d]:5.2f}" for d in range(D_AMP))
        print(f"  {name:<12s} {r_mean:8.4f}  {dims}")

    # 3. Information content
    ic = metrics["information_content"]
    print(f"\n3. Information Content (variance):")
    print(f"  {'object':<12s} {'delta var':>10s} {'surprise var':>12s} {'ratio(s/d)':>12s}")
    for obj in range(N_OBJECTS):
        dv = ic["delta_var"][obj].mean()
        sv = ic["surprise_var"][obj].mean()
        r = sv / max(dv, 1e-12)
        print(f"  {OBJECTS[obj]:<12s} {dv:10.6f} {sv:12.6f} {r:12.2f}x")

    # 4. Temporal autocorrelation
    ac = metrics["temporal_autocorr"]
    print(f"\n4. Temporal Autocorrelation (decorrelation lag):")
    print(f"  {'object':<12s} {'delta lag':>10s} {'surprise lag':>12s}")
    for obj in range(N_OBJECTS):
        dl = ac["delta_decorr_lag"][obj]
        sl = ac["surprise_decorr_lag"][obj]
        print(f"  {OBJECTS[obj]:<12s} {dl:10d} {sl:12d}")

    # 5. Action-change SNR
    snr = metrics["action_change_snr"]
    print(f"\n5. Action-Change SNR (signal at change / signal at stable):")
    print(f"  {'object':<12s} {'delta SNR':>10s} {'surprise SNR':>12s}")
    for obj in range(N_OBJECTS):
        ds = snr["delta_snr"][obj]
        ss = snr["surprise_snr"][obj]
        ds_str = f"{ds:10.2f}x" if np.isfinite(ds) else "       N/A"
        ss_str = f"{ss:10.2f}x" if np.isfinite(ss) else "         N/A"
        print(f"  {OBJECTS[obj]:<12s} {ds_str} {ss_str}")

    # 6. Redundancy
    red = metrics["redundancy"]
    print(f"\n6. Redundancy: {red['n_redundant']}/{red['n_total']} dims "
          f"with |r| > 0.95 ({red['redundancy_fraction']:.0%})")
    if red["redundancy_fraction"] > 0.5:
        print("  --> Signals are largely REDUNDANT — consider dropping one")
    elif red["redundancy_fraction"] < 0.1:
        print("  --> Signals are COMPLEMENTARY — both provide unique information")
    else:
        print("  --> Signals are PARTIALLY redundant — mixed picture")

    print(f"{'=' * 70}\n")


def save_plot(metrics: Dict, path: str) -> None:
    """Save 4-panel analysis figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Per-dim breakdown (bar chart)
    ax = axes[0, 0]
    bd = metrics["per_dim_breakdown"]
    x = np.arange(D_AMP)
    w = 0.35
    dm = bd["delta_mag"].mean(axis=0)
    sm = bd["surprise_mag"].mean(axis=0)
    ax.bar(x - w / 2, dm, w, label="delta", color="steelblue")
    ax.bar(x + w / 2, sm, w, label="surprise", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(AMP_DIM_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |signal|")
    ax.set_title("Per-Dimension Magnitude")
    ax.legend()

    # Panel 2: Correlation heatmap
    ax = axes[0, 1]
    corr = metrics["correlation"]["corr_matrix"]
    obj_names = [OBJECTS[i][:4] for i in range(N_OBJECTS)]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(D_AMP))
    ax.set_xticklabels([n[:5] for n in AMP_DIM_NAMES], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(N_OBJECTS))
    ax.set_yticklabels(obj_names, fontsize=8)
    ax.set_title("Correlation (delta vs surprise)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: Autocorrelation curves
    ax = axes[1, 0]
    ac = metrics["temporal_autocorr"]
    for obj in range(N_OBJECTS):
        if obj == _TEAM:
            continue
        lags = np.arange(1, ac["delta_autocorr"].shape[1] + 1)
        ax.plot(lags, ac["delta_autocorr"][obj], "-",
                label=f"{OBJECTS[obj]} delta", linewidth=1.2)
        ax.plot(lags, ac["surprise_autocorr"][obj], "--",
                label=f"{OBJECTS[obj]} surprise", linewidth=1.2)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Temporal Autocorrelation")
    ax.legend(fontsize=7, ncol=2)

    # Panel 4: SNR comparison
    ax = axes[1, 1]
    snr = metrics["action_change_snr"]
    x = np.arange(N_OBJECTS)
    ds = np.nan_to_num(snr["delta_snr"], nan=0)
    ss = np.nan_to_num(snr["surprise_snr"], nan=0)
    ax.bar(x - w / 2, ds, w, label="delta SNR", color="steelblue")
    ax.bar(x + w / 2, ss, w, label="surprise SNR", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([OBJECTS[i] for i in range(N_OBJECTS)], fontsize=8)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("SNR (change / stable)")
    ax.set_title("Action-Change Detection SNR")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close()


def save_json(metrics: Dict, path: str) -> None:
    """Save metrics as JSON (numpy arrays converted to lists)."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    with open(path, "w") as f:
        json.dump(_convert(metrics), f, indent=2)
    print(f"Results saved to {path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze momentum signals: delta vs surprise on replay data"
    )
    parser.add_argument(
        "--data-dir", type=str, default="training/data/parsed/",
        help="Directory with parsed .npz replay files",
    )
    parser.add_argument(
        "--encoder", type=str, default=None,
        help="Path to pretrained SE3Encoder state_dict (.pt)",
    )
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--files-per-chunk", type=int, default=10)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--plot", type=str, default=None,
                        help="Save 4-panel analysis plot")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results as JSON")
    parser.add_argument("--snr-threshold", type=float, default=0.1,
                        help="Accel residual change threshold for SNR metric")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    encoder = SE3Encoder()
    if args.encoder:
        print(f"Loading encoder from {args.encoder}...")
        state = torch.load(args.encoder, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state)

    print(f"\nCollecting momentum signals...")
    signals = collect_momentum_signals(
        encoder,
        data_dir=args.data_dir,
        window_len=args.window_len,
        stride=args.stride,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
        files_per_chunk=args.files_per_chunk,
        device=device,
    )

    delta = signals["delta"]
    surprise = signals["surprise"]
    accel_res = signals["accel_residual"]

    print("Computing metrics...")
    metrics = {
        "per_dim_breakdown": compute_per_dim_breakdown(delta, surprise),
        "correlation": compute_correlation(delta, surprise),
        "information_content": compute_information_content(delta, surprise),
        "temporal_autocorr": compute_temporal_autocorrelation(delta, surprise),
        "action_change_snr": compute_action_change_snr(
            delta, surprise, accel_res, threshold=args.snr_threshold),
    }
    metrics["redundancy"] = compute_redundancy(metrics["correlation"]["corr_matrix"])

    print_report(metrics)

    if args.plot:
        save_plot(metrics, args.plot)
    if args.output_json:
        save_json(metrics, args.output_json)


if __name__ == "__main__":
    main()
