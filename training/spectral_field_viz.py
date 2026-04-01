"""
1D Spectral Field Visualization — Bouncing Ball
=================================================
Demonstrates the SE(3) spectral field mechanism from src/se3_field.py in a
simplified 1D setting.  A ball bounces between two walls, with the bounce
emerging entirely from the wall spectral fields — no hardcoded collision logic.

1D simplification:
  - Cosine basis only (drop imaginary/sine channel)
  - Scalar positions instead of 3D vectors
  - No quaternion/orientation component
  - Same coefficient update rule as se3_field.py lines 290-343

Interaction mechanism:
  Each wall's converged spectral field F_wall(x) = sum_j c_j * cos(k_j * x)
  creates a potential landscape.  The ball experiences force from the gradient:
      force = alpha * sum_walls sum_j c_wall_j * k_j * sin(k_j * x_ball)
  Removing the walls = ball flies away freely.

Usage:
    python training/spectral_field_viz.py              # show animation
    python training/spectral_field_viz.py --save out.gif  # save as GIF
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── constants ────────────────────────────────────────────────────────────────

COEFF_CLIP = 10.0  # matches se3_field.py

# Style (matching training/scenario_visualizer.py)
BG_COLOR = '#12121f'
BALL_COLOR = '#38bdf8'
BALL_PRED_COLOR = '#7dd3fc'
LWALL_COLOR = '#f87171'
RWALL_COLOR = '#fb923c'
GRID_COLOR = '#444'
GRID_ALPHA = 0.15
LABEL_COLOR = '#aaa'
TITLE_COLOR = 'white'


# ── SpectralField1D ─────────────────────────────────────────────────────────

class SpectralField1D:
    """1D cosine-only spectral field (simplified from SE3Encoder)."""

    def __init__(self, K: int, frequencies: np.ndarray, lr: float):
        self.K = K
        self.k = np.asarray(frequencies, dtype=np.float64)  # (K,)
        self.lr = lr
        self.c = np.zeros(K, dtype=np.float64)               # coefficients

    def evaluate(self, x_domain: np.ndarray) -> np.ndarray:
        """Field value across a spatial domain.  F(x) = sum_j c_j * cos(k_j * x)."""
        # x_domain: (N,), returns (N,)
        basis = np.cos(self.k[None, :] * x_domain[:, None])  # (N, K)
        return basis @ self.c                                  # (N,)

    def predict(self, x: float) -> float:
        """Field value at a single point."""
        return float(np.sum(self.c * np.cos(self.k * x)))

    def gradient(self, x: float) -> float:
        """dF/dx = -sum_j c_j * k_j * sin(k_j * x)."""
        return float(-np.sum(self.c * self.k * np.sin(self.k * x)))

    def update(self, x: float) -> None:
        """One step of the coefficient update rule (mirrors se3_field.py)."""
        basis = np.cos(self.k * x)               # (K,)
        predicted = float(np.sum(self.c * basis))
        residual = x - predicted
        self.c += self.lr * basis * residual
        np.clip(self.c, -COEFF_CLIP, COEFF_CLIP, out=self.c)

    def reset(self) -> None:
        """Zero all coefficients (contact reset)."""
        self.c[:] = 0.0

    def get_basis_curves(self, x_domain: np.ndarray) -> list[np.ndarray]:
        """Individual weighted cosine curves: c_j * cos(k_j * x)."""
        return [self.c[j] * np.cos(self.k[j] * x_domain) for j in range(self.K)]


# ── simulation ───────────────────────────────────────────────────────────────

def run_simulation(n_frames: int, K: int, frequencies: np.ndarray,
                   ball_lr: float, wall_lr: float,
                   alpha: float, dt: float,
                   wall_left: float, wall_right: float,
                   ball_x0: float, ball_v0: float,
                   wall_warmup: int):
    """Pre-compute entire simulation trajectory.

    Returns dict of arrays, each of length n_frames.
    """
    left_field = SpectralField1D(K, frequencies, wall_lr)
    right_field = SpectralField1D(K, frequencies, wall_lr)
    ball_field = SpectralField1D(K, frequencies, ball_lr)

    # Warm up wall fields so they converge before the ball starts moving
    for _ in range(wall_warmup):
        left_field.update(wall_left)
        right_field.update(wall_right)

    ball_x = ball_x0
    ball_v = ball_v0

    # Storage
    ball_xs = np.zeros(n_frames)
    ball_vs = np.zeros(n_frames)
    ball_preds = np.zeros(n_frames)
    resets = np.zeros(n_frames, dtype=bool)

    # Snapshot field coefficients at each frame for plotting
    ball_coeffs = np.zeros((n_frames, K))
    left_coeffs = np.zeros((n_frames, K))
    right_coeffs = np.zeros((n_frames, K))

    for i in range(n_frames):
        # Record state
        ball_xs[i] = ball_x
        ball_vs[i] = ball_v
        ball_preds[i] = ball_field.predict(ball_x)
        ball_coeffs[i] = ball_field.c.copy()
        left_coeffs[i] = left_field.c.copy()
        right_coeffs[i] = right_field.c.copy()

        # Force on ball from wall field gradients
        force = -alpha * (left_field.gradient(ball_x) + right_field.gradient(ball_x))

        # Update ball dynamics
        prev_v = ball_v
        ball_v += force * dt
        ball_x += ball_v * dt

        # Contact detection: velocity sign change (mirrors se3_field.py contact reset)
        if prev_v * ball_v < 0:
            ball_field.reset()
            resets[i] = True

        # Update spectral fields
        ball_field.update(ball_x)
        left_field.update(wall_left)
        right_field.update(wall_right)

    return {
        'ball_x': ball_xs,
        'ball_v': ball_vs,
        'ball_pred': ball_preds,
        'resets': resets,
        'ball_coeffs': ball_coeffs,
        'left_coeffs': left_coeffs,
        'right_coeffs': right_coeffs,
        'left_field': left_field,
        'right_field': right_field,
        'ball_field': ball_field,
        'frequencies': frequencies,
        'wall_left': wall_left,
        'wall_right': wall_right,
    }


# ── animation ────────────────────────────────────────────────────────────────

def create_animation(sim: dict, K: int, frequencies: np.ndarray,
                     ball_lr: float, wall_lr: float, interval: int):
    """Build the FuncAnimation from pre-computed simulation data."""

    x_domain = np.linspace(-6, 6, 500)
    n_frames = len(sim['ball_x'])
    wall_left = sim['wall_left']
    wall_right = sim['wall_right']

    fig, (ax_field, ax_track) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('1D Spectral Field — Bouncing Ball via Field Interactions',
                 color=TITLE_COLOR, fontsize=13, fontweight='bold', y=0.96)

    for ax in (ax_field, ax_track):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.5)

    # ── top panel setup ──────────────────────────────────────────────────
    ax_field.set_xlim(-6, 6)
    ax_field.set_ylim(-6, 6)
    ax_field.set_xlabel('x (spatial domain)', color=LABEL_COLOR, fontsize=9)
    ax_field.set_ylabel('Field value F(x)', color=LABEL_COLOR, fontsize=9)
    ax_field.set_title('Spectral Fields & Ball Position', color=TITLE_COLOR,
                       fontsize=11, fontweight='bold', pad=8)

    # Wall position markers
    ax_field.axvline(wall_left, color=LWALL_COLOR, ls='--', lw=1, alpha=0.6)
    ax_field.axvline(wall_right, color=RWALL_COLOR, ls='--', lw=1, alpha=0.6)

    # Lines for field curves
    line_ball_field, = ax_field.plot([], [], color=BALL_COLOR, lw=2,
                                     label='Ball field', zorder=3)
    line_left_field, = ax_field.plot([], [], color=LWALL_COLOR, lw=1.5,
                                     alpha=0.8, label='Left wall field')
    line_right_field, = ax_field.plot([], [], color=RWALL_COLOR, lw=1.5,
                                      alpha=0.8, label='Right wall field')

    # Ball position dot
    ball_dot, = ax_field.plot([], [], 'o', color=BALL_COLOR, markersize=12,
                               zorder=5, markeredgecolor='white',
                               markeredgewidth=1.5)
    # Prediction marker
    pred_marker, = ax_field.plot([], [], 's', color=BALL_PRED_COLOR,
                                  markersize=8, zorder=4, alpha=0.8,
                                  markeredgecolor='white', markeredgewidth=1)

    # Equation annotation
    eq_text = (r'$F(x) = \sum_j\, c_j \cos(k_j\, x)$'
               '\n'
               r'$c_j \mathrel{+}= \alpha\, \cos(k_j\, x)\,(x - \hat{x})$')
    ax_field.text(0.02, 0.97, eq_text, transform=ax_field.transAxes,
                  color='#ccc', fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                            edgecolor='#333', alpha=0.9))

    ax_field.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    # ── bottom panel setup ───────────────────────────────────────────────
    window = min(200, n_frames)
    ax_track.set_xlim(0, window)
    ax_track.set_ylim(wall_left - 1, wall_right + 1)
    ax_track.set_xlabel('Timestep', color=LABEL_COLOR, fontsize=9)
    ax_track.set_ylabel('Position', color=LABEL_COLOR, fontsize=9)
    ax_track.set_title('Position Tracking', color=TITLE_COLOR,
                       fontsize=11, fontweight='bold', pad=8)
    ax_track.axhline(wall_left, color=LWALL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_track.axhline(wall_right, color=RWALL_COLOR, ls='--', lw=1, alpha=0.5)

    line_actual, = ax_track.plot([], [], color=BALL_COLOR, lw=1.5,
                                  label='Actual position')
    line_pred, = ax_track.plot([], [], color=BALL_PRED_COLOR, lw=1.2,
                                ls='--', alpha=0.7, label='Field prediction')
    reset_scatter = ax_track.scatter([], [], color='#ef4444', s=30,
                                      zorder=5, marker='x', linewidths=1.5,
                                      label='Coeff reset')

    ax_track.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    # Frame counter
    frame_text = ax_field.text(0.98, 0.02, '', transform=ax_field.transAxes,
                                color='#555', fontsize=8,
                                horizontalalignment='right')

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Helper: reconstruct field from saved coefficients
    def eval_field(coeffs, freqs, x_dom):
        basis = np.cos(freqs[None, :] * x_dom[:, None])
        return basis @ coeffs

    def init():
        line_ball_field.set_data([], [])
        line_left_field.set_data([], [])
        line_right_field.set_data([], [])
        ball_dot.set_data([], [])
        pred_marker.set_data([], [])
        line_actual.set_data([], [])
        line_pred.set_data([], [])
        reset_scatter.set_offsets(np.empty((0, 2)))
        frame_text.set_text('')
        return (line_ball_field, line_left_field, line_right_field,
                ball_dot, pred_marker, line_actual, line_pred,
                reset_scatter, frame_text)

    def update(frame):
        # Top panel: field curves
        bc = sim['ball_coeffs'][frame]
        lc = sim['left_coeffs'][frame]
        rc = sim['right_coeffs'][frame]

        line_ball_field.set_data(x_domain, eval_field(bc, frequencies, x_domain))
        line_left_field.set_data(x_domain, eval_field(lc, frequencies, x_domain))
        line_right_field.set_data(x_domain, eval_field(rc, frequencies, x_domain))

        bx = sim['ball_x'][frame]
        bp = sim['ball_pred'][frame]
        ball_dot.set_data([bx], [0])
        pred_marker.set_data([bx], [bp])

        frame_text.set_text(f't={frame}')

        # Bottom panel: scrolling time series
        start = max(0, frame - window)
        end = frame + 1
        ts = np.arange(start, end)

        line_actual.set_data(ts, sim['ball_x'][start:end])
        line_pred.set_data(ts, sim['ball_pred'][start:end])

        # Reset markers
        reset_idx = np.where(sim['resets'][start:end])[0] + start
        if len(reset_idx) > 0:
            offsets = np.column_stack([reset_idx, sim['ball_x'][reset_idx]])
            reset_scatter.set_offsets(offsets)
        else:
            reset_scatter.set_offsets(np.empty((0, 2)))

        ax_track.set_xlim(start, start + window)

        return (line_ball_field, line_left_field, line_right_field,
                ball_dot, pred_marker, line_actual, line_pred,
                reset_scatter, frame_text)

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=interval, blit=True)
    return fig, anim


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='1D Spectral Field Visualization — Bouncing Ball')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g. out.gif)')
    parser.add_argument('--K', type=int, default=4,
                        help='Number of spectral components (default: 4)')
    parser.add_argument('--lr', type=float, default=0.3,
                        help='Ball coefficient learning rate (default: 0.3)')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Force coupling strength (default: 0.8)')
    parser.add_argument('--frames', type=int, default=400,
                        help='Number of animation frames (default: 400)')
    args = parser.parse_args()

    K = args.K
    # Hand-picked frequencies for visual variety
    if K == 4:
        frequencies = np.array([0.3, 0.7, 1.5, 2.5])
    else:
        frequencies = np.linspace(0.3, 2.5, K)

    # Use non-interactive backend when saving
    if args.save:
        matplotlib.use('Agg')

    sim = run_simulation(
        n_frames=args.frames,
        K=K,
        frequencies=frequencies,
        ball_lr=args.lr,
        wall_lr=0.5,
        alpha=args.alpha,
        dt=0.1,
        wall_left=-4.0,
        wall_right=4.0,
        ball_x0=0.0,
        ball_v0=0.5,
        wall_warmup=100,
    )

    fig, anim = create_animation(
        sim, K, frequencies,
        ball_lr=args.lr, wall_lr=0.5, interval=50)

    if args.save:
        save_path = Path(args.save)
        if save_path.suffix == '.gif':
            anim.save(str(save_path), writer='pillow', fps=20)
        else:
            anim.save(str(save_path), fps=20)
        print(f'Saved to {save_path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
