"""
1D Spectral Field Visualization — Bouncing Ball
=================================================
Demonstrates the SE(3) spectral field mechanism from src/se3_field.py in a
simplified 1D setting.  A ball bounces between two walls, with the bounce
emerging entirely from the wall spectral fields — no hardcoded collision logic.

1D simplification of se3_field.py (lines 290-343):
  - K=8 spectral components (matching the real field)
  - Cosine + sine basis (real + imaginary channels)
  - Scalar positions instead of 3D vectors
  - No quaternion/orientation component
  - Same coefficient update rule

Interaction mechanism:
  Each wall's converged spectral field F_wall(x) = sum_j [c_j*cos(k_j*x) + s_j*sin(k_j*x)]
  creates a potential landscape.  The ball experiences force:
      force = -alpha * sum_walls F_wall(x_ball)
  The sine components provide the odd-function asymmetry that distinguishes
  left from right, creating a restoring force toward the center.
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
    """1D spectral field with cos+sin basis (simplified from SE3Encoder).

    Basis: [cos(k_0*x), ..., cos(k_{K-1}*x), sin(k_0*x), ..., sin(k_{K-1}*x)]
    This mirrors the real/imaginary channels in se3_field.py.
    """

    def __init__(self, K: int, frequencies: np.ndarray, lr: float):
        self.K = K
        self.k = np.asarray(frequencies, dtype=np.float64)  # (K,)
        self.lr = lr
        self.c = np.zeros(2 * K, dtype=np.float64)          # [cos_coeffs | sin_coeffs]

    def _basis(self, x: float) -> np.ndarray:
        """Evaluate basis functions at a single point."""
        return np.concatenate([np.cos(self.k * x), np.sin(self.k * x)])

    def _basis_domain(self, x_domain: np.ndarray) -> np.ndarray:
        """Evaluate basis functions across a domain. Returns (N, 2K)."""
        cos_b = np.cos(self.k[None, :] * x_domain[:, None])  # (N, K)
        sin_b = np.sin(self.k[None, :] * x_domain[:, None])  # (N, K)
        return np.concatenate([cos_b, sin_b], axis=1)         # (N, 2K)

    def evaluate(self, x_domain: np.ndarray) -> np.ndarray:
        """Field value across a spatial domain."""
        return self._basis_domain(x_domain) @ self.c  # (N,)

    def predict(self, x: float) -> float:
        """Field value at a single point."""
        return float(self.c @ self._basis(x))

    def gradient(self, x: float) -> float:
        """dF/dx at a single point."""
        dcos = -self.k * np.sin(self.k * x)
        dsin = self.k * np.cos(self.k * x)
        dbasis = np.concatenate([dcos, dsin])
        return float(self.c @ dbasis)

    def update(self, x: float) -> None:
        """One step of the coefficient update rule (mirrors se3_field.py)."""
        basis = self._basis(x)
        residual = x - float(self.c @ basis)
        self.c += self.lr * basis * residual
        np.clip(self.c, -COEFF_CLIP, COEFF_CLIP, out=self.c)

    def reset(self) -> None:
        """Zero all coefficients (contact reset, mirrors se3_field.py line 446)."""
        self.c[:] = 0.0

    def get_basis_curves(self, x_domain: np.ndarray) -> list[np.ndarray]:
        """Individual weighted basis curves."""
        curves = []
        for j in range(self.K):
            curves.append(self.c[j] * np.cos(self.k[j] * x_domain))
        for j in range(self.K):
            curves.append(self.c[self.K + j] * np.sin(self.k[j] * x_domain))
        return curves


# ── simulation ───────────────────────────────────────────────────────────────

def run_simulation(n_frames: int, K: int, frequencies: np.ndarray,
                   ball_lr: float, wall_lr: float,
                   alpha: float, dt: float,
                   wall_left: float, wall_right: float,
                   ball_x0: float, ball_v0: float,
                   wall_warmup: int):
    """Pre-compute entire simulation trajectory.

    The ball's dynamics are entirely field-mediated:
      force = -alpha * (F_left(x_ball) + F_right(x_ball))

    The wall fields create a potential landscape; the ball oscillates within it.
    No position-based collision checks — bouncing emerges from the spectral fields.
    """
    n_coeffs = 2 * K  # cos + sin coefficients

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
    ball_coeffs = np.zeros((n_frames, n_coeffs))
    left_coeffs = np.zeros((n_frames, n_coeffs))
    right_coeffs = np.zeros((n_frames, n_coeffs))

    for i in range(n_frames):
        # Record state
        ball_xs[i] = ball_x
        ball_vs[i] = ball_v
        ball_preds[i] = ball_field.predict(ball_x)
        ball_coeffs[i] = ball_field.c.copy()
        left_coeffs[i] = left_field.c.copy()
        right_coeffs[i] = right_field.c.copy()

        # Force on ball: field value acts as potential
        # Positive F_right near right wall pushes ball left (-alpha * positive = negative)
        # Negative F_left near left wall pushes ball right (-alpha * negative = positive)
        force = -alpha * (left_field.predict(ball_x) + right_field.predict(ball_x))

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
        'frequencies': frequencies,
        'wall_left': wall_left,
        'wall_right': wall_right,
        'K': K,
    }


# ── animation ────────────────────────────────────────────────────────────────

def create_animation(sim: dict, interval: int):
    """Build the FuncAnimation from pre-computed simulation data."""

    x_domain = np.linspace(-6, 6, 500)
    n_frames = len(sim['ball_x'])
    wall_left = sim['wall_left']
    wall_right = sim['wall_right']
    frequencies = sim['frequencies']
    K = sim['K']

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
    eq_text = (r'$F(x) = \sum_j\, c_j \cos(k_j x) + s_j \sin(k_j x)$'
               '\n'
               r'$\mathrm{force} = -\alpha \sum_{\mathrm{walls}} F_w(x_{\mathrm{ball}})$')
    ax_field.text(0.02, 0.97, eq_text, transform=ax_field.transAxes,
                  color='#ccc', fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                            edgecolor='#333', alpha=0.9))

    ax_field.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    # ── bottom panel setup ───────────────────────────────────────────────
    window = min(300, n_frames)
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

    # Pre-compute basis matrix for the domain (doesn't change per frame)
    cos_basis = np.cos(frequencies[None, :] * x_domain[:, None])  # (N, K)
    sin_basis = np.sin(frequencies[None, :] * x_domain[:, None])  # (N, K)
    full_basis = np.concatenate([cos_basis, sin_basis], axis=1)   # (N, 2K)

    def eval_field(coeffs):
        return full_basis @ coeffs

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
        line_ball_field.set_data(x_domain, eval_field(sim['ball_coeffs'][frame]))
        line_left_field.set_data(x_domain, eval_field(sim['left_coeffs'][frame]))
        line_right_field.set_data(x_domain, eval_field(sim['right_coeffs'][frame]))

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
    parser.add_argument('--K', type=int, default=8,
                        help='Number of spectral components (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.12,
                        help='Force coupling strength (default: 0.12)')
    parser.add_argument('--frames', type=int, default=600,
                        help='Number of animation frames (default: 600)')
    args = parser.parse_args()

    K = args.K
    # Hand-picked frequencies for visual variety and good field behavior
    if K == 8:
        frequencies = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    elif K == 4:
        frequencies = np.array([0.3, 0.7, 1.5, 2.5])
    else:
        frequencies = np.linspace(0.3, 3.7, K)

    # Use non-interactive backend when saving
    if args.save:
        matplotlib.use('Agg')

    # Wall lr=0.15 ensures convergence: lr * K < 2 for cos+sin basis
    sim = run_simulation(
        n_frames=args.frames,
        K=K,
        frequencies=frequencies,
        ball_lr=0.15,
        wall_lr=0.15,
        alpha=args.alpha,
        dt=0.05,
        wall_left=-4.0,
        wall_right=4.0,
        ball_x0=0.0,
        ball_v0=0.8,
        wall_warmup=300,
    )

    fig, anim = create_animation(sim, interval=50)

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
