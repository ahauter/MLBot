"""
1D Spectral Field Visualization — Bouncing Ball
=================================================
Demonstrates the SE(3) spectral field mechanism from src/se3_field.py in a
simplified 1D setting.  A ball bounces between two walls, with the bounce
emerging entirely from the wall spectral field — no hardcoded collision logic.

1D simplification of se3_field.py (lines 290-343):
  - K=8 spectral components (matching the real field)
  - Cosine + sine basis (real + imaginary channels)
  - Scalar positions instead of 3D vectors
  - No quaternion/orientation component

Key idea — coefficient-space dynamics:
  The ball's position is never tracked externally.  Instead, the ball field's
  COEFFICIENTS carry momentum (a velocity vector in coefficient space).  Each
  step, the wall field is evaluated at the ball's predicted position to compute
  a force, which is projected into coefficient space and applied as acceleration.

  Ball position at time t = F_ball(REF), where REF is a fixed reference point.
  The ball's entire state (position + velocity) lives in the field coefficients.

  force_position  = -alpha * F_wall(F_ball(REF))
  c_acceleration  = force_position * basis(REF) / ||basis(REF)||^2
  c_velocity     += c_acceleration * dt
  c_ball         += c_velocity * dt

  Removing the wall field = ball flies away.  The bouncing is field-mediated.

Usage:
    python training/spectral_field_viz.py              # runs until window closed
    python training/spectral_field_viz.py --save out.gif --frames 600  # save GIF
"""

from __future__ import annotations

import argparse
import itertools
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── constants ────────────────────────────────────────────────────────────────

COEFF_CLIP = 10.0  # matches se3_field.py
REF = 1.0          # fixed reference point for ball field evaluation

# Style (matching training/scenario_visualizer.py)
BG_COLOR = '#12121f'
BALL_COLOR = '#38bdf8'
BALL_PRED_COLOR = '#7dd3fc'
WALL_COLOR = '#f87171'
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

    def evaluate(self, x_domain: np.ndarray) -> np.ndarray:
        """Field value across a spatial domain."""
        cos_b = np.cos(self.k[None, :] * x_domain[:, None])
        sin_b = np.sin(self.k[None, :] * x_domain[:, None])
        basis = np.concatenate([cos_b, sin_b], axis=1)
        return basis @ self.c

    def predict(self, x: float) -> float:
        """Field value at a single point."""
        return float(self.c @ self._basis(x))

    def update(self, x: float) -> None:
        """One step of the coefficient update rule (mirrors se3_field.py)."""
        basis = self._basis(x)
        residual = x - float(self.c @ basis)
        self.c += self.lr * basis * residual
        np.clip(self.c, -COEFF_CLIP, COEFF_CLIP, out=self.c)


# ── animation ────────────────────────────────────────────────────────────────

def create_animation(K: int, frequencies: np.ndarray, alpha: float,
                     dt: float, ball_v0: float, wall_left: float,
                     wall_right: float, wall_warmup: int,
                     interval: int, max_frames: int | None):
    """Build a live-simulation FuncAnimation.

    If max_frames is None, runs indefinitely until the window is closed.
    """
    x_domain = np.linspace(-6, 6, 500)
    window = 300  # rolling time-series window

    # ── simulation state ─────────────────────────────────────────────────
    wall_field = SpectralField1D(K, frequencies, 0.15)
    ball_field = SpectralField1D(K, frequencies, 0.15)

    # Warm up wall field at both positions (single field, two peaks)
    for _ in range(wall_warmup):
        wall_field.update(wall_left)
        wall_field.update(wall_right)

    # Reference basis vector and its squared norm (constant)
    b_ref = ball_field._basis(REF)
    b_norm2 = np.dot(b_ref, b_ref)  # = K for cos+sin basis

    # Seed ball: position=0 via coefficients, velocity=v0 via coefficient velocity
    ball_field.c = 0.0 * b_ref / b_norm2  # F_ball(REF) = 0
    c_vel = ball_v0 * b_ref / b_norm2     # dF_ball(REF)/dt = v0

    state = {'t': 0}
    history_x = deque(maxlen=window)
    history_pred = deque(maxlen=window)
    history_t = deque(maxlen=window)

    # Pre-compute basis matrix for field curve evaluation
    cos_basis = np.cos(frequencies[None, :] * x_domain[:, None])
    sin_basis = np.sin(frequencies[None, :] * x_domain[:, None])
    full_basis = np.concatenate([cos_basis, sin_basis], axis=1)

    # ── figure setup ─────────────────────────────────────────────────────
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

    # ── top panel ────────────────────────────────────────────────────────
    ax_field.set_xlim(-6, 6)
    ax_field.set_ylim(-6, 6)
    ax_field.set_xlabel('x (spatial domain)', color=LABEL_COLOR, fontsize=9)
    ax_field.set_ylabel('Field value F(x)', color=LABEL_COLOR, fontsize=9)
    ax_field.set_title('Spectral Fields & Ball Position', color=TITLE_COLOR,
                       fontsize=11, fontweight='bold', pad=8)

    ax_field.axvline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)
    ax_field.axvline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)

    line_ball_field, = ax_field.plot([], [], color=BALL_COLOR, lw=2,
                                     label='Ball field', zorder=3)
    line_wall_field, = ax_field.plot([], [], color=WALL_COLOR, lw=1.5,
                                     alpha=0.8, label='Wall field')

    ball_dot, = ax_field.plot([], [], 'o', color=BALL_COLOR, markersize=12,
                               zorder=5, markeredgecolor='white',
                               markeredgewidth=1.5)
    pred_marker, = ax_field.plot([], [], 's', color=BALL_PRED_COLOR,
                                  markersize=8, zorder=4, alpha=0.8,
                                  markeredgecolor='white', markeredgewidth=1)

    eq_text = (r'$x_{ball} = F_{ball}(ref)$'
               '\n'
               r'$\dot{c} \leftarrow \dot{c} - \alpha\, F_{wall}(x_{ball})'
               r'\cdot \phi / \|\phi\|^2$')
    ax_field.text(0.02, 0.97, eq_text, transform=ax_field.transAxes,
                  color='#ccc', fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                            edgecolor='#333', alpha=0.9))

    ax_field.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    # ── bottom panel ─────────────────────────────────────────────────────
    ax_track.set_ylim(wall_left - 1, wall_right + 1)
    ax_track.set_xlabel('Timestep', color=LABEL_COLOR, fontsize=9)
    ax_track.set_ylabel('Position', color=LABEL_COLOR, fontsize=9)
    ax_track.set_title('Position Tracking', color=TITLE_COLOR,
                       fontsize=11, fontweight='bold', pad=8)
    ax_track.axhline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_track.axhline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.5)

    line_actual, = ax_track.plot([], [], color=BALL_COLOR, lw=1.5,
                                  label='Ball position (from field)')
    line_pred, = ax_track.plot([], [], color=BALL_PRED_COLOR, lw=1.2,
                                ls='--', alpha=0.7, label='Wall field at ball')

    ax_track.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    frame_text = ax_field.text(0.98, 0.02, '', transform=ax_field.transAxes,
                                color='#555', fontsize=8,
                                horizontalalignment='right')

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # ── animation callbacks ──────────────────────────────────────────────

    def init():
        line_ball_field.set_data([], [])
        line_wall_field.set_data([], [])
        ball_dot.set_data([], [])
        pred_marker.set_data([], [])
        line_actual.set_data([], [])
        line_pred.set_data([], [])
        frame_text.set_text('')
        return (line_ball_field, line_wall_field, ball_dot, pred_marker,
                line_actual, line_pred, frame_text)

    def step(_frame):
        nonlocal c_vel
        t = state['t']

        # ── ball position comes from field prediction ────────────
        x_ball = ball_field.predict(REF)
        wall_at_ball = wall_field.predict(x_ball)

        history_x.append(x_ball)
        history_pred.append(wall_at_ball)
        history_t.append(t)

        # ── coefficient-space dynamics (no external position tracking) ──
        # Force in position space → project into coefficient space
        force = -alpha * wall_at_ball
        c_accel = force * b_ref / b_norm2

        # Leapfrog integration in coefficient space
        c_vel += c_accel * dt
        ball_field.c += c_vel * dt
        np.clip(ball_field.c, -COEFF_CLIP, COEFF_CLIP, out=ball_field.c)

        # Wall field continues to be reinforced
        wall_field.update(wall_left)
        wall_field.update(wall_right)

        state['t'] = t + 1

        # ── draw top panel ───────────────────────────────────────
        line_wall_field.set_data(x_domain, full_basis @ wall_field.c)
        line_ball_field.set_data(x_domain, full_basis @ ball_field.c)
        ball_dot.set_data([x_ball], [0])
        pred_marker.set_data([x_ball], [ball_field.predict(REF)])
        frame_text.set_text(f't={t}')

        # ── draw bottom panel ────────────────────────────────────
        ts = np.array(history_t)
        line_actual.set_data(ts, np.array(history_x))
        line_pred.set_data(ts, np.array(history_pred))

        if len(ts) > 1:
            ax_track.set_xlim(ts[0], ts[-1] + 1)

        return (line_ball_field, line_wall_field, ball_dot, pred_marker,
                line_actual, line_pred, frame_text)

    frames = range(max_frames) if max_frames is not None else itertools.count()
    anim = FuncAnimation(fig, step, frames=frames, init_func=init,
                         interval=interval, blit=True, repeat=False)
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
    parser.add_argument('--frames', type=int, default=None,
                        help='Frame limit (default: infinite for display, '
                             'required for --save)')
    args = parser.parse_args()

    if args.save and args.frames is None:
        parser.error('--frames is required when using --save')

    K = args.K
    if K == 8:
        frequencies = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    elif K == 4:
        frequencies = np.array([0.3, 0.7, 1.5, 2.5])
    else:
        frequencies = np.linspace(0.3, 3.7, K)

    if args.save:
        matplotlib.use('Agg')

    fig, anim = create_animation(
        K=K,
        frequencies=frequencies,
        alpha=args.alpha,
        dt=0.05,
        ball_v0=0.8,
        wall_left=-4.0,
        wall_right=4.0,
        wall_warmup=300,
        interval=50,
        max_frames=args.frames,
    )

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
