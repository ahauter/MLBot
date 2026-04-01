"""
1D Spectral Field Visualization — Wavepacket Dynamics
=====================================================
Demonstrates the SE(3) spectral field mechanism from src/se3_field.py in a
simplified 1D setting.  A ball bounces between two walls, with the bounce
emerging entirely from the spectral fields — no hardcoded collision logic.

All objects (ball, left wall, right wall) are the same type: WavepacketObject.
Each has cos+sin spectral coefficients that define a field shape, a scalar
velocity, and a mass.  Dynamics use the Fourier shift theorem to move peaks
without changing field shape.

Key physics:
  - Position = wavepacket peak location (tracked via cumulative shift)
  - Velocity = shift rate of the wavepacket in position space
  - Force = -alpha * F_other(x_self) — field value of other objects at self's position
  - Mass = resistance to acceleration (walls: 1e6, ball: 1.0)

The initial parameters are tuned so a human sees Newtonian-like ball-bouncing,
but in the actual ML experiment these become learnable parameters so RL
algorithms can discover 3D physical-space representations.

1D simplification of se3_field.py (lines 290-343):
  - K=8 spectral components (matching the real field)
  - Cosine + sine basis (real + imaginary channels)
  - Scalar positions instead of 3D vectors
  - Fourier shift instead of se3_field's LMS update for ball dynamics

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

# Style (matching training/scenario_visualizer.py)
BG_COLOR = '#12121f'
BALL_COLOR = '#38bdf8'
BALL_PRED_COLOR = '#7dd3fc'
WALL_COLOR = '#f87171'
GRID_COLOR = '#444'
GRID_ALPHA = 0.15
LABEL_COLOR = '#aaa'
TITLE_COLOR = 'white'


# ── WavepacketObject ───────────────────────────────────────────────────────

class WavepacketObject:
    """Spectral wavepacket — all objects (ball, walls) use this same class.

    Represents a localized field via cos+sin coefficients.  Position is
    tracked via cumulative Fourier shifts.  The Fourier shift theorem
    rotates each (cos_j, sin_j) pair by angle k_j*delta, moving the field
    peaks by delta without changing the field shape.

    Parameters
    ----------
    K : int
        Number of spectral components.
    frequencies : ndarray (K,)
        Spatial frequencies k_j.
    x0 : float
        Initial peak position.
    mass : float
        Resistance to acceleration (large = immovable wall).
    v0 : float
        Initial velocity (shift rate).
    c_cos, c_sin : ndarray (K,), optional
        If provided, use these as initial coefficients instead of Gaussian.
    sigma : float
        Gaussian envelope width in k-space (only used if c_cos/c_sin not given).
    amplitude : float
        Gaussian envelope amplitude (only used if c_cos/c_sin not given).
    """

    def __init__(self, K: int, frequencies: np.ndarray, x0: float,
                 mass: float, v0: float = 0.0,
                 c_cos: np.ndarray | None = None,
                 c_sin: np.ndarray | None = None,
                 sigma: float = 0.8, amplitude: float = 1.5):
        self.K = K
        self.k = np.asarray(frequencies, dtype=np.float64)
        self.mass = mass
        self.v = v0
        self.x = x0

        if c_cos is not None and c_sin is not None:
            self.c_cos = np.array(c_cos, dtype=np.float64)
            self.c_sin = np.array(c_sin, dtype=np.float64)
        else:
            # Gaussian wavepacket centered at x0
            envelope = amplitude * np.exp(-self.k**2 * sigma**2 / 2)
            self.c_cos = envelope * np.cos(self.k * x0)
            self.c_sin = envelope * np.sin(self.k * x0)

    def evaluate(self, x_domain: np.ndarray) -> np.ndarray:
        """Field value F(x) across a spatial domain."""
        cos_b = np.cos(self.k[None, :] * x_domain[:, None])  # (N, K)
        sin_b = np.sin(self.k[None, :] * x_domain[:, None])
        return cos_b @ self.c_cos + sin_b @ self.c_sin

    def predict(self, x: float) -> float:
        """Field value at a single point."""
        return float(np.sum(self.c_cos * np.cos(self.k * x) +
                            self.c_sin * np.sin(self.k * x)))

    def shift(self, delta: float) -> None:
        """Fourier shift: move the wavepacket peak by delta.

        Rotates each (cos_j, sin_j) pair by angle k_j * delta.
        This preserves the field shape exactly — only the peak moves.
        """
        for j in range(self.K):
            angle = self.k[j] * delta
            ca, sa = np.cos(angle), np.sin(angle)
            oc, os = self.c_cos[j], self.c_sin[j]
            # f(x) → f(x - delta): peak shifts RIGHT by delta
            self.c_cos[j] = oc * ca - os * sa
            self.c_sin[j] = oc * sa + os * ca

    def step(self, force: float, dt: float) -> None:
        """One dynamics step: accelerate, shift, update position."""
        self.v += (force / self.mass) * dt
        delta = self.v * dt
        self.shift(delta)
        self.x += delta
        # Clip coefficients
        np.clip(self.c_cos, -COEFF_CLIP, COEFF_CLIP, out=self.c_cos)
        np.clip(self.c_sin, -COEFF_CLIP, COEFF_CLIP, out=self.c_sin)


def pretrain_wall_coefficients(K: int, frequencies: np.ndarray,
                               wall_positions: list[float],
                               lr: float = 0.15,
                               warmup_steps: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Pre-train wall coefficients via LMS to fit both wall positions.

    This is equivalent to initializing the wall wavepacket to a shape
    that encodes both wall positions.  The resulting field is antisymmetric
    (F(0)=0, F(-x)=-F(x)) when walls are symmetric about the origin.

    Returns (c_cos, c_sin) arrays.
    """
    c = np.zeros(2 * K, dtype=np.float64)
    k = np.asarray(frequencies, dtype=np.float64)

    for _ in range(warmup_steps):
        for x_w in wall_positions:
            basis = np.concatenate([np.cos(k * x_w), np.sin(k * x_w)])
            pred = c @ basis
            residual = x_w - pred
            c += lr * basis * residual
            np.clip(c, -COEFF_CLIP, COEFF_CLIP, out=c)

    return c[:K].copy(), c[K:].copy()


# ── animation ────────────────────────────────────────────────────────────────

def create_animation(K: int, frequencies: np.ndarray, alpha: float,
                     dt: float, ball_v0: float, wall_left: float,
                     wall_right: float, wall_warmup: int,
                     ball_sigma: float, ball_amplitude: float,
                     wall_mass: float, interval: int,
                     max_frames: int | None):
    """Build a live-simulation FuncAnimation."""
    x_domain = np.linspace(-6, 6, 500)
    window = 300  # rolling time-series window

    # ── create objects (all same class, different parameters) ─────────
    # Wall: LMS-pretrained coefficients, huge mass
    wall_c_cos, wall_c_sin = pretrain_wall_coefficients(
        K, frequencies, [wall_left, wall_right],
        lr=0.15, warmup_steps=wall_warmup)

    wall = WavepacketObject(
        K, frequencies, x0=0.0, mass=wall_mass, v0=0.0,
        c_cos=wall_c_cos, c_sin=wall_c_sin)

    # Ball: Gaussian wavepacket, small mass
    ball = WavepacketObject(
        K, frequencies, x0=0.0, mass=1.0, v0=ball_v0,
        sigma=ball_sigma, amplitude=ball_amplitude)

    objects = [ball, wall]

    state = {'t': 0}
    history_x = deque(maxlen=window)
    history_wall = deque(maxlen=window)
    history_t = deque(maxlen=window)

    # ── figure setup ─────────────────────────────────────────────────────
    fig, (ax_field, ax_track) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('1D Spectral Field — Wavepacket Dynamics',
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
    ax_field.set_title('Spectral Fields & Wavepacket Peaks', color=TITLE_COLOR,
                       fontsize=11, fontweight='bold', pad=8)

    ax_field.axvline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)
    ax_field.axvline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)

    line_ball_field, = ax_field.plot([], [], color=BALL_COLOR, lw=2,
                                     label='Ball wavepacket', zorder=3)
    line_wall_field, = ax_field.plot([], [], color=WALL_COLOR, lw=1.5,
                                     alpha=0.8, label='Wall field')

    ball_dot, = ax_field.plot([], [], 'o', color=BALL_COLOR, markersize=12,
                               zorder=5, markeredgecolor='white',
                               markeredgewidth=1.5)

    eq_text = (r'$\mathrm{shift}(\delta): c_j \to R(k_j \delta)\, c_j$'
               '\n'
               r'$\dot{v} = -\alpha\, F_{wall}(x_{ball}) / m$')
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
                                  label='Ball position (wavepacket peak)')
    line_wall_at_ball, = ax_track.plot([], [], color=BALL_PRED_COLOR, lw=1.2,
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
        line_actual.set_data([], [])
        line_wall_at_ball.set_data([], [])
        frame_text.set_text('')
        return (line_ball_field, line_wall_field, ball_dot,
                line_actual, line_wall_at_ball, frame_text)

    def step(_frame):
        t = state['t']

        # ── compute forces on each object ────────────────────────
        # Force on ball from wall field
        wall_at_ball = wall.predict(ball.x)
        force_on_ball = -alpha * wall_at_ball

        # Force on wall from ball field (symmetric, but wall barely moves)
        ball_at_wall = ball.predict(wall.x)
        force_on_wall = -alpha * ball_at_wall

        # ── step all objects ─────────────────────────────────────
        ball.step(force_on_ball, dt)
        wall.step(force_on_wall, dt)

        # ── record history ───────────────────────────────────────
        history_x.append(ball.x)
        history_wall.append(wall_at_ball)
        history_t.append(t)

        state['t'] = t + 1

        # ── draw top panel ───────────────────────────────────────
        ball_curve = ball.evaluate(x_domain)
        wall_curve = wall.evaluate(x_domain)

        line_ball_field.set_data(x_domain, ball_curve)
        line_wall_field.set_data(x_domain, wall_curve)
        ball_dot.set_data([ball.x], [0])
        frame_text.set_text(f't={t}')

        # ── draw bottom panel ────────────────────────────────────
        ts = np.array(history_t)
        line_actual.set_data(ts, np.array(history_x))
        line_wall_at_ball.set_data(ts, np.array(history_wall))

        if len(ts) > 1:
            ax_track.set_xlim(ts[0], ts[-1] + 1)

        return (line_ball_field, line_wall_field, ball_dot,
                line_actual, line_wall_at_ball, frame_text)

    frames = range(max_frames) if max_frames is not None else itertools.count()
    anim = FuncAnimation(fig, step, frames=frames, init_func=init,
                         interval=interval, blit=True, repeat=False)
    return fig, anim


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='1D Spectral Field Visualization — Wavepacket Dynamics')
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
        ball_sigma=0.8,
        ball_amplitude=1.5,
        wall_mass=1e6,
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
