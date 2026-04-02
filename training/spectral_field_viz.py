"""
1D Spectral Field Visualization — Newtonian to Spectral
========================================================
Starts with a standard Newtonian ball bouncing between walls, then
progressively builds the spectral field representation from the trajectory.

Flow:
  1. Newtonian ball bounces off hard walls (green) — standard physics
  2. Each timestep, LMS updates a WavepacketObject to track the ball position
  3. The spectral field curve (blue) emerges and grows richer over time
  4. Inner product force from the learned spectral representation is shown
  5. FFT comparison reveals how well the spectral basis captures the trajectory

This is the REVERSE of the encoder: position → spectral coefficients.
The real encoder does the same thing (lines 309-320 of se3_field.py) —
it receives positions and updates coefficients via LMS.  This visualization
shows what that learning looks like and how the field representation builds.

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
WALL_COLOR = '#f87171'
INTERACT_COLOR = '#a78bfa'
NEWTON_COLOR = '#4ade80'
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
    """

    def __init__(self, K: int, frequencies: np.ndarray, x0: float = 0.0,
                 mass: float = 1.0, v0: float = 0.0,
                 c_cos: np.ndarray | None = None,
                 c_sin: np.ndarray | None = None,
                 sigma: float = 0.8, amplitude: float = 1.5,
                 lr: float = 0.0):
        self.K = K
        self.k = np.asarray(frequencies, dtype=np.float64)
        self.mass = mass
        self.v = v0
        self.x = x0
        self.lr = lr  # LMS learning rate (0 = no learning)

        if c_cos is not None and c_sin is not None:
            self.c_cos = np.array(c_cos, dtype=np.float64)
            self.c_sin = np.array(c_sin, dtype=np.float64)
        else:
            # Gaussian wavepacket centered at x0
            envelope = amplitude * np.exp(-self.k**2 * sigma**2 / 2)
            self.c_cos = envelope * np.cos(self.k * x0)
            self.c_sin = envelope * np.sin(self.k * x0)

    def _basis(self, x: float) -> np.ndarray:
        """Cos+sin basis vector at a single point (2K,)."""
        return np.concatenate([np.cos(self.k * x), np.sin(self.k * x)])

    @property
    def _c(self) -> np.ndarray:
        """Full coefficient vector (2K,)."""
        return np.concatenate([self.c_cos, self.c_sin])

    @_c.setter
    def _c(self, val: np.ndarray) -> None:
        self.c_cos = val[:self.K]
        self.c_sin = val[self.K:]

    def evaluate(self, x_domain: np.ndarray) -> np.ndarray:
        """Field value F(x) across a spatial domain."""
        cos_b = np.cos(self.k[None, :] * x_domain[:, None])
        sin_b = np.sin(self.k[None, :] * x_domain[:, None])
        return cos_b @ self.c_cos + sin_b @ self.c_sin

    def predict(self, x: float) -> float:
        """Field value at a single point."""
        return float(self._c @ self._basis(x))

    def inner_product(self, other: 'WavepacketObject') -> float:
        """Overlap integral <F_self, F_other> via Parseval's theorem."""
        return float(np.sum(self.c_cos * other.c_cos +
                            self.c_sin * other.c_sin))

    def update(self, x_target: float) -> None:
        """LMS coefficient update: train field to predict x_target at x_target.

        This mirrors se3_field.py lines 309-320 — the encoder's update rule.
        """
        if self.lr <= 0:
            return
        basis = self._basis(x_target)
        pred = float(self._c @ basis)
        residual = x_target - pred
        c = self._c + self.lr * basis * residual
        np.clip(c, -COEFF_CLIP, COEFF_CLIP, out=c)
        self._c = c

    def shift(self, delta: float) -> None:
        """Fourier shift: move the wavepacket peak by delta."""
        for j in range(self.K):
            angle = self.k[j] * delta
            ca, sa = np.cos(angle), np.sin(angle)
            oc, os = self.c_cos[j], self.c_sin[j]
            self.c_cos[j] = oc * ca - os * sa
            self.c_sin[j] = oc * sa + os * ca

    def step(self, force: float, dt: float) -> None:
        """One dynamics step: accelerate, shift, update position."""
        self.v += (force / self.mass) * dt
        delta = self.v * dt
        self.shift(delta)
        self.x += delta
        np.clip(self.c_cos, -COEFF_CLIP, COEFF_CLIP, out=self.c_cos)
        np.clip(self.c_sin, -COEFF_CLIP, COEFF_CLIP, out=self.c_sin)


def pretrain_wall_coefficients(K: int, frequencies: np.ndarray,
                               wall_positions: list[float],
                               lr: float = 0.15,
                               warmup_steps: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Pre-train wall coefficients via LMS to fit both wall positions."""
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
                     ball_lr: float, wall_mass: float,
                     interval: int, max_frames: int | None):
    """Build a live-simulation FuncAnimation."""
    x_domain = np.linspace(-6, 6, 500)
    window = 300

    # ── wall: LMS-pretrained, huge mass ──────────────────────────────────
    wall_c_cos, wall_c_sin = pretrain_wall_coefficients(
        K, frequencies, [wall_left, wall_right],
        lr=0.15, warmup_steps=wall_warmup)

    wall = WavepacketObject(
        K, frequencies, x0=0.0, mass=wall_mass, v0=0.0,
        c_cos=wall_c_cos, c_sin=wall_c_sin)

    # ── ball: starts with ZERO coefficients, learns from Newtonian trajectory
    ball = WavepacketObject(
        K, frequencies, x0=0.0, mass=1.0,
        c_cos=np.zeros(K), c_sin=np.zeros(K), lr=ball_lr)

    # ── Newtonian ball — the ground truth ────────────────────────────────
    newton = {'x': 0.0, 'v': ball_v0}

    state = {'t': 0, 'fft_ymax': 1.0}
    history_newton_x = deque(maxlen=window)
    history_spectral_pred = deque(maxlen=window)
    history_force = deque(maxlen=window)
    history_t = deque(maxlen=window)

    # ── figure ───────────────────────────────────────────────────────────
    fig, (ax_field, ax_force, ax_track,
          ax_fft_spectral, ax_fft_newton) = plt.subplots(
        5, 1, figsize=(12, 14),
        gridspec_kw={'height_ratios': [2, 0.8, 1.2, 1, 1]})
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('Newtonian Physics → Spectral Field Representation',
                 color=TITLE_COLOR, fontsize=13, fontweight='bold', y=0.98)

    for ax in [ax_field, ax_force, ax_track, ax_fft_spectral, ax_fft_newton]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.5)

    # ── panel 1: spatial fields + ball positions ─────────────────────────
    ax_field.set_xlim(-6, 6)
    ax_field.set_ylim(-6, 6)
    ax_field.set_xlabel('x (spatial domain)', color=LABEL_COLOR, fontsize=9)
    ax_field.set_ylabel('Field value F(x)', color=LABEL_COLOR, fontsize=9)
    ax_field.set_title('Fields: Wall (red) + Learned Ball Representation (blue)',
                       color=TITLE_COLOR, fontsize=11, fontweight='bold', pad=8)

    ax_field.axvline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)
    ax_field.axvline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)

    line_ball_field, = ax_field.plot([], [], color=BALL_COLOR, lw=2,
                                     alpha=0.9, label='Learned ball field',
                                     zorder=3)
    line_wall_field, = ax_field.plot([], [], color=WALL_COLOR, lw=1.5,
                                     alpha=0.8, label='Wall field')

    newton_dot, = ax_field.plot([], [], 'o', color=NEWTON_COLOR, markersize=12,
                                 zorder=5, markeredgecolor='white',
                                 markeredgewidth=1.5,
                                 label='Newtonian ball')

    # Force arrow
    force_arrow = [None]

    eq_text = (r'LMS: $c \leftarrow c + \eta\, b(x)\, (x - c^T b(x))$'
               '\n'
               r'$F_{spectral} = -\alpha\, \langle F_{ball},\, F_{wall} \rangle$')
    ax_field.text(0.02, 0.97, eq_text, transform=ax_field.transAxes,
                  color='#ccc', fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                            edgecolor='#333', alpha=0.9))

    ax_field.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    # ── panel 2: forces ──────────────────────────────────────────────────
    ax_force.set_ylim(-1.5, 1.5)
    ax_force.set_xlabel('Timestep', color=LABEL_COLOR, fontsize=9)
    ax_force.set_ylabel('Force', color=LABEL_COLOR, fontsize=9)
    ax_force.set_title('Inner Product Force on Ball', color=INTERACT_COLOR,
                       fontsize=10, fontweight='bold', pad=6)
    ax_force.axhline(0, color='#555', ls='-', lw=0.5, alpha=0.3)

    line_force, = ax_force.plot([], [], color=INTERACT_COLOR, lw=1.2,
                                 label=r'$-\alpha\,\langle F_{ball}, F_{wall}\rangle$')
    ax_force.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    # ── panel 3: position comparison ─────────────────────────────────────
    ax_track.set_ylim(wall_left - 1, wall_right + 1)
    ax_track.set_xlabel('Timestep', color=LABEL_COLOR, fontsize=9)
    ax_track.set_ylabel('Position', color=LABEL_COLOR, fontsize=9)
    ax_track.set_title('Position: Newtonian vs Spectral Prediction',
                       color=TITLE_COLOR, fontsize=11, fontweight='bold', pad=8)
    ax_track.axhline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_track.axhline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_track.axhline(0, color='#555', ls='-', lw=0.5, alpha=0.3)

    line_newton_pos, = ax_track.plot([], [], color=NEWTON_COLOR, lw=1.5,
                                      label='Newtonian (ground truth)')
    line_spectral_pred, = ax_track.plot([], [], color=BALL_COLOR, lw=1.5,
                                         ls='--', alpha=0.8,
                                         label='Spectral prediction')

    ax_track.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    # ── panel 4: FFT — spectral representation ──────────────────────────
    nyquist = 1.0 / (2.0 * dt)
    ax_fft_spectral.set_xlim(0, nyquist)
    ax_fft_spectral.set_ylim(0, 50)
    ax_fft_spectral.set_xlabel('Frequency (Hz)', color=LABEL_COLOR, fontsize=9)
    ax_fft_spectral.set_ylabel('|FFT|', color=LABEL_COLOR, fontsize=9)
    ax_fft_spectral.set_title('FFT — Spectral Prediction',
                               color=BALL_COLOR, fontsize=10,
                               fontweight='bold', pad=6)

    line_fft_spectral, = ax_fft_spectral.plot([], [], color=BALL_COLOR, lw=1.5)

    # ── panel 5: FFT — Newtonian ────────────────────────────────────────
    ax_fft_newton.set_xlim(0, nyquist)
    ax_fft_newton.set_ylim(0, 50)
    ax_fft_newton.set_xlabel('Frequency (Hz)', color=LABEL_COLOR, fontsize=9)
    ax_fft_newton.set_ylabel('|FFT|', color=LABEL_COLOR, fontsize=9)
    ax_fft_newton.set_title('FFT — Newtonian Ground Truth',
                             color=NEWTON_COLOR, fontsize=10,
                             fontweight='bold', pad=6)

    line_fft_newton, = ax_fft_newton.plot([], [], color=NEWTON_COLOR, lw=1.5)

    frame_text = ax_field.text(0.98, 0.02, '', transform=ax_field.transAxes,
                                color='#555', fontsize=8,
                                horizontalalignment='right')

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── animation ────────────────────────────────────────────────────────

    def init():
        for line in [line_ball_field, line_wall_field, line_force,
                     line_newton_pos, line_spectral_pred,
                     line_fft_spectral, line_fft_newton]:
            line.set_data([], [])
        newton_dot.set_data([], [])
        frame_text.set_text('')
        return ()

    def step(_frame):
        t = state['t']

        # ── 1. Newtonian physics (ground truth) ─────────────────
        newton['x'] += newton['v'] * dt
        if newton['x'] >= wall_right:
            newton['x'] = 2 * wall_right - newton['x']
            newton['v'] = -newton['v']
        elif newton['x'] <= wall_left:
            newton['x'] = 2 * wall_left - newton['x']
            newton['v'] = -newton['v']

        nx = newton['x']

        # ── 2. LMS update: train ball field from Newtonian position
        ball.update(nx)

        # ── 3. Spectral prediction & force ───────────────────────
        spectral_pred = ball.predict(nx)
        overlap = ball.inner_product(wall)
        force_spectral = -alpha * overlap

        # ── record history ───────────────────────────────────────
        history_newton_x.append(nx)
        history_spectral_pred.append(spectral_pred)
        history_force.append(force_spectral)
        history_t.append(t)

        state['t'] = t + 1

        # ── panel 1: fields + Newtonian ball ─────────────────────
        ball_curve = ball.evaluate(x_domain)
        wall_curve = wall.evaluate(x_domain)

        line_ball_field.set_data(x_domain, ball_curve)
        line_wall_field.set_data(x_domain, wall_curve)
        newton_dot.set_data([nx], [0])
        frame_text.set_text(f't={t}')

        # Force arrow at Newtonian ball position
        if force_arrow[0] is not None:
            force_arrow[0].remove()
        arrow_dx = np.clip(force_spectral * 3.0, -2.5, 2.5)
        if abs(arrow_dx) > 0.05:
            force_arrow[0] = ax_field.annotate(
                '', xy=(nx + arrow_dx, 0),
                xytext=(nx, 0),
                arrowprops=dict(arrowstyle='->', color=INTERACT_COLOR,
                                lw=2.5, mutation_scale=15),
                zorder=6)
        else:
            force_arrow[0] = None

        # ── panel 2: force time series ───────────────────────────
        ts = np.array(history_t)
        line_force.set_data(ts, np.array(history_force))
        if len(ts) > 1:
            ax_force.set_xlim(ts[0], ts[-1] + 1)

        # ── panel 3: position comparison ─────────────────────────
        line_newton_pos.set_data(ts, np.array(history_newton_x))
        line_spectral_pred.set_data(ts, np.array(history_spectral_pred))
        if len(ts) > 1:
            ax_track.set_xlim(ts[0], ts[-1] + 1)

        # ── panels 4-5: rolling FFT ─────────────────────────────
        n = len(history_newton_x)
        if n >= 64:
            win = np.hanning(n)

            newton_arr = np.array(history_newton_x)
            newton_arr = newton_arr - newton_arr.mean()
            spectral_arr = np.array(history_spectral_pred)
            spectral_arr = spectral_arr - spectral_arr.mean()

            fft_n = np.abs(np.fft.rfft(newton_arr * win))
            fft_s = np.abs(np.fft.rfft(spectral_arr * win))
            freqs = np.fft.rfftfreq(n, d=dt)

            line_fft_newton.set_data(freqs, fft_n)
            line_fft_spectral.set_data(freqs, fft_s)

            ymax = max(fft_n.max(), fft_s.max(), 1.0) * 1.1
            ymax = max(ymax, state['fft_ymax'] * 0.95)
            state['fft_ymax'] = ymax
            ax_fft_spectral.set_ylim(0, ymax)
            ax_fft_newton.set_ylim(0, ymax)

        return ()

    frames = range(max_frames) if max_frames is not None else itertools.count()
    anim = FuncAnimation(fig, step, frames=frames, init_func=init,
                         interval=interval, blit=False, repeat=False)
    return fig, anim


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='1D Spectral Field Visualization — Newtonian to Spectral')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g. out.gif)')
    parser.add_argument('--K', type=int, default=8,
                        help='Number of spectral components (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='Force coupling strength (default: 0.15)')
    parser.add_argument('--lr', type=float, default=0.15,
                        help='LMS learning rate for ball field (default: 0.15)')
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
        ball_lr=args.lr,
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
