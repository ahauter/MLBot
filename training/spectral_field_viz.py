"""
1D Spectral Field Visualization — Two Directions
=================================================
Two parallel panels showing the same ball bouncing, computed two ways:

Panel A — Spectral → Position:
  Ball position comes FROM the spectral field.  Inner product force between
  ball and wall wavepackets drives Fourier-shift dynamics.  Position is read
  from the field.

Panel B — Position → Spectral:
  Newtonian ball bounces off hard walls.  Each timestep, LMS trains spectral
  coefficients for BOTH the ball AND the walls from the observed positions
  and bounce events.  The spectral fields emerge from the physics.

The viewer sees whether the two directions agree — does the spectral
representation faithfully capture Newtonian physics, and does spectral-native
dynamics reproduce it?

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

COEFF_CLIP = 10.0

BG_COLOR = '#12121f'
BALL_COLOR = '#38bdf8'
WALL_COLOR = '#f87171'
INTERACT_COLOR = '#a78bfa'
NEWTON_COLOR = '#4ade80'
RECON_WALL_COLOR = '#fb923c'  # orange for reconstructed wall (legacy)
RESIDUAL_COLOR = '#facc15'   # yellow for LMS residual
ENV_FIELD_COLOR = '#c084fc'  # purple for learned environment field
PAD_COLOR = '#2dd4bf'        # teal for friction pad region marker
GRID_COLOR = '#444'
GRID_ALPHA = 0.15
LABEL_COLOR = '#aaa'
TITLE_COLOR = 'white'


# ── WavepacketObject ───────────────────────────────────────────────────────

class WavepacketObject:
    """Spectral wavepacket — all objects (ball, walls) use this same class."""

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
        self.lr = lr

        if c_cos is not None and c_sin is not None:
            self.c_cos = np.array(c_cos, dtype=np.float64)
            self.c_sin = np.array(c_sin, dtype=np.float64)
        else:
            envelope = amplitude * np.exp(-self.k**2 * sigma**2 / 2)
            self.c_cos = envelope * np.cos(self.k * x0)
            self.c_sin = envelope * np.sin(self.k * x0)

    def _basis(self, x: float) -> np.ndarray:
        return np.concatenate([np.cos(self.k * x), np.sin(self.k * x)])

    @property
    def _c(self) -> np.ndarray:
        return np.concatenate([self.c_cos, self.c_sin])

    @_c.setter
    def _c(self, val: np.ndarray) -> None:
        self.c_cos = val[:self.K]
        self.c_sin = val[self.K:]

    def evaluate(self, x_domain: np.ndarray) -> np.ndarray:
        cos_b = np.cos(self.k[None, :] * x_domain[:, None])
        sin_b = np.sin(self.k[None, :] * x_domain[:, None])
        return cos_b @ self.c_cos + sin_b @ self.c_sin

    def predict(self, x: float) -> float:
        return float(self._c @ self._basis(x))

    def inner_product(self, other: 'WavepacketObject') -> float:
        return float(np.sum(self.c_cos * other.c_cos +
                            self.c_sin * other.c_sin))

    def update(self, x_target: float) -> float:
        """LMS update: train field to predict x_target at x_target.  Returns residual."""
        if self.lr <= 0:
            return 0.0
        basis = self._basis(x_target)
        pred = float(self._c @ basis)
        residual = x_target - pred
        c = self._c + self.lr * basis * residual
        np.clip(c, -COEFF_CLIP, COEFF_CLIP, out=c)
        self._c = c
        return residual

    def shift(self, delta: float) -> None:
        for j in range(self.K):
            angle = self.k[j] * delta
            ca, sa = np.cos(angle), np.sin(angle)
            oc, os = self.c_cos[j], self.c_sin[j]
            self.c_cos[j] = oc * ca - os * sa
            self.c_sin[j] = oc * sa + os * ca

    def step(self, force: float, dt: float) -> None:
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
    c = np.zeros(2 * K, dtype=np.float64)
    k = np.asarray(frequencies, dtype=np.float64)
    for _ in range(warmup_steps):
        for x_w in wall_positions:
            basis = np.concatenate([np.cos(k * x_w), np.sin(k * x_w)])
            c += lr * basis * (x_w - c @ basis)
            np.clip(c, -COEFF_CLIP, COEFF_CLIP, out=c)
    return c[:K].copy(), c[K:].copy()


# ── animation ────────────────────────────────────────────────────────────────

def create_animation(K: int, frequencies: np.ndarray, alpha: float,
                     dt: float, ball_v0: float, wall_left: float,
                     wall_right: float, wall_warmup: int,
                     ball_sigma: float, ball_amplitude: float,
                     ball_lr: float, wall_mass: float,
                     pad_center: float, pad_width: float,
                     pad_friction: float,
                     interval: int, max_frames: int | None):
    x_domain = np.linspace(-6, 6, 500)
    window = 300
    pad_left = pad_center - pad_width / 2
    pad_right = pad_center + pad_width / 2

    # ════════════════════════════════════════════════════════════════════
    # PANEL A state: Spectral → Position
    # ════════════════════════════════════════════════════════════════════
    wall_c_cos, wall_c_sin = pretrain_wall_coefficients(
        K, frequencies, [wall_left, wall_right],
        lr=0.15, warmup_steps=wall_warmup)

    a_wall = WavepacketObject(
        K, frequencies, x0=0.0, mass=wall_mass, v0=0.0,
        c_cos=wall_c_cos.copy(), c_sin=wall_c_sin.copy())

    a_ball = WavepacketObject(
        K, frequencies, x0=0.0, mass=1.0, v0=ball_v0,
        sigma=ball_sigma, amplitude=ball_amplitude)

    a_hist_x = deque(maxlen=window)

    # ════════════════════════════════════════════════════════════════════
    # PANEL B state: Position → Spectral
    # ════════════════════════════════════════════════════════════════════
    newton = {'x': 0.0, 'v': ball_v0}

    # Ball field: Gaussian wavepacket, shifted to match Newtonian position
    b_ball = WavepacketObject(
        K, frequencies, x0=0.0, mass=1.0,
        sigma=ball_sigma, amplitude=ball_amplitude)

    # Environment field: starts empty, learns from acceleration anomalies.
    # No prior knowledge of walls or pads — discovers them from velocity changes.
    b_env = WavepacketObject(
        K, frequencies, x0=0.0, mass=wall_mass,
        c_cos=np.zeros(K), c_sin=np.zeros(K), lr=ball_lr)

    b_hist_x = deque(maxlen=window)

    # Shared
    history_t = deque(maxlen=window)
    state = {'t': 0, 'residual': 0.0, 'residual_x': 0.0, 'max_residual': 1.0}

    # ── figure: 2 field panels + position comparison ──────────────────
    fig, (ax_a, ax_b, ax_track) = plt.subplots(
        3, 1, figsize=(12, 11),
        gridspec_kw={'height_ratios': [2, 2, 1]})
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('Spectral Field — Two Directions',
                 color=TITLE_COLOR, fontsize=13, fontweight='bold', y=0.98)

    for ax in [ax_a, ax_b, ax_track]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.5)

    # ── Panel A: Spectral → Position ─────────────────────────────────
    ax_a.set_xlim(-6, 6)
    ax_a.set_ylim(-6, 6)
    ax_a.set_ylabel('F(x)', color=LABEL_COLOR, fontsize=9)
    ax_a.set_title('A: Spectral → Position  (position from field)',
                   color=BALL_COLOR, fontsize=11, fontweight='bold', pad=8)
    ax_a.axvline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)
    ax_a.axvline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)

    la_ball_field, = ax_a.plot(
        [], [], color=BALL_COLOR, lw=2, label='Ball field')
    la_wall_field, = ax_a.plot([], [], color=WALL_COLOR, lw=1.5, alpha=0.8,
                               label='Wall field')
    la_dot, = ax_a.plot([], [], 'o', color=BALL_COLOR, markersize=12,
                        zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    la_arrow = [None]

    ax_a.text(0.02, 0.97,
              r'$x_{ball} = \mathrm{shift}(v \cdot dt)$'
              '\n'
              r'$F = -\alpha\,\langle F_{ball}, F_{wall}\rangle$',
              transform=ax_a.transAxes, color='#ccc', fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                        edgecolor='#333', alpha=0.9))
    ax_a.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                edgecolor='#333', labelcolor='#ccc')

    # ── Panel B: Position → Spectral ─────────────────────────────────
    ax_b.set_xlim(-6, 6)
    ax_b.set_ylim(-6, 6)
    ax_b.set_ylabel('F(x)', color=LABEL_COLOR, fontsize=9)
    ax_b.set_title('B: Position → Spectral  (fields learned from physics)',
                   color=NEWTON_COLOR, fontsize=11, fontweight='bold', pad=8)
    ax_b.axvline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)
    ax_b.axvline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.6)
    ax_b.axvspan(pad_left, pad_right, color=PAD_COLOR, alpha=0.12, zorder=0)

    lb_ball_field, = ax_b.plot([], [], color=BALL_COLOR, lw=2, alpha=0.9,
                               label='Ball wavepacket (shifted)')
    lb_env_field, = ax_b.plot([], [], color=ENV_FIELD_COLOR, lw=1.5,
                              alpha=0.8, label='Learned env field')
    lb_dot, = ax_b.plot([], [], 'o', color=NEWTON_COLOR, markersize=12,
                        zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    lb_arrow = [None]
    lb_residual_arrow = [None]

    ax_b.text(0.02, 0.97,
              'Ball: wavepacket shifted to Newtonian pos'
              '\n'
              r'Env: LMS from accel anomaly $a_{actual} - a_{predicted}$',
              transform=ax_b.transAxes, color='#ccc', fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                        edgecolor='#333', alpha=0.9))
    ax_b.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                edgecolor='#333', labelcolor='#ccc')

    # ── Panel 3: position comparison ─────────────────────────────────
    ax_track.set_ylim(wall_left - 1, wall_right + 1)
    ax_track.set_xlabel('Timestep', color=LABEL_COLOR, fontsize=9)
    ax_track.set_ylabel('Position', color=LABEL_COLOR, fontsize=9)
    ax_track.set_title('Position Comparison',
                       color=TITLE_COLOR, fontsize=11, fontweight='bold', pad=8)
    ax_track.axhline(wall_left, color=WALL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_track.axhline(wall_right, color=WALL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_track.axhline(0, color='#555', ls='-', lw=0.5, alpha=0.3)
    ax_track.axhspan(pad_left, pad_right, color=PAD_COLOR,
                     alpha=0.08, zorder=0)

    lt_spectral, = ax_track.plot([], [], color=BALL_COLOR, lw=1.5,
                                 label='A: Spectral ball')
    lt_newton, = ax_track.plot([], [], color=NEWTON_COLOR, lw=1.5,
                               alpha=0.8, label='B: Newtonian ball')
    ax_track.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#ccc')

    frame_text = ax_a.text(0.98, 0.02, '', transform=ax_a.transAxes,
                           color='#555', fontsize=8,
                           horizontalalignment='right')
    lb_vel_text = ax_b.text(0.98, 0.97, '', transform=ax_b.transAxes,
                            color=NEWTON_COLOR, fontsize=10,
                            fontweight='bold',
                            horizontalalignment='right',
                            verticalalignment='top')

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── animation ────────────────────────────────────────────────────

    def init():
        for line in [la_ball_field, la_wall_field,
                     lb_ball_field, lb_env_field,
                     lt_spectral, lt_newton]:
            line.set_data([], [])
        la_dot.set_data([], [])
        lb_dot.set_data([], [])
        frame_text.set_text('')
        lb_vel_text.set_text('')
        return ()

    def step(_frame):
        t = state['t']

        # ════════════════════════════════════════════════════════════
        # PANEL A: Spectral → Position
        # ════════════════════════════════════════════════════════════
        overlap_a = a_ball.inner_product(a_wall)
        force_a = -alpha * overlap_a

        a_ball.step(force_a, dt)
        a_wall.step(-force_a, dt)

        a_hist_x.append(a_ball.x)

        # ════════════════════════════════════════════════════════════
        # PANEL B: Position → Spectral
        # ════════════════════════════════════════════════════════════
        # Newtonian step — record velocity before and after
        v_before = newton['v']

        newton['x'] += newton['v'] * dt

        # Friction pad: decelerate proportional to velocity
        if pad_left <= newton['x'] <= pad_right:
            newton['v'] -= pad_friction * newton['v'] * dt

        # Wall bounces
        if newton['x'] >= wall_right:
            newton['x'] = 2 * wall_right - newton['x']
            newton['v'] = -newton['v']
        elif newton['x'] <= wall_left:
            newton['x'] = 2 * wall_left - newton['x']
            newton['v'] = -newton['v']

        v_after = newton['v']
        nx = newton['x']

        # Shift ball wavepacket to match Newtonian position
        delta_b = nx - b_ball.x
        b_ball.shift(delta_b)
        b_ball.x = nx

        # ── Anomaly-based learning ──────────────────────────────
        # Observed acceleration vs predicted (from learned env field)
        a_actual = (v_after - v_before) / dt
        overlap_b = b_ball.inner_product(b_env)
        force_b = -alpha * overlap_b
        a_predicted = force_b / b_ball.mass
        a_residual = a_actual - a_predicted

        # LMS: update env field at ball position, scaled by anomaly
        # No knowledge of walls or pads — only the velocity anomaly
        anomaly_mag = abs(a_residual)
        if anomaly_mag > 0.1:
            n_lms = int(np.clip(anomaly_mag * 0.5, 1, 10))
            for _ in range(n_lms):
                b_env.update(nx)

        # Store residual for visualization
        state['residual'] = a_residual
        state['residual_x'] = nx

        b_hist_x.append(nx)
        history_t.append(t)
        state['t'] = t + 1

        # ── draw panel A ─────────────────────────────────────────
        a_ball_curve = a_ball.evaluate(x_domain)
        a_wall_curve = a_wall.evaluate(x_domain)
        la_ball_field.set_data(x_domain, a_ball_curve)
        la_wall_field.set_data(x_domain, a_wall_curve)
        la_dot.set_data([a_ball.x], [0])
        frame_text.set_text(f't={t}')

        # Force arrow on panel A
        if la_arrow[0] is not None:
            la_arrow[0].remove()
        adx = np.clip(force_a * 3.0, -2.5, 2.5)
        if abs(adx) > 0.05:
            la_arrow[0] = ax_a.annotate(
                '', xy=(a_ball.x + adx, 0), xytext=(a_ball.x, 0),
                arrowprops=dict(arrowstyle='->', color=INTERACT_COLOR,
                                lw=2.5, mutation_scale=15), zorder=6)
        else:
            la_arrow[0] = None

        # ── draw panel B ─────────────────────────────────────────
        b_ball_curve = b_ball.evaluate(x_domain)
        b_env_curve = b_env.evaluate(x_domain)
        lb_ball_field.set_data(x_domain, b_ball_curve)
        lb_env_field.set_data(x_domain, b_env_curve)
        lb_dot.set_data([nx], [0])
        lb_vel_text.set_text(f'v={newton["v"]:.2f}')

        # Force arrow on panel B (reconstructed force)
        if lb_arrow[0] is not None:
            lb_arrow[0].remove()
        bdx = np.clip(force_b * 3.0, -2.5, 2.5)
        if abs(bdx) > 0.05:
            lb_arrow[0] = ax_b.annotate(
                '', xy=(nx + bdx, 0), xytext=(nx, 0),
                arrowprops=dict(arrowstyle='->', color=INTERACT_COLOR,
                                lw=2.5, mutation_scale=15), zorder=6)
        else:
            lb_arrow[0] = None

        # Acceleration anomaly arrow (vertical, at ball position)
        if lb_residual_arrow[0] is not None:
            lb_residual_arrow[0].remove()
            lb_residual_arrow[0] = None
        r = state['residual']
        rx = state['residual_x']
        # Auto-scale with exponential decay
        state['max_residual'] = max(abs(r), state['max_residual'] * 0.98)
        mr = max(state['max_residual'], 0.01)
        rdy = np.clip(abs(r) / mr * 3.0, 0, 4.0)
        if abs(rdy) > 0.05:
            lb_residual_arrow[0] = ax_b.annotate(
                '', xy=(rx, rdy), xytext=(rx, 0),
                arrowprops=dict(arrowstyle='->', color=RESIDUAL_COLOR,
                                lw=2.5, mutation_scale=15),
                zorder=7)

        # ── draw panel 3: position comparison ────────────────────
        ts = np.array(history_t)
        lt_spectral.set_data(ts, np.array(a_hist_x))
        lt_newton.set_data(ts, np.array(b_hist_x))
        if len(ts) > 1:
            ax_track.set_xlim(ts[0], ts[-1] + 1)

        return ()

    frames = range(max_frames) if max_frames is not None else itertools.count()
    anim = FuncAnimation(fig, step, frames=frames, init_func=init,
                         interval=interval, blit=False, repeat=False)
    return fig, anim


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='1D Spectral Field Visualization — Two Directions')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g. out.gif)')
    parser.add_argument('--K', type=int, default=8,
                        help='Number of spectral components (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='Force coupling strength (default: 0.15)')
    parser.add_argument('--lr', type=float, default=0.15,
                        help='LMS learning rate (default: 0.15)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Frame limit (default: infinite, required for --save)')
    parser.add_argument('--pad-center', type=float, default=1.5,
                        help='Friction pad center position (default: 1.5)')
    parser.add_argument('--pad-width', type=float, default=1.0,
                        help='Friction pad width (default: 1.0)')
    parser.add_argument('--pad-friction', type=float, default=0.3,
                        help='Friction pad coefficient mu (default: 0.3)')
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
        ball_lr=args.lr,
        wall_mass=1e6,
        pad_center=args.pad_center,
        pad_width=args.pad_width,
        pad_friction=args.pad_friction,
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
