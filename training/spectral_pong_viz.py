"""
2D Spectral Pong Visualization
===============================
Pong game where ball and paddles are observed spectral waveforms with 2D
amplitudes, and top/bottom walls are an unobserved waveform learned via LMS
from acceleration anomalies.

Two 1D domain visualizations: X-domain (horizontal) and Y-domain (vertical),
each showing one component of the 2D amplitude vectors.

Controls:
    Left paddle:  A / D
    Right paddle:  Up / Down

Usage:
    python training/spectral_pong_viz.py                              # interactive
    python training/spectral_pong_viz.py --auto-paddle                # AI paddles
    python training/spectral_pong_viz.py --save out.gif --frames 600 --auto-paddle
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation

# -- constants ----------------------------------------------------------------

COEFF_CLIP = 10.0

BG_COLOR = '#12121f'
BALL_COLOR = '#38bdf8'
PADDLE_COLOR_L = '#4ade80'
PADDLE_COLOR_R = '#f87171'
WALL_COLOR = '#333'
NET_COLOR = '#555'
SCORE_COLOR = 'white'
ENV_FIELD_COLOR = '#c084fc'
INTERACT_COLOR = '#a78bfa'
RESIDUAL_COLOR = '#facc15'
REWARD_COLOR = '#fb923c'       # orange for learned reward field
GRID_COLOR = '#444'
GRID_ALPHA = 0.15
LABEL_COLOR = '#aaa'
TITLE_COLOR = 'white'

COURT_LEFT = -5.0
COURT_RIGHT = 5.0
COURT_TOP = 3.0
COURT_BOTTOM = -3.0

WORLD_BOUNDS = [
    (COURT_LEFT, COURT_RIGHT),   # axis 0 (x)
    (COURT_BOTTOM, COURT_TOP),   # axis 1 (y)
]

PADDLE_X_OFFSET = 0.5
PADDLE_WIDTH = 0.15
PADDLE_HEIGHT = 1.0
PADDLE_SPEED = 4.0

BALL_RADIUS = 0.15
BALL_SPEED = 2.0
SPIN_FACTOR = 2.0


# -- WavepacketObject2D ------------------------------------------------------

class WavepacketObject2D:
    """Spectral wavepacket with 2D amplitude vectors.

    Each frequency k_j has coefficients c_cos[j] and c_sin[j] in R^2.
    The field is evaluated on a 1D domain per axis:
        F_d(x) = sum_j  c_cos[j, d] * cos(k_j * x) + c_sin[j, d] * sin(k_j * x)
    where d in {0, 1} selects the x or y component.
    """

    def __init__(self, K: int, frequencies: np.ndarray,
                 pos0: tuple[float, float] = (0.0, 0.0),
                 mass: float = 1.0,
                 vel0: tuple[float, float] = (0.0, 0.0),
                 c_cos: np.ndarray | None = None,
                 c_sin: np.ndarray | None = None,
                 sigma: float = 0.8, amplitude: float = 1.5,
                 lr: float = 0.0, lr_tracking: float = 0.0):
        self.K = K
        self.k = np.asarray(frequencies, dtype=np.float64)
        self.mass = mass
        self.pos = np.array(pos0, dtype=np.float64)
        self.vel = np.array(vel0, dtype=np.float64)
        self.lr = lr
        self.lr_tracking = lr_tracking

        if c_cos is not None and c_sin is not None:
            self.c_cos = np.array(c_cos, dtype=np.float64)  # (K, 2)
            self.c_sin = np.array(c_sin, dtype=np.float64)  # (K, 2)
        else:
            # Gaussian envelope, phase set from position per axis
            envelope = amplitude * np.exp(-self.k**2 * sigma**2 / 2)  # (K,)
            self.c_cos = np.zeros((K, 2), dtype=np.float64)
            self.c_sin = np.zeros((K, 2), dtype=np.float64)
            for d in range(2):
                self.c_cos[:, d] = envelope * np.cos(self.k * self.pos[d])
                self.c_sin[:, d] = envelope * np.sin(self.k * self.pos[d])

        self.normalize()

    def _basis(self, x: float) -> np.ndarray:
        """1D Fourier basis at scalar domain position x. Returns (2K,)."""
        return np.concatenate([np.cos(self.k * x), np.sin(self.k * x)])

    def _c_flat(self, axis: int) -> np.ndarray:
        """Coefficients for one axis as flat (2K,) vector."""
        return np.concatenate([self.c_cos[:, axis], self.c_sin[:, axis]])

    def _set_c_flat(self, axis: int, c: np.ndarray) -> None:
        self.c_cos[:, axis] = c[:self.K]
        self.c_sin[:, axis] = c[self.K:]

    def evaluate(self, x_domain: np.ndarray, axis: int) -> np.ndarray:
        """Evaluate 1D field for one axis over domain grid. Returns (N,)."""
        cos_b = np.cos(self.k[None, :] * x_domain[:, None])  # (N, K)
        sin_b = np.sin(self.k[None, :] * x_domain[:, None])  # (N, K)
        return cos_b @ self.c_cos[:, axis] + sin_b @ self.c_sin[:, axis]

    def predict(self, x: float, axis: int) -> float:
        """Predict field value at domain position x for one axis."""
        return float(self._c_flat(axis) @ self._basis(x))

    def integrate(self, axis: int) -> float:
        """Analytical integral of F_d over world bounds."""
        a, b = WORLD_BOUNDS[axis]
        int_cos = (np.sin(self.k * b) - np.sin(self.k * a)) / self.k  # (K,)
        int_sin = (np.cos(self.k * a) - np.cos(self.k * b)) / self.k  # (K,)
        return float(self.c_cos[:, axis] @ int_cos +
                     self.c_sin[:, axis] @ int_sin)

    def normalize(self) -> None:
        """Rescale coefficients so each axis integrates to 1 over world."""
        for d in range(2):
            I = self.integrate(d)
            if abs(I) > 1e-10:
                self.c_cos[:, d] /= I
                self.c_sin[:, d] /= I

    def gradient(self, x: float, axis: int) -> float:
        """Analytical derivative dF_d/dx at domain position x.

        dF/dx = sum_j [-c_cos[j,d] * k_j * sin(k_j * x)
                        + c_sin[j,d] * k_j * cos(k_j * x)]
        """
        return float(np.sum(
            -self.c_cos[:, axis] * self.k * np.sin(self.k * x)
            + self.c_sin[:, axis] * self.k * np.cos(self.k * x)))

    def inner_product_2d(self, other: 'WavepacketObject2D') -> np.ndarray:
        """Component-wise inner product per spatial dimension. Returns (2,)."""
        return np.array([
            float(np.sum(self.c_cos[:, d] * other.c_cos[:, d] +
                         self.c_sin[:, d] * other.c_sin[:, d]))
            for d in range(2)
        ])

    def cross_product_2d(self, other: 'WavepacketObject2D') -> np.ndarray:
        """Imaginary part of complex inner product per axis. Returns (2,).

        cross[d] = sum_j (self.c_cos[j,d] * other.c_sin[j,d]
                        - self.c_sin[j,d] * other.c_cos[j,d])

        Proportional to sin(k_j * (self.pos[d] - other.pos[d])) — encodes
        signed displacement between the two wavepackets.
        """
        return np.array([
            float(np.sum(self.c_cos[:, d] * other.c_sin[:, d] -
                         self.c_sin[:, d] * other.c_cos[:, d]))
            for d in range(2)
        ])

    def normalized_inner_product(self, other: 'WavepacketObject2D') -> float:
        """Spectral alignment score in ~[0, 1]. Attention weight for LMS."""
        ip = self.inner_product_2d(other)  # (2,)
        norm_s = np.sqrt(np.sum(self.c_cos**2 + self.c_sin**2, axis=0))  # (2,)
        norm_o = np.sqrt(np.sum(other.c_cos**2 + other.c_sin**2, axis=0))  # (2,)
        nip = ip / (norm_s * norm_o + 1e-8)  # (2,)
        return float(np.linalg.norm(nip) / np.sqrt(2))

    def shift(self, delta: float, axis: int) -> None:
        """Fourier-domain phase rotation for one axis."""
        for j in range(self.K):
            angle = self.k[j] * delta
            ca, sa = np.cos(angle), np.sin(angle)
            oc = self.c_cos[j, axis]
            os = self.c_sin[j, axis]
            self.c_cos[j, axis] = oc * ca - os * sa
            self.c_sin[j, axis] = oc * sa + os * ca

    def set_position(self, new_pos: np.ndarray) -> None:
        """Shift both axes so the wavepacket tracks new_pos."""
        for d in range(2):
            delta = new_pos[d] - self.pos[d]
            if abs(delta) > 1e-12:
                self.shift(delta, axis=d)
        self.pos[:] = new_pos
        np.clip(self.c_cos, -COEFF_CLIP, COEFF_CLIP, out=self.c_cos)
        np.clip(self.c_sin, -COEFF_CLIP, COEFF_CLIP, out=self.c_sin)

    def update_lms(self, domain_pos: np.ndarray, target_2d: np.ndarray,
                   anomaly_scale: float = 1.0) -> np.ndarray:
        """LMS update with 2D target. Returns residual (2,)."""
        if self.lr <= 0:
            return np.zeros(2)
        residual = np.zeros(2)
        for d in range(2):
            basis = self._basis(domain_pos[d])       # (2K,)
            c = self._c_flat(d)                       # (2K,)
            pred = float(c @ basis)
            res = target_2d[d] - pred
            residual[d] = res
            c_new = c + (self.lr * anomaly_scale) * basis * res
            np.clip(c_new, -COEFF_CLIP, COEFF_CLIP, out=c_new)
            self._set_c_flat(d, c_new)
        return residual

    def update_with_attention(self, domain_pos: np.ndarray,
                              target_2d: np.ndarray,
                              interaction_scales: list[float]) -> np.ndarray:
        """LMS with effective_lr = lr_tracking + lr * sum(interaction_scales)."""
        effective_lr = self.lr_tracking + self.lr * sum(interaction_scales)
        if effective_lr <= 0:
            return np.zeros(2)
        residual = np.zeros(2)
        for d in range(2):
            basis = self._basis(domain_pos[d])
            c = self._c_flat(d)
            res = target_2d[d] - float(c @ basis)
            residual[d] = res
            c_new = c + effective_lr * basis * res
            np.clip(c_new, -COEFF_CLIP, COEFF_CLIP, out=c_new)
            self._set_c_flat(d, c_new)
        return residual



# -- helpers ------------------------------------------------------------------

def reset_ball(ball: dict, toward: str = 'random',
               speed: float = BALL_SPEED) -> None:
    ball['x'] = 0.0
    ball['y'] = 0.0
    if toward == 'random':
        toward = 'left' if np.random.random() < 0.5 else 'right'
    angle = np.random.uniform(-0.4, 0.4)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
    ball['vx'] = -vx if toward == 'left' else vx
    ball['vy'] = vy


# -- game + visualization ----------------------------------------------------

def create_game(K: int, frequencies: np.ndarray, alpha: float,
                dt: float, ball_speed: float, interval: int,
                max_frames: int | None, auto_paddle: bool,
                ball_sigma: float,
                env_lr: float, spectral_paddle: bool = False,
                alpha_paddle: float = 0.5,
                paddle_damping: float = 0.85,
                beta_paddle: float = 0.3,
                gamma_paddle: float = 0.2,
                reward_lr: float = 0.3,
                reward_decay: float = 0.999,
                ball_lr: float = 0.15,
                paddle_lr: float = 0.1,
                lr_tracking: float = 0.01):

    # -- spectral setup -------------------------------------------------------
    wp_ball = WavepacketObject2D(
        K, frequencies, pos0=(0.0, 0.0), mass=1.0,
        sigma=ball_sigma,
        lr=ball_lr, lr_tracking=lr_tracking)

    wp_paddle_l = WavepacketObject2D(
        K, frequencies,
        pos0=(COURT_LEFT + PADDLE_X_OFFSET, 0.0), mass=1e6,
        sigma=0.5, amplitude=1.0,
        lr=paddle_lr, lr_tracking=lr_tracking * 2)

    wp_paddle_r = WavepacketObject2D(
        K, frequencies,
        pos0=(COURT_RIGHT - PADDLE_X_OFFSET, 0.0), mass=1e6,
        sigma=0.5, amplitude=1.0,
        lr=paddle_lr, lr_tracking=lr_tracking * 2)

    # Env and reward start with orthogonal spectral basis vectors so the
    # normalized inner product can bootstrap learning:
    #   env  — basis [1,0,0,...] on x-axis, [0,1,0,...] on y-axis
    #   reward — basis [0,0,1,...] on both axes
    env_c_cos = np.zeros((K, 2))
    env_c_cos[0, 0] = 1.0  # frequency 0 → x
    env_c_cos[1, 1] = 1.0  # frequency 1 → y
    wp_env = WavepacketObject2D(
        K, frequencies, pos0=(0.0, 0.0), mass=1e6,
        c_cos=env_c_cos, c_sin=np.zeros((K, 2)),
        lr=env_lr, lr_tracking=0.0)

    reward_c_cos = np.zeros((K, 2))
    reward_c_cos[2, 0] = 1.0  # frequency 2 → both axes
    reward_c_cos[2, 1] = 1.0
    wp_reward = WavepacketObject2D(
        K, frequencies, pos0=(0.0, 0.0), mass=1.0,
        c_cos=reward_c_cos, c_sin=np.zeros((K, 2)),
        lr=reward_lr, lr_tracking=0.0)

    # -- Newtonian state ------------------------------------------------------
    paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
    paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET

    ball = {'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0}
    reset_ball(ball, toward='right', speed=ball_speed)

    left_paddle = {'y': 0.0, 'score': 0}
    right_paddle = {'y': 0.0, 'score': 0}
    key_times: dict[str, float] = {}
    paddle_vel = {'l': 0.0, 'r': 0.0}
    state = {'freeze': 0, 't': 0,
             'anomaly': np.zeros(2), 'max_anomaly': 1.0}

    window = 200
    anomaly_hist = deque(maxlen=window)
    history_t = deque(maxlen=window)

    # -- figure layout --------------------------------------------------------
    # Layout:  Y-domain (left) | X-domain (top-right)
    #                          | Court    (bottom-right)
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('Spectral Pong — 2D Amplitude Wavepackets',
                 color=TITLE_COLOR, fontsize=14, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 2],
                          hspace=0.25, wspace=0.2,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    ax_y = fig.add_subplot(gs[:, 0])   # Y-domain (left, full height)
    ax_x = fig.add_subplot(gs[0, 1])   # X-domain (top-right)
    ax_court = fig.add_subplot(gs[1, 1])  # Court (bottom-right)

    for ax in [ax_y, ax_x, ax_court]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')

    # -- X-domain panel -------------------------------------------------------
    x_dom = np.linspace(COURT_LEFT - 1, COURT_RIGHT + 1, 400)
    ax_x.set_xlim(x_dom[0], x_dom[-1])
    ax_x.set_ylim(-1.0, 1.0)
    ax_x.set_title('X-domain spectral fields', color=TITLE_COLOR, fontsize=10)
    ax_x.set_ylabel('F_x(x)', color=LABEL_COLOR, fontsize=8)
    ax_x.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.5)

    lx_ball, = ax_x.plot([], [], color=BALL_COLOR, lw=2, label='Ball')
    lx_pad_l, = ax_x.plot([], [], color=PADDLE_COLOR_L, lw=1.5, alpha=0.7,
                           label='L paddle')
    lx_pad_r, = ax_x.plot([], [], color=PADDLE_COLOR_R, lw=1.5, alpha=0.7,
                           label='R paddle')
    lx_env, = ax_x.plot([], [], color=ENV_FIELD_COLOR, lw=1.5, alpha=0.8,
                         label='Env (learned)')
    lx_reward, = ax_x.plot([], [], color=REWARD_COLOR, lw=1.5, alpha=0.8,
                            ls='--', label='Reward (learned)')
    lx_dot, = ax_x.plot([], [], 'o', color=BALL_COLOR, markersize=8,
                         zorder=5, markeredgecolor='white', markeredgewidth=1)
    ax_x.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
                edgecolor='#333', labelcolor='#ccc')

    # -- Y-domain panel -------------------------------------------------------
    y_dom = np.linspace(COURT_BOTTOM - 1, COURT_TOP + 1, 400)
    ax_y.set_ylim(y_dom[0], y_dom[-1])
    ax_y.set_xlim(-1.5, 1.5)
    ax_y.set_title('Y-domain', color=TITLE_COLOR, fontsize=10)
    ax_y.set_xlabel('F_y(y)', color=LABEL_COLOR, fontsize=8)
    ax_y.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.5)
    # Walls as dashed lines on the Y-domain
    ax_y.axhline(COURT_TOP, color=RESIDUAL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_y.axhline(COURT_BOTTOM, color=RESIDUAL_COLOR, ls='--', lw=1, alpha=0.5)

    ly_ball, = ax_y.plot([], [], color=BALL_COLOR, lw=2, label='Ball')
    ly_pad_l, = ax_y.plot([], [], color=PADDLE_COLOR_L, lw=1.5, alpha=0.7,
                           label='L paddle')
    ly_pad_r, = ax_y.plot([], [], color=PADDLE_COLOR_R, lw=1.5, alpha=0.7,
                           label='R paddle')
    ly_env, = ax_y.plot([], [], color=ENV_FIELD_COLOR, lw=1.5, alpha=0.8,
                         label='Env (learned)')
    ly_reward, = ax_y.plot([], [], color=REWARD_COLOR, lw=1.5, alpha=0.8,
                            ls='--', label='Reward (learned)')
    ly_dot, = ax_y.plot([], [], 'o', color=BALL_COLOR, markersize=8,
                         zorder=5, markeredgecolor='white', markeredgewidth=1)
    ax_y.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
                edgecolor='#333', labelcolor='#ccc')

    # -- Court panel ----------------------------------------------------------
    ax_court.set_xlim(COURT_LEFT - 0.5, COURT_RIGHT + 0.5)
    ax_court.set_ylim(COURT_BOTTOM - 0.6, COURT_TOP + 0.6)
    ax_court.set_aspect('equal')
    ax_court.set_xticks([])
    ax_court.set_yticks([])
    for spine in ax_court.spines.values():
        spine.set_visible(False)

    court_rect = Rectangle(
        (COURT_LEFT, COURT_BOTTOM),
        COURT_RIGHT - COURT_LEFT, COURT_TOP - COURT_BOTTOM,
        linewidth=2, edgecolor=WALL_COLOR, facecolor='none', zorder=1)
    ax_court.add_patch(court_rect)

    for y_pos in np.linspace(COURT_BOTTOM + 0.15, COURT_TOP - 0.15, 15):
        ax_court.plot([0, 0], [y_pos, y_pos + 0.2], color=NET_COLOR,
                      lw=1.5, alpha=0.5, zorder=1)

    lp_patch = FancyBboxPatch(
        (paddle_lx - PADDLE_WIDTH / 2, -PADDLE_HEIGHT / 2),
        PADDLE_WIDTH, PADDLE_HEIGHT,
        boxstyle='round,pad=0.03', facecolor=PADDLE_COLOR_L,
        edgecolor='white', linewidth=1.5, zorder=3)
    rp_patch = FancyBboxPatch(
        (paddle_rx - PADDLE_WIDTH / 2, -PADDLE_HEIGHT / 2),
        PADDLE_WIDTH, PADDLE_HEIGHT,
        boxstyle='round,pad=0.03', facecolor=PADDLE_COLOR_R,
        edgecolor='white', linewidth=1.5, zorder=3)
    ax_court.add_patch(lp_patch)
    ax_court.add_patch(rp_patch)

    ball_dot, = ax_court.plot([0], [0], 'o', color=BALL_COLOR, markersize=10,
                              zorder=4, markeredgecolor='white',
                              markeredgewidth=1.0)

    score_text = ax_court.text(0, COURT_TOP + 0.35, '0 \u2014 0',
                               color=SCORE_COLOR, fontsize=18,
                               fontweight='bold',
                               horizontalalignment='center',
                               verticalalignment='center', zorder=5)

    anomaly_arrow = [None]
    frame_text = ax_court.text(0.98, 0.02, '', transform=ax_court.transAxes,
                               color='#555', fontsize=8,
                               horizontalalignment='right')

    # -- keyboard -------------------------------------------------------------
    def on_key_press(event):
        key_times[event.key] = time.time()
    fig.canvas.mpl_connect('key_press_event', on_key_press)


    # -- animation step -------------------------------------------------------
    def init():
        for ln in [lx_ball, lx_pad_l, lx_pad_r, lx_env, lx_reward,
                    ly_ball, ly_pad_l, ly_pad_r, ly_env, ly_reward]:
            ln.set_data([], [])
        lx_dot.set_data([], [])
        ly_dot.set_data([], [])
        ball_dot.set_data([], [])
        frame_text.set_text('')
        return ()

    def step(_frame):
        t = state['t']

        # Freeze after scoring
        if state['freeze'] > 0:
            state['freeze'] -= 1
            state['t'] = t + 1
            return ()

        # -- move paddles -------------------------------------------------
        half_h = PADDLE_HEIGHT / 2
        if auto_paddle:
            if spectral_paddle:
                # Three-term spectral rule:
                #   tracking     = cross(paddle, ball) — signed displacement
                #   anticipation = -cross(ball, env)   — pre-correct for wall bounce
                #   reward_grad  = dR/dy at paddle pos — move toward higher reward
                tracking_l = wp_paddle_l.cross_product_2d(wp_ball)[1]
                wall_signal = wp_ball.cross_product_2d(wp_env)[1]
                reward_grad_l = np.tanh(wp_reward.gradient(left_paddle['y'], axis=1))
                force_l = (alpha_paddle * tracking_l
                           - beta_paddle * wall_signal
                           + gamma_paddle * reward_grad_l)
                paddle_vel['l'] = paddle_damping * paddle_vel['l'] + force_l * dt
                left_paddle['y'] += paddle_vel['l'] * dt

                tracking_r = wp_paddle_r.cross_product_2d(wp_ball)[1]
                reward_grad_r = np.tanh(wp_reward.gradient(right_paddle['y'], axis=1))
                force_r = (alpha_paddle * tracking_r
                           - beta_paddle * wall_signal
                           + gamma_paddle * reward_grad_r)
                paddle_vel['r'] = paddle_damping * paddle_vel['r'] + force_r * dt
                right_paddle['y'] += paddle_vel['r'] * dt
            else:
                # Simple tracking AI
                track_speed = PADDLE_SPEED * dt * 0.8
                if ball['y'] > left_paddle['y'] + 0.1:
                    left_paddle['y'] += track_speed
                elif ball['y'] < left_paddle['y'] - 0.1:
                    left_paddle['y'] -= track_speed
                if ball['y'] > right_paddle['y'] + 0.1:
                    right_paddle['y'] += track_speed
                elif ball['y'] < right_paddle['y'] - 0.1:
                    right_paddle['y'] -= track_speed
        else:
            now = time.time()
            held = 0.15
            if now - key_times.get('a', 0) < held:
                left_paddle['y'] += PADDLE_SPEED * dt
            if now - key_times.get('d', 0) < held:
                left_paddle['y'] -= PADDLE_SPEED * dt
            if now - key_times.get('up', 0) < held:
                right_paddle['y'] += PADDLE_SPEED * dt
            if now - key_times.get('down', 0) < held:
                right_paddle['y'] -= PADDLE_SPEED * dt

        left_paddle['y'] = np.clip(left_paddle['y'],
                                    COURT_BOTTOM + half_h, COURT_TOP - half_h)
        right_paddle['y'] = np.clip(right_paddle['y'],
                                     COURT_BOTTOM + half_h, COURT_TOP - half_h)

        # Zero paddle velocity when clamped to court edge
        if spectral_paddle and auto_paddle:
            if left_paddle['y'] <= COURT_BOTTOM + half_h or left_paddle['y'] >= COURT_TOP - half_h:
                paddle_vel['l'] = 0.0
            if right_paddle['y'] <= COURT_BOTTOM + half_h or right_paddle['y'] >= COURT_TOP - half_h:
                paddle_vel['r'] = 0.0

        # -- Newtonian ball step ------------------------------------------
        vx_before, vy_before = ball['vx'], ball['vy']

        ball['x'] += ball['vx'] * dt
        ball['y'] += ball['vy'] * dt

        # Top/bottom wall bounce
        if ball['y'] >= COURT_TOP - BALL_RADIUS:
            ball['y'] = 2 * (COURT_TOP - BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']
        elif ball['y'] <= COURT_BOTTOM + BALL_RADIUS:
            ball['y'] = 2 * (COURT_BOTTOM + BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']

        # Left paddle collision
        if (ball['vx'] < 0
                and ball['x'] - BALL_RADIUS <= paddle_lx + PADDLE_WIDTH / 2
                and ball['x'] - BALL_RADIUS >= paddle_lx - PADDLE_WIDTH / 2 - abs(ball['vx'] * dt)
                and abs(ball['y'] - left_paddle['y']) <= half_h + BALL_RADIUS):
            ball['x'] = paddle_lx + PADDLE_WIDTH / 2 + BALL_RADIUS
            ball['vx'] = -ball['vx']
            offset = (ball['y'] - left_paddle['y']) / half_h
            ball['vy'] += offset * SPIN_FACTOR
            spd = np.hypot(ball['vx'], ball['vy'])
            ball['vx'] *= ball_speed / spd
            ball['vy'] *= ball_speed / spd

        # Right paddle collision
        if (ball['vx'] > 0
                and ball['x'] + BALL_RADIUS >= paddle_rx - PADDLE_WIDTH / 2
                and ball['x'] + BALL_RADIUS <= paddle_rx + PADDLE_WIDTH / 2 + abs(ball['vx'] * dt)
                and abs(ball['y'] - right_paddle['y']) <= half_h + BALL_RADIUS):
            ball['x'] = paddle_rx - PADDLE_WIDTH / 2 - BALL_RADIUS
            ball['vx'] = -ball['vx']
            offset = (ball['y'] - right_paddle['y']) / half_h
            ball['vy'] += offset * SPIN_FACTOR
            spd = np.hypot(ball['vx'], ball['vy'])
            ball['vx'] *= ball_speed / spd
            ball['vy'] *= ball_speed / spd

        vx_after, vy_after = ball['vx'], ball['vy']

        # Scoring
        scored = False
        reward = 0.0
        if ball['x'] < COURT_LEFT:
            right_paddle['score'] += 1
            reward = -1.0   # left paddle conceded
            scored = True
        elif ball['x'] > COURT_RIGHT:
            left_paddle['score'] += 1
            reward = +1.0   # left paddle scored
            scored = True

        if scored:
            reset_ball(ball, toward='left' if reward < 0 else 'right',
                       speed=ball_speed)
            state['freeze'] = int(0.5 / dt)
            paddle_vel['l'] = 0.0
            paddle_vel['r'] = 0.0
            # Re-init ball wavepacket on score
            wp_ball.__init__(K, frequencies, pos0=(ball['x'], ball['y']),
                             mass=1.0, sigma=ball_sigma,
                             lr=ball_lr, lr_tracking=lr_tracking)

        # -- update wavepackets: shift + LMS + PMF deviation learning ----
        ball_pos = np.array([ball['x'], ball['y']])
        paddle_l_pos = np.array([paddle_lx, left_paddle['y']])
        paddle_r_pos = np.array([paddle_rx, right_paddle['y']])

        # 1. Propagation: shift by velocity (free-flight kinematics)
        for d in range(2):
            wp_ball.shift([ball['vx'], ball['vy']][d] * dt, axis=d)
        delta_l = left_paddle['y'] - wp_paddle_l.pos[1]
        delta_r = right_paddle['y'] - wp_paddle_r.pos[1]
        if abs(delta_l) > 1e-12:
            wp_paddle_l.shift(delta_l, axis=1)
        if abs(delta_r) > 1e-12:
            wp_paddle_r.shift(delta_r, axis=1)

        # 2. Attention-weighted LMS correction (shape only, target=1.0)
        if not scored:
            nip_ball_env = abs(wp_ball.normalized_inner_product(wp_env))
            nip_ball_padL = abs(wp_ball.normalized_inner_product(wp_paddle_l))
            nip_ball_padR = abs(wp_ball.normalized_inner_product(wp_paddle_r))

            unity = np.array([1.0, 1.0])
            wp_ball.update_with_attention(
                ball_pos, unity,
                [nip_ball_env, nip_ball_padL, nip_ball_padR])
            wp_paddle_l.update_with_attention(
                paddle_l_pos, unity, [nip_ball_padL])
            wp_paddle_r.update_with_attention(
                paddle_r_pos, unity, [nip_ball_padR])

        # 3. Measure integral deviation BEFORE normalizing
        #    Deviation from PMF=1 reveals environmental interaction
        ball_deviation = np.array([wp_ball.integrate(d) - 1.0
                                   for d in range(2)])

        # 4. Normalize observed objects back to PMF
        wp_ball.normalize()
        wp_paddle_l.normalize()
        wp_paddle_r.normalize()

        # 5. Feed deviation into env field — "something here pushed
        #    the ball off its PMF constraint"
        deviation_mag = np.linalg.norm(ball_deviation)
        if not scored and deviation_mag > 1e-8:
            wp_env.update_lms(ball_pos, ball_deviation,
                              anomaly_scale=deviation_mag)
            wp_env.normalize()

        # 6. Reward field learns from goal events
        if scored:
            wp_reward.update_lms(
                np.array([ball['x'], ball['y']]),
                np.array([reward, reward]), anomaly_scale=1.0)
            wp_reward.normalize()

        # Decay reward field so old goals fade
        wp_reward.c_cos *= reward_decay
        wp_reward.c_sin *= reward_decay

        # 7. Update stored positions
        wp_ball.pos[:] = ball_pos
        wp_paddle_l.pos[:] = paddle_l_pos
        wp_paddle_r.pos[:] = paddle_r_pos

        # Deviation magnitude for display (replaces old anomaly)
        a_residual = ball_deviation

        state['anomaly'] = a_residual
        state['max_anomaly'] = max(np.linalg.norm(a_residual),
                                    state['max_anomaly'] * 0.98)

        anomaly_hist.append(np.linalg.norm(a_residual))
        history_t.append(t)
        state['t'] = t + 1

        # -- draw X-domain ------------------------------------------------
        lx_ball.set_data(x_dom, wp_ball.evaluate(x_dom, axis=0))
        lx_pad_l.set_data(x_dom, wp_paddle_l.evaluate(x_dom, axis=0))
        lx_pad_r.set_data(x_dom, wp_paddle_r.evaluate(x_dom, axis=0))
        lx_env.set_data(x_dom, wp_env.evaluate(x_dom, axis=0))
        lx_reward.set_data(x_dom, wp_reward.evaluate(x_dom, axis=0))
        lx_dot.set_data([ball['x']], [0])

        # -- draw Y-domain (x=field value, y=domain) ---------------------
        ly_ball.set_data(wp_ball.evaluate(y_dom, axis=1), y_dom)
        ly_pad_l.set_data(wp_paddle_l.evaluate(y_dom, axis=1), y_dom)
        ly_pad_r.set_data(wp_paddle_r.evaluate(y_dom, axis=1), y_dom)
        ly_env.set_data(wp_env.evaluate(y_dom, axis=1), y_dom)
        ly_reward.set_data(wp_reward.evaluate(y_dom, axis=1), y_dom)
        ly_dot.set_data([0], [ball['y']])

        # -- draw court ---------------------------------------------------
        ball_dot.set_data([ball['x']], [ball['y']])
        lp_patch.set_y(left_paddle['y'] - half_h)
        rp_patch.set_y(right_paddle['y'] - half_h)
        score_text.set_text(
            f'{left_paddle["score"]} \u2014 {right_paddle["score"]}')
        frame_text.set_text(f't={t}')

        # Anomaly arrow on court
        if anomaly_arrow[0] is not None:
            anomaly_arrow[0].remove()
            anomaly_arrow[0] = None
        ma = max(state['max_anomaly'], 0.01)
        a_norm = np.linalg.norm(a_residual)
        arrow_len = min(a_norm / ma * 1.5, 2.0)
        if arrow_len > 0.05 and a_norm > 0.01:
            direction = a_residual / a_norm
            anomaly_arrow[0] = ax_court.annotate(
                '', xy=(ball['x'] + direction[0] * arrow_len,
                        ball['y'] + direction[1] * arrow_len),
                xytext=(ball['x'], ball['y']),
                arrowprops=dict(arrowstyle='->', color=RESIDUAL_COLOR,
                                lw=2.5, mutation_scale=15), zorder=6)

        return ()

    frames = range(max_frames) if max_frames is not None else itertools.count()
    anim = FuncAnimation(fig, step, frames=frames, init_func=init,
                         interval=interval, blit=False, repeat=False)
    return fig, anim


# -- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='2D Spectral Pong Visualization')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g. out.gif)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Frame limit (required for --save)')
    parser.add_argument('--K', type=int, default=8,
                        help='Number of spectral components (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='Force coupling strength (default: 0.15)')
    parser.add_argument('--lr', type=float, default=0.15,
                        help='LMS learning rate (default: 0.15)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Ball speed multiplier (default: 1.0)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('--auto-paddle', action='store_true',
                        help='AI-controlled paddles (required for --save)')
    parser.add_argument('--spectral-paddle', action='store_true',
                        help='Use spectral inner-product force for paddle AI '
                             '(with --auto-paddle)')
    parser.add_argument('--alpha-paddle', type=float, default=0.5,
                        help='Paddle-ball spectral coupling strength '
                             '(default: 0.5)')
    parser.add_argument('--paddle-damping', type=float, default=0.85,
                        help='Paddle velocity damping per frame '
                             '(default: 0.85)')
    parser.add_argument('--beta-paddle', type=float, default=0.3,
                        help='Wall-anticipation coupling strength '
                             '(default: 0.3)')
    parser.add_argument('--gamma-paddle', type=float, default=0.2,
                        help='Reward-gradient coupling strength '
                             '(default: 0.2)')
    parser.add_argument('--reward-lr', type=float, default=0.3,
                        help='Reward field LMS learning rate '
                             '(default: 0.3)')
    parser.add_argument('--reward-decay', type=float, default=0.999,
                        help='Per-frame reward coefficient decay '
                             '(default: 0.999)')
    parser.add_argument('--ball-lr', type=float, default=0.15,
                        help='Ball interaction LMS learning rate '
                             '(default: 0.15)')
    parser.add_argument('--paddle-lr', type=float, default=0.1,
                        help='Paddle interaction LMS learning rate '
                             '(default: 0.1)')
    parser.add_argument('--lr-tracking', type=float, default=0.01,
                        help='Always-on tracking LMS learning rate '
                             '(default: 0.01)')
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

    interval = int(1000 / args.fps)
    dt = interval / 1000.0

    fig, anim = create_game(
        K=K,
        frequencies=frequencies,
        alpha=args.alpha,
        dt=dt,
        ball_speed=BALL_SPEED * args.speed,
        interval=interval,
        max_frames=args.frames,
        auto_paddle=args.auto_paddle,
        ball_sigma=0.8,
        env_lr=args.lr,
        spectral_paddle=args.spectral_paddle,
        alpha_paddle=args.alpha_paddle,
        paddle_damping=args.paddle_damping,
        beta_paddle=args.beta_paddle,
        gamma_paddle=args.gamma_paddle,
        reward_lr=args.reward_lr,
        reward_decay=args.reward_decay,
        ball_lr=args.ball_lr,
        paddle_lr=args.paddle_lr,
        lr_tracking=args.lr_tracking,
    )

    if args.save:
        save_path = Path(args.save)
        if save_path.suffix == '.gif':
            anim.save(str(save_path), writer='pillow', fps=args.fps)
        else:
            anim.save(str(save_path), fps=args.fps)
        print(f'Saved to {save_path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
