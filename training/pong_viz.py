"""
Newtonian Pong — matplotlib
============================
Simple 2D pong game with Newtonian ball physics.
Baseline for future spectral-field pong variant.

Controls:
    Left paddle:  W / S
    Right paddle:  Up / Down

Usage:
    python training/pong_viz.py                              # interactive
    python training/pong_viz.py --save out.gif --frames 600  # save GIF
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation

# ── constants ────────────────────────────────────────────────────────────────

BG_COLOR = '#12121f'
BALL_COLOR = '#38bdf8'
PADDLE_COLOR_L = '#4ade80'
PADDLE_COLOR_R = '#f87171'
WALL_COLOR = '#333'
NET_COLOR = '#555'
SCORE_COLOR = 'white'

COURT_LEFT = -5.0
COURT_RIGHT = 5.0
COURT_TOP = 3.0
COURT_BOTTOM = -3.0

PADDLE_X_OFFSET = 0.5      # distance from court edge
PADDLE_WIDTH = 0.15
PADDLE_HEIGHT = 1.0
PADDLE_SPEED = 4.0

BALL_RADIUS = 0.15
BALL_SPEED = 3.0
SPIN_FACTOR = 2.0           # how much paddle-hit offset affects vy


# ── helpers ──────────────────────────────────────────────────────────────────

def reset_ball(ball: dict, toward: str = 'random', speed: float = BALL_SPEED):
    """Reset ball to center with velocity toward the given side."""
    ball['x'] = 0.0
    ball['y'] = 0.0
    if toward == 'random':
        toward = 'left' if np.random.random() < 0.5 else 'right'
    angle = np.random.uniform(-0.4, 0.4)  # slight vertical spread
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
    ball['vx'] = -vx if toward == 'left' else vx
    ball['vy'] = vy


# ── game ─────────────────────────────────────────────────────────────────────

def create_game(dt: float, interval: int, max_frames: int | None,
                ball_speed: float):
    paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
    paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET

    # ── state ────────────────────────────────────────────────
    ball = {'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0}
    reset_ball(ball, toward='right', speed=ball_speed)

    left_paddle = {'y': 0.0, 'score': 0}
    right_paddle = {'y': 0.0, 'score': 0}
    keys_held: set[str] = set()
    state = {'freeze': 0}

    # ── figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(COURT_LEFT - 0.5, COURT_RIGHT + 0.5)
    ax.set_ylim(COURT_BOTTOM - 1.0, COURT_TOP + 1.0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Court border
    court = Rectangle((COURT_LEFT, COURT_BOTTOM),
                       COURT_RIGHT - COURT_LEFT,
                       COURT_TOP - COURT_BOTTOM,
                       linewidth=2, edgecolor=WALL_COLOR,
                       facecolor='none', zorder=1)
    ax.add_patch(court)

    # Center line
    for y_pos in np.linspace(COURT_BOTTOM + 0.15, COURT_TOP - 0.15, 15):
        ax.plot([0, 0], [y_pos, y_pos + 0.2], color=NET_COLOR,
                lw=1.5, alpha=0.5, zorder=1)

    # Paddles
    lp_patch = FancyBboxPatch(
        (paddle_lx - PADDLE_WIDTH / 2, -PADDLE_HEIGHT / 2),
        PADDLE_WIDTH, PADDLE_HEIGHT,
        boxstyle='round,pad=0.03',
        facecolor=PADDLE_COLOR_L, edgecolor='white',
        linewidth=1.5, zorder=3)
    rp_patch = FancyBboxPatch(
        (paddle_rx - PADDLE_WIDTH / 2, -PADDLE_HEIGHT / 2),
        PADDLE_WIDTH, PADDLE_HEIGHT,
        boxstyle='round,pad=0.03',
        facecolor=PADDLE_COLOR_R, edgecolor='white',
        linewidth=1.5, zorder=3)
    ax.add_patch(lp_patch)
    ax.add_patch(rp_patch)

    # Ball
    ball_dot, = ax.plot([0], [0], 'o', color=BALL_COLOR, markersize=10,
                         zorder=4, markeredgecolor='white',
                         markeredgewidth=1.0)

    # Score
    score_text = ax.text(0, COURT_TOP + 0.5, '0 — 0',
                          color=SCORE_COLOR, fontsize=24, fontweight='bold',
                          horizontalalignment='center',
                          verticalalignment='center', zorder=5)

    fig.tight_layout()

    # ── keyboard ─────────────────────────────────────────────
    def on_key_press(event):
        keys_held.add(event.key)

    def on_key_release(event):
        keys_held.discard(event.key)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    # ── animation ────────────────────────────────────────────

    def init():
        return ()

    def step(_frame):
        # Freeze frames after scoring
        if state['freeze'] > 0:
            state['freeze'] -= 1
            return ()

        # ── move paddles ─────────────────────────────────
        if 'w' in keys_held:
            left_paddle['y'] += PADDLE_SPEED * dt
        if 's' in keys_held:
            left_paddle['y'] -= PADDLE_SPEED * dt
        if 'up' in keys_held:
            right_paddle['y'] += PADDLE_SPEED * dt
        if 'down' in keys_held:
            right_paddle['y'] -= PADDLE_SPEED * dt

        # Clamp paddles
        half_h = PADDLE_HEIGHT / 2
        left_paddle['y'] = np.clip(left_paddle['y'],
                                    COURT_BOTTOM + half_h,
                                    COURT_TOP - half_h)
        right_paddle['y'] = np.clip(right_paddle['y'],
                                     COURT_BOTTOM + half_h,
                                     COURT_TOP - half_h)

        # ── move ball ────────────────────────────────────
        ball['x'] += ball['vx'] * dt
        ball['y'] += ball['vy'] * dt

        # Wall bounce (top/bottom)
        if ball['y'] >= COURT_TOP - BALL_RADIUS:
            ball['y'] = 2 * (COURT_TOP - BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']
        elif ball['y'] <= COURT_BOTTOM + BALL_RADIUS:
            ball['y'] = 2 * (COURT_BOTTOM + BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']

        # ── paddle collisions ────────────────────────────
        # Left paddle
        if (ball['vx'] < 0
                and ball['x'] - BALL_RADIUS <= paddle_lx + PADDLE_WIDTH / 2
                and ball['x'] - BALL_RADIUS >= paddle_lx - PADDLE_WIDTH / 2 - abs(ball['vx'] * dt)
                and abs(ball['y'] - left_paddle['y']) <= half_h + BALL_RADIUS):
            ball['x'] = paddle_lx + PADDLE_WIDTH / 2 + BALL_RADIUS
            ball['vx'] = -ball['vx']
            # Spin from hit offset
            offset = (ball['y'] - left_paddle['y']) / half_h
            ball['vy'] += offset * SPIN_FACTOR
            # Normalize speed
            spd = np.hypot(ball['vx'], ball['vy'])
            ball['vx'] *= ball_speed / spd
            ball['vy'] *= ball_speed / spd

        # Right paddle
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

        # ── scoring ──────────────────────────────────────
        scored = False
        if ball['x'] < COURT_LEFT:
            right_paddle['score'] += 1
            scored = True
            reset_ball(ball, toward='left', speed=ball_speed)
        elif ball['x'] > COURT_RIGHT:
            left_paddle['score'] += 1
            scored = True
            reset_ball(ball, toward='right', speed=ball_speed)

        if scored:
            state['freeze'] = int(0.5 / dt)  # half-second pause

        # ── draw ─────────────────────────────────────────
        ball_dot.set_data([ball['x']], [ball['y']])

        lp_patch.set_y(left_paddle['y'] - half_h)
        rp_patch.set_y(right_paddle['y'] - half_h)

        score_text.set_text(
            f'{left_paddle["score"]} — {right_paddle["score"]}')

        return ()

    frames = range(max_frames) if max_frames is not None else itertools.count()
    anim = FuncAnimation(fig, step, frames=frames, init_func=init,
                         interval=interval, blit=False, repeat=False)
    return fig, anim


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Newtonian Pong — matplotlib')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g. out.gif)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Frame limit (required for --save)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Ball speed multiplier (default: 1.0)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    args = parser.parse_args()

    if args.save and args.frames is None:
        parser.error('--frames is required when using --save')

    if args.save:
        matplotlib.use('Agg')

    interval = int(1000 / args.fps)
    dt = interval / 1000.0

    fig, anim = create_game(
        dt=dt,
        interval=interval,
        max_frames=args.frames,
        ball_speed=BALL_SPEED * args.speed,
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
