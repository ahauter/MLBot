"""Headless tuning runs for spectral pong RL.

Runs the game loop without rendering to measure whether goal intervals
grow over time (= paddles are learning to block).

Usage:
    python training/spectral_pong_tune.py                      # default sweep
    python training/spectral_pong_tune.py --lr 0.01 --std 0.3  # single run
"""
from __future__ import annotations

import argparse
import numpy as np
from spectral_pong_viz import (
    WavepacketObject2D, SimpleRLController,
    COURT_LEFT, COURT_RIGHT, COURT_TOP, COURT_BOTTOM,
    PADDLE_X_OFFSET, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED,
    BALL_RADIUS, BALL_SPEED, SPIN_FACTOR,
    COEFF_CLIP, WORLD_BOUNDS, reset_ball,
)


def run_headless(n_frames: int = 6000, lr_actor: float = 3e-3,
                 lr_critic: float = 1e-1, gamma: float = 0.95,
                 lam: float = 0.9, std: float = 0.3, seed: int = 0,
                 verbose: bool = True) -> list[int]:
    """Run pong with RL paddles, return list of goal-frame indices."""
    np.random.seed(seed)
    K = 8
    freqs = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    dt = 1 / 60
    ball_speed = BALL_SPEED

    # Wavepackets
    wp_ball = WavepacketObject2D(K, freqs, pos0=(0, 0), sigma=0.8,
                                  lr=0.15, lr_tracking=0.01)
    wp_pl = WavepacketObject2D(K, freqs, pos0=(COURT_LEFT + PADDLE_X_OFFSET, 0),
                                mass=1e6, sigma=0.5, amplitude=1.0,
                                lr=0.1, lr_tracking=0.02)
    wp_pr = WavepacketObject2D(K, freqs, pos0=(COURT_RIGHT - PADDLE_X_OFFSET, 0),
                                mass=1e6, sigma=0.5, amplitude=1.0,
                                lr=0.1, lr_tracking=0.02)
    env_c = np.zeros((K, 2))
    env_c[0, 0] = 1.0
    env_c[1, 1] = 1.0
    wp_env = WavepacketObject2D(K, freqs, pos0=(0, 0), mass=1e6,
                                 c_cos=env_c, c_sin=np.zeros((K, 2)),
                                 lr=0.15, lr_tracking=0.0)

    # RL controllers (actor-critic with TD(λ))
    rl_left = SimpleRLController(SimpleRLController.STATE_DIM,
                                  lr_actor=lr_actor, lr_critic=lr_critic,
                                  gamma=gamma, lam=lam, std=std)
    rl_right = SimpleRLController(SimpleRLController.STATE_DIM,
                                   lr_actor=lr_actor, lr_critic=lr_critic,
                                   gamma=gamma, lam=lam, std=std)

    # Game state
    ball = {'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0}
    reset_ball(ball, toward='right', speed=ball_speed)
    left_paddle = {'y': 0.0, 'score': 0}
    right_paddle = {'y': 0.0, 'score': 0}
    paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
    paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET
    half_h = PADDLE_HEIGHT / 2
    freeze = 0

    goal_frames = []

    for t in range(n_frames):
        if freeze > 0:
            freeze -= 1
            continue

        # -- RL paddle movement --
        rp_l = rl_left.reward_predict(ball['x'])
        rp_r = rl_right.reward_predict(ball['x'])
        rl_state_l = SimpleRLController.build_state(
            wp_ball, wp_pl, wp_pr, wp_env, rp_l)
        rl_state_r = SimpleRLController.build_state(
            wp_ball, wp_pl, wp_pr, wp_env, rp_r)
        act_l = rl_left.act(rl_state_l)
        act_r = rl_right.act(rl_state_r)
        left_paddle['y'] += act_l * PADDLE_SPEED * dt
        right_paddle['y'] += act_r * PADDLE_SPEED * dt
        left_paddle['y'] = np.clip(left_paddle['y'],
                                    COURT_BOTTOM + half_h, COURT_TOP - half_h)
        right_paddle['y'] = np.clip(right_paddle['y'],
                                     COURT_BOTTOM + half_h, COURT_TOP - half_h)

        # -- Ball physics --
        ball['x'] += ball['vx'] * dt
        ball['y'] += ball['vy'] * dt

        # Wall bounce
        if ball['y'] >= COURT_TOP - BALL_RADIUS:
            ball['y'] = 2 * (COURT_TOP - BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']
        elif ball['y'] <= COURT_BOTTOM + BALL_RADIUS:
            ball['y'] = 2 * (COURT_BOTTOM + BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']

        # Paddle collisions
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

        # Scoring
        scored = False
        reward = 0.0
        if ball['x'] < COURT_LEFT:
            reward = -1.0
            scored = True
        elif ball['x'] > COURT_RIGHT:
            reward = +1.0
            scored = True

        if scored:
            goal_frames.append(t)
            # Update reward wavepackets
            rl_left.reward_update(ball['x'], reward)
            rl_right.reward_update(ball['x'], -reward)
            # Terminal TD step: V(terminal)=0, so δ = reward - V(s)
            term_state = np.zeros(SimpleRLController.STATE_DIM)
            rl_left.step(term_state, reward)
            rl_right.step(term_state, -reward)
            rl_left.on_reset()
            rl_right.on_reset()
            reset_ball(ball, toward='left' if reward < 0 else 'right',
                       speed=ball_speed)
            freeze = int(0.5 / dt)
            wp_ball.__init__(K, freqs, pos0=(ball['x'], ball['y']),
                             mass=1.0, sigma=0.8,
                             lr=0.15, lr_tracking=0.01)

        # -- Wavepacket updates --
        ball_pos = np.array([ball['x'], ball['y']])
        paddle_l_pos = np.array([paddle_lx, left_paddle['y']])
        paddle_r_pos = np.array([paddle_rx, right_paddle['y']])

        for d in range(2):
            wp_ball.shift([ball['vx'], ball['vy']][d] * dt, axis=d)
        delta_l = left_paddle['y'] - wp_pl.pos[1]
        delta_r = right_paddle['y'] - wp_pr.pos[1]
        if abs(delta_l) > 1e-12:
            wp_pl.shift(delta_l, axis=1)
        if abs(delta_r) > 1e-12:
            wp_pr.shift(delta_r, axis=1)

        if not scored:
            nip_env = abs(wp_ball.normalized_inner_product(wp_env))
            nip_padL = abs(wp_ball.normalized_inner_product(wp_pl))
            nip_padR = abs(wp_ball.normalized_inner_product(wp_pr))

            unity = np.array([1.0, 1.0])
            wp_ball.update_with_attention(ball_pos, unity,
                                          [nip_env, nip_padL, nip_padR])
            wp_pl.update_with_attention(paddle_l_pos, unity, [nip_padL])
            wp_pr.update_with_attention(paddle_r_pos, unity, [nip_padR])

        ball_dev = np.array([wp_ball.integrate_squared(d) - 1.0
                             for d in range(2)])
        wp_ball.normalize()
        wp_pl.normalize()
        wp_pr.normalize()

        dev_mag = np.linalg.norm(ball_dev)
        if not scored and dev_mag > 1e-8:
            total_nip = nip_env + nip_padL + nip_padR + 1e-8
            env_frac = nip_env / total_nip
            wp_env.update_lms(ball_pos, ball_dev,
                              anomaly_scale=dev_mag * env_frac)
            wp_env.normalize()

        wp_ball.pos[:] = ball_pos
        wp_pl.pos[:] = paddle_l_pos
        wp_pr.pos[:] = paddle_r_pos

        # TD update AFTER wavepacket updates
        if not scored:
            rp_l = rl_left.reward_predict(ball['x'])
            rp_r = rl_right.reward_predict(ball['x'])
            ns_l = SimpleRLController.build_state(
                wp_ball, wp_pl, wp_pr, wp_env, rp_l)
            ns_r = SimpleRLController.build_state(
                wp_ball, wp_pl, wp_pr, wp_env, rp_r)
            rl_left.step(ns_l, reward)
            rl_right.step(ns_r, -reward)

    return goal_frames


def analyse(goal_frames: list[int], n_frames: int,
            window: int = 5, label: str = '') -> dict:
    """Compute goal interval stats in rolling windows."""
    if len(goal_frames) < 2:
        print(f'{label}: only {len(goal_frames)} goals in {n_frames} frames')
        return {}
    intervals = np.diff(goal_frames)
    n_goals = len(goal_frames)

    # Split into thirds
    third = max(len(intervals) // 3, 1)
    early = intervals[:third]
    mid = intervals[third:2*third]
    late = intervals[2*third:]

    result = {
        'n_goals': n_goals,
        'mean_interval': float(intervals.mean()),
        'early_mean': float(early.mean()),
        'mid_mean': float(mid.mean()),
        'late_mean': float(late.mean()),
        'improvement': float(late.mean() / (early.mean() + 1e-8)),
    }

    if label:
        print(f'\n{label}')
        print(f'  Goals: {n_goals} in {n_frames} frames')
        print(f'  Mean interval: {result["mean_interval"]:.0f} frames')
        print(f'  Early third:   {result["early_mean"]:.0f} frames')
        print(f'  Mid third:     {result["mid_mean"]:.0f} frames')
        print(f'  Late third:    {result["late_mean"]:.0f} frames')
        ratio = result['improvement']
        arrow = '↑' if ratio > 1.1 else ('↓' if ratio < 0.9 else '→')
        print(f'  Improvement:   {ratio:.2f}x {arrow}')

    return result


def main():
    parser = argparse.ArgumentParser(description='Headless RL tuning')
    parser.add_argument('--frames', type=int, default=6000,
                        help='Frames per run (default: 6000 = ~100s at 60fps)')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Seeds per config (default: 3)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Single LR to test (default: sweep)')
    parser.add_argument('--std', type=float, default=None,
                        help='Single exploration std (default: sweep)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Single gamma (default: sweep)')
    args = parser.parse_args()

    if args.lr is not None:
        # Single run
        configs = [{'lr_actor': args.lr,
                     'lr_critic': args.lr * 10,
                     'std': args.std or 0.3,
                     'gamma': args.gamma or 0.95}]
    else:
        # Sweep: TD(λ) actor-critic with fixed λ=0.9
        configs = [
            {'lr_actor': 3e-3, 'lr_critic': 3e-2, 'std': 0.3, 'gamma': 0.95},
            {'lr_actor': 1e-2, 'lr_critic': 1e-1, 'std': 0.3, 'gamma': 0.95},
            {'lr_actor': 3e-2, 'lr_critic': 3e-1, 'std': 0.3, 'gamma': 0.95},
            {'lr_actor': 1e-2, 'lr_critic': 1e-1, 'std': 0.1, 'gamma': 0.95},
            {'lr_actor': 1e-2, 'lr_critic': 1e-1, 'std': 0.5, 'gamma': 0.95},
            {'lr_actor': 1e-2, 'lr_critic': 1e-1, 'std': 0.3, 'gamma': 0.99},
        ]

    print(f'Running {len(configs)} configs × {args.seeds} seeds × '
          f'{args.frames} frames')
    print('=' * 60)

    best = None
    for cfg in configs:
        improvements = []
        for seed in range(args.seeds):
            goals = run_headless(n_frames=args.frames, seed=seed, **cfg,
                                 verbose=False)
            label = (f'lr_a={cfg["lr_actor"]:.0e} lr_c={cfg["lr_critic"]:.0e} '
                     f'std={cfg["std"]:.1f} γ={cfg["gamma"]:.2f} seed={seed}')
            result = analyse(goals, args.frames, label=label)
            if result:
                improvements.append(result['improvement'])

        if improvements:
            mean_imp = np.mean(improvements)
            tag = (f'lr_a={cfg["lr_actor"]:.0e} lr_c={cfg["lr_critic"]:.0e} '
                   f'std={cfg["std"]:.1f} γ={cfg["gamma"]:.2f}')
            print(f'\n  >> {tag}: mean improvement = {mean_imp:.2f}x '
                  f'({len(improvements)} seeds)')
            if best is None or mean_imp > best[1]:
                best = (tag, mean_imp)

    if best:
        print(f'\n{"=" * 60}')
        print(f'Best config: {best[0]} → {best[1]:.2f}x improvement')


if __name__ == '__main__':
    main()
