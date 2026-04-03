"""Pretraining experiment: verify the predict→correct→learn loop reduces residuals.

Runs the pong sim with varying paddle behaviors (static, random, tracking, RL)
and measures whether physics prediction residuals and reward prediction residuals
decrease over time as the encoder learns.

No RL training — this is pure encoder pretraining to validate that:
1. The env field learns wall structure (physics prediction improves)
2. The reward field learns goal locations (reward prediction improves)
3. Spatial frequency adaptation (lr_k) helps both

Usage:
    python training/pretrain_encoder.py                  # full experiment
    python training/pretrain_encoder.py --agent static   # single agent type
    python training/pretrain_encoder.py --frames 3000    # short run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure imports work from training/ directory
sys.path.insert(0, str(Path(__file__).parent))

from spectral_pong_viz import (
    WavepacketObject2D,
    COURT_LEFT, COURT_RIGHT, COURT_TOP, COURT_BOTTOM,
    PADDLE_X_OFFSET, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED,
    BALL_RADIUS, BALL_SPEED, SPIN_FACTOR,
    COEFF_CLIP, WORLD_BOUNDS, reset_ball,
)

# ---------------------------------------------------------------------------
# Paddle agent behaviours
# ---------------------------------------------------------------------------

def agent_static(ball: dict, paddle_y: float, dt: float, side: str) -> float:
    """Paddle stays at center."""
    return 0.0


def agent_random(ball: dict, paddle_y: float, dt: float, side: str) -> float:
    """Random movement each frame."""
    return np.random.uniform(-1, 1)


def agent_tracking(ball: dict, paddle_y: float, dt: float, side: str) -> float:
    """Perfect tracking: move toward ball y."""
    diff = ball['y'] - paddle_y
    return float(np.clip(diff / (PADDLE_SPEED * dt), -1, 1))


def agent_noisy_tracking(ball: dict, paddle_y: float, dt: float,
                         side: str) -> float:
    """Tracking with noise — mimics a mediocre player."""
    diff = ball['y'] - paddle_y
    signal = float(np.clip(diff / (PADDLE_SPEED * dt), -1, 1))
    return float(np.clip(signal + np.random.normal(0, 0.3), -1, 1))


AGENTS = {
    'static': agent_static,
    'random': agent_random,
    'tracking': agent_tracking,
    'noisy': agent_noisy_tracking,
}

# ---------------------------------------------------------------------------
# Main pretraining loop
# ---------------------------------------------------------------------------

def run_pretrain(n_frames: int = 6000,
                 agent_type: str = 'tracking',
                 force_scale: float = 0.5,
                 lr_k: float = 0.001,
                 seed: int = 0,
                 window: int = 200,
                 reward_replay_n: int = 5) -> dict:
    """Run pong with specified paddle agent, track residuals.

    All wavepackets are 3D (x, y, reward).  The reward field has basis
    [0,0,1] and only acts on dim 2.  The env field has basis [1,0,0]
    and [0,1,0] — physics only.  On goals, paddles are shown as
    [x, y, ±1] and the reward field gets LMS at the ball's goal position.

    Returns dict with:
        physics_residuals: per-frame ||predicted_pos - observed_pos|| (dims 0,1)
        reward_residuals:  per-frame |predicted_reward_dim - actual| (dim 2)
        goal_frames:       frame indices where goals occurred
        freq_history:      snapshots of ball.k at intervals
        env_energy:        snapshots of env field energy at intervals
    """
    np.random.seed(seed)
    K = 8
    NDIM = 3
    freqs = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    dt = 1 / 60
    ball_speed = BALL_SPEED

    agent_fn = AGENTS[agent_type]

    # Wavepackets (3D: x, y, reward)
    wp_ball = WavepacketObject2D(K, freqs, pos0=(0, 0, 0), sigma=0.8,
                                 ndim=NDIM, lr=0.15, lr_tracking=0.01)
    wp_pl = WavepacketObject2D(K, freqs,
                               pos0=(COURT_LEFT + PADDLE_X_OFFSET, 0, 0),
                               mass=1e6, sigma=0.5, amplitude=1.0,
                               ndim=NDIM, lr=0.1, lr_tracking=0.02)
    wp_pr = WavepacketObject2D(K, freqs,
                               pos0=(COURT_RIGHT - PADDLE_X_OFFSET, 0, 0),
                               mass=1e6, sigma=0.5, amplitude=1.0,
                               ndim=NDIM, lr=0.1, lr_tracking=0.02)
    # Env field: basis [1,0,0] and [0,1,0]
    env_c = np.zeros((K, NDIM))
    env_c[0, 0] = 1.0
    env_c[1, 1] = 1.0
    wp_env = WavepacketObject2D(K, freqs, pos0=(0, 0, 0), mass=1e6,
                                ndim=NDIM,
                                c_cos=env_c, c_sin=np.zeros((K, NDIM)),
                                lr=0.15, lr_tracking=0.0)
    # Per-agent reward fields: basis [0,0,1]
    rew_c = np.zeros((K, NDIM))
    rew_c[0, 2] = 1.0
    wp_reward_l = WavepacketObject2D(K, freqs, pos0=(0, 0, 0), mass=1e6,
                                      ndim=NDIM,
                                      c_cos=rew_c.copy(),
                                      c_sin=np.zeros((K, NDIM)),
                                      lr=0.15, lr_tracking=0.0)
    wp_reward_r = WavepacketObject2D(K, freqs, pos0=(0, 0, 0), mass=1e6,
                                      ndim=NDIM,
                                      c_cos=rew_c.copy(),
                                      c_sin=np.zeros((K, NDIM)),
                                      lr=0.15, lr_tracking=0.0)

    # Game state
    ball = {'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0}
    reset_ball(ball, toward='right', speed=ball_speed)
    left_paddle = {'y': 0.0}
    right_paddle = {'y': 0.0}
    paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
    paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET
    half_h = PADDLE_HEIGHT / 2
    freeze = 0

    # Tracking arrays
    physics_residuals = []
    reward_residuals = []
    goal_frames = []
    freq_history = []
    env_energy_history = []

    for t in range(n_frames):
        if freeze > 0:
            freeze -= 1
            physics_residuals.append(0.0)
            reward_residuals.append(0.0)
            continue

        # -- Paddle movement --
        act_l = agent_fn(ball, left_paddle['y'], dt, 'left')
        act_r = agent_fn(ball, right_paddle['y'], dt, 'right')
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
            # Replay reward frame N times through predict→correct→learn
            goal_pos_rew = np.array([ball['x'], ball['y'], reward])
            goal_vel = np.array([0.0, 0.0, 0.0])
            for _ in range(reward_replay_n):
                nip_rew_l = abs(wp_ball.normalized_inner_product(wp_reward_l))
                nip_env_g = abs(wp_ball.normalized_inner_product(wp_env))
                rew_force = wp_ball.predict_force(
                    wp_reward_l, nip_rew_l, force_scale=force_scale)
                env_force = wp_ball.predict_force(
                    wp_env, nip_env_g, force_scale=force_scale)
                pred_rew = wp_ball.predict_position(
                    goal_vel, dt, env_force + rew_force)
                wp_ball.update_with_attention(
                    goal_pos_rew, np.ones(NDIM),
                    [nip_env_g, nip_rew_l])
                wp_reward_l.update_lms(
                    goal_pos_rew, np.array([0.0, 0.0, reward]),
                    anomaly_scale=1.0)
                wp_reward_r.update_lms(
                    goal_pos_rew, np.array([0.0, 0.0, -reward]),
                    anomaly_scale=1.0)
                wp_reward_l.soft_normalize(max_energy=2.0)
                wp_reward_r.soft_normalize(max_energy=2.0)
                wp_ball.normalize()
                if lr_k > 0:
                    wp_ball.learn_from_residual(pred_rew, goal_pos_rew,
                                                lr_k=lr_k)
                wp_ball.pos[:] = goal_pos_rew
            # Show paddles with signed reward dim
            wp_pl.update_with_attention(
                np.array([paddle_lx, left_paddle['y'], reward]),
                np.ones(NDIM), [1.0])
            wp_pr.update_with_attention(
                np.array([paddle_rx, right_paddle['y'], -reward]),
                np.ones(NDIM), [1.0])

            reset_ball(ball, toward='left' if reward < 0 else 'right',
                       speed=ball_speed)
            freeze = int(0.5 / dt)
            wp_ball.__init__(K, freqs, pos0=(ball['x'], ball['y'], 0.0),
                             mass=1.0, sigma=0.8, ndim=NDIM,
                             lr=0.15, lr_tracking=0.01)

        # -- Wavepacket updates: predict → correct → learn --
        ball_pos = np.array([ball['x'], ball['y'], 0.0])
        paddle_l_pos = np.array([paddle_lx, left_paddle['y'], 0.0])
        paddle_r_pos = np.array([paddle_rx, right_paddle['y'], 0.0])

        # 1. PREDICT: env + reward field forces
        if not scored:
            nip_env = abs(wp_ball.normalized_inner_product(wp_env))
            nip_padL = abs(wp_ball.normalized_inner_product(wp_pl))
            nip_padR = abs(wp_ball.normalized_inner_product(wp_pr))
            nip_rew = abs(wp_ball.normalized_inner_product(wp_reward_l))

            env_force = wp_ball.predict_force(wp_env, nip_env,
                                              force_scale=force_scale)
            rew_force = wp_ball.predict_force(wp_reward_l, nip_rew,
                                              force_scale=force_scale)
            total_force = env_force + rew_force
            ball_vel = np.array([ball['vx'], ball['vy'], 0.0])
            predicted_pos = wp_ball.predict_position(ball_vel, dt, total_force)
        else:
            nip_env = nip_padL = nip_padR = nip_rew = 0.0
            predicted_pos = ball_pos.copy()

        # Shift ball by velocity (dims 0,1)
        wp_ball.shift(ball['vx'] * dt, axis=0)
        wp_ball.shift(ball['vy'] * dt, axis=1)
        delta_l = left_paddle['y'] - wp_pl.pos[1]
        delta_r = right_paddle['y'] - wp_pr.pos[1]
        if abs(delta_l) > 1e-12:
            wp_pl.shift(delta_l, axis=1)
        if abs(delta_r) > 1e-12:
            wp_pr.shift(delta_r, axis=1)

        # 2. CORRECT: LMS toward observed [x, y, 0]
        if not scored:
            unity = np.ones(NDIM)
            wp_ball.update_with_attention(ball_pos, unity,
                                          [nip_env, nip_padL, nip_padR])
            wp_pl.update_with_attention(paddle_l_pos, unity, [nip_padL])
            wp_pr.update_with_attention(paddle_r_pos, unity, [nip_padR])

        # 3. Residuals (physics = dims 0,1; reward = dim 2)
        full_residual = ball_pos - predicted_pos
        phys_res = np.linalg.norm(full_residual[:2])
        rew_res = abs(full_residual[2])
        physics_residuals.append(float(phys_res))
        reward_residuals.append(float(rew_res))

        # 4. Deviation + normalize
        ball_dev = np.array([wp_ball.integrate_squared(d) - 1.0
                             for d in range(NDIM)])
        wp_ball.normalize()
        wp_pl.normalize()
        wp_pr.normalize()

        # 5. Env learns from deviation (physics dims only)
        dev_mag = np.linalg.norm(ball_dev[:2])
        if not scored and dev_mag > 1e-8:
            total_nip = nip_env + nip_padL + nip_padR + 1e-8
            env_frac = nip_env / total_nip
            wp_env.update_lms(ball_pos, ball_dev,
                              anomaly_scale=dev_mag * env_frac)
            wp_env.normalize()

        # 6. LEARN: update frequencies from 3D prediction error
        if not scored and lr_k > 0:
            wp_ball.learn_from_residual(predicted_pos, ball_pos, lr_k=lr_k)

        # Store positions
        wp_ball.pos[:] = ball_pos
        wp_pl.pos[:] = paddle_l_pos
        wp_pr.pos[:] = paddle_r_pos

        # Periodic snapshots
        if t % 500 == 0:
            freq_history.append((t, wp_ball.k.copy()))
            env_e = sum(wp_env.integrate_squared(d) for d in range(2))
            rew_e = wp_reward_l.integrate_squared(2)
            env_energy_history.append((t, env_e, rew_e))

    return {
        'physics_residuals': physics_residuals,
        'reward_residuals': reward_residuals,
        'goal_frames': goal_frames,
        'freq_history': freq_history,
        'env_energy': env_energy_history,
        'agent': agent_type,
        'force_scale': force_scale,
        'lr_k': lr_k,
        'seed': seed,
    }


def windowed_mean(arr: list[float], window: int) -> list[float]:
    """Non-overlapping window averages."""
    out = []
    for i in range(0, len(arr), window):
        chunk = arr[i:i + window]
        if chunk:
            out.append(sum(chunk) / len(chunk))
    return out


def print_report(result: dict, window: int = 500) -> None:
    """Print human-readable report of residual evolution."""
    agent = result['agent']
    fs = result['force_scale']
    lk = result['lr_k']
    seed = result['seed']
    n_goals = len(result['goal_frames'])

    phys_win = windowed_mean(result['physics_residuals'], window)
    rew_win = windowed_mean(result['reward_residuals'], window)

    print(f"\n{'='*65}")
    print(f"  Agent: {agent:12s}  force_scale={fs:.2f}  lr_k={lk:.4f}  seed={seed}")
    print(f"  Goals: {n_goals}  ({len(result['physics_residuals'])} frames)")
    print(f"{'='*65}")

    print(f"\n  Physics residual (||predicted - observed||), {window}-frame windows:")
    for i, v in enumerate(phys_win):
        bar = '#' * min(int(v * 200), 50)
        print(f"    [{i*window:5d}-{(i+1)*window:5d}]  {v:.6f}  {bar}")

    if phys_win and len(phys_win) >= 2:
        first = phys_win[0]
        last = phys_win[-1]
        if first > 1e-8:
            pct = (last - first) / first * 100
            direction = "DECREASED" if pct < -5 else "INCREASED" if pct > 5 else "STABLE"
            print(f"    → {direction}: {first:.6f} → {last:.6f} ({pct:+.1f}%)")

    print(f"\n  Reward residual (|predicted - actual|), {window}-frame windows:")
    for i, v in enumerate(rew_win):
        bar = '#' * min(int(v * 100), 50)
        print(f"    [{i*window:5d}-{(i+1)*window:5d}]  {v:.6f}  {bar}")

    if rew_win and len(rew_win) >= 2:
        first = rew_win[0]
        last = rew_win[-1]
        if first > 1e-8:
            pct = (last - first) / first * 100
            direction = "DECREASED" if pct < -5 else "INCREASED" if pct > 5 else "STABLE"
            print(f"    → {direction}: {first:.6f} → {last:.6f} ({pct:+.1f}%)")

    print(f"\n  Frequency evolution (ball.k):")
    for t, k in result['freq_history']:
        k_str = ' '.join(f'{v:.3f}' for v in k)
        print(f"    t={t:5d}: [{k_str}]")

    print(f"\n  Env / Reward field energy:")
    for entry in result['env_energy']:
        if len(entry) == 3:
            t, env_e, rew_e = entry
            bar_e = '#' * min(int(env_e * 5), 30)
            bar_r = '#' * min(int(rew_e * 5), 30)
            print(f"    t={t:5d}: env={env_e:.4f} {bar_e}  rew={rew_e:.4f} {bar_r}")
        else:
            t, env_e = entry
            bar = '#' * min(int(env_e * 5), 40)
            print(f"    t={t:5d}: {env_e:.4f}  {bar}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Encoder pretraining experiment')
    ap.add_argument('--frames', type=int, default=6000)
    ap.add_argument('--agent', type=str, default=None,
                    choices=list(AGENTS.keys()),
                    help='Single agent type (default: run all)')
    ap.add_argument('--force-scale', type=float, default=None,
                    help='Single force_scale (default: sweep)')
    ap.add_argument('--lr-k', type=float, default=None,
                    help='Single lr_k (default: sweep)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--window', type=int, default=500)
    args = ap.parse_args()

    agents = [args.agent] if args.agent else ['static', 'random', 'noisy', 'tracking']

    if args.force_scale is not None and args.lr_k is not None:
        configs = [(args.force_scale, args.lr_k)]
    else:
        configs = [
            (0.0, 0.0),      # baseline: no prediction, no freq learning
            (0.5, 0.0),      # prediction only, no freq learning
            (0.5, 0.001),    # prediction + freq learning
            (0.5, 0.01),     # stronger freq learning
        ]

    print(f"Running pretrain experiment: {len(agents)} agents × "
          f"{len(configs)} configs × {args.seeds} seeds × {args.frames} frames")

    all_results = []
    for agent in agents:
        for fs, lk in configs:
            seed_phys_first = []
            seed_phys_last = []
            seed_rew_first = []
            seed_rew_last = []
            for s in range(args.seeds):
                result = run_pretrain(
                    n_frames=args.frames,
                    agent_type=agent,
                    force_scale=fs,
                    lr_k=lk,
                    seed=s,
                    window=args.window,
                )
                all_results.append(result)
                if args.seeds <= 3:
                    print_report(result, window=args.window)

                # Collect first/last window for summary
                phys_win = windowed_mean(result['physics_residuals'], args.window)
                rew_win = windowed_mean(result['reward_residuals'], args.window)
                if len(phys_win) >= 2:
                    seed_phys_first.append(phys_win[0])
                    seed_phys_last.append(phys_win[-1])
                if len(rew_win) >= 2:
                    seed_rew_first.append(rew_win[0])
                    seed_rew_last.append(rew_win[-1])

            # Summary across seeds
            if seed_phys_first:
                pf = np.mean(seed_phys_first)
                pl = np.mean(seed_phys_last)
                rf = np.mean(seed_rew_first)
                rl = np.mean(seed_rew_last)
                p_pct = (pl - pf) / max(pf, 1e-8) * 100
                r_pct = (rl - rf) / max(rf, 1e-8) * 100
                print(f"\n  SUMMARY [{agent:8s} fs={fs:.1f} lk={lk:.4f}] "
                      f"({args.seeds} seeds):")
                print(f"    Physics: {pf:.6f} → {pl:.6f} ({p_pct:+.1f}%)")
                print(f"    Reward:  {rf:.6f} → {rl:.6f} ({r_pct:+.1f}%)")


if __name__ == '__main__':
    main()
