"""Long-run instability analysis for spectral pong RL.

Runs 50k-frame simulations with detailed per-epoch diagnostics,
comparing pre-fix (no clipping) vs post-fix (clipped) behavior.
Outputs two text reports.
"""
from __future__ import annotations

import numpy as np
from spectral_pong_viz import (
    WavepacketObject2D, SimpleRLController,
    COURT_LEFT, COURT_RIGHT, COURT_TOP, COURT_BOTTOM,
    PADDLE_X_OFFSET, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED,
    BALL_RADIUS, BALL_SPEED, SPIN_FACTOR,
    WORLD_BOUNDS, reset_ball,
    compute_feature_maps, FM_NX, FM_NY,
)


EPOCH_SIZE = 500


def run_with_diagnostics(n_frames: int, seed: int,
                         lr_actor: float = 1e-2,
                         lr_critic: float = 1e-1,
                         gamma: float = 0.95, lam: float = 0.9,
                         std: float = 0.3,
                         disable_clipping: bool = False) -> dict:
    """Full simulation with epoch-level diagnostics."""
    np.random.seed(seed)
    K = 8
    NDIM = 3
    freqs = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    dt = 1 / 60
    ball_speed = BALL_SPEED

    # Override clipping constants
    if disable_clipping:
        SimpleRLController.WEIGHT_CLIP = 1e30
        SimpleRLController.TD_CLIP = 1e30
    else:
        SimpleRLController.WEIGHT_CLIP = 5.0
        SimpleRLController.TD_CLIP = 1.0

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
    env_c = np.zeros((K, NDIM))
    env_c[0, 0] = 1.0
    env_c[1, 1] = 1.0
    wp_env = WavepacketObject2D(K, freqs, pos0=(0, 0, 0), mass=1e6,
                                 ndim=NDIM, c_cos=env_c,
                                 c_sin=np.zeros((K, NDIM)),
                                 lr=0.15, lr_tracking=0.0)
    rew_c = np.zeros((K, NDIM))
    rew_c[0, 2] = 1.0
    wp_reward_l = WavepacketObject2D(K, freqs, pos0=(0, 0, 0), mass=1e6,
                                      ndim=NDIM, c_cos=rew_c.copy(),
                                      c_sin=np.zeros((K, NDIM)),
                                      lr=0.15, lr_tracking=0.0)
    wp_reward_r = WavepacketObject2D(K, freqs, pos0=(0, 0, 0), mass=1e6,
                                      ndim=NDIM, c_cos=rew_c.copy(),
                                      c_sin=np.zeros((K, NDIM)),
                                      lr=0.15, lr_tracking=0.0)

    rl_left = SimpleRLController(SimpleRLController.STATE_DIM,
                                  lr_actor=lr_actor, lr_critic=lr_critic,
                                  gamma=gamma, lam=lam, std=std)
    rl_right = SimpleRLController(SimpleRLController.STATE_DIM,
                                   lr_actor=lr_actor, lr_critic=lr_critic,
                                   gamma=gamma, lam=lam, std=std)

    x_fm = np.linspace(COURT_LEFT, COURT_RIGHT, FM_NX)
    y_fm = np.linspace(COURT_BOTTOM, COURT_TOP, FM_NY)
    r_fm = np.linspace(-1.0, 1.0, FM_NY)

    ball = {'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0}
    reset_ball(ball, toward='right', speed=ball_speed)
    left_paddle = {'y': 0.0, 'score': 0}
    right_paddle = {'y': 0.0, 'score': 0}
    paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
    paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET
    half_h = PADDLE_HEIGHT / 2
    freeze = 0

    # Epoch-level diagnostics (every EPOCH_SIZE frames)
    EPOCH = EPOCH_SIZE
    epochs = []
    epoch_tracking = []  # per-frame |paddle_y - ball_y| within epoch
    epoch_td_errors = []
    epoch_goals = 0
    epoch_touches = 0
    goal_frames = []
    rally_lengths = []  # touches per episode
    current_touches = 0

    for t in range(n_frames):
        if freeze > 0:
            freeze -= 1
            continue

        # RL paddle movement
        fmaps_l = compute_feature_maps(
            wp_ball, wp_env, wp_pl, wp_pr, wp_reward_l, x_fm, y_fm, r_fm)
        fmaps_r = compute_feature_maps(
            wp_ball, wp_env, wp_pl, wp_pr, wp_reward_r, x_fm, y_fm, r_fm)
        rl_state_l = SimpleRLController.build_state(fmaps_l, rl_left.conv)
        rl_state_r = SimpleRLController.build_state(fmaps_r, rl_right.conv)
        act_l = rl_left.act(rl_state_l)
        act_r = rl_right.act(rl_state_r)
        left_paddle['y'] += act_l * PADDLE_SPEED * dt
        right_paddle['y'] += act_r * PADDLE_SPEED * dt
        left_paddle['y'] = np.clip(left_paddle['y'],
                                    COURT_BOTTOM + half_h, COURT_TOP - half_h)
        right_paddle['y'] = np.clip(right_paddle['y'],
                                     COURT_BOTTOM + half_h, COURT_TOP - half_h)

        epoch_tracking.append(abs(left_paddle['y'] - ball['y']))

        # Ball physics
        ball['x'] += ball['vx'] * dt
        ball['y'] += ball['vy'] * dt
        if ball['y'] >= COURT_TOP - BALL_RADIUS:
            ball['y'] = 2 * (COURT_TOP - BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']
        elif ball['y'] <= COURT_BOTTOM + BALL_RADIUS:
            ball['y'] = 2 * (COURT_BOTTOM + BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']

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
            current_touches += 1
            epoch_touches += 1

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
            current_touches += 1
            epoch_touches += 1

        scored = False
        reward = 0.0
        if ball['x'] < COURT_LEFT:
            reward = -1.0
            scored = True
        elif ball['x'] > COURT_RIGHT:
            reward = +1.0
            scored = True

        if scored:
            epoch_goals += 1
            rally_lengths.append(current_touches)
            current_touches = 0
            goal_frames.append(t)
            goal_pos_rew = np.array([ball['x'], ball['y'], reward])
            goal_vel = np.array([0.0, 0.0, 0.0])
            for _ in range(5):
                nip_rew_l = abs(wp_ball.normalized_inner_product(wp_reward_l))
                nip_env_g = abs(wp_ball.normalized_inner_product(wp_env))
                rew_force = wp_ball.predict_force(wp_reward_l, nip_rew_l)
                env_force = wp_ball.predict_force(wp_env, nip_env_g)
                wp_ball.predict_position(goal_vel, dt, env_force + rew_force)
                wp_ball.update_with_attention(goal_pos_rew, np.ones(NDIM),
                                              [nip_env_g, nip_rew_l])
                wp_reward_l.update_lms(goal_pos_rew, np.array([0.0, 0.0, reward]),
                                        anomaly_scale=1.0)
                wp_reward_r.update_lms(goal_pos_rew, np.array([0.0, 0.0, -reward]),
                                        anomaly_scale=1.0)
                wp_reward_l.soft_normalize(max_energy=2.0)
                wp_reward_r.soft_normalize(max_energy=2.0)
                wp_ball.normalize()
                wp_ball.pos[:] = goal_pos_rew
            wp_pl.update_with_attention(
                np.array([paddle_lx, left_paddle['y'], reward]),
                np.ones(NDIM), [1.0])
            wp_pr.update_with_attention(
                np.array([paddle_rx, right_paddle['y'], -reward]),
                np.ones(NDIM), [1.0])
            rl_left.reward_update(ball['x'], reward)
            rl_right.reward_update(ball['x'], -reward)
            term_state = np.zeros(SimpleRLController.STATE_DIM)
            rl_left.step(term_state, reward)
            rl_right.step(term_state, -reward)
            rl_left.on_reset()
            rl_right.on_reset()
            reset_ball(ball, toward='left' if reward < 0 else 'right',
                       speed=ball_speed)
            freeze = int(0.5 / dt)
            wp_ball.__init__(K, freqs, pos0=(ball['x'], ball['y'], 0.0),
                             mass=1.0, sigma=0.8, ndim=NDIM,
                             lr=0.15, lr_tracking=0.01)

        ball_pos = np.array([ball['x'], ball['y'], 0.0])
        paddle_l_pos = np.array([paddle_lx, left_paddle['y'], 0.0])
        paddle_r_pos = np.array([paddle_rx, right_paddle['y'], 0.0])

        if not scored:
            nip_env = abs(wp_ball.normalized_inner_product(wp_env))
            nip_padL = abs(wp_ball.normalized_inner_product(wp_pl))
            nip_padR = abs(wp_ball.normalized_inner_product(wp_pr))
            nip_rew = abs(wp_ball.normalized_inner_product(wp_reward_l))
            env_force = wp_ball.predict_force(wp_env, nip_env)
            rew_force = wp_ball.predict_force(wp_reward_l, nip_rew)
            ball_vel = np.array([ball['vx'], ball['vy'], 0.0])
            predicted_pos = wp_ball.predict_position(ball_vel, dt,
                                                      env_force + rew_force)
        else:
            nip_env = nip_padL = nip_padR = nip_rew = 0.0
            predicted_pos = ball_pos.copy()

        wp_ball.shift(ball['vx'] * dt, axis=0)
        wp_ball.shift(ball['vy'] * dt, axis=1)
        delta_l = left_paddle['y'] - wp_pl.pos[1]
        delta_r = right_paddle['y'] - wp_pr.pos[1]
        if abs(delta_l) > 1e-12:
            wp_pl.shift(delta_l, axis=1)
        if abs(delta_r) > 1e-12:
            wp_pr.shift(delta_r, axis=1)

        if not scored:
            unity = np.ones(NDIM)
            wp_ball.update_with_attention(ball_pos, unity,
                                          [nip_env, nip_padL, nip_padR])
            wp_pl.update_with_attention(paddle_l_pos, unity, [nip_padL])
            wp_pr.update_with_attention(paddle_r_pos, unity, [nip_padR])

        ball_dev = np.array([wp_ball.integrate_squared(d) - 1.0
                             for d in range(NDIM)])
        wp_ball.normalize()
        wp_pl.normalize()
        wp_pr.normalize()

        dev_mag = np.linalg.norm(ball_dev[:2])
        if not scored and dev_mag > 1e-8:
            total_nip = nip_env + nip_padL + nip_padR + 1e-8
            env_frac = nip_env / total_nip
            wp_env.update_lms(ball_pos, ball_dev,
                              anomaly_scale=dev_mag * env_frac)
            wp_env.normalize()

        wp_ball.pos[:] = ball_pos
        wp_pl.pos[:] = paddle_l_pos
        wp_pr.pos[:] = paddle_r_pos

        if not scored:
            ns_fmaps_l = compute_feature_maps(
                wp_ball, wp_env, wp_pl, wp_pr, wp_reward_l,
                x_fm, y_fm, r_fm)
            ns_fmaps_r = compute_feature_maps(
                wp_ball, wp_env, wp_pl, wp_pr, wp_reward_r,
                x_fm, y_fm, r_fm)
            ns_l = SimpleRLController.build_state(ns_fmaps_l, rl_left.conv)
            ns_r = SimpleRLController.build_state(ns_fmaps_r, rl_right.conv)
            rl_left.step(ns_l, 0.0)
            rl_right.step(ns_r, 0.0)

        epoch_td_errors.append(rl_left.last_td_error)

        # Epoch boundary
        if (t + 1) % EPOCH == 0:
            ep = {
                'epoch': len(epochs),
                'frame': t + 1,
                'goals': epoch_goals,
                'track_err': float(np.mean(epoch_tracking)) if epoch_tracking else 0,
                'track_std': float(np.std(epoch_tracking)) if epoch_tracking else 0,
                'td_mean': float(np.mean(epoch_td_errors)) if epoch_td_errors else 0,
                'td_abs_mean': float(np.mean(np.abs(epoch_td_errors))) if epoch_td_errors else 0,
                'td_max': float(np.max(np.abs(epoch_td_errors))) if epoch_td_errors else 0,
                'w_c_norm': float(np.linalg.norm(rl_left.w_c)),
                'w_a_norm': float(np.linalg.norm(rl_left.w_a)),
                'conv_w_norm': float(np.linalg.norm(rl_left.conv.W)),
                'value': float(rl_left.last_value),
                'b_c': float(rl_left.b_c),
                'touches': epoch_touches,
            }
            epochs.append(ep)
            epoch_tracking = []
            epoch_td_errors = []
            epoch_goals = 0
            epoch_touches = 0

    # Goal interval analysis
    intervals = np.diff(goal_frames) if len(goal_frames) > 1 else np.array([n_frames])
    n_int = len(intervals)
    fifth = max(n_int // 5, 1)

    return {
        'epochs': epochs,
        'goal_frames': goal_frames,
        'rally_lengths': rally_lengths,
        'n_goals': len(goal_frames),
        'total_touches': sum(rally_lengths),
        'intervals_by_fifth': [
            float(intervals[i*fifth:(i+1)*fifth].mean())
            for i in range(5)
        ] if n_int >= 5 else [],
    }


def write_report(filename: str, title: str, all_results: list[dict],
                 n_frames: int, config: dict):
    """Write a detailed text report."""
    lines = []
    lines.append('=' * 72)
    lines.append(f'  {title}')
    lines.append('=' * 72)
    lines.append(f'  Config: lr_actor={config["lr_actor"]}, '
                 f'lr_critic={config["lr_critic"]}, '
                 f'gamma={config["gamma"]}, lam={config["lam"]}, '
                 f'std={config["std"]}')
    lines.append(f'  Frames: {n_frames} (~{n_frames/60:.0f}s game time)')
    lines.append(f'  Seeds: {len(all_results)}')
    lines.append('')

    for seed_idx, res in enumerate(all_results):
        lines.append(f'  --- Seed {seed_idx} ---')
        lines.append(f'  Total goals: {res["n_goals"]}, '
                     f'Total touches: {res["total_touches"]}, '
                     f'Mean rally: {np.mean(res["rally_lengths"]):.1f} touches')
        rl = res['rally_lengths']
        lines.append(f'  Rally lengths: {rl[:20]}{"..." if len(rl)>20 else ""}')
        if res['intervals_by_fifth']:
            fifths = res['intervals_by_fifth']
            lines.append(f'  Goal intervals by fifth:')
            for i, v in enumerate(fifths):
                label = ['1st', '2nd', '3rd', '4th', '5th'][i]
                lines.append(f'    {label}: {v:.0f} frames')
            if fifths[0] > 0:
                ratio = fifths[-1] / fifths[0]
                trend = 'IMPROVING' if ratio > 1.15 else ('DEGRADING' if ratio < 0.85 else 'FLAT')
                lines.append(f'    Last/first ratio: {ratio:.2f}x  ({trend})')

        lines.append('')
        lines.append(f'  Epoch-by-epoch diagnostics ({EPOCH_SIZE} frames per epoch):')
        lines.append(f'  {"Ep":>3} {"Frame":>6} {"Goals":>5} {"Touch":>5} {"TrackErr":>9} '
                     f'{"|TD|mean":>9} {"|TD|max":>9} {"||w_c||":>8} '
                     f'{"||w_a||":>8} {"V(s)":>8}')
        lines.append(f'  {"---":>3} {"-----":>6} {"-----":>5} {"-----":>5} {"--------":>9} '
                     f'{"--------":>9} {"--------":>9} {"-------":>8} '
                     f'{"-------":>8} {"----":>8}')
        for ep in res['epochs']:
            lines.append(
                f'  {ep["epoch"]:3d} {ep["frame"]:6d} {ep["goals"]:5d} '
                f'{ep["touches"]:5d} '
                f'{ep["track_err"]:9.3f} {ep["td_abs_mean"]:9.5f} '
                f'{ep["td_max"]:9.5f} {ep["w_c_norm"]:8.4f} '
                f'{ep["w_a_norm"]:8.4f} '
                f'{ep["value"]:8.4f}')
        lines.append('')

    # Aggregate summary
    lines.append('  === AGGREGATE SUMMARY ===')
    all_goals = [r['n_goals'] for r in all_results]
    all_touches = [r['total_touches'] for r in all_results]
    all_mean_rally = [np.mean(r['rally_lengths']) for r in all_results]
    lines.append(f'  Goals: {np.mean(all_goals):.1f} +/- {np.std(all_goals):.1f}')
    lines.append(f'  Total touches: {np.mean(all_touches):.1f} +/- {np.std(all_touches):.1f}')
    lines.append(f'  Mean rally length: {np.mean(all_mean_rally):.2f} touches')

    # Track error trajectory (averaged across seeds)
    n_epochs = min(len(r['epochs']) for r in all_results)
    if n_epochs > 0:
        track_means = np.array([
            [r['epochs'][i]['track_err'] for r in all_results]
            for i in range(n_epochs)
        ]).mean(axis=1)
        wc_means = np.array([
            [r['epochs'][i]['w_c_norm'] for r in all_results]
            for i in range(n_epochs)
        ]).mean(axis=1)
        td_max_means = np.array([
            [r['epochs'][i]['td_max'] for r in all_results]
            for i in range(n_epochs)
        ]).mean(axis=1)

        lines.append(f'  Tracking error (mean across seeds):')
        lines.append(f'    Early (ep 0-4):  {track_means[:5].mean():.3f}')
        lines.append(f'    Mid (ep 5-9):    {track_means[5:10].mean():.3f}' if n_epochs >= 10 else '')
        if n_epochs >= 15:
            lines.append(f'    Late (ep 10+):   {track_means[10:].mean():.3f}')

        lines.append(f'  ||w_c|| (mean across seeds):')
        lines.append(f'    Early: {wc_means[:5].mean():.4f}')
        lines.append(f'    Late:  {wc_means[-5:].mean():.4f}')
        lines.append(f'  |TD|max (mean across seeds):')
        lines.append(f'    Early: {td_max_means[:5].mean():.5f}')
        lines.append(f'    Late:  {td_max_means[-5:].mean():.5f}')

    lines.append('')

    report = '\n'.join(lines)
    with open(filename, 'w') as f:
        f.write(report)
    print(report)
    return report


if __name__ == '__main__':
    N_FRAMES = 6000
    SEEDS = 3
    CONFIG = {
        'lr_actor': 1e-2,
        'lr_critic': 1e-1,
        'gamma': 0.95,
        'lam': 0.9,
        'std': 0.3,
    }

    print(f'Running {N_FRAMES} frames x {SEEDS} seeds per condition...\n')

    # Pre-fix (no clipping)
    print('>>> WITHOUT clipping...')
    results_nofix = []
    for s in range(SEEDS):
        print(f'  seed {s}...', end=' ', flush=True)
        r = run_with_diagnostics(N_FRAMES, seed=s, disable_clipping=True,
                                  **CONFIG)
        results_nofix.append(r)
        print(f'{r["n_goals"]} goals')

    write_report('training/report_pre_fix.txt',
                 'PRE-FIX: No Weight/TD Clipping (Original Behavior)',
                 results_nofix, N_FRAMES, CONFIG)

    # Post-fix (with clipping)
    print('\n>>> WITH clipping...')
    results_fix = []
    for s in range(SEEDS):
        print(f'  seed {s}...', end=' ', flush=True)
        r = run_with_diagnostics(N_FRAMES, seed=s, disable_clipping=False,
                                  **CONFIG)
        results_fix.append(r)
        print(f'{r["n_goals"]} goals')

    write_report('training/report_post_fix.txt',
                 'POST-FIX: With Weight/TD Clipping (Stabilized)',
                 results_fix, N_FRAMES, CONFIG)

    # Reset
    SimpleRLController.WEIGHT_CLIP = 5.0
    SimpleRLController.TD_CLIP = 1.0
