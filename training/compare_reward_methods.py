"""
Compare wavepacket reward field inner products vs SAC critic Q-values.

Runs a short SAC training on Pong (spectral mode) and tracks:
  - r_dot_ego:  reward field . agent paddle  (spectral inner product)
  - r_dot_ball: reward field . ball          (spectral inner product)
  - Q_critic:   SAC twin-Q value estimate

Goal: see if the analytic reward field can produce a stable signal
correlated with the learned critic, as a first step toward using
wavepacket reward fields as critic substitutes.

Usage:
    python training/compare_reward_methods.py
    python training/compare_reward_methods.py --total-steps 10000 --eval-interval 1000
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

training_dir = os.path.dirname(os.path.abspath(__file__))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from pong_sac import Actor, SoftQNetwork, ReplayBuffer
from pong_trainer import PongEnv


# ---------------------------------------------------------------------------
# Wavepacket metric extraction
# ---------------------------------------------------------------------------

def compute_wp_metrics(env):
    """Extract reward field inner products from a spectral PongEnv.

    Returns dict with:
      r_dot_ego_ip2:  inner_product(wp_reward, wp_pl)[2]   (reward dim)
      r_dot_ball_ip2: inner_product(wp_reward, wp_ball)[2]  (reward dim)
      r_dot_ego_nip:  normalized_inner_product (all dims, scalar)
      r_dot_ball_nip: normalized_inner_product (all dims, scalar)
    """
    wp_reward = env._wp_reward
    wp_pl = env._wp_pl
    wp_ball = env._wp_ball

    ip_ego = wp_reward.inner_product(wp_pl)      # (3,) -> [x, y, reward]
    ip_ball = wp_reward.inner_product(wp_ball)    # (3,)
    nip_ego = wp_reward.normalized_inner_product(wp_pl)    # scalar
    nip_ball = wp_reward.normalized_inner_product(wp_ball)  # scalar

    return {
        'r_dot_ego_ip2': float(ip_ego[2]),
        'r_dot_ball_ip2': float(ip_ball[2]),
        'r_dot_ego_nip': float(nip_ego),
        'r_dot_ball_nip': float(nip_ball),
    }


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def pearson_r(x, y):
    """Pearson correlation coefficient. Returns NaN if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float('nan')
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom < 1e-12:
        return float('nan')
    return float((xm * ym).sum() / denom)


def rolling_mean(arr, window):
    """Simple rolling mean, returns array of same length (NaN-padded)."""
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = arr[i - window + 1:i + 1].mean()
    return out


# ---------------------------------------------------------------------------
# Training with metric tracking
# ---------------------------------------------------------------------------

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force spectral mode (required for wavepacket access)
    env = PongEnv(
        opp_skill=args.opp_skill,
        reward_mode=args.reward_mode,
        obs_mode='spectral',
    )
    obs_dim = env.obs_dim

    # Networks (same as pong_sac.py)
    actor = Actor(obs_dim, args.hidden).to(device)
    critic = SoftQNetwork(obs_dim, args.hidden).to(device)
    target_critic = SoftQNetwork(obs_dim, args.hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = Adam(actor.parameters(), lr=args.lr)
    critic_opt = Adam(critic.parameters(), lr=args.lr)

    # Entropy tuning
    target_entropy = -1.0
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = Adam([log_alpha], lr=args.alpha_lr)
    alpha = log_alpha.exp().item()

    buffer = ReplayBuffer(obs_dim, args.buffer_size)

    # Episode tracking
    obs = env.reset()
    ep_reward = 0.0
    ep_len = 0
    episode_count = 0

    # Logging windows
    recent_wins = []
    recent_losses = []
    recent_returns = []
    last_critic_loss = 0.0
    last_actor_loss = 0.0
    last_alpha = alpha
    last_q_batch = 0.0
    last_entropy = 0.0

    # Per-step metric storage
    rows = []  # for CSV: list of dicts

    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # Action selection
        if step <= args.random_steps:
            action = np.random.uniform(-1.0, 1.0)
        else:
            action = actor.get_action(obs, device)
            action = float(np.clip(action, -1.0, 1.0))

        # Step environment
        next_obs, reward, done = env.step(action)
        ep_reward += reward
        ep_len += 1

        buffer.add(obs, action, reward, next_obs, done)

        # Wavepacket metrics (computed after env.step updates wavepackets)
        wp = compute_wp_metrics(env)

        # Q-value for current (obs, action) — only meaningful after random phase
        q_current = float('nan')
        if step > args.random_steps:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
                act_t = torch.tensor([[action]], dtype=torch.float32, device=device)
                q1_c, q2_c = critic(obs_t, act_t)
                q_current = torch.min(q1_c, q2_c).item()

        # Store row
        rows.append({
            'step': step,
            'episode': episode_count,
            'reward': reward,
            'r_dot_ego_ip2': wp['r_dot_ego_ip2'],
            'r_dot_ball_ip2': wp['r_dot_ball_ip2'],
            'r_dot_ego_nip': wp['r_dot_ego_nip'],
            'r_dot_ball_nip': wp['r_dot_ball_nip'],
            'q_current': q_current,
            'q_batch_mean': float('nan'),
        })

        if done:
            recent_wins.append(1 if ep_reward > 0 else 0)
            recent_losses.append(1 if ep_reward < 0 else 0)
            recent_returns.append(ep_reward)
            episode_count += 1
            obs = env.reset()
            ep_reward = 0.0
            ep_len = 0
        else:
            obs = next_obs

        # SAC update
        if step > args.random_steps and step % args.update_every == 0:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(
                args.batch_size, device)

            with torch.no_grad():
                next_action, next_log_prob = actor.sample(b_next)
                q1_next, q2_next = target_critic(b_next, next_action)
                q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
                q_target = b_rew + args.gamma * (1 - b_done) * q_next

            q1_pred, q2_pred = critic(b_obs, b_act)
            critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            new_action, log_prob = actor.sample(b_obs)
            q1_new, q2_new = critic(b_obs, new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * log_prob - q_new).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()
            alpha = log_alpha.exp().item()

            with torch.no_grad():
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.lerp_(p.data, args.tau)

            last_critic_loss = critic_loss.item()
            last_actor_loss = actor_loss.item()
            last_alpha = alpha
            last_q_batch = q1_pred.mean().item()
            last_entropy = -log_prob.mean().item()

            # Record batch Q in the current row
            rows[-1]['q_batch_mean'] = last_q_batch

        # Periodic logging
        if step % args.eval_interval == 0:
            # Compute interval averages for wavepacket metrics
            interval_rows = rows[-args.eval_interval:]
            avg_ego = np.mean([r['r_dot_ego_ip2'] for r in interval_rows])
            avg_ball = np.mean([r['r_dot_ball_ip2'] for r in interval_rows])
            avg_ego_nip = np.mean([r['r_dot_ego_nip'] for r in interval_rows])
            avg_ball_nip = np.mean([r['r_dot_ball_nip'] for r in interval_rows])

            # Q stats for interval (excluding NaN)
            q_vals = [r['q_current'] for r in interval_rows
                      if np.isfinite(r['q_current'])]
            avg_q = np.mean(q_vals) if q_vals else float('nan')

            n = len(recent_wins) if recent_wins else 1
            wins = sum(recent_wins)
            losses = sum(recent_losses)
            avg_ret = np.mean(recent_returns) if recent_returns else 0.0
            elapsed = time.time() - t0
            sps = step / elapsed

            print(f'step {step:>6d} | '
                  f'ep {episode_count:>4d} | '
                  f'W/L={wins}/{losses} | '
                  f'ret={avg_ret:+.2f} | '
                  f'Qloss={last_critic_loss:.3f} | '
                  f'Q={avg_q:.3f} | '
                  f'r·ego={avg_ego:.4f} | '
                  f'r·ball={avg_ball:.4f} | '
                  f'nip_ego={avg_ego_nip:.3f} | '
                  f'nip_ball={avg_ball_nip:.3f} | '
                  f'α={last_alpha:.3f} | '
                  f'{sps:.0f} sps', flush=True)

            recent_wins.clear()
            recent_losses.clear()
            recent_returns.clear()

    total_time = time.time() - t0
    print(f'\nDone: {args.total_steps} steps in {total_time:.1f}s '
          f'({args.total_steps / total_time:.0f} sps)\n', flush=True)

    # ------------------------------------------------------------------
    # End-of-training report
    # ------------------------------------------------------------------
    print('=' * 70)
    print('REWARD FIELD vs CRITIC COMPARISON REPORT')
    print('=' * 70)

    # Extract post-random-steps data for correlation
    post_random = [r for r in rows if r['step'] > args.random_steps]
    if len(post_random) < 10:
        print('\nToo few post-random steps for meaningful analysis.')
    else:
        ego_ip2 = np.array([r['r_dot_ego_ip2'] for r in post_random])
        ball_ip2 = np.array([r['r_dot_ball_ip2'] for r in post_random])
        ego_nip = np.array([r['r_dot_ego_nip'] for r in post_random])
        ball_nip = np.array([r['r_dot_ball_nip'] for r in post_random])
        q_cur = np.array([r['q_current'] for r in post_random])
        rewards = np.array([r['reward'] for r in post_random])

        print(f'\nData points (post random exploration): {len(post_random)}')

        # Summary statistics
        print(f'\n{"Metric":<20s} {"Mean":>10s} {"Std":>10s} {"Min":>10s} {"Max":>10s}')
        print('-' * 62)
        for name, arr in [('r·ego (ip2)', ego_ip2),
                          ('r·ball (ip2)', ball_ip2),
                          ('r·ego (nip)', ego_nip),
                          ('r·ball (nip)', ball_nip),
                          ('Q_current', q_cur),
                          ('reward', rewards)]:
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                print(f'{name:<20s} {valid.mean():>10.4f} {valid.std():>10.4f} '
                      f'{valid.min():>10.4f} {valid.max():>10.4f}')

        # Pearson correlations
        print(f'\nPearson Correlations with Q_current:')
        print('-' * 40)
        for name, arr in [('r·ego (ip2)', ego_ip2),
                          ('r·ball (ip2)', ball_ip2),
                          ('r·ego (nip)', ego_nip),
                          ('r·ball (nip)', ball_nip)]:
            r = pearson_r(arr, q_cur)
            print(f'  {name:<20s}  r = {r:+.4f}')

        # Cross-correlations between reward field metrics
        print(f'\nCross-correlations (reward field metrics):')
        print('-' * 40)
        r_ego_ball = pearson_r(ego_ip2, ball_ip2)
        print(f'  r·ego vs r·ball (ip2)   r = {r_ego_ball:+.4f}')

        # Rolling-window correlation (stability check)
        window = min(200, len(post_random) // 3)
        if window >= 20:
            print(f'\nRolling correlation (window={window} steps):')
            print('-' * 40)
            for name, arr in [('r·ego (ip2)', ego_ip2),
                              ('r·ball (ip2)', ball_ip2)]:
                # Compute rolling correlations
                rolling_corrs = []
                for i in range(window, len(arr)):
                    chunk_a = arr[i - window:i]
                    chunk_q = q_cur[i - window:i]
                    rc = pearson_r(chunk_a, chunk_q)
                    if np.isfinite(rc):
                        rolling_corrs.append(rc)
                if rolling_corrs:
                    rc_arr = np.array(rolling_corrs)
                    print(f'  {name:<20s}  mean={rc_arr.mean():+.4f}  '
                          f'std={rc_arr.std():.4f}  '
                          f'range=[{rc_arr.min():+.4f}, {rc_arr.max():+.4f}]')

        # Reward-event analysis: how do metrics behave on reward steps?
        reward_steps = [r for r in post_random if abs(r['reward']) > 0.5]
        if reward_steps:
            print(f'\nReward-event analysis ({len(reward_steps)} events):')
            print('-' * 40)
            pos_events = [r for r in reward_steps if r['reward'] > 0]
            neg_events = [r for r in reward_steps if r['reward'] < 0]
            if pos_events:
                print(f'  Positive rewards ({len(pos_events)}):')
                print(f'    r·ego mean={np.mean([r["r_dot_ego_ip2"] for r in pos_events]):.4f}  '
                      f'r·ball mean={np.mean([r["r_dot_ball_ip2"] for r in pos_events]):.4f}  '
                      f'Q mean={np.nanmean([r["q_current"] for r in pos_events]):.4f}')
            if neg_events:
                print(f'  Negative rewards ({len(neg_events)}):')
                print(f'    r·ego mean={np.mean([r["r_dot_ego_ip2"] for r in neg_events]):.4f}  '
                      f'r·ball mean={np.mean([r["r_dot_ball_ip2"] for r in neg_events]):.4f}  '
                      f'Q mean={np.nanmean([r["q_current"] for r in neg_events]):.4f}')

    print(f'\n{"=" * 70}')

    # Save CSV
    if args.save_csv:
        fieldnames = ['step', 'episode', 'reward',
                      'r_dot_ego_ip2', 'r_dot_ball_ip2',
                      'r_dot_ego_nip', 'r_dot_ball_nip',
                      'q_current', 'q_batch_mean']
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved {len(rows)} rows to {args.save_csv}')

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare wavepacket reward field vs SAC critic')
    parser.add_argument('--total-steps', type=int, default=5000)
    parser.add_argument('--reward-mode', choices=['goal', 'paddle'],
                        default='paddle')
    parser.add_argument('--opp-skill', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=100_000)
    parser.add_argument('--random-steps', type=int, default=1000)
    parser.add_argument('--update-every', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--save-csv', type=str, default='compare_reward_data.csv')
    args = parser.parse_args()

    print(f'Reward field vs critic comparison')
    print(f'  obs_mode=spectral, reward_mode={args.reward_mode}')
    print(f'  total_steps={args.total_steps}, random_steps={args.random_steps}')
    print(f'  seed={args.seed}\n')

    train(args)


if __name__ == '__main__':
    main()
