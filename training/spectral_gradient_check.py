"""
Spectral Rollout Gradient Check
================================
Tests whether Q = inner_product(ego, reward) after N-step Fourier rollout
produces meaningful, directionally-correct gradients dQ/d(action).

Compares spectral-derived gradient direction against the oracle (perfect policy):
  oracle_direction = sign(ball_y - paddle_y)

Usage:
    python training/spectral_gradient_check.py
    python training/spectral_gradient_check.py --n-states 20 --seed 42
"""

import argparse
import os
import sys

import numpy as np
import torch

training_dir = os.path.dirname(os.path.abspath(__file__))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from pong_trainer import PongEnv, PADDLE_SPEED, DT


# ---------------------------------------------------------------------------
# Differentiable Fourier operations
# ---------------------------------------------------------------------------

def fourier_shift(c_cos, c_sin, delta, freqs):
    """Shift wavepacket coefficients by delta on one axis.

    Args:
        c_cos, c_sin: (K,) — Fourier coefficients for one axis
        delta: scalar tensor (requires_grad ok)
        freqs: (K,) — fixed frequencies
    Returns:
        new_cos, new_sin: (K,) shifted coefficients
    """
    angles = freqs * delta
    ca, sa = torch.cos(angles), torch.sin(angles)
    new_cos = c_cos * ca - c_sin * sa
    new_sin = c_cos * sa + c_sin * ca
    return new_cos, new_sin


def spectral_inner_product(a_cos, a_sin, b_cos, b_sin):
    """Inner product between two wavepackets across all dims.

    Args:
        a_cos, a_sin, b_cos, b_sin: (K, ndim) coefficient tensors
    Returns:
        scalar — sum of per-frequency per-dim products
    """
    return torch.sum(a_cos * b_cos + a_sin * b_sin)


def spectral_rollout(ego_cos, ego_sin, rew_cos, rew_sin,
                     ball_cos, ball_sin, action, ball_vx, ball_vy,
                     freqs, n_steps):
    """Differentiable N-step spectral rollout.

    Shifts ego paddle by action * PADDLE_SPEED * DT on y-axis each step.
    Shifts ball by (ball_vx, ball_vy) * DT on (x, y) axes each step.

    Returns Q = inner_product(ego, ball) on the y-axis after rollout.
    This measures whether the paddle is aligned with the ball vertically,
    which is the key spatial relationship for blocking.

    All inputs are torch tensors. action must have requires_grad=True for
    gradient computation.

    Args:
        ego_cos, ego_sin: (K, ndim) — ego paddle wavepacket
        rew_cos, rew_sin: (K, ndim) — reward wavepacket (unused for now)
        ball_cos, ball_sin: (K, ndim) — ball wavepacket
        action: scalar tensor — paddle action in [-1, 1]
        ball_vx, ball_vy: scalar tensors — ball velocity
        freqs: (K,) — shared frequencies
        n_steps: int — rollout horizon
    Returns:
        Q: scalar tensor — y-axis inner product after rollout
    """
    # Extract y-axis columns (axis=1) — only these matter for Q
    e_cos_y = ego_cos[:, 1]
    e_sin_y = ego_sin[:, 1]
    b_cos_y = ball_cos[:, 1]
    b_sin_y = ball_sin[:, 1]
    # Also roll ball x for completeness (not used in Q but part of dynamics)
    b_cos_x = ball_cos[:, 0]
    b_sin_x = ball_sin[:, 0]

    ego_delta_y = action * PADDLE_SPEED * DT
    ball_delta_x = ball_vx * DT
    ball_delta_y = ball_vy * DT

    for _ in range(n_steps):
        # Shift ego on y-axis
        e_cos_y, e_sin_y = fourier_shift(e_cos_y, e_sin_y, ego_delta_y, freqs)
        # Shift ball on x-axis and y-axis
        b_cos_x, b_sin_x = fourier_shift(b_cos_x, b_sin_x, ball_delta_x, freqs)
        b_cos_y, b_sin_y = fourier_shift(b_cos_y, b_sin_y, ball_delta_y, freqs)

    # Q = inner_product(ego, ball) on y-axis only
    # This measures: "after N steps, does my paddle overlap with the ball vertically?"
    Q = torch.sum(e_cos_y * b_cos_y + e_sin_y * b_sin_y)
    return Q


# ---------------------------------------------------------------------------
# Extract wavepacket state from PongEnv
# ---------------------------------------------------------------------------

def extract_spectral_state(env):
    """Pull wavepacket coefficients out of a spectral-mode PongEnv."""
    return {
        'ego_cos': torch.tensor(env._wp_pl.c_cos.copy(), dtype=torch.float32),
        'ego_sin': torch.tensor(env._wp_pl.c_sin.copy(), dtype=torch.float32),
        'rew_cos': torch.tensor(env._wp_reward.c_cos.copy(), dtype=torch.float32),
        'rew_sin': torch.tensor(env._wp_reward.c_sin.copy(), dtype=torch.float32),
        'ball_cos': torch.tensor(env._wp_ball.c_cos.copy(), dtype=torch.float32),
        'ball_sin': torch.tensor(env._wp_ball.c_sin.copy(), dtype=torch.float32),
        'freqs': torch.tensor(env._freqs.copy(), dtype=torch.float32),
        'ball_vx': env.ball_vx,
        'ball_vy': env.ball_vy,
        'ball_y': env.ball_y,
        'paddle_y': env.agent_y,
    }


# ---------------------------------------------------------------------------
# Gradient check for a single state
# ---------------------------------------------------------------------------

def check_gradients(state, n_steps, action_val=0.0):
    """Compute Q and dQ/da at a given action, plus Q at action=+1 and -1.

    Returns dict with Q, dQ_da, Q_plus, Q_minus, oracle_sign, match.
    """
    action = torch.tensor(action_val, dtype=torch.float32, requires_grad=True)
    ball_vx = torch.tensor(state['ball_vx'], dtype=torch.float32)
    ball_vy = torch.tensor(state['ball_vy'], dtype=torch.float32)

    Q = spectral_rollout(
        state['ego_cos'], state['ego_sin'],
        state['rew_cos'], state['rew_sin'],
        state['ball_cos'], state['ball_sin'],
        action, ball_vx, ball_vy,
        state['freqs'], n_steps)

    Q.backward()
    dQ_da = action.grad.item()

    # Q at extreme actions (no grad needed)
    with torch.no_grad():
        Q_plus = spectral_rollout(
            state['ego_cos'], state['ego_sin'],
            state['rew_cos'], state['rew_sin'],
            state['ball_cos'], state['ball_sin'],
            torch.tensor(1.0), ball_vx, ball_vy,
            state['freqs'], n_steps).item()

        Q_minus = spectral_rollout(
            state['ego_cos'], state['ego_sin'],
            state['rew_cos'], state['rew_sin'],
            state['ball_cos'], state['ball_sin'],
            torch.tensor(-1.0), ball_vx, ball_vy,
            state['freqs'], n_steps).item()

    # Oracle: perfect action direction
    ball_dy = state['ball_y'] - state['paddle_y']
    oracle_sign = 1.0 if ball_dy > 0 else (-1.0 if ball_dy < 0 else 0.0)

    # Does gradient agree with oracle?
    grad_sign = 1.0 if dQ_da > 0 else (-1.0 if dQ_da < 0 else 0.0)
    match = (grad_sign == oracle_sign) if oracle_sign != 0.0 else True

    return {
        'Q': Q.item(),
        'dQ_da': dQ_da,
        'Q_plus': Q_plus,
        'Q_minus': Q_minus,
        'ball_dy': ball_dy,
        'oracle_sign': oracle_sign,
        'match': match,
    }


# ---------------------------------------------------------------------------
# Collect diverse game states
# ---------------------------------------------------------------------------

def collect_states(n_states, seed=0):
    """Run PongEnv for a while, collect states at varied moments."""
    np.random.seed(seed)
    env = PongEnv(opp_skill=0.5, reward_mode='paddle', obs_mode='spectral')

    states = []
    env.reset()
    step = 0
    max_steps = n_states * 60  # ~1s of game per state

    while len(states) < n_states and step < max_steps * 5:
        action = np.random.uniform(-1.0, 1.0)
        _, reward, done = env.step(action)
        step += 1

        # Sample state every ~60 steps (1 second of game time)
        if step % 60 == 0:
            s = extract_spectral_state(env)
            # Skip if ball/paddle nearly coincident (trivial)
            if abs(s['ball_y'] - s['paddle_y']) > 0.1:
                states.append(s)

        if done:
            env.reset()

    return states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Spectral rollout gradient check')
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print("Collecting game states...")
    states = collect_states(args.n_states, seed=args.seed)
    print(f"Collected {len(states)} states\n")

    # Diagnostic: inspect first state's wavepacket structure
    s0 = states[0]
    print("=== Wavepacket Diagnostics (State 0) ===")
    for name, cos_key, sin_key in [
        ('ego',  'ego_cos',  'ego_sin'),
        ('reward', 'rew_cos', 'rew_sin'),
        ('ball', 'ball_cos', 'ball_sin'),
    ]:
        c, s = s0[cos_key].numpy(), s0[sin_key].numpy()
        energy = np.sum(c**2 + s**2, axis=0)  # per-dim energy
        print(f"  {name:>6}: energy_per_dim = [{energy[0]:.4f}, {energy[1]:.4f}, {energy[2]:.4f}]"
              f"  (x, y, reward)")
        print(f"          c_cos =\n{c}")
        print(f"          c_sin =\n{s}")
    print()

    n_values = [1, 5, 10, 20, 50]

    for N in n_values:
        print(f"{'='*80}")
        print(f"  N = {N} rollout steps ({N * DT:.3f}s lookahead)")
        print(f"{'='*80}")
        header = (f"{'St':>3} | {'ball_dy':>7} | {'Q(0)':>8} | {'dQ/da':>9} | "
                  f"{'oracle':>6} | {'match':>5} | {'Q(+1)':>8} | {'Q(-1)':>8} | "
                  f"{'Q(+1)-Q(-1)':>11}")
        print(header)
        print("-" * len(header))

        matches = 0
        nonzero = 0
        q_ranges = []

        for i, state in enumerate(states):
            r = check_gradients(state, N, action_val=0.0)
            q_range = r['Q_plus'] - r['Q_minus']
            q_ranges.append(abs(q_range))

            if r['dQ_da'] != 0.0:
                nonzero += 1
            if r['match']:
                matches += 1

            match_str = "YES" if r['match'] else "NO"
            print(f"{i:>3} | {r['ball_dy']:>7.3f} | {r['Q']:>8.4f} | "
                  f"{r['dQ_da']:>9.5f} | {r['oracle_sign']:>6.1f} | "
                  f"{match_str:>5} | {r['Q_plus']:>8.4f} | {r['Q_minus']:>8.4f} | "
                  f"{q_range:>11.5f}")

        n_total = len(states)
        print(f"\nSummary (N={N}):")
        print(f"  Non-zero gradients: {nonzero}/{n_total} ({nonzero/n_total:.0%})")
        print(f"  Oracle agreement:   {matches}/{n_total} ({matches/n_total:.0%})")
        print(f"  Mean |Q(+1)-Q(-1)|: {np.mean(q_ranges):.6f}")
        print(f"  Max  |Q(+1)-Q(-1)|: {np.max(q_ranges):.6f}")
        print()


if __name__ == '__main__':
    main()
