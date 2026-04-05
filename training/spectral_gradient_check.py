"""
Spectral Rollout Gradient Check — Conv Critic on Shifted Feature Maps
=====================================================================
Architecture:
  1. Extract decoupled wavepacket coefficients from PongEnv
  2. Shift ego's y-axis coefficients by action * PADDLE_SPEED * DT * N
  3. Compute 6-channel outer-product feature maps (differentiable in PyTorch)
  4. Run learned 2D conv → Q scalar
  5. Check dQ/d(action) agrees with oracle: sign(ball_y - paddle_y)

The conv learns which spatial patterns in the feature maps predict value.
The action enters analytically via Fourier phase rotation of the ego
wavepacket BEFORE the feature maps are computed. Gradients flow:
  actor → action → ego shift → changed feature maps → conv → Q

Usage:
    python training/spectral_gradient_check.py
    python training/spectral_gradient_check.py --n-rollout 20 --seed 42
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

training_dir = os.path.dirname(os.path.abspath(__file__))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from pong_trainer import (PongEnv, PADDLE_SPEED, DT,
                          COURT_LEFT, COURT_RIGHT, COURT_TOP, COURT_BOTTOM)


# ---------------------------------------------------------------------------
# Constants matching spectral_pong_viz.py
# ---------------------------------------------------------------------------
FM_NX, FM_NY = 24, 16
FM_CHANNELS = 6  # ball, env, padL, padR, reward, ball×reward
K = 8
NDIM = 3
BASE_FREQS = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])


# ---------------------------------------------------------------------------
# Differentiable feature map computation (PyTorch)
# ---------------------------------------------------------------------------

def make_basis_matrices(freqs, x_grid, y_grid, r_grid):
    """Precompute cos/sin basis matrices for the grids.

    Returns dict of (N, K) tensors for each (grid, trig_fn) combination.
    These are constants — no grad needed.
    """
    freqs_t = torch.tensor(freqs, dtype=torch.float32)
    x_t = torch.tensor(x_grid, dtype=torch.float32)
    y_t = torch.tensor(y_grid, dtype=torch.float32)
    r_t = torch.tensor(r_grid, dtype=torch.float32)

    return {
        'cos_x': torch.cos(x_t[:, None] * freqs_t[None, :]),  # (NX, K)
        'sin_x': torch.sin(x_t[:, None] * freqs_t[None, :]),
        'cos_y': torch.cos(y_t[:, None] * freqs_t[None, :]),  # (NY, K)
        'sin_y': torch.sin(y_t[:, None] * freqs_t[None, :]),
        'cos_r': torch.cos(r_t[:, None] * freqs_t[None, :]),  # (NY, K)  r_grid same size
        'sin_r': torch.sin(r_t[:, None] * freqs_t[None, :]),
    }


def evaluate_field(c_cos, c_sin, cos_basis, sin_basis):
    """Evaluate 1D Fourier field on a grid. All torch tensors.

    Args:
        c_cos, c_sin: (K,) — Fourier coefficients for one axis
        cos_basis, sin_basis: (N, K) — precomputed basis on grid
    Returns:
        (N,) field values
    """
    return cos_basis @ c_cos + sin_basis @ c_sin


def compute_feature_maps_torch(wavepackets, basis):
    """Compute 6-channel (FM_NY, FM_NX) feature maps, differentiable.

    Args:
        wavepackets: dict with keys 'ball', 'ego', 'opp', 'env', 'reward',
                     each a dict {'c_cos': (K, NDIM), 'c_sin': (K, NDIM)} tensors
        basis: precomputed basis matrices from make_basis_matrices()
    Returns:
        (6, FM_NY, FM_NX) tensor
    """
    maps = []

    for name in ['ball', 'env', 'ego', 'opp', 'reward']:
        wp = wavepackets[name]
        # Evaluate on x-grid (axis 0) and y-grid (axis 1)
        fx = evaluate_field(wp['c_cos'][:, 0], wp['c_sin'][:, 0],
                            basis['cos_x'], basis['sin_x'])  # (NX,)
        fy = evaluate_field(wp['c_cos'][:, 1], wp['c_sin'][:, 1],
                            basis['cos_y'], basis['sin_y'])  # (NY,)
        maps.append(torch.outer(fy, fx))  # (NY, NX)

    # Channel 6: ball outer product on (x, reward_dim)
    wp_ball = wavepackets['ball']
    fx_ball = evaluate_field(wp_ball['c_cos'][:, 0], wp_ball['c_sin'][:, 0],
                             basis['cos_x'], basis['sin_x'])
    fr_ball = evaluate_field(wp_ball['c_cos'][:, 2], wp_ball['c_sin'][:, 2],
                             basis['cos_r'], basis['sin_r'])
    maps.append(torch.outer(fr_ball, fx_ball))

    return torch.stack(maps)  # (6, NY, NX)


def fourier_shift(c_cos, c_sin, delta, freqs):
    """Shift Fourier coefficients by delta. Differentiable."""
    angles = freqs * delta
    ca, sa = torch.cos(angles), torch.sin(angles)
    return c_cos * ca - c_sin * sa, c_cos * sa + c_sin * ca


# ---------------------------------------------------------------------------
# Conv Critic
# ---------------------------------------------------------------------------

class ConvCritic(nn.Module):
    """Small conv net: (6, 16, 24) → Q scalar.

    Two conv layers with stride 2, then flatten → linear → Q.
    Matches the StridedConvExtractor architecture but learned.
    """

    def __init__(self, n_channels=FM_CHANNELS, n_filters=8, ks=3, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_filters, ks, stride=stride)
        self.conv2 = nn.Conv2d(n_filters, n_filters, ks, stride=stride)
        # Output size: (8, 3, 5) = 120
        oH1 = (FM_NY - ks) // stride + 1  # 7
        oW1 = (FM_NX - ks) // stride + 1  # 11
        oH2 = (oH1 - ks) // stride + 1    # 3
        oW2 = (oW1 - ks) // stride + 1    # 5
        flat_dim = n_filters * oH2 * oW2   # 120
        self.fc = nn.Linear(flat_dim, 1)

    def forward(self, fmaps):
        """fmaps: (6, NY, NX) or (batch, 6, NY, NX) → Q scalar."""
        if fmaps.dim() == 3:
            fmaps = fmaps.unsqueeze(0)
        x = F.relu(self.conv1(fmaps))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Extract wavepacket state from PongEnv
# ---------------------------------------------------------------------------

def extract_wavepackets(env):
    """Extract decoupled wavepacket coefficients as torch tensors."""
    def wp_to_torch(wp):
        return {
            'c_cos': torch.tensor(wp.c_cos.copy(), dtype=torch.float32),
            'c_sin': torch.tensor(wp.c_sin.copy(), dtype=torch.float32),
        }
    return {
        'ball': wp_to_torch(env._wp_ball),
        'ego': wp_to_torch(env._wp_pl),
        'opp': wp_to_torch(env._wp_pr),
        'env': wp_to_torch(env._wp_env),
        'reward': wp_to_torch(env._wp_reward),
    }


def shift_ego_y(wavepackets, delta, freqs_t):
    """Return new wavepackets dict with ego shifted on y-axis.
    Avoids in-place ops for autograd compatibility."""
    wp = {k: {kk: vv.clone() for kk, vv in v.items()}
          for k, v in wavepackets.items()}
    ego = wp['ego']
    old_cos = ego['c_cos']
    old_sin = ego['c_sin']
    # Shift only y-axis (column 1), rebuild full tensor
    new_cos_y, new_sin_y = fourier_shift(
        old_cos[:, 1], old_sin[:, 1], delta, freqs_t)
    ego['c_cos'] = torch.stack([old_cos[:, 0], new_cos_y, old_cos[:, 2]], dim=1)
    ego['c_sin'] = torch.stack([old_sin[:, 0], new_sin_y, old_sin[:, 2]], dim=1)
    return wp


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

def check_gradient(env, critic, basis, freqs_t, n_steps):
    """Compute Q and dQ/da at action=0, ±1. Compare to oracle."""
    wavepackets = extract_wavepackets(env)
    ball_y = env.ball_y
    paddle_y = env.agent_y

    results = {}
    for action_val, label in [(0.0, 'Q'), (1.0, 'Q_plus'), (-1.0, 'Q_minus')]:
        action = torch.tensor(action_val, dtype=torch.float32,
                              requires_grad=(action_val == 0.0))
        total_delta = action * PADDLE_SPEED * DT * n_steps
        wp_shifted = shift_ego_y(wavepackets, total_delta, freqs_t)
        fmaps = compute_feature_maps_torch(wp_shifted, basis)
        q = critic(fmaps)

        if action_val == 0.0:
            q.backward()
            results['dQ_da'] = action.grad.item()
            results['Q'] = q.item()
        else:
            results[label] = q.item()

    ball_dy = ball_y - paddle_y
    oracle = 1.0 if ball_dy > 0 else (-1.0 if ball_dy < 0 else 0.0)
    grad_sign = 1.0 if results['dQ_da'] > 0 else (-1.0 if results['dQ_da'] < 0 else 0.0)
    results['ball_dy'] = ball_dy
    results['oracle'] = oracle
    results['match'] = (grad_sign == oracle) if oracle != 0 else True
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-states', type=int, default=15)
    parser.add_argument('--n-rollout', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Grids for feature map evaluation
    x_grid = np.linspace(COURT_LEFT, COURT_RIGHT, FM_NX)
    y_grid = np.linspace(COURT_BOTTOM, COURT_TOP, FM_NY)
    r_grid = np.linspace(-1.0, 1.0, FM_NY)
    basis = make_basis_matrices(BASE_FREQS, x_grid, y_grid, r_grid)
    freqs_t = torch.tensor(BASE_FREQS, dtype=torch.float32)

    # Random conv critic (untrained — we're checking gradient FLOW, not quality)
    critic = ConvCritic()
    critic.eval()

    print(f"Rollout N={args.n_rollout} ({args.n_rollout * DT:.3f}s)")
    print(f"Conv critic params: {sum(p.numel() for p in critic.parameters())}")
    print(f"Feature maps: ({FM_CHANNELS}, {FM_NY}, {FM_NX})")
    print()

    # Collect states from spectral-mode env (so wavepackets exist)
    env = PongEnv(opp_skill=0.5, reward_mode='paddle', obs_mode='spectral')
    env.reset()

    states_checked = 0
    matches = 0
    nonzero = 0
    dq_list = []
    qr_list = []
    step = 0

    header = (f"{'St':>3} | {'ball_dy':>7} | {'Q(0)':>8} | {'dQ/da':>9} | "
              f"{'oracle':>6} | {'match':>5} | {'Q(+1)':>8} | {'Q(-1)':>8} | "
              f"{'Q(+1)-Q(-1)':>11}")
    print(header)
    print("-" * len(header))

    while states_checked < args.n_states:
        action = np.random.uniform(-1, 1)
        _, _, done = env.step(action)
        step += 1

        if step % 40 == 0:
            ball_dy = env.ball_y - env.agent_y
            if abs(ball_dy) > 0.15:
                with torch.no_grad():
                    pass  # critic is in eval mode, but we need grad for action
                r = check_gradient(env, critic, basis, freqs_t, args.n_rollout)

                q_range = r['Q_plus'] - r['Q_minus']
                qr_list.append(abs(q_range))
                dq_list.append(abs(r['dQ_da']))
                if abs(r['dQ_da']) > 1e-10:
                    nonzero += 1
                if r['match']:
                    matches += 1
                states_checked += 1

                m = "YES" if r['match'] else "NO"
                print(f"{states_checked:>3} | {r['ball_dy']:>7.3f} | {r['Q']:>8.4f} | "
                      f"{r['dQ_da']:>9.5f} | {r['oracle']:>6.1f} | "
                      f"{m:>5} | {r['Q_plus']:>8.4f} | {r['Q_minus']:>8.4f} | "
                      f"{q_range:>11.5f}")

        if done:
            env.reset()

    print(f"\nSummary (N={args.n_rollout}):")
    print(f"  Non-zero gradients: {nonzero}/{states_checked} ({nonzero/states_checked:.0%})")
    print(f"  Oracle agreement:   {matches}/{states_checked} ({matches/states_checked:.0%})")
    print(f"  Mean |dQ/da|:       {np.mean(dq_list):.6f}")
    print(f"  Mean |Q(+1)-Q(-1)|: {np.mean(qr_list):.6f}")

    # Also test across N values
    print(f"\n{'='*60}")
    print("Gradient strength vs rollout horizon (5 states, varying N)")
    print(f"{'='*60}")

    # Collect 5 fixed states
    env.reset()
    fixed_states = []
    for s in range(5000):
        _, _, d = env.step(np.random.uniform(-1, 1))
        if s % 200 == 100:
            if abs(env.ball_y - env.agent_y) > 0.3:
                fixed_states.append(lambda e=env: extract_wavepackets(e))
                # Actually need to snapshot — lambdas won't work
        if d:
            env.reset()

    # Re-collect properly
    env.reset()
    snapshots = []
    for s in range(3000):
        _, _, d = env.step(np.random.uniform(-1, 1))
        if s % 200 == 100 and len(snapshots) < 5:
            if abs(env.ball_y - env.agent_y) > 0.3:
                snapshots.append({
                    'wp': extract_wavepackets(env),
                    'ball_y': env.ball_y,
                    'paddle_y': env.agent_y,
                })
        if d:
            env.reset()

    if snapshots:
        print(f"\n{'N':>5} | {'mean |dQ/da|':>12} | {'mean |Qrange|':>13} | {'agree':>5}")
        print("-" * 50)
        for N in [1, 5, 10, 20, 50]:
            m_count = 0
            dq_sum = 0
            qr_sum = 0
            for snap in snapshots:
                # Rebuild env-like check using saved wavepackets
                action = torch.tensor(0.0, requires_grad=True)
                delta = action * PADDLE_SPEED * DT * N
                wp_s = shift_ego_y(snap['wp'], delta, freqs_t)
                fmaps = compute_feature_maps_torch(wp_s, basis)
                q = critic(fmaps)
                q.backward()
                dq = action.grad.item()

                with torch.no_grad():
                    d_plus = torch.tensor(PADDLE_SPEED * DT * N)
                    d_minus = torch.tensor(-PADDLE_SPEED * DT * N)
                    qp = critic(compute_feature_maps_torch(
                        shift_ego_y(snap['wp'], d_plus, freqs_t), basis)).item()
                    qm = critic(compute_feature_maps_torch(
                        shift_ego_y(snap['wp'], d_minus, freqs_t), basis)).item()

                ball_dy = snap['ball_y'] - snap['paddle_y']
                oracle = 1.0 if ball_dy > 0 else -1.0
                grad_sign = 1.0 if dq > 0 else -1.0
                if grad_sign == oracle:
                    m_count += 1
                dq_sum += abs(dq)
                qr_sum += abs(qp - qm)

            ns = len(snapshots)
            print(f"{N:>5} | {dq_sum/ns:>12.6f} | {qr_sum/ns:>13.6f} | {m_count}/{ns}")

    print("\nNote: conv weights are RANDOM (untrained). Oracle agreement reflects")
    print("gradient FLOW, not gradient QUALITY. A trained critic would learn to")
    print("make these gradients directionally correct.")


if __name__ == '__main__':
    main()
