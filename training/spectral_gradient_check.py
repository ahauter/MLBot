"""
Spectral Rollout Gradient Check — Coupled Dimensions + LMS Learning
====================================================================
Tests whether Q = inner_product(ego, reward) after N-step COUPLED Fourier
rollout produces meaningful gradients when the reward wavepacket is
LEARNED from reward events via LMS (not initialized with oracle knowledge).

Phase 1 (previous): Proved coupled representation produces action-dependent
gradients when reward wavepacket is initialized at ball position (oracle).

Phase 2 (this): Initialize reward wavepacket randomly, let LMS learn from
reward events, check if gradients become directionally correct over time.

Usage:
    python training/spectral_gradient_check.py
    python training/spectral_gradient_check.py --episodes 200 --seed 42
"""

import argparse
import os
import sys

import numpy as np
import torch

training_dir = os.path.dirname(os.path.abspath(__file__))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from pong_trainer import PongEnv, PADDLE_SPEED, DT, COURT_LEFT, COURT_RIGHT


# ---------------------------------------------------------------------------
# Coupled wavepacket primitives
# ---------------------------------------------------------------------------

def make_frequency_vectors(base_freqs, ndim=3):
    """Create coupled frequency vectors spanning R^ndim.

    For each base frequency, create vectors along:
      - each axis (K * ndim pure-axis)
      - each pair of axes (K * C(ndim,2) diagonal)
      - all axes (K all-diagonal)
    Total: K * (ndim + C(ndim,2) + 1) = 8 * 7 = 56 vectors
    """
    vecs = []
    for k in base_freqs:
        for d in range(ndim):
            v = np.zeros(ndim); v[d] = k
            vecs.append(v)
        for d1 in range(ndim):
            for d2 in range(d1 + 1, ndim):
                v = np.zeros(ndim)
                v[d1] = k / np.sqrt(2); v[d2] = k / np.sqrt(2)
                vecs.append(v)
        v = np.ones(ndim) * k / np.sqrt(ndim)
        vecs.append(v)
    return torch.tensor(np.array(vecs), dtype=torch.float32)


def init_coupled_wavepacket(position, k_vecs, sigma=0.8, amplitude=1.5):
    """Initialize coupled wavepacket centered at position.
    Returns c_cos, c_sin as (K,) numpy arrays."""
    pos = np.asarray(position, dtype=np.float64)
    kv = k_vecs.numpy()
    k_mag = np.linalg.norm(kv, axis=1)
    envelope = amplitude * np.exp(-k_mag**2 * sigma**2 / 2)
    phases = kv @ pos
    c_cos = envelope * np.cos(phases)
    c_sin = envelope * np.sin(phases)
    energy = 0.5 * np.sum(c_cos**2 + c_sin**2)
    if energy > 1e-12:
        scale = 1.0 / np.sqrt(energy)
        c_cos *= scale; c_sin *= scale
    return c_cos, c_sin


def init_random_wavepacket(K, rng, energy=1.0):
    """Initialize a random coupled wavepacket (no positional bias).
    Returns c_cos, c_sin as (K,) numpy arrays normalized to given energy."""
    c_cos = rng.randn(K).astype(np.float64)
    c_sin = rng.randn(K).astype(np.float64)
    e = 0.5 * np.sum(c_cos**2 + c_sin**2)
    if e > 1e-12:
        scale = np.sqrt(energy / e)
        c_cos *= scale; c_sin *= scale
    return c_cos, c_sin


# ---------------------------------------------------------------------------
# Coupled LMS update (numpy, for env-side learning)
# ---------------------------------------------------------------------------

def coupled_lms_update(c_cos, c_sin, k_vecs_np, position, target_value,
                       lr=0.1, clip=10.0):
    """LMS update for coupled wavepacket.

    At position p, the field predicts F(p) = Σ c_cos[j]*cos(k·p) + c_sin[j]*sin(k·p).
    Update toward target_value using gradient descent on (target - F(p))².

    Args:
        c_cos, c_sin: (K,) numpy — mutable, updated in place
        k_vecs_np: (K, ndim) numpy — frequency vectors
        position: (ndim,) — where the reward event happened
        target_value: float — what F(p) should be
        lr: learning rate
        clip: coefficient clamp
    Returns:
        residual: float
    """
    phases = k_vecs_np @ position  # (K,)
    cos_basis = np.cos(phases)
    sin_basis = np.sin(phases)
    prediction = float(np.sum(c_cos * cos_basis + c_sin * sin_basis))
    residual = target_value - prediction
    c_cos += lr * residual * cos_basis
    c_sin += lr * residual * sin_basis
    np.clip(c_cos, -clip, clip, out=c_cos)
    np.clip(c_sin, -clip, clip, out=c_sin)
    return residual


def coupled_normalize(c_cos, c_sin, target_energy=1.0):
    """Normalize so 0.5 * Σ(c²) = target_energy."""
    e = 0.5 * np.sum(c_cos**2 + c_sin**2)
    if e > 1e-12:
        s = np.sqrt(target_energy / e)
        c_cos *= s; c_sin *= s


# ---------------------------------------------------------------------------
# Differentiable rollout (torch, for gradient check)
# ---------------------------------------------------------------------------

def coupled_shift(c_cos, c_sin, delta, k_vecs, axis):
    """Shift coupled wavepacket by delta on axis. Torch tensors."""
    angles = k_vecs[:, axis] * delta
    ca, sa = torch.cos(angles), torch.sin(angles)
    return c_cos * ca - c_sin * sa, c_cos * sa + c_sin * ca


def coupled_rollout_q(ego_cos, ego_sin, rew_cos, rew_sin, action, k_vecs, n_steps):
    """N-step rollout: shift ego on y-axis, return IP(ego, reward)."""
    e_cos, e_sin = ego_cos.clone(), ego_sin.clone()
    delta_y = action * PADDLE_SPEED * DT
    for _ in range(n_steps):
        e_cos, e_sin = coupled_shift(e_cos, e_sin, delta_y, k_vecs, axis=1)
    return torch.sum(e_cos * rew_cos + e_sin * rew_sin)


# ---------------------------------------------------------------------------
# Gradient check at a single state
# ---------------------------------------------------------------------------

def check_gradient(ego_cos_np, ego_sin_np, rew_cos_np, rew_sin_np,
                   k_vecs, n_steps, ball_y, paddle_y):
    """Returns dict with Q, dQ/da, oracle comparison."""
    ego_cos = torch.tensor(ego_cos_np, dtype=torch.float32)
    ego_sin = torch.tensor(ego_sin_np, dtype=torch.float32)
    rew_cos = torch.tensor(rew_cos_np, dtype=torch.float32)
    rew_sin = torch.tensor(rew_sin_np, dtype=torch.float32)

    action = torch.tensor(0.0, requires_grad=True)
    Q = coupled_rollout_q(ego_cos, ego_sin, rew_cos, rew_sin, action, k_vecs, n_steps)
    Q.backward()
    dQ_da = action.grad.item()

    with torch.no_grad():
        Q_plus = coupled_rollout_q(ego_cos, ego_sin, rew_cos, rew_sin,
                                   torch.tensor(1.0), k_vecs, n_steps).item()
        Q_minus = coupled_rollout_q(ego_cos, ego_sin, rew_cos, rew_sin,
                                    torch.tensor(-1.0), k_vecs, n_steps).item()

    ball_dy = ball_y - paddle_y
    oracle = 1.0 if ball_dy > 0 else (-1.0 if ball_dy < 0 else 0.0)
    grad_sign = 1.0 if dQ_da > 0 else (-1.0 if dQ_da < 0 else 0.0)
    match = (grad_sign == oracle) if oracle != 0 else True

    return {
        'Q': Q.item(), 'dQ_da': dQ_da,
        'Q_plus': Q_plus, 'Q_minus': Q_minus,
        'ball_dy': ball_dy, 'oracle': oracle, 'match': match,
    }


# ---------------------------------------------------------------------------
# Main: run episodes, LMS-learn reward wavepacket, periodically check grads
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--check-interval', type=int, default=25,
                        help='Check gradients every N episodes')
    parser.add_argument('--n-checks', type=int, default=10,
                        help='Number of gradient checks per interval')
    parser.add_argument('--n-rollout', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05,
                        help='LMS learning rate for reward wavepacket')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--opp-skill', type=float, default=0.5)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    base_freqs = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    k_vecs = make_frequency_vectors(base_freqs, ndim=3)
    k_vecs_np = k_vecs.numpy()
    K = k_vecs.shape[0]

    print(f"Coupled representation: {K} frequency vectors in R^3")
    print(f"LMS lr={args.lr}, rollout N={args.n_rollout}, opp_skill={args.opp_skill}")
    print()

    # Initialize reward wavepacket RANDOMLY (no oracle knowledge)
    rew_cos, rew_sin = init_random_wavepacket(K, rng, energy=1.0)

    env = PongEnv(opp_skill=args.opp_skill, reward_mode='paddle', obs_mode='raw')

    total_pos_rewards = 0
    total_neg_rewards = 0
    episode = 0

    print(f"{'ep':>5} | {'wins':>4} | {'losses':>6} | {'+rew':>4} | {'-rew':>4} | "
          f"{'agree':>5} | {'mean_dQ':>8} | {'mean_|Qr|':>10} | {'rew_energy':>10}")
    print("-" * 85)

    MAX_EP_LEN = 600  # 10 seconds at 60fps

    while episode < args.episodes:
        obs = env.reset()
        done = False
        ep_step = 0

        while not done and ep_step < MAX_EP_LEN:
            # Very noisy — deliberately bad so we get both hits AND misses
            if rng.rand() < 0.5:
                action = float(rng.uniform(-1, 1))  # random
            else:
                ball_dy = env.ball_y - env.agent_y
                action = float(np.clip(ball_dy * 1.0 + rng.randn() * 1.5, -1, 1))
            obs, reward, done = env.step(action)
            ep_step += 1

            # LMS update on reward events
            if abs(reward) > 0.5:
                # Position where reward happened: (ball_x, ball_y, reward_sign)
                # Use ball position — that's where the scoring event occurred
                rew_pos = np.array([env.ball_x, env.ball_y, np.sign(reward)])
                coupled_lms_update(rew_cos, rew_sin, k_vecs_np, rew_pos,
                                   target_value=reward, lr=args.lr)
                coupled_normalize(rew_cos, rew_sin, target_energy=1.0)

                if reward > 0:
                    total_pos_rewards += 1
                else:
                    total_neg_rewards += 1

        episode += 1

        # Periodic gradient check
        if episode % args.check_interval == 0:
            matches = 0
            dq_sum = 0.0
            qr_sum = 0.0
            n_tested = 0

            # Collect fresh states and check gradients
            check_env = PongEnv(opp_skill=args.opp_skill, reward_mode='paddle',
                                obs_mode='raw')
            check_env.reset()
            checks_done = 0
            check_ep_step = 0
            for step in range(args.n_checks * 120):
                a = float(np.clip(rng.randn() * 0.8, -1, 1))
                _, _, d = check_env.step(a)
                check_ep_step += 1
                if d or check_ep_step > MAX_EP_LEN:
                    check_env.reset()
                    check_ep_step = 0

                if step % 60 == 0 and checks_done < args.n_checks:
                    by = check_env.ball_y
                    py = check_env.agent_y
                    if abs(by - py) > 0.2:
                        # Build ego wavepacket from current paddle position
                        ego_pos = [check_env.paddle_lx, py, 0.0]
                        ego_c, ego_s = init_coupled_wavepacket(ego_pos, k_vecs)

                        r = check_gradient(ego_c, ego_s, rew_cos, rew_sin,
                                           k_vecs, args.n_rollout, by, py)
                        if r['match']:
                            matches += 1
                        dq_sum += abs(r['dQ_da'])
                        qr_sum += abs(r['Q_plus'] - r['Q_minus'])
                        n_tested += 1
                        checks_done += 1

            if n_tested > 0:
                rew_energy = 0.5 * np.sum(rew_cos**2 + rew_sin**2)
                print(f"{episode:>5} | {total_pos_rewards:>4} | {total_neg_rewards:>6} | "
                      f"{total_pos_rewards:>4} | {total_neg_rewards:>4} | "
                      f"{matches}/{n_tested:<3} | "
                      f"{dq_sum/n_tested:>8.4f} | {qr_sum/n_tested:>10.4f} | "
                      f"{rew_energy:>10.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
