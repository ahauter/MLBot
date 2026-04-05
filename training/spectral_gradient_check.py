"""
Spectral Rollout Gradient Check — Coupled Dimensions
=====================================================
Tests whether Q = inner_product(ego, reward) after N-step COUPLED Fourier
rollout produces meaningful, directionally-correct gradients dQ/d(action).

COUPLED means: each frequency is a VECTOR k_vec[j] in R^3 (x, y, reward).
Phase = k_vec · position. Shifting on y rotates the ENTIRE coefficient
because the phase depends on ALL dimensions jointly. This means spatial
movement affects reward-dimension alignment — which is the key property
the decoupled representation lacked.

Compares spectral-derived gradient direction against the oracle:
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
# Coupled wavepacket representation
# ---------------------------------------------------------------------------
# Each wavepacket has K frequency vectors k_vec[j] in R^ndim and scalar
# coefficients c_cos[j], c_sin[j].
#
# Field at position p: F(p) = Σ_j c_cos[j]*cos(k_vec[j]·p) + c_sin[j]*sin(k_vec[j]·p)
#
# Shifting by delta on axis d:
#   angle[j] = k_vec[j, d] * delta
#   new_cos[j] = c_cos[j]*cos(angle[j]) - c_sin[j]*sin(angle[j])
#   new_sin[j] = c_cos[j]*sin(angle[j]) + c_sin[j]*cos(angle[j])
#
# Key property: shifting on y rotates ALL coefficients because k_vec[j]
# has components on all axes. This couples spatial movement to reward alignment.
# ---------------------------------------------------------------------------

def make_frequency_vectors(base_freqs, ndim=3):
    """Create coupled frequency vectors from base frequencies.

    For each base frequency, create vectors pointing in several directions
    in R^ndim so the representation spans all axis combinations.

    Strategy: for K base freqs, create vectors along:
      - each axis direction (K * ndim pure-axis vectors)
      - each pair of axes (K * C(ndim,2) diagonal vectors)
      - all axes (K all-diagonal vectors)
    Total: K * (ndim + C(ndim,2) + 1) = 8 * (3 + 3 + 1) = 56 vectors

    Returns: (K_total, ndim) tensor of frequency vectors
    """
    vecs = []
    for k in base_freqs:
        # Pure axis directions
        for d in range(ndim):
            v = np.zeros(ndim)
            v[d] = k
            vecs.append(v)
        # Pair diagonals (these create the cross-axis coupling!)
        for d1 in range(ndim):
            for d2 in range(d1 + 1, ndim):
                v = np.zeros(ndim)
                v[d1] = k / np.sqrt(2)
                v[d2] = k / np.sqrt(2)
                vecs.append(v)
        # All-axis diagonal
        v = np.ones(ndim) * k / np.sqrt(ndim)
        vecs.append(v)
    return torch.tensor(np.array(vecs), dtype=torch.float32)


def init_coupled_wavepacket(position, k_vecs, sigma=0.8, amplitude=1.5):
    """Initialize a coupled wavepacket centered at position.

    Coefficients are set so the field is a Gaussian-enveloped peak at position:
      c_cos[j] = amp[j] * cos(k_vec[j] · position)
      c_sin[j] = amp[j] * sin(k_vec[j] · position)

    where amp[j] = amplitude * exp(-|k_vec[j]|^2 * sigma^2 / 2)

    Args:
        position: (ndim,) — center position in (x, y, reward) space
        k_vecs: (K, ndim) — frequency vectors
        sigma, amplitude: envelope parameters
    Returns:
        c_cos, c_sin: (K,) tensors
    """
    pos = torch.tensor(position, dtype=torch.float32)
    k_mag = torch.norm(k_vecs, dim=1)  # (K,)
    envelope = amplitude * torch.exp(-k_mag**2 * sigma**2 / 2)  # (K,)
    phases = k_vecs @ pos  # (K,) = k_vec · position
    c_cos = envelope * torch.cos(phases)
    c_sin = envelope * torch.sin(phases)
    # Normalize so ∫F²≈1 (Parseval: energy ≈ 0.5 * Σ(c_cos² + c_sin²))
    energy = 0.5 * torch.sum(c_cos**2 + c_sin**2)
    if energy > 1e-12:
        scale = 1.0 / torch.sqrt(energy)
        c_cos = c_cos * scale
        c_sin = c_sin * scale
    return c_cos, c_sin


def coupled_shift(c_cos, c_sin, delta, k_vecs, axis):
    """Shift a coupled wavepacket by delta on one axis.

    Args:
        c_cos, c_sin: (K,) — scalar coefficients per frequency
        delta: scalar tensor (can require grad)
        k_vecs: (K, ndim) — frequency vectors
        axis: int — which axis to shift on
    Returns:
        new_cos, new_sin: (K,) shifted coefficients
    """
    angles = k_vecs[:, axis] * delta  # (K,) — only the axis-d component matters
    ca, sa = torch.cos(angles), torch.sin(angles)
    new_cos = c_cos * ca - c_sin * sa
    new_sin = c_cos * sa + c_sin * ca
    return new_cos, new_sin


def coupled_inner_product(a_cos, a_sin, b_cos, b_sin):
    """Inner product between two coupled wavepackets. Returns scalar."""
    return torch.sum(a_cos * b_cos + a_sin * b_sin)


def coupled_rollout(ego_cos, ego_sin, rew_cos, rew_sin,
                    action, k_vecs, n_steps):
    """Differentiable N-step coupled spectral rollout.

    Shifts ego by action * PADDLE_SPEED * DT on y-axis (axis=1) each step.
    Reward wavepacket is static.
    Returns Q = inner_product(ego_rolled, reward).

    Because k_vecs have components on all axes, shifting ego on y
    rotates coefficients that also encode reward-dimension alignment.
    This is the key coupling that makes Q action-dependent.
    """
    e_cos, e_sin = ego_cos.clone(), ego_sin.clone()
    ego_delta_y = action * PADDLE_SPEED * DT

    for _ in range(n_steps):
        e_cos, e_sin = coupled_shift(e_cos, e_sin, ego_delta_y, k_vecs, axis=1)

    return coupled_inner_product(e_cos, e_sin, rew_cos, rew_sin)


# ---------------------------------------------------------------------------
# Extract game state and build coupled wavepackets
# ---------------------------------------------------------------------------

def extract_state(env, k_vecs):
    """Extract game state and create coupled wavepackets from positions."""
    # Ego paddle position in (x, y, reward) space
    # reward dim = ego's accumulated reward association (use 0 as neutral)
    ego_pos = [env.paddle_lx, env.agent_y, 0.0]
    ego_cos, ego_sin = init_coupled_wavepacket(ego_pos, k_vecs)

    # Reward wavepacket: centered where positive reward happens
    # In paddle mode, +1 reward at paddle hit = agent's x position, ball's y
    # After many hits, this should be near (paddle_lx, ~0, +1)
    # For now, use a fixed "reward lives in reward-dim" position
    # but WITH spatial structure from the coupled frequencies
    rew_pos = [env.paddle_lx, env.ball_y, 1.0]
    rew_cos, rew_sin = init_coupled_wavepacket(rew_pos, k_vecs)

    return {
        'ego_cos': ego_cos, 'ego_sin': ego_sin,
        'rew_cos': rew_cos, 'rew_sin': rew_sin,
        'ball_y': env.ball_y,
        'paddle_y': env.agent_y,
        'ball_vx': env.ball_vx,
        'ball_vy': env.ball_vy,
    }


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

def check_gradients(state, k_vecs, n_steps, action_val=0.0):
    """Compute Q and dQ/da at a given action, plus Q at +1 and -1."""
    action = torch.tensor(action_val, dtype=torch.float32, requires_grad=True)

    Q = coupled_rollout(
        state['ego_cos'], state['ego_sin'],
        state['rew_cos'], state['rew_sin'],
        action, k_vecs, n_steps)
    Q.backward()
    dQ_da = action.grad.item()

    with torch.no_grad():
        Q_plus = coupled_rollout(
            state['ego_cos'], state['ego_sin'],
            state['rew_cos'], state['rew_sin'],
            torch.tensor(1.0), k_vecs, n_steps).item()
        Q_minus = coupled_rollout(
            state['ego_cos'], state['ego_sin'],
            state['rew_cos'], state['rew_sin'],
            torch.tensor(-1.0), k_vecs, n_steps).item()

    ball_dy = state['ball_y'] - state['paddle_y']
    oracle_sign = 1.0 if ball_dy > 0 else (-1.0 if ball_dy < 0 else 0.0)
    grad_sign = 1.0 if dQ_da > 0 else (-1.0 if dQ_da < 0 else 0.0)
    match = (grad_sign == oracle_sign) if oracle_sign != 0.0 else True

    return {
        'Q': Q.item(), 'dQ_da': dQ_da,
        'Q_plus': Q_plus, 'Q_minus': Q_minus,
        'ball_dy': ball_dy, 'oracle_sign': oracle_sign, 'match': match,
    }


# ---------------------------------------------------------------------------
# Collect game states
# ---------------------------------------------------------------------------

def collect_states(n_states, k_vecs, seed=0):
    """Run PongEnv, collect coupled wavepacket states at varied moments."""
    np.random.seed(seed)
    env = PongEnv(opp_skill=0.5, reward_mode='paddle', obs_mode='raw')

    states = []
    env.reset()
    step = 0

    while len(states) < n_states and step < n_states * 300:
        action = np.random.uniform(-1.0, 1.0)
        _, reward, done = env.step(action)
        step += 1

        if step % 60 == 0:
            ball_dy = env.ball_y - env.agent_y
            if abs(ball_dy) > 0.1:
                states.append(extract_state(env, k_vecs))

        if done:
            env.reset()

    return states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Coupled spectral rollout gradient check')
    parser.add_argument('--n-states', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    base_freqs = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    k_vecs = make_frequency_vectors(base_freqs, ndim=3)
    K = k_vecs.shape[0]
    print(f"Coupled representation: {K} frequency vectors in R^3")
    print(f"  (from {len(base_freqs)} base frequencies x 7 directions)\n")

    print("Collecting game states...")
    states = collect_states(args.n_states, k_vecs, seed=args.seed)
    print(f"Collected {len(states)} states\n")

    # Diagnostic: check coupling structure
    s0 = states[0]
    print("=== Coupled Wavepacket Diagnostics (State 0) ===")
    print(f"  ball_y={s0['ball_y']:.3f}  paddle_y={s0['paddle_y']:.3f}  "
          f"ball_dy={s0['ball_y']-s0['paddle_y']:.3f}")
    ego_e = 0.5 * torch.sum(s0['ego_cos']**2 + s0['ego_sin']**2).item()
    rew_e = 0.5 * torch.sum(s0['rew_cos']**2 + s0['rew_sin']**2).item()
    ip0 = coupled_inner_product(s0['ego_cos'], s0['ego_sin'],
                                s0['rew_cos'], s0['rew_sin']).item()
    print(f"  ego energy={ego_e:.4f}  reward energy={rew_e:.4f}  IP(ego,rew)={ip0:.4f}")
    # Count how many k_vecs have nonzero components on multiple axes
    multi_axis = (k_vecs.abs() > 1e-6).sum(dim=1) > 1
    print(f"  Cross-axis frequency vectors: {multi_axis.sum().item()}/{K} "
          f"(these create the coupling)")
    print()

    n_values = [1, 5, 10, 20, 50]

    for N in n_values:
        print(f"{'='*85}")
        print(f"  N = {N} rollout steps ({N * DT:.3f}s lookahead)")
        print(f"{'='*85}")
        header = (f"{'St':>3} | {'ball_dy':>7} | {'Q(0)':>8} | {'dQ/da':>9} | "
                  f"{'oracle':>6} | {'match':>5} | {'Q(+1)':>8} | {'Q(-1)':>8} | "
                  f"{'Q(+1)-Q(-1)':>11}")
        print(header)
        print("-" * len(header))

        matches = 0
        nonzero = 0
        q_ranges = []

        for i, state in enumerate(states):
            r = check_gradients(state, k_vecs, N, action_val=0.0)
            q_range = r['Q_plus'] - r['Q_minus']
            q_ranges.append(abs(q_range))

            if abs(r['dQ_da']) > 1e-10:
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
