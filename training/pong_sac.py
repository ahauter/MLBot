"""
CleanRL-style SAC for Pong — trains both raw and spectral encoders.

Single-file implementation: ReplayBuffer, Actor (squashed Gaussian),
twin Q-critics, automatic entropy tuning, per-step TD updates.

Usage:
    python training/pong_sac.py --obs-mode raw --reward-mode paddle
    python training/pong_sac.py --obs-mode spectral --reward-mode goal
    python training/pong_sac.py --total-steps 500000 --obs-mode spectral
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

training_dir = os.path.dirname(os.path.abspath(__file__))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from pong_trainer import PongEnv


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular replay buffer with pre-allocated numpy arrays."""

    def __init__(self, obs_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )


# ---------------------------------------------------------------------------
# Actor — Squashed Gaussian Policy
# ---------------------------------------------------------------------------

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """Reparameterized sample with log-prob (tanh squashing)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # Reparameterization trick
        u = normal.rsample()
        action = torch.tanh(u)
        # Log-prob with tanh correction
        log_prob = normal.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def get_action(self, obs, device):
        """Get action for env interaction (no grad)."""
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
        action, _ = self.sample(obs_t)
        return action.cpu().numpy().flatten()[0]


# ---------------------------------------------------------------------------
# Twin Q-Networks
# ---------------------------------------------------------------------------

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(obs_dim + 1, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_out = nn.Linear(hidden, 1)
        # Q2
        self.q2_fc1 = nn.Linear(obs_dim + 1, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_out = nn.Linear(hidden, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment
    env = PongEnv(
        opp_skill=args.opp_skill,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
    )
    obs_dim = env.obs_dim

    # Networks
    actor = Actor(obs_dim, args.hidden).to(device)
    critic = SoftQNetwork(obs_dim, args.hidden).to(device)
    target_critic = SoftQNetwork(obs_dim, args.hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())

    # Optimizers
    actor_opt = Adam(actor.parameters(), lr=args.lr)
    critic_opt = Adam(critic.parameters(), lr=args.lr)

    # Entropy tuning
    target_entropy = -1.0  # -action_dim
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = Adam([log_alpha], lr=args.alpha_lr)
    alpha = log_alpha.exp().item()

    # Replay buffer
    buffer = ReplayBuffer(obs_dim, args.buffer_size)

    # Episode tracking
    obs = env.reset()
    ep_reward = 0.0
    ep_len = 0
    ep_touches = 0
    episode_count = 0

    # Metrics for logging window
    recent_wins = []
    recent_losses = []
    recent_lengths = []
    recent_touches = []
    recent_returns = []
    last_critic_loss = 0.0
    last_actor_loss = 0.0
    last_alpha = alpha
    last_q_mean = 0.0
    last_entropy = 0.0

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
        ep_touches = env.agent_touches

        # Store transition
        buffer.add(obs, action, reward, next_obs, done)

        if done:
            # Record episode stats
            recent_wins.append(1 if ep_reward > 0 else 0)
            recent_losses.append(1 if ep_reward < 0 else 0)
            recent_lengths.append(ep_len)
            recent_touches.append(ep_touches)
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

            # Critic update
            q1_pred, q2_pred = critic(b_obs, b_act)
            critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # Actor update
            new_action, log_prob = actor.sample(b_obs)
            q1_new, q2_new = critic(b_obs, new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * log_prob - q_new).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Entropy coefficient update
            alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()
            alpha = log_alpha.exp().item()

            # Target network soft update
            with torch.no_grad():
                for p, tp in zip(critic.parameters(),
                                 target_critic.parameters()):
                    tp.data.lerp_(p.data, args.tau)

            last_critic_loss = critic_loss.item()
            last_actor_loss = actor_loss.item()
            last_alpha = alpha
            last_q_mean = q1_pred.mean().item()
            last_entropy = -log_prob.mean().item()

        # Logging
        if step % args.eval_interval == 0 and recent_wins:
            n = len(recent_wins)
            wins = sum(recent_wins)
            losses = sum(recent_losses)
            avg_len = np.mean(recent_lengths)
            avg_touches = np.mean(recent_touches)
            avg_ret = np.mean(recent_returns)
            elapsed = time.time() - t0
            sps = step / elapsed

            print(f'step {step:>7d} | '
                  f'ep {episode_count:>5d} | '
                  f'W/L={wins}/{losses} ({wins/n:.0%}) | '
                  f'touches={avg_touches:.1f} | '
                  f'len={avg_len:.0f} | '
                  f'ret={avg_ret:.2f} | '
                  f'Qloss={last_critic_loss:.3f} | '
                  f'Aloss={last_actor_loss:.3f} | '
                  f'α={last_alpha:.3f} | '
                  f'Q={last_q_mean:.2f} | '
                  f'H={last_entropy:.2f} | '
                  f'buf={buffer.size} | '
                  f'{sps:.0f} sps', flush=True)

            recent_wins.clear()
            recent_losses.clear()
            recent_lengths.clear()
            recent_touches.clear()
            recent_returns.clear()

    total_time = time.time() - t0
    print(f'\nDone: {args.total_steps} steps in {total_time:.0f}s '
          f'({args.total_steps/total_time:.0f} sps)', flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Pong SAC trainer')
    parser.add_argument('--total-steps', type=int, default=200_000)
    parser.add_argument('--obs-mode', choices=['raw', 'spectral'], default='raw')
    parser.add_argument('--reward-mode', choices=['goal', 'paddle'], default='paddle')
    parser.add_argument('--opp-skill', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=100_000)
    parser.add_argument('--random-steps', type=int, default=5000)
    parser.add_argument('--update-every', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=5000)
    args = parser.parse_args()

    for seed in range(args.seeds):
        if args.seeds > 1:
            print(f'\n=== Seed {seed} ===')
        args.seed = seed
        train(args)


if __name__ == '__main__':
    main()
