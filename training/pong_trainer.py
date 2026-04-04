"""
Minimal Pong RL trainer — numpy-only REINFORCE with value baseline.

Self-contained: Pong env + agent + training loop in one file.
Trains one agent against a perfect-tracker opponent.

Usage:
    python training/pong_trainer.py                    # Train with defaults
    python training/pong_trainer.py --episodes 5000    # More episodes
    python training/pong_trainer.py --hidden 32        # MLP policy
    python training/pong_trainer.py --seeds 5          # Multi-seed
"""

from __future__ import annotations
import argparse
import numpy as np

# -- Pong constants (from spectral_pong_viz.py) --------------------------------

COURT_LEFT, COURT_RIGHT = -5.0, 5.0
COURT_TOP, COURT_BOTTOM = 3.0, -3.0
PADDLE_X_OFFSET = 0.5
PADDLE_WIDTH = 0.15
PADDLE_HEIGHT = 1.0
PADDLE_SPEED = 4.0
BALL_RADIUS = 0.15
BALL_SPEED = 2.0
SPIN_FACTOR = 2.0
DT = 1.0 / 60.0


# -- PongEnv -------------------------------------------------------------------

class PongEnv:
    """Minimal Pong with egocentric observations and fixed opponent.

    Agent controls the LEFT paddle. Opponent (right) uses a noisy tracker
    with configurable skill (0.0 = random, 1.0 = perfect).

    Observation (4-dim, all roughly in [-1, 1]):
        0: ball_y - paddle_y  (normalized by court height)
        1: ball_vx sign       (-1 = approaching, +1 = going away)
        2: ball_vy             (normalized by ball speed)
        3: ball_x              (normalized, -1 = at agent, +1 = far side)
    """

    OBS_DIM = 4

    def __init__(self, opp_skill: float = 0.5):
        """opp_skill: 0.0 = random, 1.0 = perfect tracker."""
        self.paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
        self.paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET
        self.half_h = PADDLE_HEIGHT / 2.0
        self.opp_skill = opp_skill
        self.reset()

    def reset(self) -> np.ndarray:
        self.ball_x = 0.0
        self.ball_y = np.random.uniform(COURT_BOTTOM * 0.6, COURT_TOP * 0.6)
        angle = np.random.uniform(-1.0, 1.0)
        self.ball_vx = BALL_SPEED * np.cos(angle)
        self.ball_vy = BALL_SPEED * np.sin(angle)
        # Random initial direction
        if np.random.random() < 0.5:
            self.ball_vx = -abs(self.ball_vx)
        else:
            self.ball_vx = abs(self.ball_vx)
        self.agent_y = 0.0
        self.opp_y = 0.0
        self.touched = False
        return self._obs()

    def _obs(self) -> np.ndarray:
        """Egocentric observation for the left (agent) paddle."""
        return np.array([
            (self.ball_y - self.agent_y) / (COURT_TOP - COURT_BOTTOM),
            -1.0 if self.ball_vx < 0 else 1.0,  # -1 = approaching agent
            self.ball_vy / BALL_SPEED,
            (self.ball_x - COURT_LEFT) / (COURT_RIGHT - COURT_LEFT) * 2 - 1,
        ])

    def step(self, action: float):
        """Returns (obs, reward, done). Action in [-1, 1] moves paddle."""
        # Agent paddle
        action = np.clip(action, -1.0, 1.0)
        self.agent_y += action * PADDLE_SPEED * DT
        self.agent_y = np.clip(self.agent_y,
                               COURT_BOTTOM + self.half_h,
                               COURT_TOP - self.half_h)

        # Opponent: noisy tracker (skill controls tracking vs random)
        opp_target = self.ball_y
        track_action = np.clip((opp_target - self.opp_y) * 5.0, -1.0, 1.0)
        random_action = np.random.uniform(-1.0, 1.0)
        opp_action = self.opp_skill * track_action + (1 - self.opp_skill) * random_action
        self.opp_y += opp_action * PADDLE_SPEED * DT
        self.opp_y = np.clip(self.opp_y,
                             COURT_BOTTOM + self.half_h,
                             COURT_TOP - self.half_h)

        # Ball physics
        self.ball_x += self.ball_vx * DT
        self.ball_y += self.ball_vy * DT

        # Wall bounce
        if self.ball_y >= COURT_TOP - BALL_RADIUS:
            self.ball_y = 2 * (COURT_TOP - BALL_RADIUS) - self.ball_y
            self.ball_vy = -abs(self.ball_vy)
        elif self.ball_y <= COURT_BOTTOM + BALL_RADIUS:
            self.ball_y = 2 * (COURT_BOTTOM + BALL_RADIUS) - self.ball_y
            self.ball_vy = abs(self.ball_vy)

        # Agent paddle collision (left)
        if (self.ball_vx < 0
                and self.ball_x - BALL_RADIUS <= self.paddle_lx + PADDLE_WIDTH / 2
                and abs(self.ball_y - self.agent_y) <= self.half_h + BALL_RADIUS):
            self.ball_vx = abs(self.ball_vx)
            offset = (self.ball_y - self.agent_y) / (self.half_h + BALL_RADIUS)
            self.ball_vy += offset * SPIN_FACTOR
            spd = np.hypot(self.ball_vx, self.ball_vy)
            self.ball_vx *= BALL_SPEED / spd
            self.ball_vy *= BALL_SPEED / spd
            self.touched = True

        # Opponent paddle collision (right)
        if (self.ball_vx > 0
                and self.ball_x + BALL_RADIUS >= self.paddle_rx - PADDLE_WIDTH / 2
                and abs(self.ball_y - self.opp_y) <= self.half_h + BALL_RADIUS):
            self.ball_vx = -abs(self.ball_vx)
            offset = (self.ball_y - self.opp_y) / (self.half_h + BALL_RADIUS)
            self.ball_vy += offset * SPIN_FACTOR
            spd = np.hypot(self.ball_vx, self.ball_vy)
            self.ball_vx *= BALL_SPEED / spd
            self.ball_vy *= BALL_SPEED / spd

        # Scoring
        if self.ball_x < COURT_LEFT:
            return self._obs(), -1.0, True   # Agent missed
        if self.ball_x > COURT_RIGHT:
            return self._obs(), +1.0, True   # Agent scored (opponent missed)

        return self._obs(), 0.0, False


# -- REINFORCE Agent -----------------------------------------------------------

class REINFORCEAgent:
    """REINFORCE with learned value baseline. Numpy-only.

    Policy: action ~ N(mean, std^2), mean = w @ obs + b  (or MLP if hidden > 0)
    Baseline: V(s) = w_v @ obs + b_v  (always linear)
    Update: w += lr * advantage * grad_log_pi
    """

    def __init__(self, obs_dim: int = 4, hidden: int = 0,
                 lr: float = 1e-3, lr_baseline: float = 1e-2,
                 gamma: float = 0.99, std: float = 0.5):
        self.gamma = gamma
        self.std = std
        self.lr = lr
        self.lr_baseline = lr_baseline
        self.hidden = hidden

        if hidden > 0:
            # MLP: obs → hidden → 1
            self.W1 = np.random.randn(hidden, obs_dim) * np.sqrt(2.0 / obs_dim)
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden) * 0.01
            self.b2 = 0.0
        else:
            # Linear: obs → 1
            self.w = np.zeros(obs_dim)
            self.b = 0.0

        # Linear value baseline
        self.w_v = np.zeros(obs_dim)
        self.b_v = 0.0

        # Episode buffer
        self._obs_buf = []
        self._act_buf = []
        self._rew_buf = []
        # MLP internals saved per step (for gradient)
        self._h_buf = []
        self._mask_buf = []

        # Running return stats for normalization
        self._ret_mean = 0.0
        self._ret_var = 1.0
        self._ret_count = 0

    def _policy_forward(self, obs: np.ndarray) -> float:
        """Compute policy mean."""
        if self.hidden > 0:
            h = obs @ self.W1.T + self.b1
            mask = (h > 0).astype(np.float64)
            h_relu = h * mask
            mean = float(self.W2 @ h_relu + self.b2)
            self._last_h = h_relu
            self._last_mask = mask
        else:
            mean = float(self.w @ obs + self.b)
        return mean

    def act(self, obs: np.ndarray) -> float:
        mean = self._policy_forward(obs)
        action = mean + self.std * np.random.randn()
        self._obs_buf.append(obs.copy())
        self._act_buf.append(action)
        if self.hidden > 0:
            self._h_buf.append(self._last_h.copy())
            self._mask_buf.append(self._last_mask.copy())
        return float(action)

    def record_reward(self, reward: float):
        self._rew_buf.append(reward)

    def end_episode(self):
        """Compute returns, update policy and baseline."""
        if not self._rew_buf:
            return 0.0

        T = len(self._rew_buf)
        obs = np.array(self._obs_buf[:T])
        actions = np.array(self._act_buf[:T])
        rewards = np.array(self._rew_buf)

        # Monte Carlo returns
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Normalize returns (running statistics)
        self._ret_count += T
        batch_mean = returns.mean()
        batch_var = returns.var() + 1e-8
        alpha = min(T / self._ret_count, 0.1)
        self._ret_mean = (1 - alpha) * self._ret_mean + alpha * batch_mean
        self._ret_var = (1 - alpha) * self._ret_var + alpha * batch_var
        returns_norm = (returns - self._ret_mean) / (np.sqrt(self._ret_var) + 1e-8)

        # Value baseline
        values = obs @ self.w_v + self.b_v
        advantages = returns_norm - values

        # Update baseline (MSE gradient)
        for t in range(T):
            grad = advantages[t]  # d/d_params (return - V)^2 ∝ advantage * obs
            self.w_v += self.lr_baseline * grad * obs[t]
            self.b_v += self.lr_baseline * grad

        # Clip baseline weights
        np.clip(self.w_v, -10.0, 10.0, out=self.w_v)
        self.b_v = np.clip(self.b_v, -10.0, 10.0)

        # Policy gradient: d_log_pi = (action - mean) / std^2
        # Accumulate then apply averaged gradient (prevents blowup on long eps)
        if self.hidden > 0:
            gW1 = np.zeros_like(self.W1)
            gb1 = np.zeros_like(self.b1)
            gW2 = np.zeros_like(self.W2)
            gb2 = 0.0
            for t in range(T):
                h_relu = self._h_buf[t]
                mask = self._mask_buf[t]
                mean = float(self.W2 @ h_relu + self.b2)
                d_logpi = (actions[t] - mean) / (self.std ** 2)
                adv = advantages[t]
                gW2 += adv * d_logpi * h_relu
                gb2 += adv * d_logpi
                d_h = d_logpi * self.W2 * mask
                gW1 += adv * np.outer(d_h, obs[t])
                gb1 += adv * d_h
            # Apply averaged gradient
            self.W1 += self.lr * gW1 / T
            self.b1 += self.lr * gb1 / T
            self.W2 += self.lr * gW2 / T
            self.b2 += self.lr * gb2 / T
            np.clip(self.W1, -10.0, 10.0, out=self.W1)
            np.clip(self.W2, -10.0, 10.0, out=self.W2)
            self.b1 = np.clip(self.b1, -10.0, 10.0)
            self.b2 = np.clip(self.b2, -10.0, 10.0)
        else:
            gw = np.zeros_like(self.w)
            gb = 0.0
            for t in range(T):
                mean = float(self.w @ obs[t] + self.b)
                d_logpi = (actions[t] - mean) / (self.std ** 2)
                adv = advantages[t]
                gw += adv * d_logpi * obs[t]
                gb += adv * d_logpi
            self.w += self.lr * gw / T
            self.b += self.lr * gb / T
            np.clip(self.w, -10.0, 10.0, out=self.w)
            self.b = np.clip(self.b, -10.0, 10.0)

        # Clear buffers
        ep_return = returns[0]
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()
        self._h_buf.clear()
        self._mask_buf.clear()
        return float(ep_return)


# -- Training loop -------------------------------------------------------------

def train(n_episodes: int = 2000, hidden: int = 0, lr: float = 1e-3,
          lr_baseline: float = 1e-2, gamma: float = 0.99, std: float = 0.5,
          seed: int = 0, verbose: bool = True, max_steps: int = 2000,
          opp_skill: float = 0.5):
    """Train agent, return per-episode results."""
    np.random.seed(seed)
    env = PongEnv(opp_skill=opp_skill)
    agent = REINFORCEAgent(obs_dim=PongEnv.OBS_DIM, hidden=hidden,
                           lr=lr, lr_baseline=lr_baseline,
                           gamma=gamma, std=std)

    results = []  # (episode_length, reward, touched)
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, done = env.step(action)
            agent.record_reward(reward)
            ep_reward += reward
            if done:
                break
        ep_return = agent.end_episode()
        results.append({
            'length': step + 1,
            'reward': ep_reward,
            'touched': env.touched,
            'return': ep_return,
        })

        if verbose and (ep + 1) % 100 == 0:
            recent = results[-100:]
            wins = sum(1 for r in recent if r['reward'] > 0)
            losses = sum(1 for r in recent if r['reward'] < 0)
            touches = sum(1 for r in recent if r['touched'])
            avg_len = np.mean([r['length'] for r in recent])
            print(f'ep {ep+1:5d}: W/L={wins}/{losses} touches={touches}/100 '
                  f'avg_len={avg_len:.0f}')

    return results


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Pong REINFORCE trainer')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--hidden', type=int, default=0,
                        help='Hidden layer size (0 = linear policy)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-baseline', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--opp-skill', type=float, default=0.5,
                        help='Opponent skill: 0=random, 1=perfect tracker')
    args = parser.parse_args()

    for seed in range(args.seeds):
        if args.seeds > 1:
            print(f'\n=== Seed {seed} ===')
        results = train(n_episodes=args.episodes, hidden=args.hidden,
                        lr=args.lr, lr_baseline=args.lr_baseline,
                        gamma=args.gamma, std=args.std, seed=seed,
                        max_steps=args.max_steps, opp_skill=args.opp_skill)
        # Final summary
        last_200 = results[-200:] if len(results) >= 200 else results
        wins = sum(1 for r in last_200 if r['reward'] > 0)
        losses = sum(1 for r in last_200 if r['reward'] < 0)
        touches = sum(1 for r in last_200 if r['touched'])
        print(f'\nFinal (last {len(last_200)} eps): W/L={wins}/{losses} '
              f'touches={touches}/{len(last_200)}')


if __name__ == '__main__':
    main()
