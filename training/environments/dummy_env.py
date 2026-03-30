"""
Dummy Gymnasium Environment for Testing
========================================
Drop-in replacement for BaselineGymEnv that requires no rlgym-sim
or RocketSim. Produces synthetic observations with the same shape
and API, suitable for automated testing and CI.

Observation: (T_WINDOW * N_TOKENS * TOKEN_FEATURES,) = (800,) float32
Action:      (8,) float32 — 5 analog [-1,1] + 3 binary

The env simulates a simplified 'ball pursuit' dynamic:
  - Ball drifts with random velocity
  - Agent car moves toward ball based on action throttle/steer
  - Goals scored when ball crosses y threshold
  - Sparse reward: +1 blue goal, -1 orange goal
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))

from encoder import TOKEN_FEATURES, N_TOKENS

# Field dimensions (normalized to [-1, 1] in observations)
_FIELD_X = 4096.0
_FIELD_Y = 5120.0
_CEILING_Z = 2044.0
_GOAL_Y_NORM = 1.0  # normalized goal threshold


class DummyEnv(gym.Env):
    """
    Synthetic environment matching BaselineGymEnv's observation/action contract.

    No physics simulation — uses simple kinematic updates for fast testing.
    Goals occur stochastically or when the ball crosses the y-boundary.

    Parameters
    ----------
    t_window : int
        Number of frames to stack.
    max_steps : int
        Steps before episode timeout.
    goal_prob : float
        Per-step probability of a random goal event (adds stochasticity).
    reward_type : str
        'sparse' (default) or 'dense'. Dense adds small shaping rewards.
    dense_reward_weights : dict or None
        Ignored (kept for API compatibility).
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        t_window: int = 8,
        max_steps: int = 300,
        goal_prob: float = 0.005,
        reward_type: str = 'sparse',
        dense_reward_weights: dict | None = None,
        tick_skip: int = 8,
    ):
        super().__init__()
        self.t_window = t_window
        self.max_steps = max_steps
        self.goal_prob = goal_prob
        self.reward_type = reward_type

        obs_size = t_window * N_TOKENS * TOKEN_FEATURES
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32,
        )

        self._frame_buf: deque = deque(maxlen=t_window)
        self._orange_buf: deque = deque(maxlen=t_window)
        self._step_count = 0
        self._rng = np.random.default_rng()

        # Opponent (for API compat)
        self._opponent_encoder = None
        self._opponent_policy = None

        # Simple state: ball position + velocity, car position
        self._ball_pos = np.zeros(3, dtype=np.float32)
        self._ball_vel = np.zeros(3, dtype=np.float32)
        self._car_pos = np.zeros(3, dtype=np.float32)
        self._opp_pos = np.zeros(3, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0

        # Random initial positions (normalized)
        self._ball_pos = self._rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        self._ball_vel = self._rng.uniform(-0.1, 0.1, size=3).astype(np.float32)
        self._car_pos = np.array([0.0, -0.5, 0.0], dtype=np.float32)
        self._opp_pos = np.array([0.0, 0.5, 0.0], dtype=np.float32)

        tokens = self._build_tokens()
        self._frame_buf.clear()
        self._orange_buf.clear()
        for _ in range(self.t_window):
            self._frame_buf.append(tokens.copy())
            self._orange_buf.append(self._build_orange_tokens().copy())

        obs = self._get_stacked_obs()
        return obs, {'orange_obs': self._get_stacked_orange_obs()}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._step_count += 1

        # Simple dynamics
        throttle, steer = action[0], action[1]
        self._car_pos[0] = np.clip(self._car_pos[0] + steer * 0.02, -1.0, 1.0)
        self._car_pos[1] = np.clip(self._car_pos[1] + throttle * 0.02, -1.0, 1.0)

        self._ball_pos += self._ball_vel * 0.1
        self._ball_vel += self._rng.uniform(-0.01, 0.01, size=3).astype(np.float32)
        self._ball_pos = np.clip(self._ball_pos, -1.0, 1.0)
        self._ball_vel = np.clip(self._ball_vel, -0.3, 0.3)

        # Opponent drifts randomly
        self._opp_pos += self._rng.uniform(-0.01, 0.01, size=3).astype(np.float32)
        self._opp_pos = np.clip(self._opp_pos, -1.0, 1.0)

        # Goal detection
        goal = 0
        terminated = False
        if self._ball_pos[1] > _GOAL_Y_NORM * 0.95:
            goal = 1
            terminated = True
        elif self._ball_pos[1] < -_GOAL_Y_NORM * 0.95:
            goal = -1
            terminated = True
        elif self._rng.random() < self.goal_prob:
            goal = self._rng.choice([-1, 1])
            terminated = True

        # Timeout
        timed_out = self._step_count >= self.max_steps
        done = terminated or timed_out

        # Reward
        if self.reward_type == 'sparse':
            reward = float(goal) if terminated else 0.0
        else:
            # Simple dense: proximity to ball
            dist = np.linalg.norm(self._ball_pos[:2] - self._car_pos[:2])
            reward = float(goal) if terminated else -dist * 0.01

        # Build observation
        tokens = self._build_tokens()
        self._frame_buf.append(tokens)
        self._orange_buf.append(self._build_orange_tokens())

        obs = self._get_stacked_obs()
        info = {
            'goal': goal,
            'orange_obs': self._get_stacked_orange_obs(),
            'orange_action': self._rng.uniform(-1, 1, size=8).astype(np.float32),
            'orange_reward': -reward,
        }
        return obs, reward, done, False, info

    def get_opponent_obs(self) -> np.ndarray:
        """Return current stacked orange observation for external inference."""
        return self._get_stacked_orange_obs()

    def step_with_opponent_action(
        self, action: np.ndarray, opp_action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step with externally-computed opponent action. DummyEnv ignores
        opponent actions, so this delegates to step()."""
        return self.step(action)

    def close(self) -> None:
        pass

    def load_ppo_opponent(self, snap_path: str) -> None:
        """API-compatible opponent loading (no-op for dummy env)."""
        pass

    # ── internal ────────────────────────────────────────────────────────

    def _build_tokens(self) -> np.ndarray:
        """Build (N_TOKENS, TOKEN_FEATURES) observation for blue agent."""
        tokens = np.zeros((N_TOKENS, TOKEN_FEATURES), dtype=np.float32)

        # Token 0: ball
        tokens[0, :3] = self._ball_pos
        tokens[0, 3:6] = self._ball_vel

        # Token 1: own car
        tokens[1, :3] = self._car_pos
        tokens[1, 9] = self._rng.uniform(0, 1)  # boost

        # Token 2: opponent car
        tokens[2, :3] = self._opp_pos

        # Tokens 3-8: boost pads (static positions, random active flag)
        pad_positions = [
            [-0.8, -0.6, 0], [0.8, -0.6, 0], [-0.8, 0.6, 0],
            [0.8, 0.6, 0], [-0.8, 0.0, 0], [0.8, 0.0, 0],
        ]
        for i, pos in enumerate(pad_positions):
            tokens[3 + i, :3] = pos
            tokens[3 + i, 3] = float(self._rng.random() > 0.3)

        # Token 9: game state
        tokens[9, 0] = 0.0  # score diff
        tokens[9, 1] = 1.0 - self._step_count / self.max_steps  # time remaining
        tokens[9, 2] = 0.0  # overtime

        return tokens

    def _build_orange_tokens(self) -> np.ndarray:
        """Build (N_TOKENS, TOKEN_FEATURES) observation for orange agent (mirrored)."""
        tokens = self._build_tokens()
        # Swap own/opp car tokens
        tokens[1], tokens[2] = tokens[2].copy(), tokens[1].copy()
        # Flip y-axis for orange perspective
        tokens[:, 1] *= -1
        tokens[:, 4] *= -1
        return tokens

    def _get_stacked_obs(self) -> np.ndarray:
        stacked = np.stack(list(self._frame_buf), axis=0)
        return stacked.ravel().astype(np.float32)

    def _get_stacked_orange_obs(self) -> np.ndarray:
        stacked = np.stack(list(self._orange_buf), axis=0)
        return stacked.ravel().astype(np.float32)
