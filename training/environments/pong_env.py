"""
Gymnasium Environment Wrapper for Pong Experiments
===================================================
Wraps the PongEnv from pong_trainer.py as a standard gymnasium.Env
for use with the SubprocVecEnv training infrastructure.

Observation modes:
  - 'raw':      (4,)    float32 — egocentric ball/paddle features
  - 'spectral': (2304,) float32 — raw 6×16×24 wavepacket feature maps
                 (flattened; GPU conv encoder in the algorithm handles processing)

Action: (8,) float32 — only action[0] is used (paddle movement)

Usage (YAML config)
-------------------
    env_class: training.environments.pong_env.PongGymEnv
    env_params:
      obs_mode: raw       # or 'spectral'
      opp_skill: 0.0      # 0=random, 1=perfect tracker
      reward_mode: paddle  # or 'goal'
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

_TRAINING = Path(__file__).parent.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from pong_trainer import PongEnv


class PongGymEnv(gym.Env):
    """
    Gymnasium wrapper for Pong that works with SubprocVecEnv.

    The agent controls the left paddle via action[0] in [-1, 1].
    Remaining action dimensions (1-7) are ignored.

    For obs_mode='spectral', the env initializes the wavepacket system
    internally but outputs the raw 6-channel feature maps (flattened to
    2304 dims) instead of the numpy-convolved 120-dim vector. The trainable
    conv encoder lives in the algorithm on GPU.

    Parameters
    ----------
    t_window : int
        Ignored for pong (no frame stacking needed), kept for API compat.
    reward_type : str
        Maps to PongEnv reward_mode: 'sparse' -> 'goal', 'dense' -> 'paddle'.
    dense_reward_weights : dict or None
        Ignored, kept for API compatibility.
    obs_mode : str
        'raw' (4-dim) or 'spectral' (2304-dim raw feature maps).
    opp_skill : float
        Opponent skill level: 0.0 = random, 1.0 = perfect tracker.
    reward_mode : str or None
        Explicit override for reward mode. If provided, takes precedence
        over reward_type mapping.
    max_steps : int
        Maximum steps per episode before timeout.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        t_window: int = 1,
        reward_type: str = 'sparse',
        dense_reward_weights: Optional[dict] = None,
        obs_mode: str = 'raw',
        opp_skill: float = 0.0,
        reward_mode: Optional[str] = None,
        max_steps: int = 2000,
    ):
        super().__init__()
        self.t_window = t_window
        self.max_steps = max_steps
        self.obs_mode = obs_mode

        # Map framework reward_type to pong reward_mode
        if reward_mode is not None:
            self._reward_mode = reward_mode
        else:
            self._reward_mode = 'goal' if reward_type == 'sparse' else 'paddle'

        # For spectral: env uses obs_mode='spectral' internally for wavepacket
        # updates, but we extract raw feature maps ourselves
        self._env = PongEnv(
            opp_skill=opp_skill,
            reward_mode=self._reward_mode,
            obs_mode=obs_mode if obs_mode == 'raw' else 'spectral',
        )

        if obs_mode == 'spectral':
            # Raw feature maps: 6 channels × 16 height × 24 width = 2304
            obs_dim = 6 * 16 * 24
        else:
            obs_dim = self._env.obs_dim  # 4 for raw

        self._obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Framework uses 8-dim actions; only [0] is meaningful for pong
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        """Get observation in the appropriate mode."""
        if self.obs_mode == 'spectral':
            # Return raw 6×16×24 feature maps, flattened
            fmaps = self._env.get_feature_maps()  # (6, 16, 24)
            return fmaps.ravel().astype(np.float32)
        else:
            return self._env._raw_obs().astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._env.reset()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # Extract 1D paddle action from 8-dim framework action
        paddle_action = float(action[0])
        obs_internal, reward, done = self._env.step(paddle_action)
        self._step_count += 1

        # Get obs in our mode (raw feature maps for spectral)
        obs = self._get_obs()

        # Timeout handling
        truncated = False
        if not done and self._step_count >= self.max_steps:
            truncated = True

        # Goal info for eval hook compatibility
        # +1 = agent scored (opponent missed), -1 = agent missed, 0 = ongoing/timeout
        if done:
            goal = 1 if reward > 0 else -1
        else:
            goal = 0

        info = {
            'goal': goal,
            'touches': self._env.agent_touches,
        }

        # On episode end, include mean encoder residual (spectral mode only)
        if (done or truncated) and self.obs_mode == 'spectral':
            count = getattr(self._env, '_episode_residual_count', 0)
            if count > 0:
                info['encoder_residual'] = (
                    self._env._episode_residual_sum / count)
            else:
                info['encoder_residual'] = 0.0

        return (
            obs,
            float(reward),
            bool(done),
            truncated,
            info,
        )

    def close(self):
        pass

    # Compatibility methods expected by some env worker paths
    def get_opponent_obs(self):
        """Pong has a built-in opponent; return dummy obs."""
        return np.zeros(self._obs_dim, dtype=np.float32)

    def set_opponent_algo(self, algo):
        """No-op: pong uses built-in opponent."""
        pass
