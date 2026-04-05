"""
Pong Environment + Gymnasium Wrapper
=====================================
Contains the core PongEnv (game physics, wavepacket tracking, observations)
and PongGymEnv (gymnasium.Env wrapper for the training framework).

Observation modes:
  - 'raw':      (4,)    float32 — egocentric ball/paddle features
  - 'spectral': (2304,) float32 — raw 6×16×24 wavepacket feature maps
                 (flattened; GPU conv encoder in the algorithm handles processing)

Action: (8,) float32 — only action[0] is used (paddle movement [-1, 1])

Usage (YAML config)
-------------------
    env_class: training.environments.pong_env.PongGymEnv
    env_params:
      obs_mode: raw       # or 'spectral'
      opp_skill: 0.0      # 0=random, 1=perfect tracker
      reward_mode: paddle  # or 'goal'
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from training.spectral.wavepacket import (
    WavepacketObject2D, compute_feature_maps,
    FM_NX, FM_NY, FM_CHANNELS,
    COURT_LEFT, COURT_RIGHT, COURT_TOP, COURT_BOTTOM,
    PADDLE_X_OFFSET, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED,
    BALL_RADIUS, BALL_SPEED, SPIN_FACTOR,
)

DT = 1.0 / 60.0


# ---------------------------------------------------------------------------
# Strided Conv Feature Extractor (numpy, used only for legacy obs_mode path)
# ---------------------------------------------------------------------------

class StridedConvExtractor:
    """Two-layer strided conv: (6, 16, 24) -> (8, 7, 11) -> (8, 3, 5) -> 120-dim.

    Filters are He-initialized random projections (not learned).
    """

    def __init__(self, n_channels: int = 6, n_filters: int = 8,
                 kernel_size: int = 3, stride: int = 2, seed: int = 0):
        rng = np.random.RandomState(seed)
        ks = kernel_size
        fan_in1 = n_channels * ks * ks
        self.W1 = rng.randn(n_filters, n_channels, ks, ks) * np.sqrt(2.0 / fan_in1)
        self.b1 = np.zeros(n_filters)
        fan_in2 = n_filters * ks * ks
        self.W2 = rng.randn(n_filters, n_filters, ks, ks) * np.sqrt(2.0 / fan_in2)
        self.b2 = np.zeros(n_filters)
        self.stride = stride
        self.ks = ks
        oH1 = (16 - ks) // stride + 1
        oW1 = (24 - ks) // stride + 1
        oH2 = (oH1 - ks) // stride + 1
        oW2 = (oW1 - ks) // stride + 1
        self.out_dim = n_filters * oH2 * oW2

    def _strided_conv2d(self, x, W, b, stride):
        F, C, ks, _ = W.shape
        _, H, Wx = x.shape
        oH = (H - ks) // stride + 1
        oW = (Wx - ks) // stride + 1
        patches = np.lib.stride_tricks.as_strided(
            x,
            shape=(oH, oW, C, ks, ks),
            strides=(x.strides[1] * stride, x.strides[2] * stride,
                     x.strides[0], x.strides[1], x.strides[2])
        ).reshape(oH * oW, C * ks * ks)
        W_flat = W.reshape(F, -1)
        out = (patches @ W_flat.T + b).reshape(oH, oW, F).transpose(2, 0, 1)
        return out

    def forward(self, fmaps: np.ndarray) -> np.ndarray:
        h1 = self._strided_conv2d(fmaps, self.W1, self.b1, self.stride)
        h1 = np.maximum(h1, 0)
        h2 = self._strided_conv2d(h1, self.W2, self.b2, self.stride)
        h2 = np.maximum(h2, 0)
        return h2.ravel()


# ---------------------------------------------------------------------------
# PongEnv — core game physics + wavepacket tracking
# ---------------------------------------------------------------------------

class PongEnv:
    """Minimal Pong with configurable reward and observation modes.

    Agent controls the LEFT paddle. Opponent (right) uses a noisy tracker.

    Raw observation (4-dim, all roughly in [-1, 1]):
        0: (ball_y - paddle_y) / court_height
        1: ball_vx sign  (-1 = approaching agent, +1 = going away)
        2: ball_vy / ball_speed
        3: ball_x normalized  (-1 = agent side, +1 = opponent side)

    Spectral mode initializes the wavepacket system for LMS tracking.
    """

    def __init__(self, opp_skill: float = 0.0, reward_mode: str = 'goal',
                 obs_mode: str = 'raw', lr_k_env: float = 0.0,
                 lr_k: float = 0.001, lr_c_pred: float = 0.05):
        self.paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
        self.paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET
        self.half_h = PADDLE_HEIGHT / 2.0
        self.opp_skill = opp_skill
        self.reward_mode = reward_mode
        self.obs_mode = obs_mode
        self._lr_k_env = lr_k_env
        self._lr_k = lr_k
        self._lr_c_pred = lr_c_pred

        if obs_mode == 'spectral':
            self._init_spectral()

    @property
    def obs_dim(self) -> int:
        if self.obs_mode == 'spectral':
            return self._spectral_dim
        return 4

    def _init_spectral(self):
        """Initialize wavepacket system and strided conv extractor."""
        K = 8
        NDIM = 3
        self._K = K
        self._NDIM = NDIM
        self._freqs = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])

        # Feature map grids
        self._x_fm = np.linspace(COURT_LEFT, COURT_RIGHT, FM_NX)
        self._y_fm = np.linspace(COURT_BOTTOM, COURT_TOP, FM_NY)
        self._r_fm = np.linspace(-1.0, 1.0, FM_NY)

        # Strided conv: (6, 16, 24) -> 120-dim (used only for legacy obs path)
        self._strided_conv = StridedConvExtractor(n_channels=FM_CHANNELS, seed=0)
        self._spectral_dim = self._strided_conv.out_dim

    def _create_wavepackets(self):
        """Create all 6 wavepackets for a new episode."""
        WP = WavepacketObject2D
        K, NDIM, freqs = self._K, self._NDIM, self._freqs

        self._wp_ball = WP(K, freqs.copy(),
                           pos0=(self.ball_x, self.ball_y, 0),
                           sigma=0.8, ndim=NDIM, lr=0.15, lr_tracking=0.01)
        self._wp_pl = WP(K, freqs.copy(),
                         pos0=(self.paddle_lx, self.agent_y, 0),
                         mass=1e6, sigma=0.5, amplitude=1.0,
                         ndim=NDIM, lr=0.1, lr_tracking=0.02)
        self._wp_pr = WP(K, freqs.copy(),
                         pos0=(self.paddle_rx, self.opp_y, 0),
                         mass=1e6, sigma=0.5, amplitude=1.0,
                         ndim=NDIM, lr=0.1, lr_tracking=0.02)

        env_c = np.zeros((K, NDIM))
        env_c[0, 0] = 1.0
        env_c[1, 1] = 1.0
        self._wp_env = WP(K, freqs.copy(), pos0=(0, 0, 0), mass=1e6,
                          ndim=NDIM, c_cos=env_c,
                          c_sin=np.zeros((K, NDIM)),
                          lr=0.15, lr_tracking=0.0)

        rew_c = np.zeros((K, NDIM))
        rew_c[0, 2] = 1.0
        self._wp_reward = WP(K, freqs.copy(), pos0=(0, 0, 0), mass=1e6,
                             ndim=NDIM, c_cos=rew_c.copy(),
                             c_sin=np.zeros((K, NDIM)),
                             lr=0.15, lr_tracking=0.0)

    def _update_wavepackets(self, scored: bool = False, reward: float = 0.0):
        """Full wavepacket update: predict -> shift -> correct -> deviation -> env learn -> reward learn."""
        NDIM = self._NDIM
        ball_pos = np.array([self.ball_x, self.ball_y, 0.0])
        pad_l_pos = np.array([self.paddle_lx, self.agent_y, 0.0])
        pad_r_pos = np.array([self.paddle_rx, self.opp_y, 0.0])

        wp_ball = self._wp_ball
        wp_pl = self._wp_pl
        wp_pr = self._wp_pr
        wp_env = self._wp_env

        # 1. PREDICT
        if not scored:
            nip_env = abs(wp_ball.normalized_inner_product(wp_env))
            nip_padL = abs(wp_ball.normalized_inner_product(wp_pl))
            nip_padR = abs(wp_ball.normalized_inner_product(wp_pr))
            ball_vel = np.array([self.ball_vx, self.ball_vy, 0.0])
            force_env  = wp_ball.predict_force(wp_env,  nip_env,  force_scale=0.3)
            force_padL = wp_ball.predict_force(wp_pl,   nip_padL, force_scale=0.5)
            force_padR = wp_ball.predict_force(wp_pr,   nip_padR, force_scale=0.5)
            total_force = force_env + force_padL + force_padR
            predicted_pos = wp_ball.predict_position(ball_vel, DT, total_force)
        else:
            nip_env = nip_padL = nip_padR = 0.0
            predicted_pos = ball_pos.copy()

        # 2. SHIFT wavepackets by velocity
        wp_ball.shift(self.ball_vx * DT, axis=0)
        wp_ball.shift(self.ball_vy * DT, axis=1)
        delta_l = self.agent_y - wp_pl.pos[1]
        delta_r = self.opp_y - wp_pr.pos[1]
        if abs(delta_l) > 1e-12:
            wp_pl.shift(delta_l, axis=1)
        if abs(delta_r) > 1e-12:
            wp_pr.shift(delta_r, axis=1)

        # 3. CORRECT: LMS toward observed positions — capture residual
        if not scored:
            unity = np.ones(NDIM)
            ball_residual = wp_ball.update_with_attention(
                ball_pos, unity, [nip_env, nip_padL, nip_padR])
            wp_pl.update_with_attention(pad_l_pos, unity, [nip_padL])
            wp_pr.update_with_attention(pad_r_pos, unity, [nip_padR])
            self._last_residual = float(np.linalg.norm(ball_residual))
            self._episode_residual_sum += self._last_residual
            self._episode_residual_count += 1
        else:
            self._last_residual = 0.0

        # 3b. Amplitude update from prediction residual
        # Teach the field to encode the prediction error at each location so
        # that predict_force produces corrective forces in future steps.
        if not scored and self._lr_c_pred > 0:
            pred_residual = ball_pos - predicted_pos
            pred_mag = np.linalg.norm(pred_residual[:2])
            if pred_mag > 1e-8:
                wp_ball.update_lms(ball_pos,
                                   pred_residual / (pred_mag + 1e-8),
                                   anomaly_scale=pred_mag,
                                   lr=self._lr_c_pred)

        # 4. Deviation + normalize
        ball_dev = np.array([wp_ball.integrate_squared(d) - 1.0 for d in range(NDIM)])
        wp_ball.normalize()
        wp_pl.normalize()
        wp_pr.normalize()

        # 5. Env learns from deviation
        dev_mag = np.linalg.norm(ball_dev[:2])
        if not scored and dev_mag > 1e-8:
            total_nip = nip_env + nip_padL + nip_padR + 1e-8
            env_frac = nip_env / total_nip
            wp_env.update_lms(ball_pos, ball_dev, anomaly_scale=dev_mag * env_frac)
            wp_env.normalize()

        # 6. Reward wavepacket learns from reward signal
        if abs(reward) > 1e-8:
            wp_reward = self._wp_reward
            reward_pos = np.array([self.ball_x, self.ball_y, 0.0])
            reward_target = np.array([0.0, 0.0, reward])
            wp_reward.update_lms(reward_pos, reward_target, anomaly_scale=1.0)
            wp_reward.soft_normalize(max_energy=2.0)
            ball_reward_pos = np.array([self.ball_x, self.ball_y, reward])
            nip_rew = abs(wp_ball.normalized_inner_product(wp_reward))
            wp_ball.update_with_attention(ball_reward_pos, np.ones(NDIM), [nip_rew])
            wp_ball.normalize()
            wp_pl.update_with_attention(
                np.array([self.paddle_lx, self.agent_y, reward]), np.ones(NDIM), [1.0])
            wp_pr.update_with_attention(
                np.array([self.paddle_rx, self.opp_y, -reward]), np.ones(NDIM), [1.0])
            wp_pl.normalize()
            wp_pr.normalize()

        # 7. Frequency learning from prediction residual
        if not scored and self._lr_k > 0:
            wp_ball.learn_from_residual(predicted_pos, ball_pos, lr_k=self._lr_k)

        # 8. Store positions
        wp_ball.pos[:] = ball_pos
        wp_pl.pos[:] = pad_l_pos
        wp_pr.pos[:] = pad_r_pos

    def _compute_fmaps(self) -> np.ndarray:
        return compute_feature_maps(
            self._wp_ball, self._wp_env, self._wp_pl, self._wp_pr,
            self._wp_reward, self._x_fm, self._y_fm, self._r_fm)

    def _spectral_obs(self) -> np.ndarray:
        fmaps = self._compute_fmaps()
        return self._strided_conv.forward(fmaps)

    def get_feature_maps(self) -> np.ndarray:
        """Return raw (6, 16, 24) outer-product maps."""
        return self._compute_fmaps()

    def _raw_obs(self) -> np.ndarray:
        return np.array([
            (self.ball_y - self.agent_y) / (COURT_TOP - COURT_BOTTOM),
            -1.0 if self.ball_vx < 0 else 1.0,
            self.ball_vy / BALL_SPEED,
            (self.ball_x - COURT_LEFT) / (COURT_RIGHT - COURT_LEFT) * 2 - 1,
        ])

    def _obs(self) -> np.ndarray:
        if self.obs_mode == 'spectral':
            return self._spectral_obs()
        return self._raw_obs()

    def reset(self) -> np.ndarray:
        self.ball_x = 0.0
        self.ball_y = np.random.uniform(COURT_BOTTOM * 0.6, COURT_TOP * 0.6)
        angle = np.random.uniform(-1.0, 1.0)
        self.ball_vx = BALL_SPEED * np.cos(angle)
        self.ball_vy = BALL_SPEED * np.sin(angle)
        if np.random.random() < 0.5:
            self.ball_vx = -abs(self.ball_vx)
        else:
            self.ball_vx = abs(self.ball_vx)
        self.agent_y = 0.0
        self.opp_y = 0.0
        self.agent_touches = 0
        self.touched = False
        self._last_residual = 0.0
        self._episode_residual_sum = 0.0
        self._episode_residual_count = 0

        if self.obs_mode == 'spectral':
            self._create_wavepackets()

        return self._obs()

    def step(self, action: float):
        """Returns (obs, reward, done). Action in [-1, 1] moves paddle."""
        action = np.clip(action, -1.0, 1.0)
        self.agent_y += action * PADDLE_SPEED * DT
        self.agent_y = np.clip(self.agent_y,
                               COURT_BOTTOM + self.half_h,
                               COURT_TOP - self.half_h)

        # Opponent: noisy tracker
        opp_target = self.ball_y
        track_action = np.clip((opp_target - self.opp_y) * 5.0, -1.0, 1.0)
        random_action = np.random.uniform(-1.0, 1.0)
        opp_action = (self.opp_skill * track_action
                      + (1 - self.opp_skill) * random_action)
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

        agent_hit = False

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
            self.agent_touches += 1
            self.touched = True
            agent_hit = True

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

        agent_missed = self.ball_x < COURT_LEFT
        opp_missed = self.ball_x > COURT_RIGHT
        scored = agent_missed or opp_missed

        # Reward for wavepacket update
        if self.reward_mode == 'goal':
            step_reward = -1.0 if agent_missed else (1.0 if opp_missed else 0.0)
        else:
            step_reward = -1.0 if agent_missed else (1.0 if agent_hit else 0.0)

        # Obs before wavepacket update for spectral
        if self.obs_mode == 'spectral':
            pre_update_obs = self._spectral_obs()
            self._update_wavepackets(scored=scored, reward=step_reward)
        else:
            pre_update_obs = None

        obs_out = pre_update_obs if pre_update_obs is not None else self._obs()

        if self.reward_mode == 'goal':
            if agent_missed:
                return obs_out, -1.0, True
            if opp_missed:
                return obs_out, +1.0, True
            return obs_out, 0.0, False
        else:
            if agent_missed:
                return obs_out, -1.0, True
            if opp_missed:
                return obs_out, 0.0, True
            reward = 1.0 if agent_hit else 0.0
            return obs_out, reward, False


# ---------------------------------------------------------------------------
# PongGymEnv — gymnasium.Env wrapper for SubprocVecEnv
# ---------------------------------------------------------------------------

class PongGymEnv(gym.Env):
    """
    Gymnasium wrapper for Pong that works with SubprocVecEnv.

    The agent controls the left paddle via action[0] in [-1, 1].
    Remaining action dimensions (1-7) are ignored.

    For obs_mode='spectral', the env outputs raw 6-channel feature maps
    (flattened to 2304 dims). The trainable conv encoder lives in the
    algorithm on GPU.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        t_window: int = 1,
        reward_type: str = 'sparse',
        dense_reward_weights: Optional[dict] = None,
    ):
        super().__init__()
        self.t_window = t_window

        # Pong-specific params come through dense_reward_weights dict
        # (the only custom dict the framework passes to env constructors)
        params = dense_reward_weights or {}
        obs_mode = params.get('obs_mode', 'raw')
        opp_skill = params.get('opp_skill', 0.0)
        reward_mode = params.get('reward_mode', None)
        self.max_steps = params.get('max_steps', 2000)
        self.obs_mode = obs_mode

        if reward_mode is not None:
            self._reward_mode = reward_mode
        else:
            self._reward_mode = 'goal' if reward_type == 'sparse' else 'paddle'

        self._env = PongEnv(
            opp_skill=opp_skill,
            reward_mode=self._reward_mode,
            obs_mode=obs_mode if obs_mode == 'raw' else 'spectral',
            lr_k=params.get('lr_k', 0.001),
            lr_c_pred=params.get('lr_c_pred', 0.05),
        )

        if obs_mode == 'spectral':
            # 2304 feature map + 1 encoder residual appended
            obs_dim = FM_CHANNELS * FM_NY * FM_NX + 1
        else:
            obs_dim = self._env.obs_dim

        self._obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == 'spectral':
            fmaps = self._env.get_feature_maps()
            flat = fmaps.ravel().astype(np.float32)
            # Append LMS residual as last element
            residual = getattr(self._env, '_last_residual', 0.0)
            return np.append(flat, np.float32(residual))
        else:
            return self._env._raw_obs().astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._env.reset()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        paddle_action = float(action[0])
        obs_internal, reward, done = self._env.step(paddle_action)
        self._step_count += 1

        obs = self._get_obs()

        truncated = False
        if not done and self._step_count >= self.max_steps:
            truncated = True

        if done:
            goal = -1 if reward < 0 else 1
        else:
            goal = 0

        info = {
            'goal': goal,
            'touches': self._env.agent_touches,
        }

        return obs, float(reward), bool(done), truncated, info

    def step_with_opponent_action(self, action, opp_action):
        """Ignore external opponent action; PongEnv controls its own opponent."""
        return self.step(action)

    def close(self):
        pass

    def get_opponent_obs(self):
        return np.zeros(self._obs_dim, dtype=np.float32)

    def set_opponent_algo(self, algo):
        pass
