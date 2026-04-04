"""
Minimal Pong RL trainer — numpy-only REINFORCE with value baseline.

Self-contained: Pong env + agent + training loop in one file.
Trains one agent against a configurable opponent.

Supports two reward modes:
  - 'goal':   +1 opponent miss, -1 agent miss (sparse, episode-terminal)
  - 'paddle': +1 agent paddle hit, -1 agent miss (frequent signal)

Supports two observation modes:
  - 'raw':      4-dim egocentric [ball_dy, ball_vx_sign, ball_vy, ball_x]
  - 'spectral': 8-dim from wavepacket outer-product maps → conv features

Usage:
    python training/pong_trainer.py --reward-mode paddle --obs-mode raw
    python training/pong_trainer.py --reward-mode goal --obs-mode spectral --hidden 32
"""

from __future__ import annotations
import argparse
import numpy as np
import sys
import os

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
    """Minimal Pong with configurable reward and observation modes.

    Agent controls the LEFT paddle. Opponent (right) uses a noisy tracker.

    Raw observation (4-dim, all roughly in [-1, 1]):
        0: (ball_y - paddle_y) / court_height
        1: ball_vx sign  (-1 = approaching agent, +1 = going away)
        2: ball_vy / ball_speed
        3: ball_x normalized  (-1 = agent side, +1 = opponent side)

    Spectral observation (8-dim):
        Conv features from wavepacket outer-product maps.
    """

    def __init__(self, opp_skill: float = 0.0, reward_mode: str = 'goal',
                 obs_mode: str = 'raw', pool_mode: str = 'avg'):
        self.paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
        self.paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET
        self.half_h = PADDLE_HEIGHT / 2.0
        self.opp_skill = opp_skill
        self.reward_mode = reward_mode
        self.obs_mode = obs_mode
        self._pool_mode = pool_mode

        if obs_mode == 'spectral':
            self._init_spectral()

    @property
    def obs_dim(self) -> int:
        return 8 if self.obs_mode == 'spectral' else 4

    def _init_spectral(self):
        """Initialize wavepacket objects and conv feature extractor."""
        # Add training dir to path for imports
        training_dir = os.path.dirname(os.path.abspath(__file__))
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        from spectral_pong_viz import (
            WavepacketObject2D, compute_feature_maps, ConvFeatureExtractor,
            FM_NX, FM_NY, FM_CHANNELS
        )
        self._compute_feature_maps = compute_feature_maps
        self._FM_NX = FM_NX
        self._FM_NY = FM_NY

        K = 8
        NDIM = 3
        self._K = K
        self._NDIM = NDIM
        self._freqs = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
        self._WavepacketObject2D = WavepacketObject2D

        # Feature map grids
        self._x_fm = np.linspace(COURT_LEFT, COURT_RIGHT, FM_NX)
        self._y_fm = np.linspace(COURT_BOTTOM, COURT_TOP, FM_NY)
        self._r_fm = np.linspace(-1.0, 1.0, FM_NY)

        # Conv feature extractor (fixed random projections, shared across resets)
        self._conv = ConvFeatureExtractor(
            n_channels=FM_CHANNELS, n_filters=8, kernel_size=3, seed=0)

    def _create_wavepackets(self):
        """Create fresh wavepackets for a new episode."""
        WP = self._WavepacketObject2D
        K, NDIM, freqs = self._K, self._NDIM, self._freqs

        self._wp_ball = WP(K, freqs, pos0=(self.ball_x, self.ball_y, 0),
                           sigma=0.8, ndim=NDIM, lr=0.15, lr_tracking=0.01)
        self._wp_pl = WP(K, freqs,
                         pos0=(self.paddle_lx, self.agent_y, 0),
                         mass=1e6, sigma=0.5, amplitude=1.0,
                         ndim=NDIM, lr=0.1, lr_tracking=0.02)
        self._wp_pr = WP(K, freqs,
                         pos0=(self.paddle_rx, self.opp_y, 0),
                         mass=1e6, sigma=0.5, amplitude=1.0,
                         ndim=NDIM, lr=0.1, lr_tracking=0.02)

        # Env field: basis on physics dims
        env_c = np.zeros((K, NDIM))
        env_c[0, 0] = 1.0
        env_c[1, 1] = 1.0
        self._wp_env = WP(K, freqs, pos0=(0, 0, 0), mass=1e6, ndim=NDIM,
                          c_cos=env_c, c_sin=np.zeros((K, NDIM)),
                          lr=0.15, lr_tracking=0.0)

        # Reward field: basis on reward dim
        rew_c = np.zeros((K, NDIM))
        rew_c[0, 2] = 1.0
        self._wp_reward = WP(K, freqs, pos0=(0, 0, 0), mass=1e6, ndim=NDIM,
                             c_cos=rew_c.copy(), c_sin=np.zeros((K, NDIM)),
                             lr=0.15, lr_tracking=0.0)

    def _update_wavepackets(self):
        """Per-frame wavepacket update: shift, correct, normalize."""
        NDIM = self._NDIM
        ball_pos = np.array([self.ball_x, self.ball_y, 0.0])
        pad_l_pos = np.array([self.paddle_lx, self.agent_y, 0.0])
        pad_r_pos = np.array([self.paddle_rx, self.opp_y, 0.0])

        wp_ball = self._wp_ball
        wp_pl = self._wp_pl
        wp_pr = self._wp_pr
        wp_env = self._wp_env

        # Normalized inner products for attention
        nip_env = abs(wp_ball.normalized_inner_product(wp_env))
        nip_padL = abs(wp_ball.normalized_inner_product(wp_pl))
        nip_padR = abs(wp_ball.normalized_inner_product(wp_pr))

        # Shift wavepackets by velocity
        wp_ball.shift(self.ball_vx * DT, axis=0)
        wp_ball.shift(self.ball_vy * DT, axis=1)
        delta_l = self.agent_y - wp_pl.pos[1]
        delta_r = self.opp_y - wp_pr.pos[1]
        if abs(delta_l) > 1e-12:
            wp_pl.shift(delta_l, axis=1)
        if abs(delta_r) > 1e-12:
            wp_pr.shift(delta_r, axis=1)

        # LMS correction toward observed positions
        unity = np.ones(NDIM)
        wp_ball.update_with_attention(ball_pos, unity,
                                      [nip_env, nip_padL, nip_padR])
        wp_pl.update_with_attention(pad_l_pos, unity, [nip_padL])
        wp_pr.update_with_attention(pad_r_pos, unity, [nip_padR])

        # Deviation + normalize
        ball_dev = np.array([wp_ball.integrate_squared(d) - 1.0
                             for d in range(NDIM)])
        wp_ball.normalize()
        wp_pl.normalize()
        wp_pr.normalize()

        # Env learns from deviation
        dev_mag = np.linalg.norm(ball_dev[:2])
        if dev_mag > 1e-8:
            total_nip = nip_env + nip_padL + nip_padR + 1e-8
            env_frac = nip_env / total_nip
            wp_env.update_lms(ball_pos, ball_dev,
                              anomaly_scale=dev_mag * env_frac)
            wp_env.normalize()

        # Store positions
        wp_ball.pos[:] = ball_pos
        wp_pl.pos[:] = pad_l_pos
        wp_pr.pos[:] = pad_r_pos

    def _spectral_obs(self) -> np.ndarray:
        """8-dim spectral features from current wavepacket state."""
        fmaps = self._compute_feature_maps(
            self._wp_ball, self._wp_env, self._wp_pl, self._wp_pr,
            self._wp_reward, self._x_fm, self._y_fm, self._r_fm)
        if self._pool_mode == 'max':
            return self._conv_maxpool(fmaps)
        return self._conv.forward_fast(fmaps)

    def _conv_maxpool(self, fmaps: np.ndarray) -> np.ndarray:
        """Conv → ReLU → max pool (instead of avg pool)."""
        conv = self._conv
        W_flat = conv.W.reshape(conv.n_filters, -1)
        ks = conv.ks
        C, H, W = fmaps.shape
        oH, oW = H - ks + 1, W - ks + 1
        patches = np.lib.stride_tricks.as_strided(
            fmaps, shape=(oH, oW, C, ks, ks),
            strides=(fmaps.strides[1], fmaps.strides[2],
                     fmaps.strides[0], fmaps.strides[1], fmaps.strides[2])
        ).reshape(oH * oW, -1)
        pre_relu = patches @ W_flat.T + conv.b
        return np.maximum(pre_relu, 0).max(axis=0)

    def get_feature_maps(self) -> np.ndarray:
        """Return raw (6, 16, 24) outer-product maps. Only valid for spectral mode."""
        return self._compute_feature_maps(
            self._wp_ball, self._wp_env, self._wp_pl, self._wp_pr,
            self._wp_reward, self._x_fm, self._y_fm, self._r_fm)

    def _raw_obs(self) -> np.ndarray:
        """4-dim egocentric observation."""
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
        self.touched = False  # at least one touch this episode

        if self.obs_mode == 'spectral':
            self._create_wavepackets()

        return self._obs()

    def step(self, action: float):
        """Returns (obs, reward, done). Action in [-1, 1] moves paddle."""
        # Agent paddle
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

        # Track paddle reward before collision detection
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

        # Update wavepackets AFTER physics (they track the new positions)
        if self.obs_mode == 'spectral':
            self._update_wavepackets()

        # Scoring / reward
        agent_missed = self.ball_x < COURT_LEFT
        opp_missed = self.ball_x > COURT_RIGHT

        if self.reward_mode == 'goal':
            # Sparse: only terminal reward on scoring
            if agent_missed:
                return self._obs(), -1.0, True
            if opp_missed:
                return self._obs(), +1.0, True
            return self._obs(), 0.0, False

        else:  # reward_mode == 'paddle'
            # +1 on agent paddle hit, -1 on agent miss, episode ends on any goal
            if agent_missed:
                return self._obs(), -1.0, True
            if opp_missed:
                return self._obs(), 0.0, True  # opponent miss = neutral for agent
            reward = 1.0 if agent_hit else 0.0
            return self._obs(), reward, False


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
        self.obs_dim = obs_dim

        if hidden > 0:
            self.W1 = np.random.randn(hidden, obs_dim) * np.sqrt(2.0 / obs_dim)
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden) * 0.01
            self.b2 = 0.0
        else:
            self.w = np.zeros(obs_dim)
            self.b = 0.0

        self.w_v = np.zeros(obs_dim)
        self.b_v = 0.0

        self._obs_buf = []
        self._act_buf = []
        self._rew_buf = []
        self._h_buf = []
        self._mask_buf = []

        self._ret_mean = 0.0
        self._ret_var = 1.0
        self._ret_count = 0

    def _policy_forward(self, obs: np.ndarray) -> float:
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

        # Update baseline (MSE gradient, averaged over episode)
        grad_wv = np.zeros_like(self.w_v)
        grad_bv = 0.0
        for t in range(T):
            grad_wv += advantages[t] * obs[t]
            grad_bv += advantages[t]
        self.w_v += self.lr_baseline * grad_wv / T
        self.b_v += self.lr_baseline * grad_bv / T
        np.clip(self.w_v, -10.0, 10.0, out=self.w_v)
        self.b_v = np.clip(self.b_v, -10.0, 10.0)

        # Policy gradient (averaged over episode)
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

        ep_return = returns[0]
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()
        self._h_buf.clear()
        self._mask_buf.clear()
        return float(ep_return)


# -- Conv Policy Agent ---------------------------------------------------------

class ConvPolicyAgent:
    """Policy that learns a single conv filter over spectral feature maps.

    The filter convolves (6, H, W) outer-product maps down to a (1, oH, oW)
    activation map. The y-axis center-of-mass of that map is the action mean.
    REINFORCE updates the 6*ks*ks filter weights + bias directly.

    This tests whether the spectral encoder's spatial structure can serve
    as a direct policy representation with minimal learned parameters.
    """

    def __init__(self, n_channels: int = 6, kernel_size: int = 3,
                 lr: float = 1e-3, gamma: float = 0.99, std: float = 0.3):
        self.ks = kernel_size
        self.lr = lr
        self.gamma = gamma
        self.std = std
        # Single conv filter: (1, C, ks, ks)
        fan_in = n_channels * kernel_size * kernel_size
        self.W = np.random.randn(n_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.bias = 0.0
        # Episode buffers
        self._fmaps_buf = []
        self._act_buf = []
        self._mean_buf = []
        self._rew_buf = []
        # Running return stats
        self._ret_mean = 0.0
        self._ret_var = 1.0
        self._ret_count = 0

    def _forward(self, fmaps: np.ndarray) -> float:
        """Conv → ReLU → y-center-of-mass → action mean in [-1, 1]."""
        C, H, W = fmaps.shape
        ks = self.ks
        oH, oW = H - ks + 1, W - ks + 1
        # im2col
        patches = np.lib.stride_tricks.as_strided(
            fmaps, shape=(oH, oW, C, ks, ks),
            strides=(fmaps.strides[1], fmaps.strides[2],
                     fmaps.strides[0], fmaps.strides[1], fmaps.strides[2])
        ).reshape(oH * oW, -1)
        W_flat = self.W.reshape(1, -1)
        pre_relu = (patches @ W_flat.T + self.bias).reshape(oH, oW)
        activated = np.maximum(pre_relu, 0)  # (oH, oW)

        # y-axis center of mass
        y_grid = np.linspace(-1.0, 1.0, oH)
        y_marginal = activated.sum(axis=1)  # (oH,)
        total = y_marginal.sum()
        if total > 1e-8:
            mean = float((y_grid * y_marginal).sum() / total)
        else:
            mean = 0.0
        return np.clip(mean, -1.0, 1.0)

    def act(self, fmaps: np.ndarray) -> float:
        mean = self._forward(fmaps)
        action = mean + self.std * np.random.randn()
        self._fmaps_buf.append(fmaps.copy())
        self._act_buf.append(action)
        self._mean_buf.append(mean)
        return float(action)

    def record_reward(self, reward: float):
        self._rew_buf.append(reward)

    def _forward_with_grad(self, fmaps: np.ndarray):
        """Forward pass returning (mean, d_mean_d_W, d_mean_d_bias).

        Chain rule through: patches → pre_relu → ReLU → y_marginal → COM.
        """
        C, H, W_ = fmaps.shape
        ks = self.ks
        oH, oW = H - ks + 1, W_ - ks + 1
        # im2col: (oH*oW, C*ks*ks)
        patches = np.lib.stride_tricks.as_strided(
            fmaps, shape=(oH, oW, C, ks, ks),
            strides=(fmaps.strides[1], fmaps.strides[2],
                     fmaps.strides[0], fmaps.strides[1], fmaps.strides[2])
        ).reshape(oH * oW, -1)
        W_flat = self.W.reshape(1, -1)
        pre_relu = (patches @ W_flat.T + self.bias).ravel()  # (oH*oW,)
        mask = (pre_relu > 0).astype(np.float64)
        activated = pre_relu * mask  # (oH*oW,)
        act_2d = activated.reshape(oH, oW)

        # y-marginal and COM
        y_grid = np.linspace(-1.0, 1.0, oH)
        y_marginal = act_2d.sum(axis=1)  # (oH,)
        total = y_marginal.sum()
        if total < 1e-8:
            return 0.0, np.zeros_like(self.W), 0.0

        mean = float((y_grid * y_marginal).sum() / total)

        # Backward: d_mean / d_activated
        # mean = sum(y_i * m_i) / sum(m_i) where m_i = sum_j(act[i,j])
        # d_mean/d_act[i,j] = (y_i - mean) / total
        d_mean_d_act = np.zeros(oH * oW)
        for i in range(oH):
            for j in range(oW):
                d_mean_d_act[i * oW + j] = (y_grid[i] - mean) / total

        # Through ReLU: d_act/d_pre = mask
        d_mean_d_pre = d_mean_d_act * mask  # (oH*oW,)

        # Through conv: pre = patches @ W_flat.T + bias
        # d_pre/d_W_flat = patches, d_pre/d_bias = 1
        d_mean_d_W_flat = patches.T @ d_mean_d_pre  # (C*ks*ks,)
        d_mean_d_bias = d_mean_d_pre.sum()

        return mean, d_mean_d_W_flat.reshape(self.W.shape), d_mean_d_bias

    def end_episode(self):
        if not self._rew_buf:
            return 0.0
        T = len(self._rew_buf)
        rewards = np.array(self._rew_buf)
        actions = np.array(self._act_buf[:T])
        means = np.array(self._mean_buf[:T])

        # MC returns
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Normalize returns
        self._ret_count += T
        alpha = min(T / self._ret_count, 0.1)
        self._ret_mean = (1 - alpha) * self._ret_mean + alpha * returns.mean()
        self._ret_var = (1 - alpha) * self._ret_var + alpha * (returns.var() + 1e-8)
        returns_norm = (returns - self._ret_mean) / (np.sqrt(self._ret_var) + 1e-8)

        # REINFORCE with analytic gradient through COM
        gW = np.zeros_like(self.W)
        g_bias = 0.0
        for t in range(T):
            adv = returns_norm[t]
            d_logpi = (actions[t] - means[t]) / (self.std ** 2)
            _, d_mean_dW, d_mean_db = self._forward_with_grad(self._fmaps_buf[t])
            gW += adv * d_logpi * d_mean_dW
            g_bias += adv * d_logpi * d_mean_db

        self.W += self.lr * gW / T
        self.bias += self.lr * g_bias / T
        np.clip(self.W, -5.0, 5.0, out=self.W)
        self.bias = np.clip(self.bias, -5.0, 5.0)

        ep_return = returns[0]
        self._fmaps_buf.clear()
        self._act_buf.clear()
        self._mean_buf.clear()
        self._rew_buf.clear()
        return float(ep_return)


# -- Training loop -------------------------------------------------------------

def train(n_episodes: int = 2000, hidden: int = 0, lr: float = 1e-3,
          lr_baseline: float = 1e-2, gamma: float = 0.99, std: float = 0.5,
          seed: int = 0, verbose: bool = True, max_steps: int = 2000,
          opp_skill: float = 0.0, reward_mode: str = 'goal',
          obs_mode: str = 'raw', pool_mode: str = 'avg',
          agent_type: str = 'reinforce'):
    """Train agent, return per-episode results.

    agent_type: 'reinforce' (standard) or 'conv_policy' (learns conv filter
                over spectral feature maps, requires obs_mode='spectral').
    """
    np.random.seed(seed)
    env = PongEnv(opp_skill=opp_skill, reward_mode=reward_mode,
                  obs_mode=obs_mode, pool_mode=pool_mode)

    use_conv_policy = (agent_type == 'conv_policy')
    if use_conv_policy:
        assert obs_mode == 'spectral', 'conv_policy requires obs_mode=spectral'
        agent = ConvPolicyAgent(lr=lr, gamma=gamma, std=std)
    else:
        agent = REINFORCEAgent(obs_dim=env.obs_dim, hidden=hidden,
                               lr=lr, lr_baseline=lr_baseline,
                               gamma=gamma, std=std)

    results = []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            if use_conv_policy:
                fmaps = env.get_feature_maps()
                action = agent.act(fmaps)
            else:
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
            'touches': env.agent_touches,
            'touched': env.touched,
            'return': ep_return,
        })

        if verbose and (ep + 1) % 100 == 0:
            recent = results[-100:]
            wins = sum(1 for r in recent if r['reward'] > 0)
            losses = sum(1 for r in recent if r['reward'] < 0)
            avg_touches = np.mean([r['touches'] for r in recent])
            avg_len = np.mean([r['length'] for r in recent])
            print(f'ep {ep+1:5d}: W/L={wins}/{losses} '
                  f'touches={avg_touches:.1f}/ep avg_len={avg_len:.0f}')

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
    parser.add_argument('--opp-skill', type=float, default=0.0,
                        help='Opponent skill: 0=random, 1=perfect tracker')
    parser.add_argument('--reward-mode', choices=['goal', 'paddle'],
                        default='goal')
    parser.add_argument('--obs-mode', choices=['raw', 'spectral'],
                        default='raw')
    parser.add_argument('--agent-type', choices=['reinforce', 'conv_policy'],
                        default='reinforce',
                        help='conv_policy: learn conv filter as policy head')
    args = parser.parse_args()

    for seed in range(args.seeds):
        if args.seeds > 1:
            print(f'\n=== Seed {seed} ===')
        results = train(n_episodes=args.episodes, hidden=args.hidden,
                        lr=args.lr, lr_baseline=args.lr_baseline,
                        gamma=args.gamma, std=args.std, seed=seed,
                        max_steps=args.max_steps, opp_skill=args.opp_skill,
                        reward_mode=args.reward_mode, obs_mode=args.obs_mode,
                        agent_type=args.agent_type)
        last_200 = results[-200:] if len(results) >= 200 else results
        wins = sum(1 for r in last_200 if r['reward'] > 0)
        losses = sum(1 for r in last_200 if r['reward'] < 0)
        avg_touches = np.mean([r['touches'] for r in last_200])
        print(f'\nFinal (last {len(last_200)} eps): W/L={wins}/{losses} '
              f'touches={avg_touches:.1f}/ep')


if __name__ == '__main__':
    main()
