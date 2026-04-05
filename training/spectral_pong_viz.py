"""
2D Spectral Pong Visualization
===============================
Pong game where ball and paddles are observed spectral waveforms with 2D
amplitudes, and top/bottom walls are an unobserved waveform learned via LMS
from acceleration anomalies.

Two 1D domain visualizations: X-domain (horizontal) and Y-domain (vertical),
each showing one component of the 2D amplitude vectors.

Controls:
    Left paddle:  A / D
    Right paddle:  Up / Down

Usage:
    python training/spectral_pong_viz.py                              # interactive
    python training/spectral_pong_viz.py --auto-paddle                # AI paddles
    python training/spectral_pong_viz.py --save out.gif --frames 600 --auto-paddle
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpecFromSubplotSpec

# -- Import wavepacket library (canonical source: training/spectral/wavepacket.py)
from training.spectral.wavepacket import (  # noqa: E402
    WavepacketObject2D, compute_feature_maps, ConvFeatureExtractor,
    FM_NX, FM_NY, FM_CHANNELS, FM_LABELS, COEFF_CLIP, WORLD_BOUNDS,
    COURT_LEFT, COURT_RIGHT, COURT_TOP, COURT_BOTTOM,
    PADDLE_X_OFFSET, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED,
    BALL_RADIUS, BALL_SPEED, SPIN_FACTOR,
)

# -- Visualization color constants --------------------------------------------

BG_COLOR = '#12121f'
BALL_COLOR = '#38bdf8'
PADDLE_COLOR_L = '#4ade80'
PADDLE_COLOR_R = '#f87171'
WALL_COLOR = '#333'
NET_COLOR = '#555'
SCORE_COLOR = 'white'
ENV_FIELD_COLOR = '#c084fc'
INTERACT_COLOR = '#a78bfa'
RESIDUAL_COLOR = '#facc15'
REWARD_COLOR = '#fb923c'
GRID_COLOR = '#444'
GRID_ALPHA = 0.15
LABEL_COLOR = '#aaa'
TITLE_COLOR = 'white'


# -- Simple RL paddle controller -----------------------------------------------

class SimpleRLController:
    """Linear actor-critic: TD(0) with eligibility traces.

    Actor:  action = tanh(w_a · state + b_a) + noise
    Critic: V(s)   = w_c · state + b_c
    TD error: δ = reward + γ·V(s') − V(s)
    Updates every frame via δ, not just on goals.
    ~22 parameters per paddle (10+1 actor, 10+1 critic).
    Includes a 1D reward wavepacket that learns where goals happen.
    """

    TRACE_CLIP = 1.0
    WEIGHT_CLIP = 5.0
    TD_CLIP = 1.0

    # Conv over 6-channel outer-product maps → 8-dim state
    STATE_DIM = 8
    STATE_LABELS = [f'conv_{i}' for i in range(8)]

    def __init__(self, state_dim: int, lr_actor: float = 1e-7,
                 lr_critic: float = 1e-6, gamma: float = 0.95,
                 lam: float = 0.9, std: float = 0.3,
                 K: int = 8, frequencies: np.ndarray | None = None,
                 batch_size: int = 1, **_kwargs):
        # Actor (policy)
        self.w_a = np.random.randn(state_dim) * 0.1
        self.b_a = 0.0
        # Critic (value)
        self.w_c = np.zeros(state_dim)
        self.b_c = 0.0
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.lam = lam  # trace decay for TD(λ)
        self.std = std
        # Actor eligibility trace (∇logπ accumulated)
        self.trace_a_w = np.zeros(state_dim)
        self.trace_a_b = 0.0
        # Critic eligibility trace (state features accumulated)
        self.trace_c_w = np.zeros(state_dim)
        self.trace_c_b = 0.0
        # Previous state for TD
        self.prev_state = None
        # 1D reward wavepacket: learns where goals happen along x-axis
        if frequencies is None:
            frequencies = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
        self.rw_k = np.asarray(frequencies, dtype=np.float64)
        self.rw_c_cos = np.zeros(K, dtype=np.float64)
        self.rw_c_sin = np.zeros(K, dtype=np.float64)
        self.rw_lr = 0.0001
        # EMA feature normalization
        self.feat_mean = np.zeros(state_dim)
        self.feat_var = np.ones(state_dim)
        self._feat_ema_alpha = 0.01
        # Conv feature extractor for 2D outer-product maps
        self.conv = ConvFeatureExtractor(
            n_channels=FM_CHANNELS, n_filters=8, kernel_size=3)
        # Conv eligibility traces + learning rate
        self.trace_conv_w = np.zeros_like(self.conv.W)   # (n_filters, C, ks, ks)
        self.trace_conv_b = np.zeros(self.conv.n_filters)
        self.lr_conv = 1e-9
        # Activations saved in act() for critic gradient in step()
        self._act_patches = None
        self._act_gate    = None
        self._act_n_pos   = None
        # Diagnostics
        self.last_mean = 0.0
        self.last_action = 0.0
        self.last_td_error = 0.0
        self.last_value = 0.0
        # Batched update support (batch_size=1 uses legacy trace path)
        self.batch_size = batch_size
        self._buffer = []
        self._pending_act = None

    def reward_predict(self, x: float) -> float:
        """Evaluate 1D reward wavepacket at x position."""
        basis = np.concatenate([np.cos(self.rw_k * x),
                                np.sin(self.rw_k * x)])
        coeffs = np.concatenate([self.rw_c_cos, self.rw_c_sin])
        return float(coeffs @ basis)

    def reward_update(self, x: float, reward: float) -> None:
        """LMS update: push reward wavepacket toward observed reward at x."""
        basis_cos = np.cos(self.rw_k * x)
        basis_sin = np.sin(self.rw_k * x)
        pred = float(self.rw_c_cos @ basis_cos + self.rw_c_sin @ basis_sin)
        error = reward - pred
        self.rw_c_cos += self.rw_lr * error * basis_cos
        self.rw_c_sin += self.rw_lr * error * basis_sin

    @staticmethod
    def build_state(feature_maps: np.ndarray,
                    conv: 'ConvFeatureExtractor', **_) -> np.ndarray:
        """8-dim state from 2D conv over 6-channel outer-product feature maps."""
        return conv.forward_fast(feature_maps)

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """EMA per-feature standardization: zero mean, unit variance."""
        a = self._feat_ema_alpha
        self.feat_mean = (1 - a) * self.feat_mean + a * state
        self.feat_var = (1 - a) * self.feat_var + a * (state - self.feat_mean) ** 2
        return (state - self.feat_mean) / (np.sqrt(self.feat_var) + 1e-8)

    def _value(self, state: np.ndarray) -> float:
        return float(self.w_c @ state + self.b_c)

    def act(self, raw_state: np.ndarray) -> float:
        """Actor forward pass + accumulate trace. Call step() after."""
        state = self._normalize(raw_state)
        z = float(self.w_a @ state + self.b_a)
        mean = np.tanh(z)
        action = float(np.clip(mean + self.std * np.random.randn(), -1, 1))

        # ∇ log π through tanh
        d_logpi = (action - mean) / (self.std ** 2)
        d_z = d_logpi * (1 - mean ** 2)

        if self.batch_size <= 1:
            # Legacy per-frame trace path
            grad_w = d_z * state
            grad_b = d_z
            self.trace_a_w = np.clip(
                self.gamma * self.lam * self.trace_a_w + grad_w,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.trace_a_b = np.clip(
                self.gamma * self.lam * self.trace_a_b + grad_b,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            # Conv actor gradient (only if forward_fast has been called)
            p = getattr(self.conv, '_last_patches', None)
            g = getattr(self.conv, '_last_gate', None)
            n = getattr(self.conv, '_last_n_pos', None)
            if p is not None:
                scale = d_z * self.w_a
                grad_conv_w = ((g * scale[None, :]).T @ p) / n
                grad_conv_b = scale * g.mean(axis=0)
                self.trace_conv_w = np.clip(
                    self.gamma * self.lam * self.trace_conv_w
                    + grad_conv_w.reshape(self.conv.W.shape),
                    -self.TRACE_CLIP, self.TRACE_CLIP)
                self.trace_conv_b = np.clip(
                    self.gamma * self.lam * self.trace_conv_b + grad_conv_b,
                    -self.TRACE_CLIP, self.TRACE_CLIP)
            self._act_patches = p
            self._act_gate    = g
            self._act_n_pos   = n
        else:
            # Batch mode: save pending data for step() to buffer
            self._pending_act = {
                'state': state.copy(),
                'd_z': d_z,
            }

        # Diagnostics
        self.last_mean = mean
        self.last_action = action
        self.last_value = self._value(state)
        return action

    def step(self, raw_state: np.ndarray, reward: float,
             terminal: bool = False) -> None:
        """TD update — per-frame (batch_size=1) or buffered (batch_size>1)."""
        state = self._normalize(raw_state)
        if self.prev_state is None:
            self.prev_state = state.copy()
            return

        # TD error (always computed for diagnostics)
        v_prev = self._value(self.prev_state)
        v_curr = 0.0 if terminal else self._value(state)
        td_error = reward + self.gamma * v_curr - v_prev
        td_error = np.clip(td_error, -self.TD_CLIP, self.TD_CLIP)
        self.last_td_error = td_error

        if self.batch_size <= 1:
            # Legacy per-frame trace update
            self.trace_c_w = np.clip(
                self.gamma * self.lam * self.trace_c_w + self.prev_state,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.trace_c_b = np.clip(
                self.gamma * self.lam * self.trace_c_b + 1.0,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.w_c += self.lr_critic * td_error * self.trace_c_w
            self.b_c += self.lr_critic * td_error * self.trace_c_b
            np.clip(self.w_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w_c)
            self.b_c = np.clip(self.b_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
            self.w_a += self.lr_actor * td_error * self.trace_a_w
            self.b_a += self.lr_actor * td_error * self.trace_a_b
            np.clip(self.w_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w_a)
            self.b_a = np.clip(self.b_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
            if self._act_patches is not None:
                p, g, n = self._act_patches, self._act_gate, self._act_n_pos
                scale_c = self.w_c
                grad_conv_c_w = ((g * scale_c[None, :]).T @ p) / n
                grad_conv_c_b = scale_c * g.mean(axis=0)
                self.conv.W += self.lr_conv * td_error * (
                    self.trace_conv_w + grad_conv_c_w.reshape(self.conv.W.shape))
                self.conv.b += self.lr_conv * td_error * (
                    self.trace_conv_b + grad_conv_c_b)
                np.clip(self.conv.W, -self.WEIGHT_CLIP, self.WEIGHT_CLIP,
                        out=self.conv.W)
                np.clip(self.conv.b, -self.WEIGHT_CLIP, self.WEIGHT_CLIP,
                        out=self.conv.b)
        else:
            # Batch mode: buffer transition, flush when full
            if self._pending_act is not None:
                self._buffer.append({
                    'state': self._pending_act['state'],
                    'd_z': self._pending_act['d_z'],
                    'reward': reward,
                    'next_state': state.copy(),
                    'terminal': terminal,
                })
                self._pending_act = None
                if len(self._buffer) >= self.batch_size:
                    self._flush_batch()

        self.prev_state = state.copy()

    def _flush_batch(self) -> None:
        """Compute summed TD(0) gradients and apply."""
        if not self._buffer:
            return
        grad_wc = np.zeros_like(self.w_c)
        grad_bc = 0.0
        grad_wa = np.zeros_like(self.w_a)
        grad_ba = 0.0

        for t in self._buffer:
            s, ns, r = t['state'], t['next_state'], t['reward']
            v_s = self._value(s)
            v_ns = 0.0 if t['terminal'] else self._value(ns)
            delta = np.clip(r + self.gamma * v_ns - v_s,
                            -self.TD_CLIP, self.TD_CLIP)
            # Critic: ∇V(s) = s for linear critic
            grad_wc += delta * s
            grad_bc += delta
            # Actor: delta * d_z * s
            d_z = t['d_z']
            grad_wa += delta * d_z * s
            grad_ba += delta * d_z

        # Apply summed gradients (no averaging)
        self.w_c += self.lr_critic * grad_wc
        self.b_c += self.lr_critic * grad_bc
        self.w_a += self.lr_actor * grad_wa
        self.b_a += self.lr_actor * grad_ba
        np.clip(self.w_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w_c)
        self.b_c = np.clip(self.b_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        np.clip(self.w_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w_a)
        self.b_a = np.clip(self.b_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        self._buffer.clear()

    def on_reset(self) -> None:
        """Call on episode boundary (goal scored) to clear traces."""
        if self.batch_size > 1:
            # Mark last buffer entry as terminal
            if self._buffer:
                self._buffer[-1]['terminal'] = True
            self._pending_act = None
        else:
            self.trace_a_w[:] = 0
            self.trace_a_b = 0.0
            self.trace_c_w[:] = 0
            self.trace_c_b = 0.0
            self.trace_conv_w[:] = 0
            self.trace_conv_b[:] = 0
            self._act_patches = None
            self._act_gate    = None
            self._act_n_pos   = None
        self.prev_state = None


# -- MLP baseline (no spectral encoder) ----------------------------------------

def build_raw_state(ball: dict, own_paddle: dict, opp_paddle: dict) -> np.ndarray:
    """6-dim raw observation: [ball_x, ball_y, ball_vx, ball_vy, own_y, opp_y].
    Pre-normalized to ~[-1, 1] using known court/speed bounds."""
    return np.array([
        ball['x'] / 5.0,
        ball['y'] / 3.0,
        ball['vx'] / BALL_SPEED,
        ball['vy'] / BALL_SPEED,
        own_paddle['y'] / 2.5,
        opp_paddle['y'] / 2.5,
    ])


class MLPController(SimpleRLController):
    """Linear actor-critic on raw game state (no spectral encoder).
    Identical RL algorithm, just 6-dim direct observation instead of
    8-dim conv features from spectral outer-product maps."""

    STATE_DIM = 6

    def __init__(self, state_dim: int = 6, **kwargs):
        kwargs.pop('K', None)
        kwargs.pop('frequencies', None)
        super().__init__(state_dim=state_dim, **kwargs)
        self.conv = None
        self.trace_conv_w = None
        self.trace_conv_b = None
        self.lr_conv = 0.0

    def act(self, raw_state: np.ndarray) -> float:
        state = self._normalize(raw_state)
        z = float(self.w_a @ state + self.b_a)
        mean = np.tanh(z)
        action = float(np.clip(mean + self.std * np.random.randn(), -1, 1))
        d_logpi = (action - mean) / (self.std ** 2)
        d_z = d_logpi * (1 - mean ** 2)

        if self.batch_size <= 1:
            grad_w = d_z * state
            grad_b = d_z
            self.trace_a_w = np.clip(
                self.gamma * self.lam * self.trace_a_w + grad_w,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.trace_a_b = np.clip(
                self.gamma * self.lam * self.trace_a_b + grad_b,
                -self.TRACE_CLIP, self.TRACE_CLIP)
        else:
            self._pending_act = {'state': state.copy(), 'd_z': d_z}

        self.last_mean = mean
        self.last_action = action
        self.last_value = self._value(state)
        return action

    def step(self, raw_state: np.ndarray, reward: float) -> None:
        state = self._normalize(raw_state)
        if self.prev_state is None:
            self.prev_state = state.copy()
            return
        v_prev = self._value(self.prev_state)
        v_curr = self._value(state)
        td_error = reward + self.gamma * v_curr - v_prev
        td_error = np.clip(td_error, -self.TD_CLIP, self.TD_CLIP)
        self.last_td_error = td_error

        if self.batch_size <= 1:
            self.trace_c_w = np.clip(
                self.gamma * self.lam * self.trace_c_w + self.prev_state,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.trace_c_b = np.clip(
                self.gamma * self.lam * self.trace_c_b + 1.0,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.w_c += self.lr_critic * td_error * self.trace_c_w
            self.b_c += self.lr_critic * td_error * self.trace_c_b
            np.clip(self.w_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w_c)
            self.b_c = np.clip(self.b_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
            self.w_a += self.lr_actor * td_error * self.trace_a_w
            self.b_a += self.lr_actor * td_error * self.trace_a_b
            np.clip(self.w_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=self.w_a)
            self.b_a = np.clip(self.b_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        else:
            if self._pending_act is not None:
                self._buffer.append({
                    'state': self._pending_act['state'],
                    'd_z': self._pending_act['d_z'],
                    'reward': reward,
                    'next_state': state.copy(),
                    'terminal': False,
                })
                self._pending_act = None
                if len(self._buffer) >= self.batch_size:
                    self._flush_batch()
        self.prev_state = state.copy()

    def on_reset(self) -> None:
        if self.batch_size > 1:
            if self._buffer:
                self._buffer[-1]['terminal'] = True
            self._pending_act = None
        else:
            self.trace_a_w[:] = 0
            self.trace_a_b = 0.0
            self.trace_c_w[:] = 0
            self.trace_c_b = 0.0
        self.prev_state = None


# -- MLP actor-critic (nonlinear policy) ---------------------------------------

class MLPActorCriticController(SimpleRLController):
    """Nonlinear actor-critic with hidden layer.

    Actor:  state → W1(H×D)+b1 → ReLU → W2(1×H)+b2 → tanh → mean
    Critic: state → W1(H×D)+b1 → ReLU → W2(1×H)+b2 → value

    Supports both per-frame TD(λ) traces (batch_size=1) and batched TD(0).
    """

    def __init__(self, state_dim: int = 8, hidden: int = 32,
                 lr_actor: float = 3e-3, lr_critic: float = 3e-2,
                 gamma: float = 0.99, lam: float = 0.92,
                 std: float = 0.3, batch_size: int = 1, **kwargs):
        kwargs.pop('K', None)
        kwargs.pop('frequencies', None)
        super().__init__(state_dim=state_dim, lr_actor=lr_actor,
                         lr_critic=lr_critic, gamma=gamma, lam=lam,
                         std=std, batch_size=batch_size, **kwargs)
        # Override linear weights with MLP
        self.hidden = hidden
        # Actor MLP
        self.W1_a = np.random.randn(hidden, state_dim) * np.sqrt(2.0 / state_dim)
        self.b1_a = np.zeros(hidden)
        self.W2_a = np.random.randn(1, hidden) * 0.01
        self.b2_a = 0.0
        # Critic MLP
        self.W1_c = np.random.randn(hidden, state_dim) * np.sqrt(2.0 / state_dim)
        self.b1_c = np.zeros(hidden)
        self.W2_c = np.random.randn(1, hidden) * 0.01
        self.b2_c = 0.0
        # Eligibility traces for all 8 MLP parameter groups
        self.tr_W1_a = np.zeros_like(self.W1_a)
        self.tr_b1_a = np.zeros(hidden)
        self.tr_W2_a = np.zeros_like(self.W2_a)
        self.tr_b2_a = 0.0
        self.tr_W1_c = np.zeros_like(self.W1_c)
        self.tr_b1_c = np.zeros(hidden)
        self.tr_W2_c = np.zeros_like(self.W2_c)
        self.tr_b2_c = 0.0

    def _value(self, state: np.ndarray) -> float:
        h = state @ self.W1_c.T + self.b1_c
        h_relu = np.maximum(h, 0)
        return float(self.W2_c.ravel() @ h_relu + self.b2_c)

    def _critic_grads(self, state: np.ndarray):
        """Return (gW1, gb1, gW2, gb2) = ∇_params V(state)."""
        h = state @ self.W1_c.T + self.b1_c
        mask = (h > 0).astype(np.float64)
        h_relu = h * mask
        # dV/dW2 = h_relu, dV/db2 = 1
        gW2 = h_relu.reshape(1, -1)
        gb2 = 1.0
        # dV/dh = W2 * mask, dh/dW1 = outer(mask*W2, state)
        d_h = self.W2_c.ravel() * mask
        gW1 = np.outer(d_h, state)
        gb1 = d_h
        return gW1, gb1, gW2, gb2

    def act(self, raw_state: np.ndarray) -> float:
        state = self._normalize(raw_state)
        # MLP forward
        h = state @ self.W1_a.T + self.b1_a
        mask = (h > 0).astype(np.float64)
        h_relu = h * mask
        z = float(self.W2_a.ravel() @ h_relu + self.b2_a)
        mean = np.tanh(z)
        action = float(np.clip(mean + self.std * np.random.randn(), -1, 1))
        # Policy gradient components
        d_logpi = (action - mean) / (self.std ** 2)
        d_z = d_logpi * (1 - mean ** 2)

        if self.batch_size <= 1:
            # Per-frame: accumulate actor traces
            gl = self.gamma * self.lam
            # dz/dW2 = h_relu, dz/db2 = 1
            self.tr_W2_a = np.clip(
                gl * self.tr_W2_a + d_z * h_relu.reshape(1, -1),
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.tr_b2_a = np.clip(
                gl * self.tr_b2_a + d_z,
                -self.TRACE_CLIP, self.TRACE_CLIP)
            # dz/dW1 = outer(d_z * W2 * mask, state)
            d_h = d_z * self.W2_a.ravel() * mask
            self.tr_W1_a = np.clip(
                gl * self.tr_W1_a + np.outer(d_h, state),
                -self.TRACE_CLIP, self.TRACE_CLIP)
            self.tr_b1_a = np.clip(
                gl * self.tr_b1_a + d_h,
                -self.TRACE_CLIP, self.TRACE_CLIP)
        else:
            self._pending_act = {
                'state': state.copy(),
                'd_z': d_z,
                'h_relu': h_relu.copy(),
                'mask': mask.copy(),
            }

        self.last_mean = mean
        self.last_action = action
        self.last_value = self._value(state)
        return action

    def step(self, raw_state: np.ndarray, reward: float,
             terminal: bool = False) -> None:
        state = self._normalize(raw_state)
        if self.prev_state is None:
            self.prev_state = state.copy()
            return
        v_prev = self._value(self.prev_state)
        v_curr = 0.0 if terminal else self._value(state)
        td_error = reward + self.gamma * v_curr - v_prev
        td_error = np.clip(td_error, -self.TD_CLIP, self.TD_CLIP)
        self.last_td_error = td_error

        if self.batch_size <= 1:
            # Per-frame trace update for critic
            gl = self.gamma * self.lam
            gW1, gb1, gW2, gb2 = self._critic_grads(self.prev_state)
            self.tr_W1_c = np.clip(gl * self.tr_W1_c + gW1,
                                   -self.TRACE_CLIP, self.TRACE_CLIP)
            self.tr_b1_c = np.clip(gl * self.tr_b1_c + gb1,
                                   -self.TRACE_CLIP, self.TRACE_CLIP)
            self.tr_W2_c = np.clip(gl * self.tr_W2_c + gW2,
                                   -self.TRACE_CLIP, self.TRACE_CLIP)
            self.tr_b2_c = np.clip(gl * self.tr_b2_c + gb2,
                                   -self.TRACE_CLIP, self.TRACE_CLIP)
            # Apply: param += lr * delta * trace
            self.W1_c += self.lr_critic * td_error * self.tr_W1_c
            self.b1_c += self.lr_critic * td_error * self.tr_b1_c
            self.W2_c += self.lr_critic * td_error * self.tr_W2_c
            self.b2_c += self.lr_critic * td_error * self.tr_b2_c
            self.W1_a += self.lr_actor * td_error * self.tr_W1_a
            self.b1_a += self.lr_actor * td_error * self.tr_b1_a
            self.W2_a += self.lr_actor * td_error * self.tr_W2_a
            self.b2_a += self.lr_actor * td_error * self.tr_b2_a
            # Clip all weights
            for arr in [self.W1_c, self.b1_c, self.W2_c,
                        self.W1_a, self.b1_a, self.W2_a]:
                np.clip(arr, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=arr)
            self.b2_c = np.clip(self.b2_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
            self.b2_a = np.clip(self.b2_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        else:
            # Batch mode: buffer transition, flush when full
            if self._pending_act is not None:
                self._buffer.append({
                    'state': self._pending_act['state'],
                    'd_z': self._pending_act['d_z'],
                    'h_relu': self._pending_act['h_relu'],
                    'mask': self._pending_act['mask'],
                    'reward': reward,
                    'next_state': state.copy(),
                    'terminal': terminal,
                })
                self._pending_act = None
                if len(self._buffer) >= self.batch_size:
                    self._flush_batch()
        self.prev_state = state.copy()

    def _flush_batch(self) -> None:
        if not self._buffer:
            return
        gW1_c = np.zeros_like(self.W1_c)
        gb1_c = np.zeros_like(self.b1_c)
        gW2_c = np.zeros_like(self.W2_c)
        gb2_c = 0.0
        gW1_a = np.zeros_like(self.W1_a)
        gb1_a = np.zeros_like(self.b1_a)
        gW2_a = np.zeros_like(self.W2_a)
        gb2_a = 0.0

        for t in self._buffer:
            s, ns, r = t['state'], t['next_state'], t['reward']
            v_s = self._value(s)
            v_ns = 0.0 if t['terminal'] else self._value(ns)
            delta = np.clip(r + self.gamma * v_ns - v_s,
                            -self.TD_CLIP, self.TD_CLIP)
            # Critic gradient: delta * dV/dparams
            h_c = s @ self.W1_c.T + self.b1_c
            mask_c = (h_c > 0).astype(np.float64)
            h_relu_c = h_c * mask_c
            gW2_c += delta * h_relu_c
            gb2_c += delta
            d_h_c = self.W2_c.ravel() * mask_c
            gW1_c += delta * np.outer(d_h_c, s)
            gb1_c += delta * d_h_c
            # Actor gradient: delta * d_z * dz/dparams
            d_z = t['d_z']
            h_relu_a = t['h_relu']
            mask_a = t['mask']
            gW2_a += delta * d_z * h_relu_a
            gb2_a += delta * d_z
            d_h_a = d_z * self.W2_a.ravel() * mask_a
            gW1_a += delta * np.outer(d_h_a, s)
            gb1_a += delta * d_h_a

        # Sum of gradients (no averaging)
        self.W1_c += self.lr_critic * gW1_c
        self.b1_c += self.lr_critic * gb1_c
        self.W2_c += self.lr_critic * gW2_c.reshape(self.W2_c.shape)
        self.b2_c += self.lr_critic * gb2_c
        self.W1_a += self.lr_actor * gW1_a
        self.b1_a += self.lr_actor * gb1_a
        self.W2_a += self.lr_actor * gW2_a.reshape(self.W2_a.shape)
        self.b2_a += self.lr_actor * gb2_a
        # Clip all weights
        for arr in [self.W1_c, self.b1_c, self.W2_c,
                    self.W1_a, self.b1_a, self.W2_a]:
            np.clip(arr, -self.WEIGHT_CLIP, self.WEIGHT_CLIP, out=arr)
        self.b2_c = np.clip(self.b2_c, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        self.b2_a = np.clip(self.b2_a, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        self._buffer.clear()

    def on_reset(self) -> None:
        if self.batch_size > 1:
            if self._buffer:
                self._buffer[-1]['terminal'] = True
            self._pending_act = None
        else:
            self.tr_W1_a[:] = 0; self.tr_b1_a[:] = 0
            self.tr_W2_a[:] = 0; self.tr_b2_a = 0.0
            self.tr_W1_c[:] = 0; self.tr_b1_c[:] = 0
            self.tr_W2_c[:] = 0; self.tr_b2_c = 0.0
        self.prev_state = None


class MLPRawController(MLPActorCriticController):
    """MLP actor-critic on raw game state (no spectral encoder)."""

    STATE_DIM = 6

    def __init__(self, state_dim: int = 6, **kwargs):
        super().__init__(state_dim=state_dim, **kwargs)
        self.conv = None
        self.lr_conv = 0.0


# -- helpers ------------------------------------------------------------------

def reset_ball(ball: dict, toward: str = 'random',
               speed: float = BALL_SPEED) -> None:
    ball['x'] = 0.0
    ball['y'] = np.random.uniform(COURT_BOTTOM * 0.6, COURT_TOP * 0.6)
    if toward == 'random':
        toward = 'left' if np.random.random() < 0.5 else 'right'
    angle = np.random.uniform(-1.0, 1.0)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
    ball['vx'] = -vx if toward == 'left' else vx
    ball['vy'] = vy


# -- game + visualization ----------------------------------------------------

def create_game(K: int, frequencies: np.ndarray, alpha: float,
                dt: float, ball_speed: float, interval: int,
                max_frames: int | None, auto_paddle: bool,
                ball_sigma: float,
                env_lr: float, spectral_paddle: bool = False,
                alpha_paddle: float = 0.5,
                paddle_damping: float = 0.85,
                beta_paddle: float = 0.3,
                gamma_paddle: float = 0.2,
                ball_lr: float = 0.15,
                paddle_lr: float = 0.1,
                lr_tracking: float = 0.01,
                force_scale: float = 0.0,
                lr_k: float = 0.0,
                reward_replay_n: int = 1):

    # -- spectral setup (3D: x, y, reward) ------------------------------------
    NDIM = 3

    wp_ball = WavepacketObject2D(
        K, frequencies, pos0=(0.0, 0.0, 0.0), mass=1.0,
        sigma=ball_sigma, ndim=NDIM,
        lr=ball_lr, lr_tracking=lr_tracking)

    wp_paddle_l = WavepacketObject2D(
        K, frequencies,
        pos0=(COURT_LEFT + PADDLE_X_OFFSET, 0.0, 0.0), mass=1e6,
        sigma=0.5, amplitude=1.0, ndim=NDIM,
        lr=paddle_lr, lr_tracking=lr_tracking * 2)

    wp_paddle_r = WavepacketObject2D(
        K, frequencies,
        pos0=(COURT_RIGHT - PADDLE_X_OFFSET, 0.0, 0.0), mass=1e6,
        sigma=0.5, amplitude=1.0, ndim=NDIM,
        lr=paddle_lr, lr_tracking=lr_tracking * 2)

    # Env field: basis [0,1,0] only — acts on y only (cheat: x axis disabled)
    env_c_cos = np.zeros((K, NDIM))
    env_c_cos[0, 0] = 1.0  # frequency 0 → x
    env_c_cos[1, 1] = 1.0  # frequency 1 → y
    wp_env = WavepacketObject2D(
        K, frequencies, pos0=(0.0, 0.0, 0.0), mass=1e6, ndim=NDIM,
        c_cos=env_c_cos, c_sin=np.zeros((K, NDIM)),
        lr=env_lr, lr_tracking=0.0)

    # Per-agent reward fields: basis [0,0,1] — acts on reward dim only
    rew_c_cos = np.zeros((K, NDIM))
    rew_c_cos[0, 2] = 1.0  # frequency 0 → reward dimension
    wp_reward_l = WavepacketObject2D(
        K, frequencies, pos0=(0.0, 0.0, 0.0), mass=1e6, ndim=NDIM,
        c_cos=rew_c_cos.copy(), c_sin=np.zeros((K, NDIM)),
        lr=0.15, lr_tracking=0.0)
    wp_reward_r = WavepacketObject2D(
        K, frequencies, pos0=(0.0, 0.0, 0.0), mass=1e6, ndim=NDIM,
        c_cos=rew_c_cos.copy(), c_sin=np.zeros((K, NDIM)),
        lr=0.15, lr_tracking=0.0)

    # -- RL paddle controllers (one per paddle, opposite rewards) ------------
    rl_left = None
    rl_right = None
    if spectral_paddle:
        # State: 5 wavepackets × (K×2 cos + K×2 sin) + 6 game-state floats
        rl_left = SimpleRLController(SimpleRLController.STATE_DIM)
        rl_right = SimpleRLController(SimpleRLController.STATE_DIM)

    # -- Newtonian state ------------------------------------------------------
    paddle_lx = COURT_LEFT + PADDLE_X_OFFSET
    paddle_rx = COURT_RIGHT - PADDLE_X_OFFSET

    ball = {'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0}
    reset_ball(ball, toward='right', speed=ball_speed)

    left_paddle = {'y': 0.0, 'score': 0}
    right_paddle = {'y': 0.0, 'score': 0}
    key_times: dict[str, float] = {}
    paddle_vel = {'l': 0.0, 'r': 0.0}
    state = {'freeze': 0, 't': 0,
             'anomaly': np.zeros(NDIM), 'max_anomaly': 1.0,
             'rally_touches': 0, 'total_touches': 0,
             'last_rally': 0, 'rally_lengths': []}

    # Sample-efficiency tracking: measure when ball-tracking behavior emerges.
    # Metric: mean |left_paddle_y - ball_y| per episode (lower = better tracking).
    # Random baseline ~2.0 (uniform over ±3 court); threshold 1.0 = one paddle-height.
    tracking_buf: list[float] = []          # per-frame errors this episode
    tracking_history: deque = deque(maxlen=10)  # per-episode means (rolling)
    tracking_total_goals = 0
    tracking_emerged = False
    TRACKING_THRESHOLD = 1.0               # units; court half-height = 3.0
    TRACKING_WINDOW = 5                    # episodes for rolling mean
    if spectral_paddle:
        print('goal,episode_mean_err,rolling5_mean_err,emerged')

    window = 200
    anomaly_hist = deque(maxlen=window)
    history_t = deque(maxlen=window)

    # -- figure layout --------------------------------------------------------
    # Layout:  [ball] [env] [padL] [padR] [rew] [b×r]   FEATURE MAPS
    #          [            Court (game)              ]
    #          [Debug strip — only when spectral_paddle]
    #
    # Feature map grids for outer-product evaluation
    x_fm = np.linspace(COURT_LEFT, COURT_RIGHT, FM_NX)
    y_fm = np.linspace(COURT_BOTTOM, COURT_TOP, FM_NY)
    r_fm = np.linspace(-1.0, 1.0, FM_NY)  # reward domain, same height

    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('Spectral Pong — Wavepackets + Conv Feature Maps',
                 color=TITLE_COLOR, fontsize=14, fontweight='bold', y=0.98)
    # Layout:  [  feature maps (6 panels, full width)  ]   row 0 (thin)
    #          [ Y-wave | X-wave                        ]   row 1
    #          [ Y-wave | Court                         ]   row 2
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 3],
                          height_ratios=[0.55, 1, 2],
                          hspace=0.3, wspace=0.2,
                          left=0.06, right=0.97, top=0.93, bottom=0.04)
    gs_fm = GridSpecFromSubplotSpec(1, FM_CHANNELS, subplot_spec=gs[0, :],
                                    wspace=0.15)
    ax_fmaps = [fig.add_subplot(gs_fm[0, i]) for i in range(FM_CHANNELS)]
    ax_y     = fig.add_subplot(gs[1:3, 0])
    ax_x     = fig.add_subplot(gs[1, 1])
    ax_court = fig.add_subplot(gs[2, 1])

    for ax in ax_fmaps + [ax_y, ax_x, ax_court]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')

    # -- Feature map thumbnails (outer product 2D maps) -----------------------
    fm_artists = []
    fm_cmaps = ['inferno', 'viridis', PADDLE_COLOR_L, PADDLE_COLOR_R,
                'RdYlGn', 'coolwarm']
    # Use named colormaps for all; paddle colors handled below
    for i, ax_fm in enumerate(ax_fmaps):
        ax_fm.set_xticks([])
        ax_fm.set_yticks([])
        ax_fm.set_title(FM_LABELS[i], color=TITLE_COLOR, fontsize=9)
        cmap = 'inferno' if i <= 1 else ('RdYlGn' if i == 4 else
                'coolwarm' if i == 5 else 'viridis')
        im = ax_fm.imshow(np.zeros((FM_NY, FM_NX)),
                          origin='lower', aspect='auto', cmap=cmap,
                          vmin=-0.5, vmax=0.5, interpolation='bilinear')
        fm_artists.append(im)

    # -- X-domain wave panel --------------------------------------------------
    x_dom = np.linspace(COURT_LEFT - 1, COURT_RIGHT + 1, 400)
    ax_x.set_xlim(x_dom[0], x_dom[-1])
    ax_x.set_ylim(-0.05, 1.0)
    ax_x.set_title('X-domain spectral fields', color=TITLE_COLOR, fontsize=10)
    ax_x.set_ylabel('F²(x)', color=LABEL_COLOR, fontsize=8)
    ax_x.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.5)
    lx_ball,  = ax_x.plot([], [], color=BALL_COLOR,      lw=2,   label='Ball')
    lx_pad_l, = ax_x.plot([], [], color=PADDLE_COLOR_L,  lw=1.5, alpha=0.7, label='L paddle')
    lx_pad_r, = ax_x.plot([], [], color=PADDLE_COLOR_R,  lw=1.5, alpha=0.7, label='R paddle')
    lx_env,   = ax_x.plot([], [], color=ENV_FIELD_COLOR, lw=1.5, alpha=0.8, label='Env')
    lx_dot,   = ax_x.plot([], [], 'o', color=BALL_COLOR, markersize=8,
                          zorder=5, markeredgecolor='white', markeredgewidth=1)
    ax_x.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
                edgecolor='#333', labelcolor='#ccc')

    # -- Y-domain wave panel --------------------------------------------------
    y_dom = np.linspace(COURT_BOTTOM - 1, COURT_TOP + 1, 400)
    ax_y.set_ylim(y_dom[0], y_dom[-1])
    ax_y.set_xlim(-0.05, 1.5)
    ax_y.set_title('Y-domain', color=TITLE_COLOR, fontsize=10)
    ax_y.set_xlabel('F²(y)', color=LABEL_COLOR, fontsize=8)
    ax_y.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.5)
    ax_y.axhline(COURT_TOP,    color=RESIDUAL_COLOR, ls='--', lw=1, alpha=0.5)
    ax_y.axhline(COURT_BOTTOM, color=RESIDUAL_COLOR, ls='--', lw=1, alpha=0.5)
    ly_ball,  = ax_y.plot([], [], color=BALL_COLOR,      lw=2,   label='Ball')
    ly_pad_l, = ax_y.plot([], [], color=PADDLE_COLOR_L,  lw=1.5, alpha=0.7, label='L paddle')
    ly_pad_r, = ax_y.plot([], [], color=PADDLE_COLOR_R,  lw=1.5, alpha=0.7, label='R paddle')
    ly_env,   = ax_y.plot([], [], color=ENV_FIELD_COLOR, lw=1.5, alpha=0.8, label='Env')
    ly_dot,   = ax_y.plot([], [], 'o', color=BALL_COLOR, markersize=8,
                          zorder=5, markeredgecolor='white', markeredgewidth=1)
    ax_y.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e',
                edgecolor='#333', labelcolor='#ccc')

    # -- Court panel ----------------------------------------------------------
    ax_court.set_xlim(COURT_LEFT - 0.5, COURT_RIGHT + 0.5)
    ax_court.set_ylim(COURT_BOTTOM - 0.6, COURT_TOP + 0.6)
    ax_court.set_aspect('equal')
    ax_court.set_xticks([])
    ax_court.set_yticks([])
    for spine in ax_court.spines.values():
        spine.set_visible(False)

    court_rect = Rectangle(
        (COURT_LEFT, COURT_BOTTOM),
        COURT_RIGHT - COURT_LEFT, COURT_TOP - COURT_BOTTOM,
        linewidth=2, edgecolor=WALL_COLOR, facecolor='none', zorder=1)
    ax_court.add_patch(court_rect)

    for y_pos in np.linspace(COURT_BOTTOM + 0.15, COURT_TOP - 0.15, 15):
        ax_court.plot([0, 0], [y_pos, y_pos + 0.2], color=NET_COLOR,
                      lw=1.5, alpha=0.5, zorder=1)

    lp_patch = FancyBboxPatch(
        (paddle_lx - PADDLE_WIDTH / 2, -PADDLE_HEIGHT / 2),
        PADDLE_WIDTH, PADDLE_HEIGHT,
        boxstyle='round,pad=0.03', facecolor=PADDLE_COLOR_L,
        edgecolor='white', linewidth=1.5, zorder=3)
    rp_patch = FancyBboxPatch(
        (paddle_rx - PADDLE_WIDTH / 2, -PADDLE_HEIGHT / 2),
        PADDLE_WIDTH, PADDLE_HEIGHT,
        boxstyle='round,pad=0.03', facecolor=PADDLE_COLOR_R,
        edgecolor='white', linewidth=1.5, zorder=3)
    ax_court.add_patch(lp_patch)
    ax_court.add_patch(rp_patch)

    ball_dot, = ax_court.plot([0], [0], 'o', color=BALL_COLOR, markersize=10,
                              zorder=4, markeredgecolor='white',
                              markeredgewidth=1.0)

    score_text = ax_court.text(0, COURT_TOP + 0.35, '0 \u2014 0',
                               color=SCORE_COLOR, fontsize=18,
                               fontweight='bold',
                               horizontalalignment='center',
                               verticalalignment='center', zorder=5)

    anomaly_arrow = [None]
    frame_text = ax_court.text(0.98, 0.02, '', transform=ax_court.transAxes,
                               color='#555', fontsize=8,
                               horizontalalignment='right')
    stats_text = ax_court.text(0.02, 0.02, '', transform=ax_court.transAxes,
                               color='#aaa', fontsize=8,
                               horizontalalignment='left')

    # -- reward field heatmap overlay -------------------------------------------
    HM_NX = 50
    r_dom_hm = np.linspace(-1.0, 1.0, HM_NX)
    reward_heatmap = ax_court.imshow(
        np.zeros((1, HM_NX)),
        extent=[COURT_LEFT, COURT_RIGHT, COURT_BOTTOM, COURT_TOP],
        origin='lower', aspect='auto', cmap='RdYlGn',
        alpha=0.25, vmin=-2.0, vmax=2.0, zorder=0,
        interpolation='bilinear')

    # -- keyboard -------------------------------------------------------------
    def on_key_press(event):
        key_times[event.key] = time.time()
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # -- animation step -------------------------------------------------------

    def init():
        ball_dot.set_data([], [])
        frame_text.set_text('')
        stats_text.set_text('')
        return ()

    def step(_frame):
        nonlocal tracking_total_goals, tracking_emerged
        t = state['t']

        # Freeze after scoring
        if state['freeze'] > 0:
            state['freeze'] -= 1
            state['t'] = t + 1
            return ()

        # -- move paddles -------------------------------------------------
        half_h = PADDLE_HEIGHT / 2
        if auto_paddle:
            if spectral_paddle and rl_left is not None:
                # RL policy: wavepacket state → paddle velocity (one per paddle)
                fmaps_l = compute_feature_maps(
                    wp_ball, wp_env, wp_paddle_l, wp_paddle_r,
                    wp_reward_l, x_fm, y_fm, r_fm)
                fmaps_r = compute_feature_maps(
                    wp_ball, wp_env, wp_paddle_l, wp_paddle_r,
                    wp_reward_r, x_fm, y_fm, r_fm)
                rl_state_l = SimpleRLController.build_state(fmaps_l, rl_left.conv)
                rl_state_r = SimpleRLController.build_state(fmaps_r, rl_right.conv)
                act_l = rl_left.act(rl_state_l)
                act_r = rl_right.act(rl_state_r)
                left_paddle['y'] += act_l * PADDLE_SPEED * dt
                right_paddle['y'] += act_r * PADDLE_SPEED * dt
            else:
                # Simple tracking AI
                track_speed = PADDLE_SPEED * dt * 0.8
                if ball['y'] > left_paddle['y'] + 0.1:
                    left_paddle['y'] += track_speed
                elif ball['y'] < left_paddle['y'] - 0.1:
                    left_paddle['y'] -= track_speed
                if ball['y'] > right_paddle['y'] + 0.1:
                    right_paddle['y'] += track_speed
                elif ball['y'] < right_paddle['y'] - 0.1:
                    right_paddle['y'] -= track_speed
        else:
            now = time.time()
            held = 0.15
            if now - key_times.get('a', 0) < held:
                left_paddle['y'] += PADDLE_SPEED * dt
            if now - key_times.get('d', 0) < held:
                left_paddle['y'] -= PADDLE_SPEED * dt
            if now - key_times.get('up', 0) < held:
                right_paddle['y'] += PADDLE_SPEED * dt
            if now - key_times.get('down', 0) < held:
                right_paddle['y'] -= PADDLE_SPEED * dt

        left_paddle['y'] = np.clip(left_paddle['y'],
                                   COURT_BOTTOM + half_h, COURT_TOP - half_h)
        right_paddle['y'] = np.clip(right_paddle['y'],
                                    COURT_BOTTOM + half_h, COURT_TOP - half_h)

        # Accumulate per-frame tracking error (left paddle vs ball y)
        if spectral_paddle:
            tracking_buf.append(abs(left_paddle['y'] - ball['y']))

        # -- Newtonian ball step ------------------------------------------
        vx_before, vy_before = ball['vx'], ball['vy']

        ball['x'] += ball['vx'] * dt
        ball['y'] += ball['vy'] * dt

        # Top/bottom wall bounce
        if ball['y'] >= COURT_TOP - BALL_RADIUS:
            ball['y'] = 2 * (COURT_TOP - BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']
        elif ball['y'] <= COURT_BOTTOM + BALL_RADIUS:
            ball['y'] = 2 * (COURT_BOTTOM + BALL_RADIUS) - ball['y']
            ball['vy'] = -ball['vy']

        # Left paddle collision
        if (ball['vx'] < 0
                and ball['x'] - BALL_RADIUS <= paddle_lx + PADDLE_WIDTH / 2
                and ball['x'] - BALL_RADIUS >= paddle_lx - PADDLE_WIDTH / 2 - abs(ball['vx'] * dt)
                and abs(ball['y'] - left_paddle['y']) <= half_h + BALL_RADIUS):
            ball['x'] = paddle_lx + PADDLE_WIDTH / 2 + BALL_RADIUS
            ball['vx'] = -ball['vx']
            offset = (ball['y'] - left_paddle['y']) / half_h
            ball['vy'] += offset * SPIN_FACTOR
            spd = np.hypot(ball['vx'], ball['vy'])
            ball['vx'] *= ball_speed / spd
            ball['vy'] *= ball_speed / spd
            state['rally_touches'] += 1
            state['total_touches'] += 1

        # Right paddle collision
        if (ball['vx'] > 0
                and ball['x'] + BALL_RADIUS >= paddle_rx - PADDLE_WIDTH / 2
                and ball['x'] + BALL_RADIUS <= paddle_rx + PADDLE_WIDTH / 2 + abs(ball['vx'] * dt)
                and abs(ball['y'] - right_paddle['y']) <= half_h + BALL_RADIUS):
            ball['x'] = paddle_rx - PADDLE_WIDTH / 2 - BALL_RADIUS
            ball['vx'] = -ball['vx']
            offset = (ball['y'] - right_paddle['y']) / half_h
            ball['vy'] += offset * SPIN_FACTOR
            spd = np.hypot(ball['vx'], ball['vy'])
            ball['vx'] *= ball_speed / spd
            ball['vy'] *= ball_speed / spd
            state['rally_touches'] += 1
            state['total_touches'] += 1

        vx_after, vy_after = ball['vx'], ball['vy']

        # Scoring
        scored = False
        reward = 0.0
        if ball['x'] < COURT_LEFT:
            right_paddle['score'] += 1
            reward = -1.0   # left paddle conceded
            scored = True
        elif ball['x'] > COURT_RIGHT:
            left_paddle['score'] += 1
            reward = +1.0   # left paddle scored
            scored = True

        if scored:
            state['last_rally'] = state['rally_touches']
            state['rally_lengths'].append(state['rally_touches'])
            state['rally_touches'] = 0

            # Sample-efficiency measurement: log tracking error per episode
            if spectral_paddle and tracking_buf:
                ep_err = float(np.mean(tracking_buf))
                tracking_buf.clear()
                tracking_history.append(ep_err)
                tracking_total_goals += 1
                rolling = (float(np.mean(list(tracking_history)[-TRACKING_WINDOW:]))
                           if len(tracking_history) >= TRACKING_WINDOW else float('nan'))
                emerged_flag = ''
                if (not tracking_emerged
                        and len(tracking_history) >= TRACKING_WINDOW
                        and rolling < TRACKING_THRESHOLD):
                    tracking_emerged = True
                    emerged_flag = '*EMERGED*'
                    print(f'[TRACKING EMERGED] goal={tracking_total_goals}  '
                          f'rolling{TRACKING_WINDOW}={rolling:.3f}')
                print(f'{tracking_total_goals},{ep_err:.4f},'
                      f'{rolling:.4f},{emerged_flag}')
            # Replay reward frame N times through predict→correct→learn
            goal_pos_rew = np.array([ball['x'], ball['y'], reward])
            goal_vel = np.array([0.0, 0.0, 0.0])  # ball stopped at goal
            for _ in range(reward_replay_n):
                # 1. PREDICT with reward field forces
                nip_rew_l = abs(wp_ball.normalized_inner_product(wp_reward_l))
                nip_env_g = abs(wp_ball.normalized_inner_product(wp_env))
                rew_force = wp_ball.predict_force(
                    wp_reward_l, nip_rew_l, force_scale=force_scale)
                env_force = wp_ball.predict_force(
                    wp_env, nip_env_g, force_scale=force_scale)
                pred_rew = wp_ball.predict_position(
                    goal_vel, dt, env_force + rew_force)
                # 2. CORRECT: LMS toward goal pos with reward in dim 2
                wp_ball.update_with_attention(
                    goal_pos_rew, np.ones(NDIM),
                    [nip_env_g, nip_rew_l])
                wp_reward_l.update_lms(
                    goal_pos_rew, np.array([0.0, 0.0, reward]),
                    anomaly_scale=1.0)
                wp_reward_r.update_lms(
                    goal_pos_rew, np.array([0.0, 0.0, -reward]),
                    anomaly_scale=1.0)
                wp_reward_l.soft_normalize(max_energy=2.0)
                wp_reward_r.soft_normalize(max_energy=2.0)
                wp_ball.normalize()
                # 3. LEARN: residual in dim 2 trains frequencies
                if lr_k > 0:
                    wp_ball.learn_from_residual(pred_rew, goal_pos_rew,
                                                lr_k=lr_k)
                wp_ball.pos[:] = goal_pos_rew
            # Show paddles with signed reward dim for NIP learning
            paddle_l_goal = np.array([paddle_lx, left_paddle['y'], reward])
            paddle_r_goal = np.array([paddle_rx, right_paddle['y'], -reward])
            wp_paddle_l.update_with_attention(
                paddle_l_goal, np.ones(NDIM), [1.0])
            wp_paddle_r.update_with_attention(
                paddle_r_goal, np.ones(NDIM), [1.0])

            if rl_left is not None:
                rl_left.reward_update(ball['x'], reward)
                rl_right.reward_update(ball['x'], -reward)
                term_state = np.zeros(SimpleRLController.STATE_DIM)
                rl_left.step(term_state, reward)
                rl_right.step(term_state, -reward)
                rl_left.on_reset()
                rl_right.on_reset()
            reset_ball(ball, toward='random', speed=ball_speed)
            state['freeze'] = int(0.5 / dt)
            paddle_vel['l'] = 0.0
            paddle_vel['r'] = 0.0
            # Re-init ball wavepacket on score (3D)
            wp_ball.__init__(K, frequencies,
                             pos0=(ball['x'], ball['y'], 0.0),
                             mass=1.0, sigma=ball_sigma, ndim=NDIM,
                             lr=ball_lr, lr_tracking=lr_tracking)

        # -- update wavepackets: predict → correct → learn ----------------
        ball_pos = np.array([ball['x'], ball['y'], 0.0])
        paddle_l_pos = np.array([paddle_lx, left_paddle['y'], 0.0])
        paddle_r_pos = np.array([paddle_rx, right_paddle['y'], 0.0])

        # 1. PREDICT: env + reward field forces, step forward
        if not scored:
            nip_ball_env = abs(wp_ball.normalized_inner_product(wp_env))
            nip_ball_padL = abs(wp_ball.normalized_inner_product(wp_paddle_l))
            nip_ball_padR = abs(wp_ball.normalized_inner_product(wp_paddle_r))
            nip_ball_rew = abs(wp_ball.normalized_inner_product(wp_reward_l))

            # Env force: dims 0,1 (walls). Reward force: dim 2 only.
            env_force = wp_ball.predict_force(wp_env, nip_ball_env,
                                              force_scale=force_scale)
            rew_force = wp_ball.predict_force(wp_reward_l, nip_ball_rew,
                                              force_scale=force_scale)
            total_force = env_force + rew_force
            ball_vel = np.array([ball['vx'], ball['vy'], 0.0])
            predicted_pos = wp_ball.predict_position(ball_vel, dt, total_force)
        else:
            nip_ball_env = nip_ball_padL = nip_ball_padR = 0.0
            nip_ball_rew = 0.0
            predicted_pos = ball_pos.copy()

        # Shift ball wavepacket by velocity (dims 0,1 only; dim 2 = 0)
        wp_ball.shift(ball['vx'] * dt, axis=0)
        wp_ball.shift(ball['vy'] * dt, axis=1)
        # Paddle shifts (y-axis only)
        delta_l = left_paddle['y'] - wp_paddle_l.pos[1]
        delta_r = right_paddle['y'] - wp_paddle_r.pos[1]
        if abs(delta_l) > 1e-12:
            wp_paddle_l.shift(delta_l, axis=1)
        if abs(delta_r) > 1e-12:
            wp_paddle_r.shift(delta_r, axis=1)

        # 2. CORRECT: LMS toward observed position [x, y, 0]
        if not scored:
            unity = np.ones(NDIM)
            wp_ball.update_with_attention(
                ball_pos, unity,
                [nip_ball_env, nip_ball_padL, nip_ball_padR])
            wp_paddle_l.update_with_attention(
                paddle_l_pos, unity, [nip_ball_padL])
            wp_paddle_r.update_with_attention(
                paddle_r_pos, unity, [nip_ball_padR])

        # 3. Prediction residual (3D: physics + reward)
        prediction_residual = ball_pos - predicted_pos

        # 4. Measure integral deviation BEFORE normalizing
        ball_deviation = np.array([wp_ball.integrate_squared(d) - 1.0
                                   for d in range(NDIM)])

        # 5. Normalize observed objects back to PMF
        wp_ball.normalize()
        wp_paddle_l.normalize()
        wp_paddle_r.normalize()

        # 6. Env learns from prediction residual (velocity explained → only structural
        #    events like wall bounces drive env, not free-flight kinetic motion).
        #    Restrict to Y axis only to match env basis [0,1,0].
        pred_res_y = np.zeros(NDIM)
        pred_res_y[1] = prediction_residual[1]  # Y only
        pred_res_mag = abs(prediction_residual[1])
        if not scored and pred_res_mag > 1e-8:
            total_nip = nip_ball_env + nip_ball_padL + nip_ball_padR + 1e-8
            env_fraction = nip_ball_env / total_nip
            wp_env.update_lms(ball_pos, pred_res_y,
                              anomaly_scale=pred_res_mag * env_fraction)
            # Hard-enforce Y-only basis: zero out X and reward dims
            wp_env.c_cos[:, 0] = 0.0
            wp_env.c_sin[:, 0] = 0.0
            wp_env.c_cos[:, 2] = 0.0
            wp_env.c_sin[:, 2] = 0.0
            wp_env.normalize()

        # 7. LEARN: update spatial frequencies from prediction residual
        if not scored and lr_k > 0:
            wp_ball.learn_from_residual(predicted_pos, ball_pos, lr_k=lr_k)

        # 8. Update stored positions
        wp_ball.pos[:] = ball_pos
        wp_paddle_l.pos[:] = paddle_l_pos
        wp_paddle_r.pos[:] = paddle_r_pos

        # TD update AFTER wavepacket updates
        fmaps_ns_l = compute_feature_maps(
            wp_ball, wp_env, wp_paddle_l, wp_paddle_r,
            wp_reward_l, x_fm, y_fm, r_fm)
        if rl_left is not None and not scored:
            fmaps_ns_r = compute_feature_maps(
                wp_ball, wp_env, wp_paddle_l, wp_paddle_r,
                wp_reward_r, x_fm, y_fm, r_fm)
            ns_l = SimpleRLController.build_state(fmaps_ns_l, rl_left.conv)
            ns_r = SimpleRLController.build_state(fmaps_ns_r, rl_right.conv)
            rl_left.step(ns_l, 0.0)
            rl_right.step(ns_r, 0.0)

        # Deviation magnitude for display
        a_residual = ball_deviation

        state['anomaly'] = a_residual
        state['max_anomaly'] = max(np.linalg.norm(a_residual),
                                   state['max_anomaly'] * 0.98)

        anomaly_hist.append(np.linalg.norm(a_residual))
        history_t.append(t)
        state['t'] = t + 1

        # -- draw 2D feature maps (outer products) -------------------------
        for i in range(FM_CHANNELS):
            fm_artists[i].set_array(fmaps_ns_l[i])

        # -- draw X-domain wave graphs ------------------------------------
        lx_ball.set_data(x_dom,  wp_ball.evaluate(x_dom, axis=0) ** 2)
        lx_pad_l.set_data(x_dom, wp_paddle_l.evaluate(x_dom, axis=0) ** 2)
        lx_pad_r.set_data(x_dom, wp_paddle_r.evaluate(x_dom, axis=0) ** 2)
        lx_env.set_data(x_dom,   wp_env.evaluate(x_dom, axis=0) ** 2)
        lx_dot.set_data([ball['x']], [0])

        # -- draw Y-domain wave graphs ------------------------------------
        ly_ball.set_data(wp_ball.evaluate(y_dom, axis=1) ** 2,     y_dom)
        ly_pad_l.set_data(wp_paddle_l.evaluate(y_dom, axis=1) ** 2, y_dom)
        ly_pad_r.set_data(wp_paddle_r.evaluate(y_dom, axis=1) ** 2, y_dom)
        ly_env.set_data(wp_env.evaluate(y_dom, axis=1) ** 2,        y_dom)
        ly_dot.set_data([0], [ball['y']])

        # -- draw court ---------------------------------------------------
        ball_dot.set_data([ball['x']], [ball['y']])
        lp_patch.set_y(left_paddle['y'] - half_h)
        rp_patch.set_y(right_paddle['y'] - half_h)
        score_text.set_text(
            f'{left_paddle["score"]} \u2014 {right_paddle["score"]}')
        frame_text.set_text(f't={t}')
        rally_lengths = state['rally_lengths']
        avg_rally = (sum(rally_lengths) / len(rally_lengths)
                     if rally_lengths else 0.0)
        stats_text.set_text(
            f'touches: {state["total_touches"]}  '
            f'rally: {state["rally_touches"]}  '
            f'last: {state["last_rally"]}  '
            f'avg: {avg_rally:.1f}'
        )

        # Anomaly arrow on court
        if anomaly_arrow[0] is not None:
            anomaly_arrow[0].remove()
            anomaly_arrow[0] = None
        ma = max(state['max_anomaly'], 0.01)
        a_norm = np.linalg.norm(a_residual)
        arrow_len = min(a_norm / ma * 1.5, 2.0)
        if arrow_len > 0.05 and a_norm > 0.01:
            direction = a_residual / a_norm
            anomaly_arrow[0] = ax_court.annotate(
                '', xy=(ball['x'] + direction[0] * arrow_len,
                        ball['y'] + direction[1] * arrow_len),
                xytext=(ball['x'], ball['y']),
                arrowprops=dict(arrowstyle='->', color=RESIDUAL_COLOR,
                                lw=2.5, mutation_scale=15), zorder=6)

        # -- update reward heatmap -----------------------------------------
        rew_vals = wp_reward_l.evaluate(r_dom_hm, axis=2)
        reward_heatmap.set_array(rew_vals.reshape(1, -1))



        return ()

    frames = range(max_frames) if max_frames is not None else itertools.count()
    anim = FuncAnimation(fig, step, frames=frames, init_func=init,
                         interval=interval, blit=False, repeat=False)
    return fig, anim


# -- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='2D Spectral Pong Visualization')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g. out.gif)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Frame limit (required for --save)')
    parser.add_argument('--K', type=int, default=8,
                        help='Number of spectral components (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='Force coupling strength (default: 0.15)')
    parser.add_argument('--lr', type=float, default=0.0015,
                        help='LMS learning rate (default: 0.0015)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Ball speed multiplier (default: 1.0)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('--auto-paddle', action='store_true',
                        help='AI-controlled paddles (required for --save)')
    parser.add_argument('--spectral-paddle', action='store_true',
                        help='Use spectral inner-product force for paddle AI '
                             '(with --auto-paddle)')
    parser.add_argument('--alpha-paddle', type=float, default=0.5,
                        help='Paddle-ball spectral coupling strength '
                             '(default: 0.5)')
    parser.add_argument('--paddle-damping', type=float, default=0.85,
                        help='Paddle velocity damping per frame '
                             '(default: 0.85)')
    parser.add_argument('--beta-paddle', type=float, default=0.3,
                        help='Wall-anticipation coupling strength '
                             '(default: 0.3)')
    parser.add_argument('--gamma-paddle', type=float, default=0.2,
                        help='Reward-gradient coupling strength '
                             '(default: 0.2)')
    parser.add_argument('--ball-lr', type=float, default=0.0015,
                        help='Ball interaction LMS learning rate '
                             '(default: 0.0015)')
    parser.add_argument('--paddle-lr', type=float, default=0.001,
                        help='Paddle interaction LMS learning rate '
                             '(default: 0.001)')
    parser.add_argument('--lr-tracking', type=float, default=0.0001,
                        help='Always-on tracking LMS learning rate '
                             '(default: 0.0001)')
    args = parser.parse_args()

    if args.save and args.frames is None:
        parser.error('--frames is required when using --save')

    K = args.K
    if K == 8:
        frequencies = np.array([0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 3.0, 3.7])
    elif K == 4:
        frequencies = np.array([0.3, 0.7, 1.5, 2.5])
    else:
        frequencies = np.linspace(0.3, 3.7, K)

    if args.save:
        matplotlib.use('Agg')

    interval = int(1000 / args.fps)
    dt = interval / 1000.0

    fig, anim = create_game(
        K=K,
        frequencies=frequencies,
        alpha=args.alpha,
        dt=dt,
        ball_speed=BALL_SPEED * args.speed,
        interval=interval,
        max_frames=args.frames,
        auto_paddle=args.auto_paddle,
        ball_sigma=0.8,
        env_lr=args.lr,
        spectral_paddle=args.spectral_paddle,
        alpha_paddle=args.alpha_paddle,
        paddle_damping=args.paddle_damping,
        beta_paddle=args.beta_paddle,
        gamma_paddle=args.gamma_paddle,
        ball_lr=args.ball_lr,
        paddle_lr=args.paddle_lr,
        lr_tracking=args.lr_tracking,
    )

    if args.save:
        save_path = Path(args.save)
        if save_path.suffix == '.gif':
            anim.save(str(save_path), writer='pillow', fps=args.fps)
        else:
            anim.save(str(save_path), fps=args.fps)
        print(f'Saved to {save_path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
