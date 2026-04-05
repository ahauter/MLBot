"""
Spectral Wavepacket Library
============================
Core wavepacket representation and feature map computation for spectral
encoding experiments. Domain-agnostic — used by both the Pong environment
(wavepacket tracking + LMS updates) and the GPU conv encoder (feature maps).

Classes:
  WavepacketObject2D   - N-dimensional Fourier wavepacket with LMS learning
  ConvFeatureExtractor  - Numpy conv on outer-product maps (used by viz only)

Functions:
  compute_feature_maps  - 6-channel outer-product maps from wavepackets
"""
from __future__ import annotations

import numpy as np


# -- Physical constants (shared with pong environment) -------------------------

COEFF_CLIP = 10.0

COURT_LEFT = -5.0
COURT_RIGHT = 5.0
COURT_TOP = 3.0
COURT_BOTTOM = -3.0

WORLD_BOUNDS = [
    (COURT_LEFT, COURT_RIGHT),   # axis 0 (x)
    (COURT_BOTTOM, COURT_TOP),   # axis 1 (y)
    (-1.0, 1.0),                 # axis 2 (reward)
]

PADDLE_X_OFFSET = 0.5
PADDLE_WIDTH = 0.15
PADDLE_HEIGHT = 1.0
PADDLE_SPEED = 4.0

BALL_RADIUS = 0.15
BALL_SPEED = 2.0
SPIN_FACTOR = 2.0

# Feature map grid constants
FM_NX, FM_NY = 24, 16  # outer-product grid resolution (court aspect ~5:3)
FM_CHANNELS = 6        # ball, env, padL, padR, reward, ball×reward
FM_LABELS = ['ball', 'env', 'padL', 'padR', 'rew', 'b×r']


# -- WavepacketObject2D -------------------------------------------------------

class WavepacketObject2D:
    """Spectral wavepacket with N-dimensional amplitude vectors.

    Each frequency k_j has coefficients c_cos[j] and c_sin[j] in R^ndim.
    The field is evaluated on a 1D domain per axis:
        F_d(x) = sum_j  c_cos[j, d] * cos(k_j * x) + c_sin[j, d] * sin(k_j * x)
    where d in {0, ..., ndim-1} selects the component.

    Default ndim=2 (x, y) for backward compatibility.  Use ndim=3 to add
    a reward dimension: dims 0,1 = physics, dim 2 = reward.
    """

    def __init__(self, K: int, frequencies: np.ndarray,
                 pos0: tuple = (0.0, 0.0),
                 mass: float = 1.0,
                 vel0: tuple = (0.0, 0.0),
                 c_cos: np.ndarray | None = None,
                 c_sin: np.ndarray | None = None,
                 sigma: float = 0.8, amplitude: float = 1.5,
                 lr: float = 0.0, lr_tracking: float = 0.0,
                 ndim: int | None = None):
        self.K = K
        self.k = np.asarray(frequencies, dtype=np.float64)
        self.mass = mass
        self.pos = np.array(pos0, dtype=np.float64)
        self.vel = np.array(vel0, dtype=np.float64)
        self.lr = lr
        self.lr_tracking = lr_tracking

        if ndim is not None:
            self.ndim = ndim
        else:
            self.ndim = len(self.pos)

        if len(self.pos) < self.ndim:
            self.pos = np.concatenate([self.pos,
                                       np.zeros(self.ndim - len(self.pos))])
        if len(self.vel) < self.ndim:
            self.vel = np.concatenate([self.vel,
                                       np.zeros(self.ndim - len(self.vel))])

        if c_cos is not None and c_sin is not None:
            self.c_cos = np.array(c_cos, dtype=np.float64)
            self.c_sin = np.array(c_sin, dtype=np.float64)
        else:
            envelope = amplitude * np.exp(-self.k**2 * sigma**2 / 2)
            self.c_cos = np.zeros((K, self.ndim), dtype=np.float64)
            self.c_sin = np.zeros((K, self.ndim), dtype=np.float64)
            for d in range(self.ndim):
                self.c_cos[:, d] = envelope * np.cos(self.k * self.pos[d])
                self.c_sin[:, d] = envelope * np.sin(self.k * self.pos[d])

        self.normalize()

    def _basis(self, x: float) -> np.ndarray:
        return np.concatenate([np.cos(self.k * x), np.sin(self.k * x)])

    def _c_flat(self, axis: int) -> np.ndarray:
        return np.concatenate([self.c_cos[:, axis], self.c_sin[:, axis]])

    def _set_c_flat(self, axis: int, c: np.ndarray) -> None:
        self.c_cos[:, axis] = c[:self.K]
        self.c_sin[:, axis] = c[self.K:]

    def evaluate(self, x_domain: np.ndarray, axis: int) -> np.ndarray:
        cos_b = np.cos(self.k[None, :] * x_domain[:, None])
        sin_b = np.sin(self.k[None, :] * x_domain[:, None])
        return cos_b @ self.c_cos[:, axis] + sin_b @ self.c_sin[:, axis]

    def predict(self, x: float, axis: int) -> float:
        return float(self._c_flat(axis) @ self._basis(x))

    def integrate(self, axis: int) -> float:
        a, b = WORLD_BOUNDS[axis]
        int_cos = (np.sin(self.k * b) - np.sin(self.k * a)) / self.k
        int_sin = (np.cos(self.k * a) - np.cos(self.k * b)) / self.k
        return float(self.c_cos[:, axis] @ int_cos +
                     self.c_sin[:, axis] @ int_sin)

    def integrate_squared(self, axis: int) -> float:
        a, b = WORLD_BOUNDS[axis]
        L = b - a
        cc = self.c_cos[:, axis]
        ss = self.c_sin[:, axis]
        return float(0.5 * L * np.sum(cc**2 + ss**2))

    def normalize(self) -> None:
        for d in range(self.ndim):
            S = self.integrate_squared(d)
            if S > 1e-20:
                self.c_cos[:, d] /= np.sqrt(S)
                self.c_sin[:, d] /= np.sqrt(S)

    def soft_normalize(self, max_energy: float = 2.0) -> None:
        for d in range(self.ndim):
            E = self.integrate_squared(d)
            if E > max_energy:
                s = np.sqrt(max_energy / E)
                self.c_cos[:, d] *= s
                self.c_sin[:, d] *= s

    def outer_product_map(self, x_grid: np.ndarray, y_grid: np.ndarray,
                          ax0: int = 0, ax1: int = 1) -> np.ndarray:
        fx = self.evaluate(x_grid, axis=ax0)
        fy = self.evaluate(y_grid, axis=ax1)
        return np.outer(fy, fx)

    def gradient(self, x: float, axis: int) -> float:
        return float(np.sum(
            -self.c_cos[:, axis] * self.k * np.sin(self.k * x)
            + self.c_sin[:, axis] * self.k * np.cos(self.k * x)))

    def inner_product(self, other: 'WavepacketObject2D') -> np.ndarray:
        return np.array([
            float(np.sum(self.c_cos[:, d] * other.c_cos[:, d] +
                         self.c_sin[:, d] * other.c_sin[:, d]))
            for d in range(self.ndim)
        ])

    inner_product_2d = inner_product

    def cross_product(self, other: 'WavepacketObject2D') -> np.ndarray:
        return np.array([
            float(np.sum(self.c_cos[:, d] * other.c_sin[:, d] -
                         self.c_sin[:, d] * other.c_cos[:, d]))
            for d in range(self.ndim)
        ])

    cross_product_2d = cross_product

    def normalized_inner_product(self, other: 'WavepacketObject2D') -> float:
        ip = self.inner_product(other)
        norm_s = np.sqrt(np.sum(self.c_cos**2 + self.c_sin**2, axis=0))
        norm_o = np.sqrt(np.sum(other.c_cos**2 + other.c_sin**2, axis=0))
        nip = ip / (norm_s * norm_o + 1e-8)
        return float(np.linalg.norm(nip) / np.sqrt(self.ndim))

    def shift(self, delta: float, axis: int) -> None:
        angles = self.k * delta
        ca = np.cos(angles)
        sa = np.sin(angles)
        oc = self.c_cos[:, axis].copy()
        os = self.c_sin[:, axis].copy()
        self.c_cos[:, axis] = oc * ca - os * sa
        self.c_sin[:, axis] = oc * sa + os * ca

    def predict_force(self, field_wp: 'WavepacketObject2D',
                      nip: float,
                      force_scale: float = 0.5) -> np.ndarray:
        force = np.zeros(self.ndim)
        for d in range(self.ndim):
            grad = field_wp.gradient(self.pos[d], axis=d)
            force[d] = -force_scale * nip * grad
        return force

    def predict_position(self, vel: np.ndarray, dt: float,
                         total_force: np.ndarray) -> np.ndarray:
        corrected_vel = vel + total_force * dt
        return self.pos + corrected_vel * dt

    def learn_from_residual(self, predicted_pos: np.ndarray,
                            observed_pos: np.ndarray,
                            lr_k: float = 0.001) -> np.ndarray:
        residual = observed_pos - predicted_pos
        for d in range(self.ndim):
            x = self.pos[d]
            dF_dk = (-self.c_cos[:, d] * x * np.sin(self.k * x)
                     + self.c_sin[:, d] * x * np.cos(self.k * x))
            self.k += lr_k * residual[d] * dF_dk
        self.k = np.clip(self.k, 0.1, 5.0)
        return residual

    def set_position(self, new_pos: np.ndarray) -> None:
        for d in range(self.ndim):
            delta = new_pos[d] - self.pos[d]
            if abs(delta) > 1e-12:
                self.shift(delta, axis=d)
        self.pos[:] = new_pos
        np.clip(self.c_cos, -COEFF_CLIP, COEFF_CLIP, out=self.c_cos)
        np.clip(self.c_sin, -COEFF_CLIP, COEFF_CLIP, out=self.c_sin)

    def update_lms(self, domain_pos: np.ndarray, target: np.ndarray,
                   anomaly_scale: float = 1.0,
                   lr: float | None = None) -> np.ndarray:
        effective_lr = (lr if lr is not None else self.lr) * anomaly_scale
        if effective_lr <= 0:
            return np.zeros(self.ndim)
        residual = np.zeros(self.ndim)
        for d in range(self.ndim):
            basis = self._basis(domain_pos[d])
            c = self._c_flat(d)
            pred = float(c @ basis)
            res = target[d] - pred
            residual[d] = res
            c_new = c + effective_lr * basis * res
            np.clip(c_new, -COEFF_CLIP, COEFF_CLIP, out=c_new)
            self._set_c_flat(d, c_new)
        return residual

    def update_with_attention(self, domain_pos: np.ndarray,
                              target: np.ndarray,
                              interaction_scales: list[float]) -> np.ndarray:
        effective_lr = self.lr_tracking + self.lr * sum(interaction_scales)
        if effective_lr <= 0:
            return np.zeros(self.ndim)
        residual = np.zeros(self.ndim)
        for d in range(self.ndim):
            basis = self._basis(domain_pos[d])
            c = self._c_flat(d)
            res = target[d] - float(c @ basis)
            residual[d] = res
            c_new = c + effective_lr * basis * res
            np.clip(c_new, -COEFF_CLIP, COEFF_CLIP, out=c_new)
            self._set_c_flat(d, c_new)
        return residual


# -- Feature map computation ---------------------------------------------------

def compute_feature_maps(wp_ball, wp_env, wp_paddle_l, wp_paddle_r,
                         wp_reward_l, x_fm, y_fm, r_fm):
    """Compute 6-channel outer-product feature maps. Returns (6, Ny, Nx)."""
    maps = np.empty((FM_CHANNELS, FM_NY, FM_NX))
    maps[0] = wp_ball.outer_product_map(x_fm, y_fm, ax0=0, ax1=1)
    maps[1] = wp_env.outer_product_map(x_fm, y_fm, ax0=0, ax1=1)
    maps[2] = wp_paddle_l.outer_product_map(x_fm, y_fm, ax0=0, ax1=1)
    maps[3] = wp_paddle_r.outer_product_map(x_fm, y_fm, ax0=0, ax1=1)
    maps[4] = wp_reward_l.outer_product_map(x_fm, y_fm, ax0=0, ax1=1)
    maps[5] = wp_ball.outer_product_map(x_fm, r_fm, ax0=0, ax1=2)
    return maps


# -- Numpy conv feature extractor (used by visualization only) ----------------

class ConvFeatureExtractor:
    """2D convolution on outer-product feature maps. Numpy-only.

    Forward: valid conv2d -> ReLU -> global avg pool -> (n_filters,) vector.
    Filters are fixed random projections (not learned via TD).
    """

    def __init__(self, n_channels: int = FM_CHANNELS, n_filters: int = 8,
                 kernel_size: int = 3, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_filters = n_filters
        self.ks = kernel_size
        fan_in = n_channels * kernel_size * kernel_size
        self.W = rng.randn(n_filters, n_channels,
                           kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(n_filters)

    def forward(self, maps: np.ndarray) -> np.ndarray:
        """maps: (C, H, W) -> feature vector (n_filters,)."""
        C, H, W = maps.shape
        oH = H - self.ks + 1
        oW = W - self.ks + 1
        out = np.zeros((self.n_filters, oH, oW))
        for f in range(self.n_filters):
            for c in range(C):
                for i in range(oH):
                    for j in range(oW):
                        out[f, i, j] += np.sum(
                            maps[c, i:i+self.ks, j:j+self.ks] * self.W[f, c])
            out[f] += self.b[f]
        out = np.maximum(out, 0.0)
        return out.mean(axis=(1, 2))

    def forward_fast(self, maps: np.ndarray) -> np.ndarray:
        """Vectorized valid conv2d -> ReLU -> global avg pool."""
        C, H, W = maps.shape
        ks = self.ks
        oH, oW = H - ks + 1, W - ks + 1
        patches = np.empty((oH * oW, C * ks * ks))
        idx = 0
        for i in range(oH):
            for j in range(oW):
                patches[idx] = maps[:, i:i+ks, j:j+ks].ravel()
                idx += 1
        W_flat = self.W.reshape(self.n_filters, -1)
        pre_relu = patches @ W_flat.T + self.b
        gate = (pre_relu > 0).astype(np.float64)
        self._last_patches = patches
        self._last_gate = gate
        self._last_n_pos = oH * oW
        return (pre_relu * gate).mean(axis=0)
