"""
SE(3) spectral field primitives.

All operations are pure functions on tensors — no nn.Module needed.
Coefficients use real spherical harmonics with layout:
    [l=0 (1 coeff), l=1 (3 coeffs), l=2 (5 coeffs)] = 9 total
"""

from typing import NamedTuple

import torch
from torch import Tensor

from rlbot.constants import L_MAX, N_COEFFS, DEGREE_SLICES, DEGREE_SIZES


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class SE3Field(NamedTuple):
    position: Tensor      # (3,)   center in normalized coordinates
    coefficients: Tensor  # (9,)   real spherical harmonic coefficients
    covariance: Tensor    # (9,)   diagonal covariance per coefficient


def make_field(position: Tensor, coefficients: Tensor,
               covariance: Tensor | None = None) -> SE3Field:
    """Construct an SE3Field with default covariance if not provided."""
    if covariance is None:
        covariance = torch.ones(N_COEFFS, dtype=coefficients.dtype)
    return SE3Field(position, coefficients, covariance)


# ---------------------------------------------------------------------------
# Wigner D-matrices (explicit for l=0, 1, 2)
# ---------------------------------------------------------------------------

def _rotation_matrix_from_euler(yaw: Tensor, pitch: Tensor,
                                roll: Tensor) -> Tensor:
    """ZYX Euler angles → 3×3 rotation matrix.

    Convention matches src/util/orientation.py:
        forward = (cp*cy, cp*sy, sp)
    where cy=cos(yaw), sy=sin(yaw), cp=cos(pitch), sp=sin(pitch).
    """
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    # Row-major rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = torch.stack([
        torch.stack([cp*cy, cy*sp*sr - cr*sy, -cr*cy*sp - sr*sy]),
        torch.stack([cp*sy, sy*sp*sr + cr*cy, -cr*sy*sp + sr*cy]),
        torch.stack([sp,    -cp*sr,            cp*cr]),
    ])
    return R


def wigner_d_l0() -> Tensor:
    """Wigner D-matrix for l=0: trivially [[1]]."""
    return torch.ones(1, 1)


def wigner_d_l1(yaw: Tensor, pitch: Tensor, roll: Tensor) -> Tensor:
    """Wigner D-matrix for l=1: the 3×3 rotation matrix.

    We use the convention where l=1 coefficients represent (x, y, z) directly,
    so D^1 = R (the standard rotation matrix).
    """
    return _rotation_matrix_from_euler(yaw, pitch, roll)


def wigner_d_l2(yaw: Tensor, pitch: Tensor, roll: Tensor) -> Tensor:
    """Wigner D-matrix for l=2: 5×5 orthogonal matrix.

    Computed from the rotation matrix R via the rank-2 tensor representation.

    The l=2 real SH basis functions are (in our ordering m=-2,-1,0,1,2):
        f_{-2} = xy,  f_{-1} = yz,  f_0 = (3z²-r²)/2,  f_1 = xz,  f_2 = (x²-y²)/2

    Under rotation x → Rx, each quadratic form transforms. We compute how
    each basis function maps, extracting coefficients in terms of the same basis.

    The normalization is chosen so that D^2 is orthogonal. This requires using
    the standard real SH basis where the (3z²-r²)/2 term has the same inner
    product norm as the off-diagonal terms. We achieve this by tracking the
    overlap of the rotated Cartesian products with each basis function.
    """
    R = _rotation_matrix_from_euler(yaw, pitch, roll)

    # Under rotation x_i → R_{ij} x_j, a product x_a x_b → (Rx)_a (Rx)_b
    # = Σ_{ij} R_{ai} R_{bj} x_i x_j
    #
    # We express each rotated basis function in terms of {x_i x_j} products,
    # then project back onto our 5 basis functions.
    #
    # The 5 basis functions span the traceless symmetric rank-2 tensor space.
    # The trace part (r² = x²+y²+z²) is invariant and decouples.
    #
    # For each column m (input basis), we apply the rotation and extract
    # the coefficient in each row m' (output basis).
    #
    # Input basis vectors as symmetric tensors (upper triangle):
    #   m=-2: T^{-2}_{ij} = (δ_{i0}δ_{j1} + δ_{i1}δ_{j0})/2  → xy
    #   m=-1: T^{-1}_{ij} = (δ_{i1}δ_{j2} + δ_{i2}δ_{j1})/2  → yz
    #   m=0:  T^0_{ij}    = δ_{i2}δ_{j2} - δ_{ij}/3           → (3z²-r²)/3
    #   m=1:  T^1_{ij}    = (δ_{i0}δ_{j2} + δ_{i2}δ_{j0})/2  → xz
    #   m=2:  T^2_{ij}    = (δ_{i0}δ_{j0} - δ_{i1}δ_{j1})/2  → (x²-y²)/2
    #
    # Under rotation: T'_{ij} = R_{ia} R_{jb} T_{ab}
    #
    # Extraction:
    #   coeff of f_{-2} in T' = T'_{01} + T'_{10}  = 2*T'_{01}
    #   coeff of f_{-1} in T' = T'_{12} + T'_{21}  = 2*T'_{12}
    #   coeff of f_0 in T'    = T'_{22} - (T'_{00} + T'_{11} + T'_{22})/3
    #                         = (2*T'_{22} - T'_{00} - T'_{11}) / 3 * ... hmm
    #
    # Actually, let's use the direct inner product approach.
    # D^2_{m', m} = Σ_{ij} T^{m'}_{ij} * (R @ T^m @ R^T)_{ij}
    #             / (Σ_{ij} T^{m'}_{ij} * T^{m'}_{ij})
    #
    # But since we want an orthogonal matrix and our basis is orthogonal
    # with equal norms, we just need:
    # D^2_{m',m} = Σ_{ij} T^{m'}_{ij} * Σ_{ab} R_{ia} R_{jb} T^m_{ab}
    #
    # divided by the common norm (which cancels in an orthogonal matrix).
    #
    # For unnormalized basis functions, the norms are:
    #   ||xy||² = Σ T^{-2}_{ij}² = 2*(1/2)² = 1/2
    #   ||yz||² = 1/2, ||xz||² = 1/2, ||(x²-y²)/2||² = 2*(1/2)² = 1/2
    #   ||(3z²-r²)/3||² = Σ (δ_{i2}δ_{j2} - δ_{ij}/3)²
    #     = (1-1/3)² + (0-1/3)² + (0-1/3)² = (2/3)² + 2*(1/3)² = 4/9+2/9 = 6/9 = 2/3
    #
    # So the m=0 basis has norm 2/3 while others have norm 1/2.
    # To make D^2 orthogonal, we need the basis to be orthonormal.
    # Scale factor: T^0 → T^0 * sqrt(1/2) / sqrt(2/3) = T^0 * sqrt(3/4) = T^0 * √3/2

    # I'll compute D^2 in the unnormalized basis first, then apply the
    # similarity transform to make it orthogonal.

    D2 = torch.zeros(5, 5, dtype=R.dtype)

    # Helper to compute: Σ_{ij} A_{ij} * Σ_{ab} R_{ia} R_{jb} B_{ab}
    # = Σ_{ijab} A_{ij} R_{ia} R_{jb} B_{ab}
    # = Σ_{ab} (R^T A R)_{ab} * B_{ab}  ... wait
    # = Σ_{ij} A_{ij} (R B R^T)_{ij}
    # = tr(A^T @ R @ B @ R^T)
    # For symmetric A, B: = tr(A @ R @ B @ R^T)

    # Build the 5 basis tensors as 3x3 symmetric matrices
    T = torch.zeros(5, 3, 3, dtype=R.dtype)
    # m=-2: xy
    T[0, 0, 1] = 0.5;  T[0, 1, 0] = 0.5
    # m=-1: yz
    T[1, 1, 2] = 0.5;  T[1, 2, 1] = 0.5
    # m=0: z² - r²/3 (traceless z²)
    T[2, 0, 0] = -1.0/3;  T[2, 1, 1] = -1.0/3;  T[2, 2, 2] = 2.0/3
    # m=1: xz
    T[3, 0, 2] = 0.5;  T[3, 2, 0] = 0.5
    # m=2: (x²-y²)/2
    T[4, 0, 0] = 0.5;  T[4, 1, 1] = -0.5

    # Compute norms for normalization
    norms_sq = torch.zeros(5)
    for m in range(5):
        norms_sq[m] = (T[m] * T[m]).sum()
    # norms_sq = [0.5, 0.5, 2/3, 0.5, 0.5]

    # Compute D^2_{m', m} = tr(T^{m'} @ R @ T^m @ R^T) / norm^2
    # For orthogonal D, we need: / sqrt(norm_m' * norm_m)
    for m_out in range(5):
        for m_in in range(5):
            # Rotated input: R @ T^{m_in} @ R^T
            rotated = R @ T[m_in] @ R.T
            # Inner product with output basis
            D2[m_out, m_in] = (T[m_out] * rotated).sum()

    # Now D2 is computed in the raw basis.
    # To make it orthogonal, normalize: D2[m',m] /= sqrt(norm_m' * norm_m)
    # But actually D2[m',m] already equals sum T_{m'} * (R T_m R^T).
    # If all norms were equal (say 1/2), D2 would be orthogonal * (1/2).
    # Since m=0 has norm 2/3 instead of 1/2, we need to adjust.
    #
    # D2_ortho[m',m] = D2[m',m] / sqrt(norms_sq[m'] * norms_sq[m])
    #                  * common_norm  (any constant to make det=1)
    #
    # Actually: in the orthonormal basis, D2_ortho = N @ D2_raw @ N^{-1}
    # where N = diag(1/sqrt(norms_sq[m])) (up to common scale).
    # Since we want D2_ortho @ D2_ortho^T = I, we need:
    # D2_ortho[m',m] = D2_raw[m',m] / (norms_sq[m'])
    # ... no, let's just do it properly.
    #
    # In the orthonormal basis e_m = T_m / ||T_m||:
    # <e_{m'}, R @ e_m @ R^T> = tr(T_{m'}/||T_{m'}|| @ R @ T_m/||T_m|| @ R^T)
    #                         = D2_raw[m',m] / (||T_{m'}|| * ||T_m||)
    # But ||T_m|| = sqrt(norms_sq[m]).

    inv_norms = 1.0 / torch.sqrt(norms_sq)
    D2_ortho = D2 * inv_norms.unsqueeze(1) * inv_norms.unsqueeze(0)

    # D2_ortho should now be orthogonal. But we need to check if the
    # common factor is right. For the identity rotation, D2_raw[m,m] = norms_sq[m],
    # so D2_ortho[m,m] = norms_sq[m] / (norms_sq[m]) = 1. And off-diagonal = 0. ✓

    return D2_ortho


def wigner_d_matrix(l: int, yaw: Tensor, pitch: Tensor,
                    roll: Tensor) -> Tensor:
    """Wigner D-matrix for degree l from Euler angles.

    Returns:
        (2l+1, 2l+1) rotation matrix in the real SH basis.
    """
    if l == 0:
        return wigner_d_l0()
    elif l == 1:
        return wigner_d_l1(yaw, pitch, roll)
    elif l == 2:
        return wigner_d_l2(yaw, pitch, roll)
    else:
        raise ValueError(f"Only l=0,1,2 supported, got l={l}")


def rotate_coefficients(coeffs: Tensor, yaw: Tensor, pitch: Tensor,
                        roll: Tensor) -> Tensor:
    """Apply block-diagonal Wigner D rotation to coefficient vector.

    Args:
        coeffs: (9,) real SH coefficients [l=0 (1), l=1 (3), l=2 (5)]
        yaw, pitch, roll: scalar Euler angles

    Returns:
        (9,) rotated coefficients
    """
    result = torch.empty_like(coeffs)
    for l in range(L_MAX + 1):
        s = DEGREE_SLICES[l]
        D = wigner_d_matrix(l, yaw, pitch, roll)
        result[s] = D @ coeffs[s]
    return result


# ---------------------------------------------------------------------------
# Interaction affinity
# ---------------------------------------------------------------------------

def degree_affinity(c_a: Tensor, c_b: Tensor, l: int,
                    eps: float = 1e-8) -> Tensor:
    """Squared cosine similarity for a single degree l.

    Returns a scalar in [0, 1].
    """
    s = DEGREE_SLICES[l]
    a, b = c_a[s], c_b[s]
    dot = torch.dot(a, b)
    norm_a = torch.dot(a, a)
    norm_b = torch.dot(b, b)
    denom = norm_a * norm_b + eps
    return (dot ** 2) / denom


def interaction_affinity(field_a: SE3Field, field_b: SE3Field,
                         sigma: float = 0.5, eps: float = 1e-8) -> Tensor:
    """Per-degree squared cosine similarity averaged over degrees,
    gated by spatial Gaussian decay.

    Args:
        field_a, field_b: SE3Fields to compare
        sigma: spatial decay scale (in normalized coordinates)

    Returns:
        Scalar affinity in [0, 1].
    """
    # Average per-degree affinity
    aff_sum = torch.tensor(0.0)
    n_active = 0
    for l in range(L_MAX + 1):
        s = DEGREE_SLICES[l]
        norm_a = torch.dot(field_a.coefficients[s], field_a.coefficients[s])
        norm_b = torch.dot(field_b.coefficients[s], field_b.coefficients[s])
        # Skip degrees where either field has zero coefficients
        if norm_a > eps and norm_b > eps:
            aff_sum = aff_sum + degree_affinity(
                field_a.coefficients, field_b.coefficients, l, eps
            )
            n_active += 1

    if n_active == 0:
        avg_aff = torch.tensor(0.0)
    else:
        avg_aff = aff_sum / n_active

    # Spatial Gaussian gate
    diff = field_a.position - field_b.position
    dist_sq = torch.dot(diff, diff)
    spatial = torch.exp(-dist_sq / (2 * sigma ** 2))

    return avg_aff * spatial


def interaction_matrix(fields: list[SE3Field], tau: float = 1.0,
                       sigma: float = 0.5) -> Tensor:
    """Compute N×N interaction matrix with softmax-normalized rows.

    Args:
        fields: list of N SE3Fields
        tau: temperature for softmax
        sigma: spatial decay scale

    Returns:
        (N, N) tensor where each row sums to 1.
    """
    n = len(fields)
    raw = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            raw[i, j] = interaction_affinity(fields[i], fields[j], sigma)

    # Row-wise softmax with temperature
    return torch.softmax(raw / tau, dim=-1)


# ---------------------------------------------------------------------------
# Kalman update (per-degree)
# ---------------------------------------------------------------------------

def kalman_update(prior: SE3Field, observed_coeffs: Tensor,
                  noise_per_degree: Tensor | None = None) -> SE3Field:
    """Per-degree diagonal Kalman update.

    Args:
        prior: SE3Field with current estimate and covariance
        observed_coeffs: (9,) observed coefficient vector
        noise_per_degree: (L_MAX+1,) observation noise variance per degree.
            If None, uses 1.0 for all degrees.

    Returns:
        Updated SE3Field with reduced covariance.
    """
    if noise_per_degree is None:
        noise_per_degree = torch.ones(L_MAX + 1)

    new_coeffs = torch.empty_like(prior.coefficients)
    new_cov = torch.empty_like(prior.covariance)

    for l in range(L_MAX + 1):
        s = DEGREE_SLICES[l]
        R_l = noise_per_degree[l]

        sigma_l = prior.covariance[s]
        K_l = sigma_l / (sigma_l + R_l)  # element-wise Kalman gain

        innovation = observed_coeffs[s] - prior.coefficients[s]
        new_coeffs[s] = prior.coefficients[s] + K_l * innovation
        new_cov[s] = (1 - K_l) * sigma_l

    return SE3Field(prior.position, new_coeffs, new_cov)


# ---------------------------------------------------------------------------
# Spectral coefficient fitting from trajectory
# ---------------------------------------------------------------------------

def fit_spectral_coefficients(positions: Tensor, dt: float = 1.0 / 120.0,
                              center: Tensor | None = None) -> Tensor:
    """Fit l=0, l=1, l=2 coefficients from a short position trajectory.

    Given T positions in R³, computes:
        l=0: 1.0 (presence)
        l=1: average velocity direction (3 components)
        l=2: velocity outer product (spread/shape, 5 components)

    Args:
        positions: (T, 3) position trajectory in normalized coordinates
        dt: time step between frames
        center: optional (3,) center to subtract. If None, uses mean position.

    Returns:
        (9,) real SH coefficient vector.
    """
    T = positions.shape[0]
    coeffs = torch.zeros(N_COEFFS, dtype=positions.dtype)

    # l=0: presence (always 1)
    coeffs[0] = 1.0

    if T < 2:
        return coeffs

    # Compute velocities via finite differences
    velocities = (positions[1:] - positions[:-1]) / dt
    avg_vel = velocities.mean(dim=0)  # (3,)

    # l=1: velocity direction (normalized, but keep magnitude info in scale)
    speed = torch.norm(avg_vel)
    if speed > 1e-6:
        coeffs[1:4] = avg_vel / speed  # direction
        # Scale by normalized speed so magnitude isn't lost
        coeffs[0] = speed  # override l=0 with speed magnitude

    # l=2: velocity spread — symmetric traceless tensor from velocity covariance
    if T >= 3:
        vel_centered = velocities - avg_vel.unsqueeze(0)
        # Covariance matrix (3×3)
        cov = (vel_centered.T @ vel_centered) / (T - 2)

        # Extract 5 independent components of the symmetric traceless part
        # Traceless: subtract (trace/3) * I
        trace = cov[0, 0] + cov[1, 1] + cov[2, 2]
        traceless = cov - (trace / 3) * torch.eye(3, dtype=cov.dtype)

        # Map to l=2 real SH basis: (xy, yz, z², xz, x²-y²)
        coeffs[4] = traceless[0, 1]                       # xy
        coeffs[5] = traceless[1, 2]                       # yz
        coeffs[6] = traceless[2, 2] * torch.sqrt(torch.tensor(2.0/3.0))  # z² (normalized)
        coeffs[7] = traceless[0, 2]                       # xz
        coeffs[8] = (traceless[0, 0] - traceless[1, 1]) / 2  # x²-y²

    return coeffs


def reconstruct_position(coeffs: Tensor, t: float,
                         center: Tensor | None = None) -> Tensor:
    """Reconstruct position from spectral coefficients at time t.

    Uses l=0 as speed magnitude and l=1 as direction.
    position(t) = center + direction * speed * t

    This is a simple linear reconstruction; higher-order reconstruction
    would use the l=2 components for acceleration.

    Args:
        coeffs: (9,) coefficient vector
        t: time offset from center of trajectory
        center: (3,) center position

    Returns:
        (3,) predicted position
    """
    if center is None:
        center = torch.zeros(3, dtype=coeffs.dtype)

    speed = coeffs[0]        # l=0 encodes speed magnitude
    direction = coeffs[1:4]  # l=1 encodes direction
    return center + direction * speed * t
