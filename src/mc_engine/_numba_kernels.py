"""Numba kernels for the Heston time-stepping loops.

Why these loops and not others (from profiling, see benchmarks/performance.py):

* Heston QE spends ~90% of its time in per-step numpy arithmetic: each step allocates
  ~30 temporary arrays of length n_paths and walks memory once per operation. A compiled
  loop does the whole step for one path in registers, and ``prange`` spreads paths
  over cores. This is the textbook case for numba.
* GBM is already one vectorised cumsum and spends ~40% of its time *generating random
  numbers*, which numba cannot speed up; multiprocessing over independent seed streams
  is the better tool there (:mod:`mc_engine.parallel`).

Amdahl's law, learned the hard way: a first version compiled only the recursion and left
``ndtri`` (inverse normal CDF) and ``exp`` in single-threaded numpy. The kernel was then
8% of the run time and the overall speed-up was a disappointing 2.3x. Everything per
element now runs inside the parallel loop. The only serial part left is drawing the
uniforms, which stays in the RandomSource so both backends see identical random numbers.

Correctness contract: kernels receive exactly the random numbers the numpy code uses and
perform the same arithmetic, so the two backends agree to floating-point round-off
(tested). The one non-identical piece is :func:`norm_ppf`, which matches scipy's
``ndtri`` to ~1e-15 (also tested).

numba is optional (``pip install mc-engine[fast]``); without it, the numpy backend is used
and requesting ``backend="numba"`` raises a clear ImportError.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    numba = None
    NUMBA_AVAILABLE = False


def require_numba() -> None:
    if not NUMBA_AVAILABLE:  # pragma: no cover
        raise ImportError("backend='numba' needs numba: pip install mc-engine[fast]")


# Acklam's rational approximation to the inverse normal CDF (relative error < 1.15e-9),
# followed by one Halley step on Phi(x) - p using erfc, which is cubically convergent and
# takes the result to full double precision. scipy.special.ndtri cannot be called from
# numba, and it was 58% of the run time when called from numpy outside the kernel.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _ppf_lower(p: float) -> float:
    """Inverse normal CDF for p <= 0.5 (x <= 0)."""
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    # Halley refinement. For x <= 0, Phi(x) = erfc(|x|/sqrt 2)/2 is small, so Phi(x) - p
    # has no cancellation and the step gains full relative precision.
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * _SQRT_2PI * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def _norm_ppf(p: float) -> float:
    # Upper half by symmetry. A first version refined Phi(x) - p directly for p near 1,
    # where both terms are ~1 and cancel catastrophically (error stuck at ~1e-9 at
    # p = 1 - 1e-15; caught by the tests). 1 - p is *exact* for p in [0.5, 1] (Sterbenz).
    if p > 0.5:
        return -ppf_lower(1.0 - p)
    return ppf_lower(p)


def _heston_qe(log_s0, v0, kappa, theta, xi, rho, mu, psi_c, correction, dt, u):
    """QE paths from uniforms ``u`` of shape (n_paths, n_steps, 2); returns (S, v)."""
    n_paths, n_steps = u.shape[0], u.shape[1]
    paths = np.empty((n_paths, n_steps + 1))
    var = np.empty((n_paths, n_steps + 1))
    # Per-step constants are shared by all paths: compute them once.
    e = np.exp(-kappa * dt)
    a_ = kappa * rho / xi - 0.5
    k1 = 0.5 * dt * a_ - rho / xi
    k2 = 0.5 * dt * a_ + rho / xi
    k3 = 0.5 * dt * (1.0 - rho * rho)
    k4 = k3
    big_a = k2 + 0.5 * k4
    k0_plain = -rho * kappa * theta * dt / xi
    s0 = math.exp(log_s0)
    for i in numba.prange(n_paths):
        x = log_s0
        v = v0
        paths[i, 0] = s0
        var[i, 0] = v
        for n in range(n_steps):
            en = e[n]
            m = theta + (v - theta) * en
            s2 = (
                v * xi * xi * en / kappa * (1.0 - en)
                + theta * xi * xi / (2.0 * kappa) * (1.0 - en) ** 2
            )
            psi = s2 / (m * m)
            base = -(k1[n] + 0.5 * k3[n]) * v
            k0 = k0_plain[n]
            uv = u[i, n, 0]
            if psi <= psi_c:
                inv = 2.0 / psi
                b2 = inv - 1.0 + math.sqrt(inv) * math.sqrt(inv - 1.0)
                a = m / (1.0 + b2)
                vn = a * (math.sqrt(b2) + norm_ppf(uv)) ** 2  # normal only needed here
                if correction and 2.0 * big_a[n] * a < 1.0:
                    one_m = 1.0 - 2.0 * big_a[n] * a
                    k0 = -big_a[n] * b2 * a / one_m + 0.5 * math.log(one_m) + base
            else:
                p = (psi - 1.0) / (psi + 1.0)
                beta = (1.0 - p) / m
                vn = 0.0 if uv <= p else math.log((1.0 - p) / (1.0 - uv)) / beta
                if correction and big_a[n] < beta:
                    k0 = -math.log(p + beta * (1.0 - p) / (beta - big_a[n])) + base
            diff = max(k3[n] * v + k4[n] * vn, 0.0)
            zs = norm_ppf(u[i, n, 1])
            x = x + mu * dt[n] + k0 + k1[n] * v + k2[n] * vn + math.sqrt(diff) * zs
            v = vn
            paths[i, n + 1] = math.exp(x)
            var[i, n + 1] = v
    return paths, var


def _heston_euler(log_s0, v0, kappa, theta, xi, rho, mu, dt, z):
    """Full-truncation Euler from normals ``z`` (n_paths, n_steps, 2); returns (S, v)."""
    n_paths, n_steps = z.shape[0], z.shape[1]
    paths = np.empty((n_paths, n_steps + 1))
    var = np.empty((n_paths, n_steps + 1))
    rho_perp = math.sqrt(1.0 - rho * rho)
    s0 = math.exp(log_s0)
    for i in numba.prange(n_paths):
        x = log_s0
        v = v0  # raw (possibly negative) Euler state
        paths[i, 0] = s0
        var[i, 0] = v0
        for n in range(n_steps):
            zv = z[i, n, 0]
            zs = rho * zv + rho_perp * z[i, n, 1]
            v_pos = max(v, 0.0)
            sq = math.sqrt(v_pos * dt[n])
            x = x + (mu - 0.5 * v_pos) * dt[n] + sq * zs
            v = v + kappa * (theta - v_pos) * dt[n] + xi * sq * zv
            paths[i, n + 1] = math.exp(x)
            var[i, n + 1] = max(v, 0.0)
    return paths, var


if NUMBA_AVAILABLE:
    # cache=True stores the compiled machine code next to the source, so the ~1-2 s JIT
    # cost is paid once per machine rather than once per process.
    ppf_lower = numba.njit(cache=True)(_ppf_lower)
    norm_ppf = numba.njit(cache=True)(_norm_ppf)
    heston_qe_kernel = numba.njit(parallel=True, cache=True)(_heston_qe)
    heston_euler_kernel = numba.njit(parallel=True, cache=True)(_heston_euler)
else:  # pragma: no cover
    ppf_lower = _ppf_lower
    norm_ppf = _norm_ppf
    heston_qe_kernel = heston_euler_kernel = None
