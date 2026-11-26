r"""Merton (1976) jump-diffusion, simulated exactly.

.. math::
    \frac{dS_t}{S_{t^-}} = (r - q - \lambda\bar k)\,dt + \sigma\,dW_t + (J - 1)\,dN_t,

with :math:`N` a Poisson process of intensity :math:`\lambda`, i.i.d. jump multipliers
:math:`\ln J \sim N(\mu_J, \delta^2)`, and :math:`\bar k = E[J-1] = e^{\mu_J + \delta^2/2} - 1`.
The :math:`-\lambda\bar k` drift compensator makes :math:`e^{-(r-q)t}S_t` a martingale.

Over a step of length :math:`\Delta`, with :math:`N_\Delta \sim \text{Poisson}(\lambda\Delta)`,

.. math::
    \ln\frac{S_{t+\Delta}}{S_t} = (r - q - \lambda\bar k - \tfrac12\sigma^2)\Delta
      + \sigma\sqrt\Delta\,Z_1 + N_\Delta\mu_J + \sqrt{N_\Delta}\,\delta\,Z_2,

using that a sum of :math:`n` i.i.d. :math:`N(\mu_J, \delta^2)` jumps is
:math:`N(n\mu_J, n\delta^2)`. This is exact in distribution at the grid points, so,
like GBM, there is no discretisation bias. Each step consumes exactly two normals and
one uniform regardless of how many jumps occur, which keeps the random-number layout
fixed (needed for CRN Greeks and QMC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mc_engine.models.base import Model, SimResult, validate_n_paths, validate_times
from mc_engine.rng import SeedLike, as_random_source

_MAX_POISSON_TERMS = 10_000


def poisson_inverse_cdf(u: np.ndarray, mean: np.ndarray | float) -> np.ndarray:
    """Poisson quantile :math:`N = \\min\\{k : F(k) \\ge u\\}`, vectorised.

    Computed as :math:`N = \\#\\{k \\ge 0 : F(k) < u\\}` by accumulating the pmf with the
    recursion :math:`p_{k+1} = p_k\\,\\mu/(k+1)`. Inversion (rather than
    ``Generator.poisson``) keeps the map from uniforms monotone and one-to-one, which is
    what antithetic and quasi-random sampling require. Efficient for the small
    per-step means typical of jump models; cost is O(max N).
    """
    u = np.asarray(u, dtype=float)
    mean = np.broadcast_to(np.asarray(mean, dtype=float), u.shape)
    pmf = np.exp(-mean)
    cdf = pmf.copy()
    counts = np.zeros(u.shape, dtype=np.int64)
    for k in range(_MAX_POISSON_TERMS):
        above = u > cdf
        if not above.any():
            break
        counts += above
        pmf = pmf * mean / (k + 1)
        cdf = cdf + pmf
    return counts


@dataclass(frozen=True)
class Merton(Model):
    S0: float
    r: float
    sigma: float
    lam: float
    mu_j: float
    sigma_j: float
    q: float = 0.0

    def __post_init__(self) -> None:
        if self.S0 <= 0 or self.sigma <= 0:
            raise ValueError("S0 and sigma must be strictly positive")
        if self.lam < 0 or self.sigma_j < 0:
            raise ValueError("lam and sigma_j must be non-negative")

    @property
    def kbar(self) -> float:
        """Mean relative jump size :math:`E[J - 1]`."""
        return float(np.expm1(self.mu_j + 0.5 * self.sigma_j**2))

    def simulate(self, n_paths: int, times: ArrayLike, rng: SeedLike = None) -> SimResult:
        n_paths = validate_n_paths(n_paths)
        t = validate_times(times)
        dt = np.diff(t)
        source = as_random_source(rng)
        z = source.normals(n_paths, dt.size, 2)
        u = source.uniforms(n_paths, dt.size, 1)[..., 0]

        counts = poisson_inverse_cdf(u, self.lam * dt)
        drift = (self.r - self.q - self.lam * self.kbar - 0.5 * self.sigma**2) * dt
        incr = (
            drift
            + self.sigma * np.sqrt(dt) * z[..., 0]
            + counts * self.mu_j
            + np.sqrt(counts) * self.sigma_j * z[..., 1]
        )

        log_s = np.empty((n_paths, t.size))
        log_s[:, 0] = np.log(self.S0)
        np.cumsum(incr, axis=1, out=log_s[:, 1:])
        log_s[:, 1:] += log_s[:, :1]
        return SimResult(times=t, paths=np.exp(log_s))
