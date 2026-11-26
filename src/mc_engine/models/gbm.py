r"""Geometric Brownian motion with exact discretisation.

:math:`dS_t = (r - q)S_t\,dt + \sigma S_t\,dW_t` has the explicit solution
:math:`S_t = S_0\exp((r - q - \tfrac12\sigma^2)t + \sigma W_t)`, so on any grid

.. math::
    \ln S_{t_{k+1}} - \ln S_{t_k} = (r - q - \tfrac12\sigma^2)\Delta_k + \sigma\sqrt{\Delta_k}\,Z_k,
    \qquad Z_k \overset{iid}{\sim} N(0,1).

This is *exact in distribution* at the grid points: there is no time-discretisation
bias, so for European payoffs one step is enough, and the only error is statistical.
(Euler on S itself would introduce an O(dt) bias for no benefit.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mc_engine.models.base import Model, SimResult, validate_n_paths, validate_times
from mc_engine.rng import SeedLike, as_random_source


@dataclass(frozen=True)
class GBM(Model):
    """Black-Scholes dynamics. ``sigma`` is the (constant) lognormal volatility."""

    S0: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self) -> None:
        if self.S0 <= 0 or self.sigma <= 0:
            raise ValueError("S0 and sigma must be strictly positive")

    def log_increments(self, dt: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Map standard normals ``z`` (shape ``(n_paths, n_steps)``) to log-price increments.

        Exposed separately because the pathwise / likelihood-ratio Greeks (Phase 3)
        need the same map, and the Brownian bridge (Phase 2) supplies ``z`` directly.
        """
        return (self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z

    def simulate(self, n_paths: int, times: ArrayLike, rng: SeedLike = None) -> SimResult:
        n_paths = validate_n_paths(n_paths)
        t = validate_times(times)
        dt = np.diff(t)
        z = as_random_source(rng).normals(n_paths, dt.size, 1)[..., 0]
        log_s = np.empty((n_paths, t.size))
        log_s[:, 0] = np.log(self.S0)
        np.cumsum(self.log_increments(dt, z), axis=1, out=log_s[:, 1:])
        log_s[:, 1:] += log_s[:, :1]
        return SimResult(times=t, paths=np.exp(log_s))
