"""Abstract model interface shared by all stochastic models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mc_engine.rng import FloatArray, SeedLike


@dataclass(frozen=True)
class SimResult:
    """Output of :meth:`Model.simulate`.

    Attributes
    ----------
    times
        Simulation grid, shape ``(n_steps + 1,)``, with ``times[0] == 0``.
    paths
        Underlying price paths, shape ``(n_paths, n_steps + 1)``; ``paths[:, 0] == S0``.
    variance
        Instantaneous variance paths for stochastic-volatility models, else ``None``.
    """

    times: FloatArray
    paths: FloatArray
    variance: FloatArray | None = None

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def terminal(self) -> FloatArray:
        return self.paths[:, -1]


class Model(ABC):
    """A risk-neutral model of one underlying.

    Concrete models are frozen dataclasses exposing ``S0`` (spot), ``r`` (continuously
    compounded risk-free rate) and ``q`` (continuous dividend yield). Under the
    risk-neutral measure every model satisfies :math:`E[S_t] = S_0 e^{(r-q)t}`; the test
    suite checks this martingale property for each one.
    """

    S0: float
    r: float
    q: float

    @abstractmethod
    def simulate(self, n_paths: int, times: ArrayLike, rng: SeedLike = None) -> SimResult:
        """Simulate ``n_paths`` paths on the grid ``times`` (must start at 0)."""

    def discount(self, t: float) -> float:
        """Risk-free discount factor :math:`e^{-rt}`."""
        return float(np.exp(-self.r * t))


def validate_times(times: ArrayLike) -> FloatArray:
    """Return ``times`` as a float array after checking it is a valid simulation grid."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a 1-D array with at least two points")
    if t[0] != 0.0:
        raise ValueError("times must start at 0")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    return t


def validate_n_paths(n_paths: int) -> int:
    if int(n_paths) != n_paths or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    return int(n_paths)
