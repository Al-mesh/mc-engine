"""Payoffs as pure functions of simulated paths.

A payoff maps a path matrix ``paths`` of shape ``(n_paths, n_steps + 1)`` (column 0 is
``S0``) and its time grid to an *undiscounted* cash flow at maturity, one per path.
Discounting belongs to the pricer, so payoffs stay trivially unit-testable.

Monitoring convention: path-dependent features (averages, barriers, extrema) are
observed **on the simulation grid only**, i.e. discretely. Averages use the fixings
``t_1..t_n`` and exclude ``t_0``; barriers and extrema include ``t_0``. The difference
between discrete and continuous monitoring is a real economic effect, not a numerical
error, and the tests treat it as such (see the Broadie-Glasserman-Kou test).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mc_engine.rng import FloatArray


class Payoff(ABC):
    path_dependent: ClassVar[bool] = False

    @abstractmethod
    def __call__(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        """Undiscounted payoff per path, shape ``(n_paths,)``."""


@dataclass(frozen=True)
class European(Payoff):
    r"""Vanilla :math:`(S_T - K)^+` (call) or :math:`(K - S_T)^+` (put)."""

    K: float
    call: bool = True

    def __call__(self, paths, times):
        s_t = paths[:, -1]
        return np.maximum(s_t - self.K, 0.0) if self.call else np.maximum(self.K - s_t, 0.0)


@dataclass(frozen=True)
class Digital(Payoff):
    r"""Cash-or-nothing: ``payout`` if :math:`S_T > K` (call) or :math:`S_T < K` (put).

    The payoff is discontinuous at K, so its pathwise derivative in :math:`S_0` is zero
    almost surely, which is the textbook failure case for pathwise Greeks (Phase 3).
    """

    K: float
    call: bool = True
    payout: float = 1.0

    def __call__(self, paths, times):
        s_t = paths[:, -1]
        hit = s_t > self.K if self.call else s_t < self.K
        return self.payout * hit.astype(float)


@dataclass(frozen=True)
class ArithmeticAsian(Payoff):
    r"""Fixed-strike arithmetic-average Asian :math:`(\bar A - K)^+`.

    :math:`\bar A = \frac1n\sum_{i=1}^n S_{t_i}`. No closed form exists (a sum of lognormals
    is not lognormal), which is why it is the classic control-variate example.
    """

    K: float
    call: bool = True
    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        avg = paths[:, 1:].mean(axis=1)
        return np.maximum(avg - self.K, 0.0) if self.call else np.maximum(self.K - avg, 0.0)


@dataclass(frozen=True)
class GeometricAsian(Payoff):
    r"""Fixed-strike geometric-average Asian :math:`(G - K)^+`.

    :math:`G = \exp(\frac1n\sum_{i=1}^n \ln S_{t_i})`; closed form under GBM in
    :func:`mc_engine.analytic.geometric_asian_price`.
    """

    K: float
    call: bool = True
    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
        return np.maximum(geo - self.K, 0.0) if self.call else np.maximum(self.K - geo, 0.0)


@dataclass(frozen=True)
class UpAndOutCall(Payoff):
    r"""Up-and-out call: :math:`(S_T - K)^+\,\mathbb 1\{\max_{0\le i\le n} S_{t_i} < B\}`."""

    K: float
    barrier: float
    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        alive = paths.max(axis=1) < self.barrier
        return np.where(alive, np.maximum(paths[:, -1] - self.K, 0.0), 0.0)


@dataclass(frozen=True)
class FloatingLookbackCall(Payoff):
    r"""Floating-strike lookback call :math:`S_T - \min_{0\le i\le n} S_{t_i}` (always >= 0)."""

    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        return paths[:, -1] - paths.min(axis=1)


@dataclass(frozen=True)
class FixedLookbackCall(Payoff):
    r"""Fixed-strike lookback call: :math:`(\max_{0\le i\le n} S_{t_i} - K)^+`."""

    K: float
    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        return np.maximum(paths.max(axis=1) - self.K, 0.0)
