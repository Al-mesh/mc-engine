"""Payoffs as pure functions of simulated paths.

A payoff maps a path matrix ``paths`` of shape ``(n_paths, n_steps + 1)`` (column 0 is
``S0``) and its time grid to an *undiscounted* cash flow at maturity, one per path.
Discounting belongs to the pricer, so payoffs stay trivially unit-testable.

Monitoring convention: path-dependent features (averages, barriers, extrema) are
observed **on the simulation grid only**, i.e. discretely. Averages use the fixings
``t_1..t_n`` and exclude ``t_0``; barriers and extrema include ``t_0``. The difference
between discrete and continuous monitoring is a real economic effect, not a numerical
error, and the tests treat it as such (see the Broadie-Glasserman-Kou test).

Pathwise derivatives
--------------------
``path_gradient`` returns :math:`\\partial h/\\partial S_{t_i}` per path, the derivative
of the payoff with respect to each observed price, **where it exists**. Payoffs are only
piecewise differentiable, and the gradient is the almost-sure derivative, which ignores
jumps. That is exactly what pathwise Greeks use (:mod:`mc_engine.greeks`), and exactly
why they fail for discontinuous payoffs: the jump's contribution to
:math:`\\frac{d}{d\\theta}E[h]` is a boundary term that no a.s. derivative can see.
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

    def path_gradient(self, paths: FloatArray, times: FloatArray) -> FloatArray:
        """Almost-sure derivative :math:`\\partial h/\\partial S_{t_i}`, shape like ``paths``."""
        raise NotImplementedError(f"{type(self).__name__} has no pathwise derivative")


def _terminal_only(paths: FloatArray, last_col: FloatArray) -> FloatArray:
    g = np.zeros_like(paths)
    g[:, -1] = last_col
    return g


@dataclass(frozen=True)
class European(Payoff):
    r"""Vanilla :math:`(S_T - K)^+` (call) or :math:`(K - S_T)^+` (put)."""

    K: float
    call: bool = True

    def __call__(self, paths, times):
        s_t = paths[:, -1]
        return np.maximum(s_t - self.K, 0.0) if self.call else np.maximum(self.K - s_t, 0.0)

    def path_gradient(self, paths, times):
        s_t = paths[:, -1]
        d = (s_t > self.K).astype(float) if self.call else -(s_t < self.K).astype(float)
        return _terminal_only(paths, d)


@dataclass(frozen=True)
class Digital(Payoff):
    r"""Cash-or-nothing: ``payout`` if :math:`S_T > K` (call) or :math:`S_T < K` (put).

    The payoff is a step function of :math:`S_T`, so its a.s. derivative is **zero**: the
    whole sensitivity sits in the jump at K, which a pathwise estimator never sees. See
    :class:`SmoothedDigital` and the likelihood-ratio method for the two standard fixes.
    """

    K: float
    call: bool = True
    payout: float = 1.0

    def __call__(self, paths, times):
        s_t = paths[:, -1]
        hit = s_t > self.K if self.call else s_t < self.K
        return self.payout * hit.astype(float)

    def path_gradient(self, paths, times):
        return np.zeros_like(paths)


@dataclass(frozen=True)
class SmoothedDigital(Payoff):
    r"""Digital replaced by a call spread of half-width :math:`\varepsilon`.

    :math:`h_\varepsilon(S) = \text{payout}\cdot
    \min(\max(\frac{S - K + \varepsilon}{2\varepsilon}, 0), 1)` (call). It is Lipschitz,
    so pathwise differentiation is valid for it, at the price of a
    deterministic bias :math:`E[h_\varepsilon] - E[h] = O(\varepsilon^2)` (for a smooth
    density of :math:`S_T`, by symmetry of the ramp around K). The pathwise estimator's
    variance grows like :math:`1/\varepsilon`, so :math:`\varepsilon` trades bias against variance.
    This is the same trade-off a trader faces when over-hedging a digital with a call spread.
    """

    K: float
    eps: float
    call: bool = True
    payout: float = 1.0

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def __call__(self, paths, times):
        s_t = paths[:, -1]
        x = (s_t - self.K) if self.call else (self.K - s_t)
        return self.payout * np.clip((x + self.eps) / (2 * self.eps), 0.0, 1.0)

    def path_gradient(self, paths, times):
        s_t = paths[:, -1]
        ramp = (np.abs(s_t - self.K) < self.eps) * self.payout / (2 * self.eps)
        return _terminal_only(paths, ramp if self.call else -ramp)


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

    def path_gradient(self, paths, times):
        avg = paths[:, 1:].mean(axis=1)
        n = paths.shape[1] - 1
        d = (avg > self.K).astype(float) if self.call else -(avg < self.K).astype(float)
        g = np.zeros_like(paths)
        g[:, 1:] = (d / n)[:, None]
        return g


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

    def path_gradient(self, paths, times):
        geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
        n = paths.shape[1] - 1
        d = (geo > self.K).astype(float) if self.call else -(geo < self.K).astype(float)
        g = np.zeros_like(paths)
        g[:, 1:] = (d * geo)[:, None] / (n * paths[:, 1:])  # dG/dS_i = G / (n S_i)
        return g


@dataclass(frozen=True)
class UpAndOutCall(Payoff):
    r"""Up-and-out call: :math:`(S_T - K)^+\,\mathbb 1\{\max_{0\le i\le n} S_{t_i} < B\}`.

    The a.s. gradient is :math:`\mathbb 1\{\text{alive}\}\mathbb 1\{S_T > K\}` on
    :math:`S_T`. It ignores the knock-out discontinuity, where a small bump in :math:`S_0`
    pushes some paths over the barrier and *destroys* a payoff of size :math:`B - K`, so
    the pathwise delta is biased upward.
    """

    K: float
    barrier: float
    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        alive = paths.max(axis=1) < self.barrier
        return np.where(alive, np.maximum(paths[:, -1] - self.K, 0.0), 0.0)

    def path_gradient(self, paths, times):
        alive = paths.max(axis=1) < self.barrier
        return _terminal_only(paths, (alive & (paths[:, -1] > self.K)).astype(float))


@dataclass(frozen=True)
class FloatingLookbackCall(Payoff):
    r"""Floating-strike lookback call :math:`S_T - \min_{0\le i\le n} S_{t_i}` (always >= 0)."""

    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        return paths[:, -1] - paths.min(axis=1)

    def path_gradient(self, paths, times):
        g = np.zeros_like(paths)
        g[:, -1] = 1.0
        rows = np.arange(paths.shape[0])
        g[rows, paths.argmin(axis=1)] -= 1.0  # the minimum is a.s. attained at one date
        return g


@dataclass(frozen=True)
class FixedLookbackCall(Payoff):
    r"""Fixed-strike lookback call: :math:`(\max_{0\le i\le n} S_{t_i} - K)^+`."""

    K: float
    path_dependent: ClassVar[bool] = True

    def __call__(self, paths, times):
        return np.maximum(paths.max(axis=1) - self.K, 0.0)

    def path_gradient(self, paths, times):
        g = np.zeros_like(paths)
        rows = np.arange(paths.shape[0])
        g[rows, paths.argmax(axis=1)] = (paths.max(axis=1) > self.K).astype(float)
        return g
