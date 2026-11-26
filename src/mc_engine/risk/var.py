r"""Value-at-Risk and Expected Shortfall.

With P&L :math:`X` and loss :math:`L = -X`, at confidence level :math:`\alpha` (e.g. 0.99):

* :math:`\mathrm{VaR}_\alpha = \inf\{\ell : P(L \le \ell) \ge \alpha\}`, the
  :math:`\alpha`-quantile of the loss.
* :math:`\mathrm{ES}_\alpha = \frac{1}{1-\alpha}\int_\alpha^1 \mathrm{VaR}_u\,du`, the average
  loss in the worst :math:`1-\alpha` of outcomes (for continuous :math:`L`,
  :math:`E[L \mid L \ge \mathrm{VaR}_\alpha]`).

Why both: VaR says nothing about *how bad* the tail is beyond the quantile, and it is
not subadditive (diversification can appear to increase VaR). ES is coherent (subadditive)
and tail-sensitive, which is why Basel FRTB moved the capital measure to 97.5% ES. The
price is that ES is harder to backtest, since it is not *elicitable*.

Empirical estimators from n scenarios, with order statistics :math:`L_{(1)} \le \dots \le L_{(n)}`
and :math:`m = \lceil n\alpha\rceil`: :math:`\widehat{\mathrm{VaR}} = L_{(m)}` (the empirical
:math:`\alpha`-quantile, matching the inf-definition above) and :math:`\widehat{\mathrm{ES}}` the
mean of the :math:`n - m` losses beyond it (for integer :math:`n\alpha`, exactly the worst
:math:`n(1-\alpha)`). The quantile estimator's standard error is roughly
:math:`\sqrt{\alpha(1-\alpha)/n}/f(\mathrm{VaR})`: tail quantiles need *many* scenarios,
because the density f is small there.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def var_es(pnl: ArrayLike, alpha: float) -> tuple[float, float]:
    """Empirical (VaR, ES) of a P&L sample, both reported as positive losses."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    losses = -np.asarray(pnl, dtype=float).ravel()
    n = losses.size
    # The epsilon guards against floating point: 100 * 0.95 is exact, but e.g.
    # 100 * (1 - 0.95) = 5.000000000000004 would round *up* to 6 (a real bug found by the
    # tests in an earlier version that counted from the top).
    m = min(n, max(1, math.ceil(n * alpha - 1e-9)))
    # np.partition places the m-th smallest correctly and everything larger after it,
    # in O(n) without a full sort of a large scenario set.
    part = np.partition(losses, m - 1)
    var = float(part[m - 1])
    beyond = part[m:]
    es = float(beyond.mean()) if beyond.size else var
    return var, es


def gaussian_var_es(mean: float, std: float, alpha: float) -> tuple[float, float]:
    r"""Closed form for normal P&L :math:`N(m, s^2)`.

    :math:`\mathrm{VaR} = -m + s\,z_\alpha` and
    :math:`\mathrm{ES} = -m + s\,\varphi(z_\alpha)/(1-\alpha)`.
    """
    z = norm.ppf(alpha)
    return float(-mean + std * z), float(-mean + std * norm.pdf(z) / (1.0 - alpha))
