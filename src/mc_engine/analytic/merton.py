r"""Merton (1976) jump-diffusion: closed-form series for European options.

Conditioning on :math:`n` jumps by :math:`T`, :math:`\ln S_T` is Gaussian, so the price is a
Poisson mixture of Black-Scholes prices:

.. math::
    V = \sum_{n=0}^\infty \frac{e^{-\lambda' T}(\lambda' T)^n}{n!}\,
        V_{BS}(S_0, K, T, r_n, \sigma_n, q),

    \lambda' = \lambda(1 + \bar k), \quad
    \sigma_n^2 = \sigma^2 + \frac{n\delta^2}{T}, \quad
    r_n = r - \lambda\bar k + \frac{n\ln(1+\bar k)}{T}.

The change from :math:`\lambda` to :math:`\lambda'` and from :math:`r` to :math:`r_n` is
bookkeeping: :math:`V_{BS}` discounts at :math:`r_n`, and the reweighting by
:math:`e^{-\lambda\bar k T}(1+\bar k)^n` exactly converts that back to discounting at
:math:`r`. The series converges super-exponentially (Poisson tail), so it is summed
until the weights fall below ``tol`` past the mode.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from mc_engine.analytic.black_scholes import bs_price
from mc_engine.models.merton import Merton


def merton_price(
    model: Merton, K: float, T: float, call: bool = True, tol: float = 1e-16, n_max: int = 1000
) -> float:
    lam_p = model.lam * (1.0 + model.kbar)
    total = 0.0
    for n in range(n_max):
        w = poisson.pmf(n, lam_p * T)
        sigma_n = np.sqrt(model.sigma**2 + n * model.sigma_j**2 / T)
        r_n = model.r - model.lam * model.kbar + n * np.log1p(model.kbar) / T
        total += w * float(bs_price(model.S0, K, T, r_n, sigma_n, model.q, call))
        if n > lam_p * T and w < tol:
            break
    return total
