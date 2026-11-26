r"""Correlated multi-asset return scenarios: Gaussian (Cholesky) and Student-t.

Gaussian dependence
-------------------
If :math:`Z \sim N(0, I_d)` and :math:`\Sigma = LL^\top` (Cholesky), then
:math:`LZ \sim N(0, \Sigma)`. With a *correlation* matrix :math:`R = LL^\top` the
innovations :math:`\varepsilon = LZ` have unit variances and correlation :math:`R`.
Cholesky needs :math:`R` to be positive definite; a sample correlation of d assets from
fewer than d observations, or a hand-typed matrix, may not be, and we refuse it rather
than silently "fix" it.

Fat tails: Student-t
--------------------
With :math:`W \sim \chi^2_\nu/\nu` independent of :math:`Z`,
:math:`T = LZ/\sqrt W` is multivariate Student-t: its copula is the t-copula with
:math:`\nu` degrees of freedom and its marginals are :math:`t_\nu`. Two effects:

* **Fat marginal tails**: excess kurtosis :math:`6/(\nu - 4)` for :math:`\nu > 4`.
* **Tail dependence**: the *shared* mixing variable :math:`W` makes all assets crash
  together (a small :math:`W` scales every component up). Gaussian dependence has zero
  tail dependence however high the correlation, which is why it understates joint crashes.

:math:`\mathrm{Var}(T_i) = \nu/(\nu-2)`, so we rescale by :math:`\sqrt{(\nu-2)/\nu}` to keep
unit variance. That way :math:`\sigma` means the same thing in both models, and any
difference in VaR is due to the *shape* of the distribution, not its scale.

Multi-day horizons are simulated as sums of daily steps, not by scaling one-day draws.
For Student-t innovations the sum is *less* fat-tailed than each step (CLT), so
:math:`\sqrt{h}`-scaling of a one-day VaR overstates the h-day VaR. The tests check this.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from mc_engine.rng import FloatArray, SeedLike, as_random_source

TRADING_DAYS = 252
Dist = Literal["normal", "t"]


def cholesky_factor(corr: ArrayLike) -> FloatArray:
    """Lower-triangular L with L L^T = corr, after validating the matrix."""
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("correlation matrix must be square")
    if not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
        raise ValueError("correlation matrix must be symmetric with unit diagonal")
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation matrix is not positive definite") from exc


def correlated_innovations(
    n_sims: int,
    n_steps: int,
    chol: FloatArray,
    rng: SeedLike = None,
    dist: Dist = "normal",
    nu: float | None = None,
) -> FloatArray:
    """Unit-variance innovations with correlation ``chol @ chol.T``, shape ``(n, steps, d)``.

    For ``dist="t"`` each (path, step) draws one chi-square mixing variable shared by all
    assets (that sharing *is* the t-copula's tail dependence). It comes from the
    RandomSource like every other draw (direct sampling for pseudo-random sources,
    inversion otherwise).
    """
    source = as_random_source(rng)
    d = chol.shape[0]
    z = source.normals(n_sims, n_steps, d) @ chol.T
    if dist == "normal":
        return z
    if dist != "t":
        raise ValueError(f"unknown dist {dist!r}")
    if nu is None or nu <= 2:
        raise ValueError("Student-t needs nu > 2 for a finite variance")
    w = source.chisquare(nu, n_sims, n_steps, 1) / nu
    return z / np.sqrt(w) * np.sqrt((nu - 2.0) / nu)


def simulate_log_returns(
    mu: ArrayLike,
    sigma: ArrayLike,
    corr: ArrayLike,
    horizon_days: int,
    n_sims: int,
    rng: SeedLike = None,
    dist: Dist = "normal",
    nu: float | None = None,
) -> FloatArray:
    r"""Cumulative log returns over ``horizon_days``, shape ``(n_sims, d)``.

    Daily step :math:`x = (\mu - \tfrac12\sigma^2)\Delta + \sigma\sqrt\Delta\,\varepsilon`, with
    :math:`\Delta = 1/252` and annualised ``mu`` and ``sigma``. With ``dist="normal"`` this
    is exactly correlated GBM. With ``dist="t"`` it is a GBM-like model with Student-t
    shocks, and :math:`E[e^x]` then differs from :math:`e^{\mu\Delta}` by
    :math:`O(\Delta^2)`, which is negligible at daily steps.
    """
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    dt = 1.0 / TRADING_DAYS
    eps = correlated_innovations(n_sims, horizon_days, cholesky_factor(corr), rng, dist, nu)
    daily = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
    return daily.sum(axis=1)
