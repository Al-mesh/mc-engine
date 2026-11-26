r"""VaR backtesting: rolling out-of-sample forecasts, Kupiec POF and Christoffersen tests.

Kupiec (1995) proportion-of-failures
------------------------------------
If the :math:`\alpha`-VaR is correct, exceedances ("the loss was bigger than VaR") are
Bernoulli with :math:`p = 1 - \alpha`, so over :math:`n` days the count :math:`x` is
Binomial(n, p). The likelihood-ratio statistic for :math:`H_0: \pi = p` against the MLE
:math:`\hat\pi = x/n` is

.. math::
    LR_{POF} = -2\ln\frac{(1-p)^{n-x}p^x}{(1-\hat\pi)^{n-x}\hat\pi^x}
    \ \xrightarrow{d}\ \chi^2_1.

It is two-sided: too *few* exceedances (an over-conservative model) are also rejected.
Its power is low at 99%: in 250 days only 2.5 exceedances are expected, which is why
regulators look at multi-year windows (and why we use ~13 years of data).

Christoffersen (1998) independence
----------------------------------
Kupiec only counts exceedances. A model can have the right *number* but have them
clustered (all in 2008 and 2020), which is precisely the failure of a static-volatility
model. With :math:`n_{ij}` the number of transitions from state i to state j (1 =
exceedance), compare a first-order Markov chain with :math:`\pi_{01} \ne \pi_{11}` to the
i.i.d. model :math:`\pi`: :math:`LR_{ind} \to \chi^2_1`.

The forecasting setup
---------------------
For each day t, parameters are estimated from the previous ``window`` days only (no
look-ahead), a one-day VaR forecast is made, and it is compared with the realised return
on day t. The mean daily return is set to zero: over one day it is two orders of magnitude
smaller than the volatility and very noisily estimated, so including it adds noise, not
information. That is standard practice.

Volatility estimators:

* ``"window"``: equally weighted sample covariance of the last ``window`` days. It reacts
  slowly to regime changes.
* ``"ewma"``: RiskMetrics, :math:`\Sigma_t = \lambda\Sigma_{t-1} + (1-\lambda)r_{t-1}r_{t-1}^\top`
  with :math:`\lambda = 0.94`. Effective memory is about :math:`1/(1-\lambda) \approx 17` days, so
  it tracks volatility clustering.

For Student-t innovations :math:`\nu` is set from the pooled excess kurtosis :math:`k` of the
*volatility-standardised* window returns, via :math:`k = 6/(\nu - 4)`. Standardising by
the volatility in force on each day matters: raw returns look fat-tailed partly *because*
volatility varies, and counting that twice would over-fatten the tails.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import xlogy
from scipy.stats import chi2

from mc_engine.risk.scenarios import correlated_innovations
from mc_engine.risk.var import var_es
from mc_engine.rng import PseudoRandomSource

Method = Literal["historical", "normal", "t"]
VolModel = Literal["window", "ewma"]


@dataclass(frozen=True)
class LRTest:
    statistic: float
    p_value: float
    n_obs: int
    n_exceed: int
    expected: float


def kupiec_pof(exceedances: ArrayLike, alpha: float) -> LRTest:
    """Kupiec proportion-of-failures test for a boolean exceedance series."""
    e = np.asarray(exceedances, dtype=bool)
    n, x = e.size, int(e.sum())
    p = 1.0 - alpha
    pi_hat = x / n
    # xlogy(0, 0) = 0 handles x = 0 and x = n (log-likelihood of the MLE is then 0).
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - pi_hat) + xlogy(x, pi_hat)
    lr = float(-2.0 * (ll_null - ll_alt))
    return LRTest(lr, float(chi2.sf(lr, 1)), n, x, n * p)


def christoffersen_independence(exceedances: ArrayLike) -> LRTest:
    """Christoffersen's Markov test that exceedances do not cluster."""
    e = np.asarray(exceedances, dtype=int)
    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = float(-2.0 * (ll_null - ll_alt))
    return LRTest(lr, float(chi2.sf(lr, 1)), int(e.size), int(e.sum()), float("nan"))


def ewma_covariances(returns: np.ndarray, lam: float, init: np.ndarray) -> np.ndarray:
    """Forecast covariances: entry t uses returns up to t-1 only. Shape (T, d, d)."""
    t_len, d = returns.shape
    out = np.empty((t_len, d, d))
    cov = init.copy()
    for t in range(t_len):
        out[t] = cov
        r = returns[t]
        cov = lam * cov + (1.0 - lam) * np.outer(r, r)
    return out


def _nu_from_kurtosis(u: np.ndarray, nu_max: float = 100.0) -> float:
    k = float(np.mean(u**4) - 3.0)  # pooled excess kurtosis of standardised returns
    if k <= 6.0 / (nu_max - 4.0):
        return nu_max
    return float(np.clip(4.0 + 6.0 / k, 4.5, nu_max))


def rolling_var_backtest(
    log_returns: pd.DataFrame,
    weights: ArrayLike,
    method: Method = "normal",
    vol_model: VolModel = "window",
    window: int = 500,
    alphas: Sequence[float] = (0.95, 0.99),
    lam: float = 0.94,
    n_sims: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    """One-day-ahead VaR forecasts of a constant-weight portfolio, as return fractions.

    Returns a frame indexed by forecast date with the realised portfolio return, one
    ``var_<alpha>`` column per alpha (positive = loss), and boolean ``exceed_<alpha>``.
    """
    r = log_returns.to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.shape[1] != w.size:
        raise ValueError("one weight per asset")
    if r.shape[0] <= window:
        raise ValueError("need more observations than the estimation window")
    port_simple = (np.expm1(r)) @ w  # realised constant-weight portfolio returns
    source = PseudoRandomSource(seed)
    covs = ewma_covariances(r, lam, np.cov(r[:window].T)) if vol_model == "ewma" else None
    if vol_model not in ("window", "ewma"):
        raise ValueError(f"unknown vol_model {vol_model!r}")

    rows = []
    for t in range(window, r.shape[0]):
        hist = r[t - window : t]
        row: dict[str, float] = {"realized": float(port_simple[t])}
        if method == "historical":
            sample = port_simple[t - window : t]
            for a in alphas:
                row[f"var_{a}"] = var_es(sample, a)[0]
        else:
            if vol_model == "ewma":
                cov = covs[t]
                vol_path = np.sqrt(np.diagonal(covs[t - window : t], axis1=1, axis2=2))
            else:
                cov = np.cov(hist.T)
                vol_path = np.sqrt(np.diag(cov))[None, :]
            sd = np.sqrt(np.diag(cov))
            chol = np.linalg.cholesky(cov / np.outer(sd, sd))
            if method == "t":
                nu = _nu_from_kurtosis(hist / vol_path)
                eps = correlated_innovations(n_sims, 1, chol, source, "t", nu)[:, 0]
            elif method == "normal":
                eps = correlated_innovations(n_sims, 1, chol, source, "normal")[:, 0]
            else:
                raise ValueError(f"unknown method {method!r}")
            pnl = np.expm1(eps * sd) @ w
            for a in alphas:
                row[f"var_{a}"] = var_es(pnl, a)[0]
        rows.append(row)

    out = pd.DataFrame(rows, index=log_returns.index[window:])
    for a in alphas:
        out[f"exceed_{a}"] = -out["realized"] > out[f"var_{a}"]
    return out


def backtest_summary(bt: pd.DataFrame, alphas: Sequence[float] = (0.95, 0.99)) -> pd.DataFrame:
    """Kupiec and Christoffersen results per alpha for a :func:`rolling_var_backtest` frame."""
    rows = []
    for a in alphas:
        e = bt[f"exceed_{a}"].to_numpy()
        k = kupiec_pof(e, a)
        c = christoffersen_independence(e)
        rows.append(
            dict(
                alpha=a,
                n_obs=k.n_obs,
                exceedances=k.n_exceed,
                expected=k.expected,
                rate=k.n_exceed / k.n_obs,
                kupiec_lr=k.statistic,
                kupiec_p=k.p_value,
                christoffersen_lr=c.statistic,
                christoffersen_p=c.p_value,
            )
        )
    return pd.DataFrame(rows)
