r"""Longstaff-Schwartz (2001) least-squares Monte Carlo for Bermudan/American options.

The problem
-----------
An American option is worth :math:`V_0 = \sup_\tau E[e^{-r\tau}h(S_\tau)]` over stopping
times. On a grid of exercise dates :math:`t_1 < \dots < t_M` the optimal rule is
"exercise at :math:`t_j` iff :math:`h(S_{t_j}) \ge C_j(S_{t_j})`", where the continuation value
:math:`C_j(s) = E[e^{-r\Delta}V_{j+1} \mid S_{t_j} = s]` is a conditional expectation that
plain simulation cannot evaluate (it would need simulations within simulations).

The LSM idea
------------
A conditional expectation is the :math:`L^2`-projection onto functions of :math:`S_{t_j}`.
Approximate it by projecting onto a finite basis :math:`\{\psi_k\}`: regress the realised
discounted future cash flows :math:`Y_i` on :math:`\psi_k(S^{(i)}_{t_j})` across paths,

.. math::
    \hat\beta_j = \arg\min_\beta \sum_i
        \big(Y_i - \textstyle\sum_k \beta_k\psi_k(S^{(i)}_{t_j})\big)^2,
    \qquad \hat C_j(s) = \sum_k \hat\beta_{j,k}\psi_k(s).

Working backwards from :math:`t_M`, a path is exercised at :math:`t_j` when
:math:`h > \hat C_j`, and its cash flow is then replaced by :math:`h(S_{t_j})`. Two key details:

* **Regress on realised cash flows, not on fitted values.** :math:`Y_i` is the cash flow
  actually received under the estimated *future* policy. Errors in :math:`\hat C` only
  affect the exercise *decision*, never the value credited to the path, which makes LSM
  far more robust than regression-based value iteration (Tsitsiklis-Van Roy).
* **Regress on in-the-money paths only.** Exercise is only ever considered where
  :math:`h > 0`, so fitting :math:`C` where it matters improves the fit at equal degree.

Bias, in both directions
------------------------
* **Out-of-sample (low-biased).** Estimate :math:`\hat\beta` on training paths, then apply
  the frozen rule to *independent* pricing paths. Any stopping rule is sub-optimal or
  optimal, so this estimates :math:`E[\text{value of a feasible policy}] \le V_0`: a true
  lower bound, up to MC noise. This is the number to report.
* **In-sample (typically high-biased).** Fitting and pricing on the same paths lets the
  regression "see" each path's future, a foresight bias that pushes the estimate up.
  It shrinks as N grows relative to the number of basis functions.

Bases
-----
Any polynomial family of degree :math:`\le d` (monomials, Chebyshev, Hermite, plain
Laguerre) spans the *same* space, so in exact arithmetic it gives the *same* fit. They
differ only in numerical conditioning. What changes the answer is the span: the degree,
or a non-polynomial basis like Longstaff-Schwartz's weighted Laguerre functions
:math:`e^{-x/2}L_k(x)`. Inputs are normalised to :math:`x = S/K` for conditioning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial import chebyshev, laguerre, polynomial

from mc_engine.analytic.black_scholes import bs_price
from mc_engine.models.base import Model, validate_n_paths
from mc_engine.models.gbm import GBM
from mc_engine.pricer import PriceResult, time_grid

Basis = Literal["monomial", "chebyshev", "laguerre"]


def basis_matrix(x: np.ndarray, basis: Basis, degree: int) -> np.ndarray:
    """Design matrix ``(len(x), degree + 1)`` evaluated at normalised spot ``x = S/K``."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if basis == "monomial":
        return polynomial.polyvander(x, degree)
    if basis == "chebyshev":
        # Chebyshev polynomials are well conditioned on [-1, 1]; map the bulk of x = S/K
        # (roughly 0..2) there. Same span as monomials, so same fit up to roundoff.
        return chebyshev.chebvander(x - 1.0, degree)
    if basis == "laguerre":
        # Longstaff-Schwartz's weighted Laguerre functions exp(-x/2) L_k(x). The weight
        # makes this a different function space from plain polynomials.
        return np.exp(-0.5 * x)[:, None] * laguerre.lagvander(x, degree)
    raise ValueError(f"unknown basis {basis!r}")


@dataclass(frozen=True)
class LSMResult(PriceResult):
    """``price``/``std_error`` are the out-of-sample (low-biased) estimate."""

    in_sample_price: float
    n_train: int
    early_exercise_fraction: float


def _intrinsic(s: np.ndarray, K: float, call: bool) -> np.ndarray:
    return np.maximum(s - K, 0.0) if call else np.maximum(K - s, 0.0)


def _backward_induction(
    s: np.ndarray,
    K: float,
    call: bool,
    df: float,
    basis: Basis,
    degree: int,
    coefs: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, list[np.ndarray | None], np.ndarray]:
    """Run the LSM recursion on spot values ``s`` at exercise dates ``t_0..t_M``.

    If ``coefs`` is None, fit regressions (training / in-sample). Otherwise apply the given
    coefficients (out-of-sample). Returns (cash flows discounted to t_0 *before* the t_0
    decision, fitted coefficients indexed by date, exercised-early flag per path).
    """
    n_paths, n_dates = s.shape
    m = n_dates - 1
    ex = _intrinsic(s, K, call)
    v = ex[:, -1].copy()
    early = np.zeros(n_paths, dtype=bool)
    fitted: list[np.ndarray | None] = [None] * n_dates
    for j in range(m - 1, 0, -1):
        v *= df  # discount the path's future cash flow back to t_j
        itm = np.flatnonzero(ex[:, j] > 0)
        if coefs is None:
            # Need more ITM points than parameters for a meaningful regression; otherwise
            # leave the rule as "never exercise here" (conservative, still a valid policy).
            if itm.size <= degree + 1:
                continue
            x = basis_matrix(s[itm, j] / K, basis, degree)
            beta, *_ = np.linalg.lstsq(x, v[itm], rcond=None)
            fitted[j] = beta
        else:
            beta = coefs[j]
            if beta is None or itm.size == 0:
                continue
            x = basis_matrix(s[itm, j] / K, basis, degree)
        continuation = x @ beta
        stop = itm[ex[itm, j] > continuation]
        v[stop] = ex[stop, j]
        early[stop] = True
    v *= df  # t_1 -> t_0
    return v, fitted, early


def lsm_price(
    model: Model,
    K: float,
    maturity: float,
    n_exercise: int,
    n_paths: int,
    n_train: int | None = None,
    basis: Basis = "laguerre",
    degree: int = 3,
    call: bool = False,
    steps_per_exercise: int = 1,
    seed: int | np.random.SeedSequence | None = None,
    control_variate: bool = False,
) -> LSMResult:
    """Bermudan option with ``n_exercise`` equally spaced exercise dates (plus t = 0).

    Parameters
    ----------
    n_paths, n_train
        Pricing paths and (independent) training paths; ``n_train`` defaults to ``n_paths``.
    steps_per_exercise
        Simulation sub-steps between exercise dates. 1 is exact for GBM; Heston needs more.
    seed
        Spawns independent training and pricing streams.
    control_variate
        GBM only. Use the European option (same strike and maturity, Black-Scholes mean)
        as a control. The gain depends on how often the option is exercised early: paths
        stopped early receive :math:`K - S_\\tau`, only loosely correlated with
        :math:`(K - S_T)^+`. For the ITM put S0=36, K=40 (about 70% early exercise) it is a
        modest ~1.5x variance reduction; nearer the money it is larger.
        :math:`\\beta` is estimated on the *training* paths,
        hence is independent of the pricing paths, and the estimator stays exactly
        unbiased for the value of the frozen policy.

    The regression state is the spot only. For stochastic-volatility models the true
    continuation value also depends on :math:`v_t`, so a spot-only basis gives a valid
    but more sub-optimal (lower) policy; that is a documented limitation.
    """
    n_paths = validate_n_paths(n_paths)
    n_train = n_paths if n_train is None else validate_n_paths(n_train)
    if n_exercise < 1 or steps_per_exercise < 1:
        raise ValueError("n_exercise and steps_per_exercise must be >= 1")
    times = time_grid(maturity, n_exercise * steps_per_exercise)
    cols = slice(None, None, steps_per_exercise)  # exercise dates within the sim grid
    df = float(np.exp(-model.r * maturity / n_exercise))
    train_seed, price_seed = np.random.SeedSequence(seed).spawn(2)

    if control_variate and not isinstance(model, GBM):
        raise TypeError("the European control variate needs a closed form, i.e. GBM")

    start = time.perf_counter()
    s_train = model.simulate(n_train, times, train_seed).paths[:, cols]
    v_in, coefs, _ = _backward_induction(s_train, K, call, df, basis, degree)
    s_price = model.simulate(n_paths, times, price_seed).paths[:, cols]
    v_out, _, early = _backward_induction(s_price, K, call, df, basis, degree, coefs)

    if control_variate:
        disc_t = np.exp(-model.r * maturity)
        euro_mean = float(bs_price(model.S0, K, maturity, model.r, model.sigma, model.q, call))
        x_train = disc_t * _intrinsic(s_train[:, -1], K, call)
        cov = np.cov(v_in, x_train)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        v_in = v_in - beta * (x_train - euro_mean)
        v_out = v_out - beta * (disc_t * _intrinsic(s_price[:, -1], K, call) - euro_mean)
    elapsed = time.perf_counter() - start

    # t = 0: every path shares S0, so the decision is a comparison with the estimate itself.
    h0 = float(_intrinsic(np.array([model.S0]), K, call)[0])
    cont_out, cont_in = float(v_out.mean()), float(v_in.mean())
    price = max(h0, cont_out)
    se = 0.0 if h0 > cont_out else float(v_out.std(ddof=1) / np.sqrt(n_paths))
    return LSMResult(
        price=price,
        std_error=se,
        n_paths=n_paths,
        runtime_s=elapsed,
        in_sample_price=max(h0, cont_in),
        n_train=n_train,
        early_exercise_fraction=float(early.mean()),
    )
