r"""Variance reduction: control variates, randomised QMC, and an efficiency report.

Antithetic variates live in :func:`mc_engine.pricer.mc_price` (``antithetic=True``)
because they only change the random source. The tools here need extra structure: a
control with known mean, or independent replications of a quasi-random estimator.

How to compare techniques fairly
--------------------------------
A technique that halves the variance but triples the runtime is a net loss. The relevant
figure of merit (Glasserman 2004, section 1.1.3) is the *work-normalised variance*
:math:`\sigma^2 \times \tau`, the per-sample variance times the per-sample cost, or
equivalently :math:`\mathrm{SE}^2 \times \text{runtime}` for a given run. Its inverse is
the efficiency. :func:`efficiency_table` reports both the raw variance-reduction ratio
(same number of paths) and this efficiency gain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from mc_engine.analytic.black_scholes import geometric_asian_price
from mc_engine.brownian_bridge import BrownianBridgeSource
from mc_engine.models.base import Model, validate_n_paths
from mc_engine.models.gbm import GBM
from mc_engine.payoffs import GeometricAsian, Payoff
from mc_engine.pricer import PriceResult, RunningMoments, time_grid
from mc_engine.rng import RandomSource, SeedLike, SobolSource, as_random_source

# --------------------------------------------------------------------------- controls


@dataclass(frozen=True)
class TerminalSpot(Payoff):
    r"""The payoff :math:`S_T`. Its discounted mean is :math:`S_0e^{-qT}` under *any*
    risk-neutral model, which makes it a model-free (if weak) control variate."""

    path_dependent: ClassVar[bool] = False

    def __call__(self, paths, times):
        return paths[:, -1].copy()


def terminal_spot_control(model: Model, maturity: float) -> tuple[Payoff, float]:
    """Control ``S_T`` with known discounted mean ``S0 * exp(-q T)``."""
    return TerminalSpot(), float(model.S0 * np.exp(-model.q * maturity))


def geometric_asian_control(
    model: GBM, K: float, maturity: float, n_steps: int, call: bool = True
) -> tuple[Payoff, float]:
    """Geometric Asian control for an arithmetic Asian on the same equally spaced fixings.

    Kemna & Vorst (1990): :math:`G \\le A` path by path and the two averages are almost
    perfectly correlated (rho ~ 0.99+ for typical parameters), while :math:`E[(G-K)^+]` is
    known in closed form under GBM. Only valid under GBM, so other models are rejected.
    """
    if not isinstance(model, GBM):
        raise TypeError("the geometric-Asian closed form requires a GBM model")
    fixings = time_grid(maturity, n_steps)[1:]
    mean = geometric_asian_price(model.S0, K, model.r, model.sigma, fixings, model.q, call)
    return GeometricAsian(K, call), mean


@dataclass(frozen=True)
class ControlVariateResult(PriceResult):
    beta: float
    pilot_correlation: float


def control_variate_price(
    model: Model,
    payoff: Payoff,
    control: Payoff,
    control_mean: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: SeedLike = None,
    n_pilot: int = 10_000,
    batch_size: int = 65_536,
) -> ControlVariateResult:
    r"""Control-variate estimator :math:`\hat V_{cv} = \bar Y - \beta(\bar X - \mu_X)`.

    For any fixed :math:`\beta` this is unbiased, with variance
    :math:`(\sigma_Y^2 - 2\beta\sigma_{XY} + \beta^2\sigma_X^2)/N`, minimised at
    :math:`\beta^* = \sigma_{XY}/\sigma_X^2`, where it equals :math:`\sigma_Y^2(1 - \rho_{XY}^2)/N`.
    So the variance-reduction ratio is :math:`1/(1-\rho^2)`: rho = 0.99 gives about 50x.

    Design choice: :math:`\beta` is estimated on an **independent pilot run** and then held
    fixed. Estimating it from the same paths as the main estimator makes the estimator
    slightly biased (O(1/N)), because beta-hat and X-bar are correlated. The bias is
    negligible in practice, but the pilot approach is exactly unbiased, which keeps the
    CI-coverage tests clean and keeps the main loop batchable. The pilot's cost is
    included in ``runtime_s`` but its paths are not counted in ``n_paths``.

    ``control_mean`` must be the *discounted* expectation of the control payoff.
    """
    n_paths = validate_n_paths(n_paths)
    source = as_random_source(rng)
    times = time_grid(maturity, n_steps)
    disc = model.discount(maturity)

    start = time.perf_counter()
    pilot = model.simulate(validate_n_paths(n_pilot), times, source)
    y_p = disc * payoff(pilot.paths, pilot.times)
    x_p = disc * control(pilot.paths, pilot.times)
    cov = np.cov(y_p, x_p)
    var_x = cov[1, 1]
    # A (numerically) constant control carries no information. Its sample variance is
    # roundoff (~1e-34), not 0, and dividing by it would give an arbitrary beta, so treat
    # anything below a relative threshold of the control's second moment as degenerate.
    degenerate = var_x <= 1e-12 * float(np.mean(x_p**2)) or cov[0, 0] <= 0
    beta = 0.0 if degenerate else float(cov[0, 1] / var_x)
    corr = 0.0 if degenerate else float(cov[0, 1] / np.sqrt(cov[0, 0] * var_x))

    stats = RunningMoments()
    remaining = n_paths
    while remaining > 0:
        m = min(batch_size, remaining)
        sim = model.simulate(m, times, source)
        y = disc * payoff(sim.paths, sim.times)
        x = disc * control(sim.paths, sim.times)
        stats.update(y - beta * (x - control_mean))
        remaining -= m
    elapsed = time.perf_counter() - start

    se = float(np.sqrt(stats.variance / n_paths)) if n_paths > 1 else float("nan")
    return ControlVariateResult(
        price=stats.mean,
        std_error=se,
        n_paths=n_paths,
        runtime_s=elapsed,
        beta=beta,
        pilot_correlation=corr,
    )


# ------------------------------------------------------------------------------ RQMC


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def rqmc_price(
    model: Model,
    payoff: Payoff,
    maturity: float,
    n_steps: int,
    n_paths: int,
    n_replications: int = 16,
    seed: int | np.random.SeedSequence | None = None,
    brownian_bridge: bool = True,
    batch_size: int = 65_536,
) -> PriceResult:
    r"""Randomised quasi-Monte Carlo with independent scramblings.

    Each of the ``R = n_replications`` scramblings yields an unbiased estimate
    :math:`\hat V_r` from ``n_paths`` Sobol points. The :math:`\hat V_r` are i.i.d., so the
    final estimate is their mean, with standard error :math:`\mathrm{sd}(\hat V_r)/\sqrt R`.
    That is a valid CI, unlike a single QMC run, whose points are deterministic given the
    scramble and so have no meaningful sample variance. Trade-off: more replications give a
    better SE estimate; more points per replication give more QMC benefit. R = 16 to 32 is
    standard.

    Total paths = ``n_paths * n_replications`` (reported in the result).
    """
    n_paths = validate_n_paths(n_paths)
    if not _is_power_of_two(n_paths) or not _is_power_of_two(batch_size):
        raise ValueError("n_paths and batch_size must be powers of two for Sobol points")
    if n_replications < 2:
        raise ValueError("need at least 2 replications to estimate the standard error")
    times = time_grid(maturity, n_steps)
    disc = model.discount(maturity)
    children = np.random.SeedSequence(seed).spawn(n_replications)

    start = time.perf_counter()
    estimates = np.empty(n_replications)
    for k, child in enumerate(children):
        source: RandomSource = SobolSource(child)
        if brownian_bridge:
            source = BrownianBridgeSource(source)
        stats = RunningMoments()
        remaining = n_paths
        while remaining > 0:
            m = min(batch_size, remaining)
            sim = model.simulate(m, times, source)
            stats.update(disc * payoff(sim.paths, sim.times))
            remaining -= m
        estimates[k] = stats.mean
    elapsed = time.perf_counter() - start

    se = float(estimates.std(ddof=1) / np.sqrt(n_replications))
    return PriceResult(
        price=float(estimates.mean()),
        std_error=se,
        n_paths=n_paths * n_replications,
        runtime_s=elapsed,
    )


# --------------------------------------------------------------- convergence studies


def convergence_study(
    model: Model,
    payoff: Payoff,
    maturity: float,
    n_steps: int,
    exact: float,
    path_counts: list[int],
    n_reps: int,
    method: str,
    seed: int = 0,
) -> pd.DataFrame:
    r"""Root-mean-square error of a *single* estimate as a function of N.

    For each N, ``n_reps`` independent estimates are made (independent seeds for MC,
    independent scramblings for QMC) and :math:`\mathrm{RMSE} = \sqrt{\frac1R\sum_r
    (\hat V_r - V)^2}` against the exact price, which captures bias and variance together.

    ``method``: ``"mc"``, ``"qmc"`` (scrambled Sobol, forward construction) or
    ``"qmc_bb"`` (scrambled Sobol + Brownian bridge).
    """
    disc = model.discount(maturity)
    times = time_grid(maturity, n_steps)
    seeds = np.random.SeedSequence(seed).spawn(n_reps)
    rows = []
    for n in path_counts:
        errs = np.empty(n_reps)
        for k, child in enumerate(seeds):
            if method == "mc":
                source: RandomSource = as_random_source(child)
            elif method in ("qmc", "qmc_bb"):
                source = SobolSource(child)
                if method == "qmc_bb":
                    source = BrownianBridgeSource(source)
            else:
                raise ValueError(f"unknown method {method!r}")
            sim = model.simulate(n, times, source)
            errs[k] = disc * payoff(sim.paths, sim.times).mean() - exact
        rows.append({"method": method, "n_paths": n, "rmse": float(np.sqrt(np.mean(errs**2)))})
    return pd.DataFrame(rows)


def loglog_slope(n_paths: np.ndarray, rmse: np.ndarray) -> float:
    """Least-squares slope of log(RMSE) on log(N): -0.5 for MC, approaching -1 for QMC."""
    slope, _ = np.polyfit(np.log(n_paths), np.log(rmse), 1)
    return float(slope)


# ------------------------------------------------------------------- comparison table


def efficiency_table(results: dict[str, PriceResult], baseline: str = "plain") -> pd.DataFrame:
    r"""Tabulate variance-reduction ratio and efficiency gain relative to ``baseline``.

    * ``var_per_path`` = :math:`\mathrm{SE}^2 \times N`: the variance of one path's worth
      of effort. Comparable across techniques regardless of N.
    * ``vr_ratio`` = baseline var_per_path / technique var_per_path: how many times more
      plain-MC paths you would need for the same SE.
    * ``work_norm_var`` = :math:`\mathrm{SE}^2 \times` runtime (lower is better).
    * ``efficiency_gain`` = baseline work_norm_var / technique work_norm_var: the speed-up
      in wall-clock time to reach a given accuracy, which is the number that matters.
    """
    rows = {
        name: {
            "price": r.price,
            "std_error": r.std_error,
            "n_paths": r.n_paths,
            "runtime_s": r.runtime_s,
            "var_per_path": r.std_error**2 * r.n_paths,
            "work_norm_var": r.std_error**2 * r.runtime_s,
        }
        for name, r in results.items()
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    base = df.loc[baseline]
    df["vr_ratio"] = base.var_per_path / df.var_per_path
    df["efficiency_gain"] = base.work_norm_var / df.work_norm_var
    return df
