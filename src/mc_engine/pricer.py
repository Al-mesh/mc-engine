r"""Monte Carlo pricer: price, standard error and 95% confidence interval.

Risk-neutral pricing: :math:`V_0 = e^{-rT}E^Q[h(S)]`. The estimator is the sample mean
:math:`\hat V = \frac1N\sum_i Y_i` of i.i.d. discounted payoffs :math:`Y_i = e^{-rT}h(S^{(i)})`.
By the CLT, :math:`\sqrt N(\hat V - V_0) \Rightarrow N(0, \sigma_Y^2)`, so the standard error
is :math:`\hat\sigma_Y/\sqrt N` (sample std with Bessel's correction) and the asymptotic
95% CI is :math:`\hat V \pm z_{0.975}\,\hat\sigma_Y/\sqrt N`.

Paths are simulated in batches so that memory stays O(batch x steps) for any N. Batch
statistics are merged with Chan et al.'s parallel update, which is numerically stable
(no catastrophic cancellation from accumulating :math:`\sum Y^2`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from mc_engine.models.base import Model, validate_n_paths
from mc_engine.payoffs import Payoff
from mc_engine.rng import FloatArray, SeedLike, as_random_source

Z_95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class PriceResult:
    price: float
    std_error: float
    n_paths: int
    runtime_s: float

    @property
    def ci95(self) -> tuple[float, float]:
        half = Z_95 * self.std_error
        return self.price - half, self.price + half

    def __str__(self) -> str:
        lo, hi = self.ci95
        return (
            f"price={self.price:.6f}  se={self.std_error:.6f}  "
            f"95% CI=[{lo:.6f}, {hi:.6f}]  N={self.n_paths:,}  ({self.runtime_s:.3f}s)"
        )


class RunningMoments:
    """Streaming mean / variance via the pairwise update of Chan, Golub & LeVeque (1979).

    Merging a batch with count :math:`n_b`, mean :math:`\\bar x_b`, and sum of squared
    deviations :math:`M_b` into the running totals:
    :math:`\\delta = \\bar x_b - \\bar x_a`,
    :math:`\\bar x = \\bar x_a + \\delta n_b/n`,
    :math:`M = M_a + M_b + \\delta^2 n_a n_b / n`.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: FloatArray) -> None:
        x = np.asarray(x, dtype=float).ravel()
        n_b = x.size
        if n_b == 0:
            return
        mean_b = float(x.mean())
        m2_b = float(((x - mean_b) ** 2).sum())
        n = self.n + n_b
        delta = mean_b - self.mean
        self.mean += delta * n_b / n
        self.m2 += m2_b + delta**2 * self.n * n_b / n
        self.n = n

    @property
    def variance(self) -> float:
        """Unbiased sample variance (ddof = 1)."""
        return self.m2 / (self.n - 1) if self.n > 1 else float("nan")


def time_grid(maturity: float, n_steps: int) -> FloatArray:
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    return np.linspace(0.0, maturity, int(n_steps) + 1)


def discounted_payoffs(
    model: Model, payoff: Payoff, maturity: float, n_steps: int, n_paths: int, rng: SeedLike = None
) -> FloatArray:
    """Return the ``n_paths`` discounted payoff samples (no batching).

    Used when the per-path samples themselves are needed, e.g. for control variates or
    for inspecting the distribution.
    """
    times = time_grid(maturity, n_steps)
    sim = model.simulate(n_paths, times, as_random_source(rng))
    return model.discount(maturity) * payoff(sim.paths, sim.times)


def mc_price(
    model: Model,
    payoff: Payoff,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: SeedLike = None,
    batch_size: int = 65_536,
) -> PriceResult:
    """Plain Monte Carlo price of ``payoff`` under ``model``.

    Parameters
    ----------
    maturity
        Payment date T in years.
    n_steps
        Number of equal time steps. Exact schemes (GBM, Merton) need only one step for
        European payoffs; Heston needs a fine grid to control discretisation bias; for
        path-dependent payoffs the grid *is* the monitoring schedule.
    rng
        Seed, ``Generator`` or ``RandomSource``. With a pseudo-random source, results do
        not depend on ``batch_size`` for models that make one draw per simulate call (GBM),
        because numpy fills arrays in C order from one continuous stream.
    """
    n_paths = validate_n_paths(n_paths)
    source = as_random_source(rng)
    times = time_grid(maturity, n_steps)
    disc = model.discount(maturity)
    stats = RunningMoments()

    start = time.perf_counter()
    remaining = n_paths
    while remaining > 0:
        m = min(batch_size, remaining)
        sim = model.simulate(m, times, source)
        stats.update(disc * payoff(sim.paths, sim.times))
        remaining -= m
    elapsed = time.perf_counter() - start

    se = float(np.sqrt(stats.variance / n_paths)) if n_paths > 1 else float("nan")
    return PriceResult(price=stats.mean, std_error=se, n_paths=n_paths, runtime_s=elapsed)
