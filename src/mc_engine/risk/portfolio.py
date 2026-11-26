"""A small multi-asset portfolio of stocks and European options, with full revaluation.

Full revaluation means that each scenario's P&L is computed by *repricing* every position at
the horizon, as opposed to a delta (or delta-gamma) approximation. For options this
matters: a short option position's loss is convex in the move, so delta-normal VaR
understates it. Options are repriced with Black-Scholes at the asset's own volatility
("sticky" vol, no vol shock). That is a modelling simplification, and the README says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from mc_engine.analytic.black_scholes import bs_price
from mc_engine.risk.scenarios import TRADING_DAYS, Dist, simulate_log_returns
from mc_engine.risk.var import var_es
from mc_engine.rng import FloatArray


@dataclass(frozen=True)
class OptionPosition:
    asset: int
    K: float
    maturity: float  # years from today
    call: bool
    quantity: float  # contracts (negative = short)


@dataclass(frozen=True)
class Portfolio:
    spot: FloatArray
    vol: FloatArray
    shares: FloatArray
    options: Sequence[OptionPosition] = field(default_factory=tuple)
    r: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "spot", np.asarray(self.spot, dtype=float))
        object.__setattr__(self, "vol", np.asarray(self.vol, dtype=float))
        object.__setattr__(self, "shares", np.asarray(self.shares, dtype=float))
        if not self.spot.shape == self.vol.shape == self.shares.shape:
            raise ValueError("spot, vol and shares must have the same length")

    def value(self, spot: ArrayLike, elapsed: float = 0.0) -> FloatArray:
        """Portfolio value for each row of ``spot`` (shape (n, d)), ``elapsed`` years later."""
        s = np.atleast_2d(np.asarray(spot, dtype=float))
        total = s @ self.shares
        for opt in self.options:
            tau = opt.maturity - elapsed
            if tau <= 0:
                raise ValueError("risk horizon must end before every option's expiry")
            px = bs_price(s[:, opt.asset], opt.K, tau, self.r, self.vol[opt.asset], call=opt.call)
            total = total + opt.quantity * px
        return total

    def pnl(self, log_returns: ArrayLike, horizon_days: int) -> FloatArray:
        """Scenario P&L: revalue at S0 * exp(x) after the horizon, minus today's value."""
        x = np.asarray(log_returns, dtype=float)
        elapsed = horizon_days / TRADING_DAYS
        return self.value(self.spot * np.exp(x), elapsed) - self.value(self.spot)[0]


def risk_report(
    portfolio: Portfolio,
    mu: ArrayLike,
    corr: ArrayLike,
    alphas: Sequence[float] = (0.95, 0.99),
    horizons: Sequence[int] = (1, 10),
    dist: Dist = "normal",
    nu: float | None = None,
    n_sims: int = 200_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """VaR and ES table for each (horizon, alpha), plus the sqrt(h)-scaled one-day number.

    Each horizon uses independent scenarios simulated over its own number of daily steps.
    The ``sqrt_time_var`` column shows what the common shortcut
    :math:`\\mathrm{VaR}_h \\approx \\sqrt h\\,\\mathrm{VaR}_1` would have said. It is exact
    only for i.i.d. zero-mean normal P&L that is linear in the returns.
    """
    seeds = np.random.SeedSequence(seed).spawn(len(horizons))
    rows = []
    one_day: dict[float, float] = {}
    for h, s in zip(horizons, seeds, strict=True):
        x = simulate_log_returns(mu, portfolio.vol, corr, h, n_sims, s, dist, nu)
        pnl = portfolio.pnl(x, h)
        for a in alphas:
            v, e = var_es(pnl, a)
            if h == 1:
                one_day[a] = v
            rows.append(dict(horizon_days=h, alpha=a, VaR=v, ES=e))
    df = pd.DataFrame(rows)
    if one_day:
        df["sqrt_time_var"] = [
            one_day[a] * np.sqrt(h) for h, a in zip(df.horizon_days, df.alpha, strict=True)
        ]
    return df
