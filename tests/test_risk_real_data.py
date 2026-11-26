"""Backtest on the committed real-price cache (AAPL, MSFT, JPM, XOM, JNJ, 2010-2024).

These tests read data/ only; they never touch the network. They pin down the empirical
findings reported in the README, so a change to the data or the backtest code that
alters a conclusion shows up as a test failure rather than a silently different plot.
"""

from pathlib import Path

import numpy as np
import pytest

from mc_engine.risk import backtest_summary, rolling_var_backtest
from mc_engine.risk.data import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKERS,
    cache_path,
    load_prices,
    log_returns,
)

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(
    not cache_path(DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END, DATA).exists(),
    reason="price cache not present",
)


@pytest.fixture(scope="module")
def returns():
    return log_returns(load_prices(cache_dir=DATA))


def test_cached_data_integrity(returns):
    assert list(returns.columns) == list(DEFAULT_TICKERS)
    assert len(returns) > 3700 and not returns.isna().any().any()
    assert returns.index.is_monotonic_increasing
    # No absurd prints: every daily log return within +/- 30%.
    assert (returns.abs() < 0.3).all().all()


@pytest.mark.statistical
@pytest.mark.slow
def test_real_data_backtest_findings(returns):
    """Documented findings (see README): a static-window Gaussian VaR fails at 99% on both
    count and clustering; EWMA removes the clustering; fat tails reduce, but do not
    eliminate, 99% exceedances."""
    w = np.ones(5) / 5

    def run(method, vol):
        bt = rolling_var_backtest(returns, w, method, vol, window=500, n_sims=5_000, seed=1)
        return backtest_summary(bt).set_index("alpha")

    gauss_win = run("normal", "window")
    gauss_ewma = run("normal", "ewma")
    t_ewma = run("t", "ewma")
    assert gauss_win.loc[0.99, "kupiec_p"] < 1e-3
    assert gauss_win.loc[0.99, "christoffersen_p"] < 1e-3
    assert gauss_ewma.loc[0.99, "christoffersen_p"] > 0.05
    assert t_ewma.loc[0.99, "christoffersen_p"] > 0.05
    assert t_ewma.loc[0.99, "exceedances"] < gauss_ewma.loc[0.99, "exceedances"]
    assert t_ewma.loc[0.95, "kupiec_p"] > 0.01
