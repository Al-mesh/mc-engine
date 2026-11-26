"""Risk module: scenarios, VaR/ES, portfolio revaluation, backtests (synthetic data only)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, norm

from mc_engine.analytic import bs_price
from mc_engine.risk import (
    OptionPosition,
    Portfolio,
    backtest_summary,
    cholesky_factor,
    christoffersen_independence,
    correlated_innovations,
    gaussian_var_es,
    kupiec_pof,
    risk_report,
    rolling_var_backtest,
    simulate_log_returns,
    var_es,
)
from mc_engine.risk.backtest import ewma_covariances
from mc_engine.risk.data import cache_path, load_prices, log_returns

CORR = np.array([[1.0, 0.5], [0.5, 1.0]])
L = cholesky_factor(CORR)

# ------------------------------------------------------------------------ scenarios


def test_cholesky_validation():
    np.testing.assert_allclose(L @ L.T, CORR)
    with pytest.raises(ValueError, match="positive definite"):
        cholesky_factor([[1.0, 1.2], [1.2, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        cholesky_factor([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(ValueError, match="square"):
        cholesky_factor([1.0, 0.5])


@pytest.mark.statistical
@pytest.mark.parametrize("dist,nu", [("normal", None), ("t", 6.0)])
def test_innovations_have_unit_variance_and_target_correlation(dist, nu):
    e = correlated_innovations(400_000, 1, L, 1, dist, nu)[:, 0]
    np.testing.assert_allclose(e.var(axis=0), 1.0, atol=0.02)
    assert np.corrcoef(e.T)[0, 1] == pytest.approx(0.5, abs=0.01)


@pytest.mark.statistical
def test_student_t_marginals_have_the_theoretical_kurtosis():
    """Excess kurtosis of t_nu is 6/(nu - 4). nu = 12 keeps the 8th moment finite so the
    sample kurtosis itself has finite variance (it would not for nu <= 8)."""
    e = correlated_innovations(400_000, 1, L, 1, "t", 12.0)[:, 0]
    assert kurtosis(e, axis=0) == pytest.approx([0.75, 0.75], abs=0.08)


@pytest.mark.statistical
def test_t_copula_has_tail_dependence_gaussian_does_not():
    """Same correlation, same marginal quantile: joint 1% crashes are ~2x more likely
    under the t copula, because the shared chi-square mixing variable moves all assets."""

    def joint_crash(e):
        q = np.quantile(e, 0.01, axis=0)
        return np.mean((e[:, 0] < q[0]) & (e[:, 1] < q[1]))

    gauss = joint_crash(correlated_innovations(400_000, 1, L, 2, "normal")[:, 0])
    student = joint_crash(correlated_innovations(400_000, 1, L, 2, "t", 4.5)[:, 0])
    assert student > 1.6 * gauss


def test_innovation_argument_checks():
    with pytest.raises(ValueError):
        correlated_innovations(10, 1, L, 0, "t", None)
    with pytest.raises(ValueError):
        correlated_innovations(10, 1, L, 0, "t", 2.0)
    with pytest.raises(ValueError):
        correlated_innovations(10, 1, L, 0, "laplace")
    with pytest.raises(ValueError):
        simulate_log_returns([0, 0], [0.2, 0.2], CORR, 0, 10)


@pytest.mark.statistical
def test_multi_day_log_returns_have_gbm_moments():
    mu, sig, h = np.array([0.08, 0.02]), np.array([0.2, 0.35]), 10
    x = simulate_log_returns(mu, sig, CORR, h, 400_000, rng=3)
    dt = h / 252
    se_mean = sig * np.sqrt(dt / 4e5)
    assert np.all(np.abs(x.mean(axis=0) - (mu - sig**2 / 2) * dt) < 4 * se_mean)
    np.testing.assert_allclose(x.std(axis=0), sig * np.sqrt(dt), rtol=0.01)
    assert np.corrcoef(x.T)[0, 1] == pytest.approx(0.5, abs=0.01)


# ------------------------------------------------------------------------- VaR / ES


def test_empirical_var_es_definition():
    """Losses 1..100 at 95%: VaR = inf{l : F(l) >= 0.95} = 95 (F(95) = 0.95 exactly), and
    ES is the mean of the 5 losses beyond it, 96..100 = 98. This also guards the
    floating-point trap 100 * (1 - 0.95) = 5.000000000000004."""
    pnl = -np.arange(1, 101, dtype=float)
    var, es = var_es(pnl, 0.95)
    assert var == 95.0 and es == pytest.approx(98.0)
    assert var_es(pnl, 0.99) == (99.0, 100.0)
    assert var_es([-3.0], 0.99) == (3.0, 3.0)  # single scenario: ES falls back to VaR
    with pytest.raises(ValueError):
        var_es(pnl, 1.0)


@pytest.mark.statistical
def test_gaussian_closed_form_matches_simulation():
    x = np.random.default_rng(4).normal(0.1, 2.0, 1_000_000)
    for a in (0.95, 0.99):
        v, e = var_es(x, a)
        cv, ce = gaussian_var_es(0.1, 2.0, a)
        assert v == pytest.approx(cv, rel=0.01) and e == pytest.approx(ce, rel=0.01)


@pytest.mark.statistical
@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_single_stock_var_es_match_exact_lognormal(alpha):
    """For one stock the loss V(1 - e^X), X ~ N(m, s^2), has exact quantiles:
    VaR = V(1 - e^{m + s z_{1-a}}),
    ES  = V(1 - e^{m + s^2/2} Phi(z_{1-a} - s)/(1 - a)).
    Tolerance: the 99% quantile SE at n = 400k is ~0.25% of VaR; 1% is ~4 SE."""
    sigma, h, n = 0.25, 10, 400_000
    p = Portfolio(spot=[50.0], vol=[sigma], shares=[100.0])
    x = simulate_log_returns([0.05], [sigma], [[1.0]], h, n, rng=5)
    v, e = var_es(p.pnl(x, h), alpha)
    dt = h / 252
    m, s = (0.05 - sigma**2 / 2) * dt, sigma * np.sqrt(dt)
    z = norm.ppf(1 - alpha)
    exact_var = 5000 * (1 - np.exp(m + s * z))
    exact_es = 5000 * (1 - np.exp(m + s**2 / 2) * norm.cdf(z - s) / (1 - alpha))
    assert v == pytest.approx(exact_var, rel=0.01)
    assert e == pytest.approx(exact_es, rel=0.01)


# ------------------------------------------------------------------------- portfolio


def test_portfolio_valuation_and_theta():
    opt = OptionPosition(asset=1, K=100.0, maturity=0.5, call=True, quantity=-10)
    p = Portfolio(spot=[50.0, 100.0], vol=[0.2, 0.3], shares=[10.0, 5.0], options=(opt,), r=0.03)
    v0 = p.value(p.spot)[0]
    assert v0 == pytest.approx(1000.0 - 10 * float(bs_price(100, 100, 0.5, 0.03, 0.3)))
    # Zero price move over 10 days: P&L is pure theta, positive for a short call.
    theta_pnl = p.pnl(np.zeros((1, 2)), 10)[0]
    expected = -10 * (
        float(bs_price(100, 100, 0.5 - 10 / 252, 0.03, 0.3))
        - float(bs_price(100, 100, 0.5, 0.03, 0.3))
    )
    assert theta_pnl == pytest.approx(expected) and theta_pnl > 0
    with pytest.raises(ValueError, match="expiry"):
        p.value(p.spot, elapsed=0.6)
    with pytest.raises(ValueError):
        Portfolio(spot=[1.0, 2.0], vol=[0.2], shares=[1.0, 1.0])


@pytest.mark.statistical
def test_short_option_tail_is_worse_than_delta_equivalent():
    """Full revaluation sees the convexity: a short call loses more in the tail than a
    delta-equivalent short stock position with the same small-move sensitivity."""
    from mc_engine.analytic import bs_delta

    delta = float(bs_delta(100, 100, 0.25, 0.0, 0.3))
    short_call = Portfolio([100.0], [0.3], [0.0], (OptionPosition(0, 100.0, 0.25, True, -100),))
    delta_equiv = Portfolio([100.0], [0.3], [-100 * delta])
    x = simulate_log_returns([0.0], [0.3], [[1.0]], 10, 200_000, rng=6)
    v_opt, _ = var_es(short_call.pnl(x, 10), 0.99)
    v_lin, _ = var_es(delta_equiv.pnl(x, 10), 0.99)
    assert v_opt > 1.15 * v_lin


@pytest.mark.statistical
def test_risk_report_sqrt_time_rule():
    """sqrt(10) scaling roughly holds for Gaussian shocks but overstates the 99% 10-day VaR
    for fat-tailed daily shocks, since a sum of 10 t-shocks is closer to normal (CLT)."""
    p = Portfolio(spot=[100.0, 100.0], vol=[0.2, 0.3], shares=[1.0, 1.0])
    g = risk_report(p, [0, 0], CORR, n_sims=400_000, seed=7).set_index(["horizon_days", "alpha"])
    t = risk_report(p, [0, 0], CORR, dist="t", nu=4.5, n_sims=400_000, seed=7).set_index(
        ["horizon_days", "alpha"]
    )
    assert (g.ES >= g.VaR).all() and (t.ES >= t.VaR).all()
    ratio_g = g.loc[(10, 0.99), "sqrt_time_var"] / g.loc[(10, 0.99), "VaR"]
    ratio_t = t.loc[(10, 0.99), "sqrt_time_var"] / t.loc[(10, 0.99), "VaR"]
    assert 1.0 < ratio_g < 1.06  # lognormal loss is concave in the move: slight overstatement
    assert ratio_t > 1.08
    # Fat tails: higher 99% one-day VaR and ES than Gaussian at the same volatility.
    assert t.loc[(1, 0.99), "VaR"] > g.loc[(1, 0.99), "VaR"]
    assert t.loc[(1, 0.99), "ES"] > 1.2 * g.loc[(1, 0.99), "ES"]


# ------------------------------------------------------------------- backtest tests


def test_kupiec_exact_rate_and_edge_cases():
    e = np.zeros(1000, dtype=bool)
    e[:10] = True
    res = kupiec_pof(e, 0.99)
    assert res.statistic == pytest.approx(0.0, abs=1e-10) and res.p_value == pytest.approx(1.0)
    assert res.expected == pytest.approx(10.0)
    none = kupiec_pof(np.zeros(1000, dtype=bool), 0.99)  # too few is also a failure
    assert none.statistic == pytest.approx(-2 * 1000 * np.log(0.99)) and none.p_value < 1e-4
    allx = kupiec_pof(np.ones(50, dtype=bool), 0.99)
    assert np.isfinite(allx.statistic) and allx.p_value < 1e-10


@pytest.mark.statistical
def test_kupiec_and_christoffersen_have_correct_size_under_the_null():
    """Under i.i.d. Bernoulli(p) exceedances each test should reject ~5% of the time at
    the 5% level (chi-square approximation plus discreteness, so allow 2.5% to 8%)."""
    rng = np.random.default_rng(8)
    rej_k = rej_c = 0
    n_trials = 1000
    for _ in range(n_trials):
        e = rng.random(1000) < 0.05
        rej_k += kupiec_pof(e, 0.95).p_value < 0.05
        rej_c += christoffersen_independence(e).p_value < 0.05
    assert 0.025 < rej_k / n_trials < 0.08
    assert 0.025 < rej_c / n_trials < 0.08


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(9)
    e = np.zeros(2000, dtype=bool)
    for t in range(1, 2000):  # Markov chain: P(1|0) = 0.03, P(1|1) = 0.5
        e[t] = rng.random() < (0.5 if e[t - 1] else 0.03)
    assert christoffersen_independence(e).p_value < 1e-6


def test_ewma_has_no_look_ahead():
    r = np.random.default_rng(10).normal(size=(50, 2))
    a = ewma_covariances(r, 0.94, np.eye(2))
    r2 = r.copy()
    r2[30] *= 100  # a shock on day 30 may only affect forecasts from day 31 on
    b = ewma_covariances(r2, 0.94, np.eye(2))
    np.testing.assert_array_equal(a[:31], b[:31])
    assert not np.allclose(a[31], b[31])
    np.testing.assert_array_equal(a[0], np.eye(2))


def _synthetic_returns(regimes: bool, seed: int = 0, n_days: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.array([[1, 0.4, 0.3], [0.4, 1, 0.5], [0.3, 0.5, 1]])
    vol = np.where((np.arange(n_days) // 250) % 2 == 0, 0.01, 0.03) if regimes else 0.01
    r = (rng.standard_normal((n_days, 3)) @ np.linalg.cholesky(c).T) * np.reshape(vol, (-1, 1))
    return pd.DataFrame(r, index=pd.bdate_range("2000-01-03", periods=n_days))


@pytest.mark.statistical
@pytest.mark.parametrize("method", ["normal", "historical"])
def test_backtest_passes_when_the_model_is_right(method):
    """i.i.d. Gaussian returns with constant vol: a static-window model is correctly
    specified, so neither test should reject."""
    bt = rolling_var_backtest(
        _synthetic_returns(False), np.ones(3) / 3, method, window=250, n_sims=10_000
    )
    summary = backtest_summary(bt)
    assert (summary.kupiec_p > 0.01).all()
    assert (summary.christoffersen_p > 0.01).all()


@pytest.mark.statistical
def test_static_window_fails_under_volatility_regimes_ewma_does_not_cluster():
    """Volatility alternating 1% / 3% each year: the 250-day window lags, so exceedances
    are too many and clustered (both tests reject). EWMA tracks the regime: no clustering."""
    df = _synthetic_returns(True)
    w = np.ones(3) / 3
    static = backtest_summary(rolling_var_backtest(df, w, "normal", "window", 250, n_sims=10_000))
    ewma = backtest_summary(rolling_var_backtest(df, w, "normal", "ewma", 250, n_sims=10_000))
    assert (static.kupiec_p < 1e-4).all()
    assert (static.christoffersen_p < 0.01).all()
    assert (ewma.christoffersen_p > 0.05).all()
    assert (ewma.kupiec_lr < static.kupiec_lr).all()


@pytest.mark.statistical
def test_t_backtest_runs_and_estimates_heavier_tails_than_normal():
    df = _synthetic_returns(True)
    w = np.ones(3) / 3
    bt_t = rolling_var_backtest(df, w, "t", "ewma", 250, n_sims=10_000)
    bt_n = rolling_var_backtest(df, w, "normal", "ewma", 250, n_sims=10_000)
    assert (bt_t["var_0.99"] > 0).all()
    # Regime switches leave residual kurtosis after EWMA scaling, so t VaR >= normal VaR
    # at 99% on average.
    assert bt_t["var_0.99"].mean() > bt_n["var_0.99"].mean()


def test_backtest_argument_checks():
    df = _synthetic_returns(False, n_days=300)
    with pytest.raises(ValueError):
        rolling_var_backtest(df, [0.5, 0.5], window=250)
    with pytest.raises(ValueError):
        rolling_var_backtest(df, np.ones(3) / 3, window=400)
    with pytest.raises(ValueError):
        rolling_var_backtest(df, np.ones(3) / 3, vol_model="garch", window=250)
    with pytest.raises(ValueError):
        rolling_var_backtest(df, np.ones(3) / 3, method="evt", window=250, n_sims=100)


# -------------------------------------------------------------------------------- data


def test_load_prices_reads_the_cache_without_network(tmp_path: Path):
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        {"AAA": [1.0, 1.1, 1.2, 1.1, 1.3], "BBB": [2.0, 2.1, 2.0, 2.2, 2.3]}, index=idx
    )
    prices.to_csv(cache_path(("AAA", "BBB"), "s", "e", tmp_path))
    loaded = load_prices(("AAA", "BBB"), "s", "e", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(loaded, prices, check_freq=False)
    lr = log_returns(loaded)
    assert lr.shape == (4, 2)
    assert lr.iloc[0, 0] == pytest.approx(np.log(1.1))
