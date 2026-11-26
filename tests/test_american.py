"""American / Bermudan options: CRR tree and Longstaff-Schwartz."""

import numpy as np
import pytest

from mc_engine import GBM, Heston
from mc_engine.american import basis_matrix, lsm_price
from mc_engine.analytic import bs_price, heston_price
from mc_engine.analytic.binomial import crr_price

# Longstaff-Schwartz (2001) Table 1 setting: K = 40, r = 6%, 50 exercise dates per year.
K, R = 40.0, 0.06


# ------------------------------------------------------------------------- CRR tree


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("S0", [36.0, 40.0, 44.0])
def test_crr_european_converges_to_black_scholes(S0, call):
    tree = crr_price(S0, K, 1.0, R, 0.2, q=0.02, n_steps=2000, call=call, exercise="european")
    assert tree == pytest.approx(float(bs_price(S0, K, 1.0, R, 0.2, 0.02, call)), abs=5e-4)


def test_crr_american_put_dominates_european_and_intrinsic():
    for S0 in (30.0, 36.0, 40.0, 44.0):
        am = crr_price(S0, K, 1.0, R, 0.2, n_steps=1000)
        eu = crr_price(S0, K, 1.0, R, 0.2, n_steps=1000, exercise="european")
        assert am > eu
        assert am >= max(K - S0, 0.0)


def test_crr_american_call_without_dividends_is_european():
    """Merton: never exercise a call early on a non-dividend stock (with r >= 0), since
    C >= S - K e^{-rT} > S - K. With dividends early exercise can be optimal."""
    am = crr_price(40, K, 1.0, R, 0.3, n_steps=1000, call=True)
    eu = crr_price(40, K, 1.0, R, 0.3, n_steps=1000, call=True, exercise="european")
    assert am == pytest.approx(eu, abs=1e-10)
    am_q = crr_price(40, K, 1.0, R, 0.3, q=0.08, n_steps=1000, call=True)
    eu_q = crr_price(40, K, 1.0, R, 0.3, q=0.08, n_steps=1000, call=True, exercise="european")
    assert am_q > eu_q + 1e-3


def test_crr_bermudan_is_between_european_and_american_and_increasing_in_dates():
    kw = dict(S0=36.0, K=K, T=1.0, r=R, sigma=0.2, n_steps=1200)
    eu = crr_price(**kw, exercise="european")
    am = crr_price(**kw)
    berm = [crr_price(**kw, exercise="bermudan", n_exercise_dates=m) for m in (4, 12, 50)]
    assert eu < berm[0] < berm[1] < berm[2] < am


def test_crr_bermudan_with_every_step_is_american():
    kw = dict(S0=36.0, K=K, T=1.0, r=R, sigma=0.2, n_steps=500, smooth=False)
    assert crr_price(**kw, exercise="bermudan", n_exercise_dates=500) == crr_price(**kw)


def test_crr_argument_checks():
    with pytest.raises(ValueError):
        crr_price(36, K, 1.0, R, 0.2, n_steps=100, exercise="bermudan", n_exercise_dates=7)
    with pytest.raises(ValueError):
        crr_price(36, K, 1.0, R, 0.2, exercise="asian")
    with pytest.raises(ValueError):
        crr_price(36, K, 1.0, 0.5, 0.001, n_steps=1)  # p > 1: tree inconsistent


# ------------------------------------------------------------------ Longstaff-Schwartz


def bermudan_ref(S0, sigma, T, m=50):
    return crr_price(S0, K, T, R, sigma, n_steps=100 * m, exercise="bermudan", n_exercise_dates=m)


@pytest.mark.statistical
@pytest.mark.parametrize(
    "S0,sigma,T", [(36.0, 0.2, 1.0), (40.0, 0.2, 1.0), (44.0, 0.4, 1.0), (36.0, 0.4, 2.0)]
)
def test_lsm_out_of_sample_matches_bermudan_tree(S0, sigma, T):
    """Same exercise schedule (50 dates / year) for LSM and tree, so the only differences
    are MC noise and the (small, negative) sub-optimality of the regressed policy."""
    m = int(50 * T)
    res = lsm_price(GBM(S0, R, sigma), K, T, m, 100_000, degree=3, seed=int(S0 + 10 * sigma))
    ref = bermudan_ref(S0, sigma, T, m)
    z = (res.price - ref) / res.std_error
    assert abs(z) < 4, f"LSM {res.price:.4f} vs tree {ref:.4f} ({z:+.1f} se)"


@pytest.mark.statistical
def test_linear_basis_is_a_visibly_suboptimal_policy():
    """Any stopping rule gives a lower bound; a crude one gives a *noticeably* lower one."""
    res = lsm_price(GBM(36.0, R, 0.2), K, 1.0, 50, 100_000, basis="monomial", degree=1, seed=2)
    assert bermudan_ref(36.0, 0.2, 1.0) - res.price > 4 * res.std_error


@pytest.mark.statistical
def test_in_sample_estimate_has_foresight_bias():
    """With few training paths and many regressors the in-sample estimate is pushed *up*
    (the regression overfits each path's own future), while the out-of-sample one on
    independent paths stays below the true value."""
    g = GBM(36.0, R, 0.2)
    ref = bermudan_ref(36.0, 0.2, 1.0)
    gaps = []
    for seed in range(8):
        res = lsm_price(g, K, 1.0, 50, 20_000, n_train=300, degree=8, basis="monomial", seed=seed)
        gaps.append(res.in_sample_price - res.price)
        assert res.price < ref + 4 * res.std_error
    assert np.mean(gaps) > 0.02


def test_polynomial_families_of_equal_degree_give_identical_policies():
    """Monomials and Chebyshev polynomials span the same space: same fit, same price."""
    g = GBM(36.0, R, 0.2)
    a = lsm_price(g, K, 1.0, 20, 5000, basis="monomial", degree=4, seed=3)
    b = lsm_price(g, K, 1.0, 20, 5000, basis="chebyshev", degree=4, seed=3)
    assert a.price == pytest.approx(b.price, abs=1e-9)
    assert a.in_sample_price == pytest.approx(b.in_sample_price, abs=1e-9)


@pytest.mark.statistical
def test_american_call_without_dividends_matches_european():
    """Early exercise is never optimal here, yet the regressed policy *does* stop ~10% of
    paths. Diagnosis (seed 4): all at the last date before expiry, deep ITM (S/K ~ 1.35-1.55),
    where true continuation exceeds intrinsic by only K(1 - e^{-r dt}) ~ 0.05, less than the
    cubic fit's error in the tail. The value lost was 0.004 +/- 0.004 per path: decisions
    are wrong only where the holder is nearly indifferent, so they cost almost nothing.
    The price, which is what matters, is therefore checked, not the exercise count."""
    g = GBM(40.0, R, 0.3)
    res = lsm_price(g, K, 1.0, 50, 100_000, call=True, seed=4)
    euro = float(bs_price(40.0, K, 1.0, R, 0.3))
    assert abs(res.price - euro) < 4 * res.std_error


@pytest.mark.statistical
def test_control_variate_reduces_se_and_stays_consistent():
    g = GBM(40.0, R, 0.2)
    plain = lsm_price(g, K, 1.0, 50, 100_000, seed=5)
    cv = lsm_price(g, K, 1.0, 50, 100_000, seed=5, control_variate=True)
    assert cv.std_error < 0.8 * plain.std_error
    assert abs(cv.price - bermudan_ref(40.0, 0.2, 1.0)) < 4 * cv.std_error


def test_deep_in_the_money_put_is_exercised_immediately():
    res = lsm_price(GBM(10.0, R, 0.2), K, 1.0, 10, 2000, seed=6)
    assert res.price == pytest.approx(30.0)
    assert res.std_error == 0.0


@pytest.mark.statistical
def test_lsm_heston_put_is_worth_at_least_the_european():
    h = Heston(S0=36.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, r=R)
    res = lsm_price(h, K, 1.0, 25, 50_000, steps_per_exercise=4, seed=7)
    euro = heston_price(h, K, 1.0, call=False)
    assert res.price > euro + 4 * res.std_error


def test_lsm_argument_checks():
    g = GBM(36.0, R, 0.2)
    with pytest.raises(ValueError):
        lsm_price(g, K, 1.0, 0, 100)
    with pytest.raises(ValueError):
        basis_matrix(np.ones(3), "monomial", -1)
    with pytest.raises(ValueError):
        basis_matrix(np.ones(3), "legendre", 2)
    h = Heston(S0=36.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, r=R)
    with pytest.raises(TypeError):
        lsm_price(h, K, 1.0, 5, 100, control_variate=True)


def test_too_few_itm_paths_means_no_exercise_at_that_date():
    """Deep OTM put with tiny training set: regressions are skipped, policy never exercises,
    and the price is just the discounted European cash flow on the pricing paths."""
    res = lsm_price(GBM(80.0, R, 0.1), K, 1.0, 10, 1000, n_train=20, degree=3, seed=8)
    assert res.early_exercise_fraction == 0.0
    assert res.price >= 0.0
