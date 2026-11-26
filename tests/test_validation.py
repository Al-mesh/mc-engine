"""Monte Carlo prices against analytic / semi-analytic benchmarks.

These are the tests that justify trusting the engine. Each compares an MC estimate to an
independently computed reference and requires agreement to within 4 standard errors
(see conftest.py for why 4). Where a scheme has a known discretisation bias, the test
uses a grid fine enough that the bias is small relative to the SE, and the bias itself is
studied in ``benchmarks/heston_bias.py`` rather than hidden by a loose tolerance.
"""

import dataclasses

import numpy as np
import pytest
from scipy.stats import binom

from mc_engine import GBM, Heston, Merton, mc_price
from mc_engine.analytic import (
    bgk_shifted_barrier,
    bs_digital_price,
    bs_price,
    geometric_asian_price,
    heston_price,
    merton_price,
    up_and_out_call_price,
)
from mc_engine.payoffs import Digital, European, GeometricAsian, UpAndOutCall
from tests.conftest import assert_close_to_reference

pytestmark = pytest.mark.statistical

GBM_MODEL = GBM(S0=100, r=0.05, sigma=0.2, q=0.01)
HESTON = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03, q=0.0)


@pytest.mark.parametrize("call", [True, False])
def test_confidence_interval_coverage_over_many_seeds(call):
    """The 95% CI must contain the BS price in ~95% of independent runs.

    This tests the *uncertainty quantification*, not just the point estimate: a biased
    estimator, or a wrong SE formula, both show up as miscoverage. Under correct coverage
    the hit count is Binomial(n_runs, 0.95); we accept the central 99.9% region of that law
    (for 500 runs: about [462, 489]), so the test is exact rather than "close enough".
    """
    n_runs, n_paths = 500, 20_000
    ref = float(bs_price(100, 100, 1.0, 0.05, 0.2, 0.01, call))
    hits = 0
    for seed in range(n_runs):
        lo, hi = mc_price(GBM_MODEL, European(100, call), 1.0, 1, n_paths, rng=seed).ci95
        hits += lo <= ref <= hi
    lo_ok, hi_ok = binom.ppf(0.0005, n_runs, 0.95), binom.ppf(0.9995, n_runs, 0.95)
    assert lo_ok <= hits <= hi_ok, f"coverage {hits}/{n_runs} outside [{lo_ok}, {hi_ok}]"


@pytest.mark.parametrize("K", [80, 100, 120])
@pytest.mark.parametrize("call", [True, False])
def test_gbm_european_matches_black_scholes(K, call):
    r = mc_price(GBM_MODEL, European(K, call), 1.0, 1, 400_000, rng=K + call)
    assert_close_to_reference(r, float(bs_price(100, K, 1.0, 0.05, 0.2, 0.01, call)))


@pytest.mark.parametrize("call", [True, False])
def test_gbm_digital_matches_closed_form(call):
    r = mc_price(GBM_MODEL, Digital(105, call, payout=10.0), 1.0, 1, 400_000, rng=10 + call)
    ref = float(bs_digital_price(100, 105, 1.0, 0.05, 0.2, 0.01, call, payout=10.0))
    assert_close_to_reference(r, ref)


def test_geometric_asian_matches_closed_form():
    """Validates the *path* simulation (joint law of the fixings), not just S_T."""
    n_fix = 12
    r = mc_price(GBM_MODEL, GeometricAsian(100), 1.0, n_fix, 400_000, rng=20)
    ref = geometric_asian_price(100, 100, 0.05, 0.2, np.linspace(0, 1, n_fix + 1)[1:], q=0.01)
    assert_close_to_reference(r, ref)


def test_discrete_barrier_matches_continuous_formula_with_bgk_correction():
    """Discrete daily monitoring vs Broadie-Glasserman-Kou-shifted continuous price.

    Also checks the *direction* of the monitoring effect: a discretely monitored knock-out
    is worth more than the continuously monitored one, because paths can cross and come
    back between dates without being observed.
    """
    S0, K, B, T, r_, sigma, n = 100, 100, 130, 1.0, 0.05, 0.2, 250
    m = GBM(S0, r_, sigma)
    res = mc_price(m, UpAndOutCall(K, B), T, n, 400_000, rng=30)
    bgk = up_and_out_call_price(S0, K, bgk_shifted_barrier(B, sigma, T / n), T, r_, sigma)
    continuous = up_and_out_call_price(S0, K, B, T, r_, sigma)
    assert_close_to_reference(res, bgk)
    assert res.price > continuous + 4 * res.std_error


@pytest.mark.parametrize("K", [80, 100, 120])
def test_merton_matches_series_solution(K):
    m = Merton(S0=100, r=0.05, sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15, q=0.01)
    res = mc_price(m, European(K), 1.0, 4, 400_000, rng=40 + K)
    assert_close_to_reference(res, merton_price(m, K, 1.0))


@pytest.mark.parametrize("K", [80, 100, 120])
@pytest.mark.parametrize("call", [True, False])
def test_heston_qe_matches_characteristic_function(K, call):
    """QE with dt = 1/50 in a Feller-violating regime: bias is far below the SE here."""
    res = mc_price(HESTON, European(K, call), 1.0, 50, 400_000, rng=50 + K + call)
    assert_close_to_reference(res, heston_price(HESTON, K, 1.0, call=call))


@pytest.mark.slow
@pytest.mark.parametrize("K", [80, 100, 120])
def test_heston_euler_converges_to_characteristic_function(K):
    """Full-truncation Euler on a fine grid (dt = 1/200)."""
    m = dataclasses.replace(HESTON, scheme="euler")
    res = mc_price(m, European(K), 1.0, 200, 400_000, rng=60 + K)
    assert_close_to_reference(res, heston_price(HESTON, K, 1.0))
