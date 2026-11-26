"""Structural and distributional tests of the path simulators."""

import dataclasses

import numpy as np
import pytest
from scipy import stats

from mc_engine import GBM, Heston, Merton
from mc_engine.models.base import validate_times
from mc_engine.models.merton import poisson_inverse_cdf
from mc_engine.rng import PseudoRandomSource

HESTON = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03, q=0.01)
MODELS = [
    GBM(S0=100, r=0.03, sigma=0.25, q=0.01),
    HESTON,
    dataclasses.replace(HESTON, scheme="euler"),
    Merton(S0=100, r=0.03, sigma=0.2, lam=0.8, mu_j=-0.1, sigma_j=0.15, q=0.01),
]
IDS = ["gbm", "heston-qe", "heston-euler", "merton"]


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_shapes_initial_value_and_positivity(model):
    times = np.linspace(0, 1, 11)
    sim = model.simulate(500, times, rng=0)
    assert sim.paths.shape == (500, 11)
    assert sim.n_paths == 500
    np.testing.assert_allclose(sim.paths[:, 0], 100.0)
    assert np.all(sim.paths > 0)
    np.testing.assert_array_equal(sim.times, times)
    np.testing.assert_array_equal(sim.terminal, sim.paths[:, -1])


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_reproducible_given_seed(model):
    a = model.simulate(50, [0, 0.5, 1], rng=11).paths
    b = model.simulate(50, [0, 0.5, 1], rng=11).paths
    np.testing.assert_array_equal(a, b)


@pytest.mark.statistical
@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_discounted_price_is_martingale(model):
    """E[S_t] = S0 exp((r-q)t) at every grid date (risk-neutral drift is right)."""
    times = np.linspace(0, 2, 9)
    s = model.simulate(200_000, times, rng=1).paths
    expected = 100 * np.exp((model.r - model.q) * times[1:])
    mean = s[:, 1:].mean(axis=0)
    se = s[:, 1:].std(axis=0, ddof=1) / np.sqrt(s.shape[0])
    z = (mean - expected) / se
    assert np.all(np.abs(z) < 4), z


@pytest.mark.statistical
def test_gbm_log_returns_are_exactly_normal_on_irregular_grid():
    """Exact scheme: log increments are N((r-q-s^2/2)dt, s^2 dt) for *any* step size."""
    m = GBM(S0=100, r=0.05, sigma=0.3)
    times = np.array([0.0, 0.01, 0.3, 1.7])
    log_s = np.log(m.simulate(100_000, times, rng=2).paths)
    dt = np.diff(times)
    incr = np.diff(log_s, axis=1)
    for k in range(dt.size):
        mu, sd = (0.05 - 0.045) * dt[k], 0.3 * np.sqrt(dt[k])
        # KS test against the exact law; p-value threshold is deliberately small since
        # we test several distributions.
        assert stats.kstest(incr[:, k], stats.norm(mu, sd).cdf).pvalue > 1e-3


def test_qe_variance_is_nonnegative_and_euler_stores_truncated():
    harsh = Heston(S0=100, v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, r=0.0)
    for scheme in ("qe", "euler"):
        sim = dataclasses.replace(harsh, scheme=scheme).simulate(
            20_000, np.linspace(0, 5, 11), rng=3
        )
        assert np.all(sim.variance >= 0)
        assert np.all(np.isfinite(sim.paths))
    # Harsh Feller violation: QE must actually put mass at zero (exponential branch).
    qe = harsh.simulate(20_000, np.linspace(0, 5, 11), rng=3)
    assert np.mean(qe.variance[:, -1] == 0.0) > 0.01


@pytest.mark.statistical
def test_qe_variance_matches_exact_cir_moments():
    """The QE step is moment matched, so E[v_T] and Var[v_T] should match CIR for one step."""
    m = HESTON
    dt = 0.25
    v = m.simulate(400_000, [0, dt], rng=4).variance[:, 1]
    e = np.exp(-m.kappa * dt)
    mean = m.theta + (m.v0 - m.theta) * e
    var = m.v0 * m.xi**2 * e / m.kappa * (1 - e) + m.theta * m.xi**2 / (2 * m.kappa) * (1 - e) ** 2
    assert abs(v.mean() - mean) < 4 * v.std() / np.sqrt(v.size)
    # SE of the sample variance ~ sqrt((m4 - s^4)/n); estimate it from the sample.
    se_var = np.sqrt(np.var((v - v.mean()) ** 2) / v.size)
    assert abs(v.var() - var) < 4 * se_var


def test_heston_without_vol_of_vol_is_gbm_in_distribution():
    # v0 = theta and tiny xi: QE log-price should match GBM with sigma = sqrt(theta).
    m = Heston(S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=1e-6, rho=0.0, r=0.02)
    x = np.log(m.simulate(100_000, np.linspace(0, 1, 5), rng=5).terminal / 100)
    # ln(S_T/S0) ~ N((r - theta/2) T, theta T) = N(0, 0.2^2).
    assert stats.kstest(x, stats.norm(0.0, 0.2).cdf).pvalue > 1e-3


def test_qe_without_martingale_correction_runs():
    m = dataclasses.replace(HESTON, martingale_correction=False)
    assert np.all(np.isfinite(m.simulate(1000, [0, 0.5, 1], rng=6).paths))


def test_poisson_inverse_cdf_matches_scipy():
    u = PseudoRandomSource(8).uniforms(50_000, 1)[..., 0].ravel()
    for mean in (0.01, 0.5, 4.0):
        np.testing.assert_array_equal(
            poisson_inverse_cdf(u, mean), stats.poisson.ppf(u, mean).astype(int)
        )


def test_poisson_inverse_cdf_edge_cases():
    assert poisson_inverse_cdf(np.array([0.3]), 0.0)[0] == 0
    # Monotone in u (required for antithetic / QMC).
    u = np.linspace(0.001, 0.999, 999)
    assert np.all(np.diff(poisson_inverse_cdf(u, 2.0)) >= 0)


@pytest.mark.parametrize("bad", [[0.5, 1.0], [0.0], [0.0, 1.0, 1.0], [[0.0, 1.0]]])
def test_validate_times_rejects_bad_grids(bad):
    with pytest.raises(ValueError):
        validate_times(bad)


def test_simulate_rejects_bad_n_paths():
    with pytest.raises(ValueError):
        GBM(100, 0.0, 0.2).simulate(0, [0, 1])


def test_discount_factor():
    assert GBM(100, 0.05, 0.2).discount(2.0) == pytest.approx(np.exp(-0.1))
