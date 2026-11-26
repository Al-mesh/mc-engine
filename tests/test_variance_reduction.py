"""Variance reduction: unbiasedness, correctness of error estimates, and actual reduction."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import binom

from mc_engine import GBM, Heston, Merton, mc_price
from mc_engine.analytic import bs_price, geometric_asian_price, heston_price, merton_price
from mc_engine.brownian_bridge import (
    BrownianBridgeSource,
    bridge_schedule,
    brownian_bridge_increments,
)
from mc_engine.payoffs import ArithmeticAsian, European, GeometricAsian, Payoff
from mc_engine.pricer import time_grid
from mc_engine.rng import AntitheticSource, PseudoRandomSource, SobolSource
from mc_engine.variance_reduction import (
    control_variate_price,
    efficiency_table,
    geometric_asian_control,
    rqmc_price,
    terminal_spot_control,
)
from tests.conftest import assert_close_to_reference

GBM_MODEL = GBM(S0=100, r=0.05, sigma=0.2)
BS_ATM = float(bs_price(100, 100, 1.0, 0.05, 0.2))
HESTON = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03)


@dataclass(frozen=True)
class LogSquare(Payoff):
    """(ln S_T - E ln S_T)^2: an *even* function of the driving normal under GBM."""

    centre: float

    def __call__(self, paths, times):
        return (np.log(paths[:, -1]) - self.centre) ** 2


# ------------------------------------------------------------------------ antithetic


def test_antithetic_source_reflects_pairs():
    src = AntitheticSource(PseudoRandomSource(0))
    z = src.normals(6, 2, 1)
    np.testing.assert_array_equal(z[3:], -z[:3])
    u = src.uniforms(4, 3, 2)
    np.testing.assert_allclose(u[2:], 1.0 - u[:2])
    with pytest.raises(ValueError):
        src.normals(5, 1)


@pytest.mark.statistical
def test_antithetic_unbiased_and_reduces_variance_for_monotone_payoff():
    plain = mc_price(GBM_MODEL, European(100), 1.0, 1, 200_000, rng=1)
    anti = mc_price(GBM_MODEL, European(100), 1.0, 1, 200_000, rng=1, antithetic=True)
    assert_close_to_reference(anti, BS_ATM)
    # Same number of paths (same cost): a call is monotone in Z, so the pair correlation
    # is negative and the SE must fall. Observed ratio is ~2x in variance.
    assert anti.std_error < 0.8 * plain.std_error


@pytest.mark.statistical
def test_antithetic_hurts_for_even_payoff():
    """For a payoff symmetric in Z, Y(Z) = Y(-Z): the pair is perfectly correlated, so a
    pair carries the information of one path at the cost of two. Variance per path doubles."""
    centre = np.log(100) + (0.05 - 0.02) * 1.0
    p = LogSquare(centre)
    plain = mc_price(GBM_MODEL, p, 1.0, 1, 200_000, rng=2)
    anti = mc_price(GBM_MODEL, p, 1.0, 1, 200_000, rng=2, antithetic=True)
    ratio = anti.std_error**2 / plain.std_error**2
    assert 1.8 < ratio < 2.2
    # Still unbiased: E[(sigma W_T)^2] = sigma^2 T, discounted.
    assert_close_to_reference(anti, np.exp(-0.05) * 0.04)


@pytest.mark.statistical
def test_antithetic_ci_coverage():
    """SE from pair averages must give correct coverage (treating the 2m dependent paths as
    independent would understate the SE for monotone payoffs' complement, etc.)."""
    n_runs = 300
    hits = 0
    for seed in range(n_runs):
        lo, hi = mc_price(GBM_MODEL, European(100), 1.0, 1, 10_000, rng=seed, antithetic=True).ci95
        hits += lo <= BS_ATM <= hi
    assert binom.ppf(0.0005, n_runs, 0.95) <= hits <= binom.ppf(0.9995, n_runs, 0.95)


@pytest.mark.statistical
@pytest.mark.parametrize(
    "model,ref",
    [
        (HESTON, heston_price(HESTON, 100, 1.0)),
        (
            Merton(S0=100, r=0.05, sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15),
            merton_price(
                Merton(S0=100, r=0.05, sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15), 100, 1
            ),
        ),
    ],
    ids=["heston", "merton"],
)
def test_antithetic_unbiased_for_other_models(model, ref):
    res = mc_price(model, European(100), 1.0, 50, 200_000, rng=3, antithetic=True)
    assert_close_to_reference(res, ref)


def test_antithetic_requires_even_counts():
    with pytest.raises(ValueError):
        mc_price(GBM_MODEL, European(100), 1.0, 1, 1001, rng=0, antithetic=True)


# ------------------------------------------------------------------ control variates


def test_control_variate_with_itself_has_zero_variance():
    """If the control *is* the target, beta = 1 and every adjusted sample equals the mean."""
    ctrl, mu = geometric_asian_control(GBM_MODEL, 100, 1.0, 12)
    res = control_variate_price(GBM_MODEL, ctrl, ctrl, mu, 1.0, 12, 10_000, rng=4)
    assert res.beta == pytest.approx(1.0)
    assert res.price == pytest.approx(mu, abs=1e-10)
    assert res.std_error < 1e-10


@pytest.mark.statistical
def test_geometric_control_for_arithmetic_asian():
    ctrl, mu = geometric_asian_control(GBM_MODEL, 100, 1.0, 12)
    cv = control_variate_price(GBM_MODEL, ArithmeticAsian(100), ctrl, mu, 1.0, 12, 100_000, rng=5)
    plain = mc_price(GBM_MODEL, ArithmeticAsian(100), 1.0, 12, 100_000, rng=6)
    assert cv.pilot_correlation > 0.99
    # 1/(1 - rho^2) with rho > 0.99 implies > 50x variance reduction.
    assert (plain.std_error / cv.std_error) ** 2 > 50
    # The two estimators are independent, so their difference has SE sqrt(se1^2 + se2^2).
    diff_se = np.hypot(plain.std_error, cv.std_error)
    assert abs(cv.price - plain.price) < 4 * diff_se


@pytest.mark.statistical
def test_control_variate_unbiased_against_closed_form():
    """CV estimate of a *European* with S_T as control, checked against Black-Scholes."""
    ctrl, mu = terminal_spot_control(GBM_MODEL, 1.0)
    assert mu == pytest.approx(100.0)
    res = control_variate_price(GBM_MODEL, European(100), ctrl, mu, 1.0, 1, 200_000, rng=7)
    assert_close_to_reference(res, BS_ATM)
    assert 0 < res.beta < 1  # roughly the (discounted) delta of the option


@pytest.mark.statistical
def test_control_variate_ci_coverage():
    """Pilot-estimated beta keeps the estimator exactly unbiased, so coverage must be ~95%."""
    ctrl, mu = terminal_spot_control(GBM_MODEL, 1.0)
    n_runs = 300
    hits = 0
    for seed in range(n_runs):
        res = control_variate_price(
            GBM_MODEL, European(100), ctrl, mu, 1.0, 1, 10_000, rng=seed, n_pilot=2_000
        )
        lo, hi = res.ci95
        hits += lo <= BS_ATM <= hi
    assert binom.ppf(0.0005, n_runs, 0.95) <= hits <= binom.ppf(0.9995, n_runs, 0.95)


@pytest.mark.statistical
def test_terminal_spot_control_works_for_heston():
    ctrl, mu = terminal_spot_control(HESTON, 1.0)
    res = control_variate_price(HESTON, European(100), ctrl, mu, 1.0, 50, 200_000, rng=8)
    assert_close_to_reference(res, heston_price(HESTON, 100, 1.0))


def test_geometric_control_rejects_non_gbm():
    with pytest.raises(TypeError):
        geometric_asian_control(HESTON, 100, 1.0, 12)


def test_constant_control_gives_zero_beta():
    @dataclass(frozen=True)
    class Const(Payoff):
        def __call__(self, paths, times):
            return np.ones(paths.shape[0])

    res = control_variate_price(
        GBM_MODEL, European(100), Const(), np.exp(-0.05), 1.0, 1, 1000, rng=9, n_pilot=500
    )
    assert res.beta == 0.0 and res.pilot_correlation == 0.0


# --------------------------------------------------------------------- brownian bridge


def test_bridge_schedule_visits_every_point_once_terminal_first():
    for m in (1, 2, 5, 12, 64):
        sched = bridge_schedule(m)
        assert sched[0][0] == m
        assert sorted(i for i, _, _ in sched) == list(range(1, m + 1))


def test_bridge_terminal_value_uses_first_normal_only():
    z = PseudoRandomSource(10).normals(1000, 16, 2)
    incr = brownian_bridge_increments(z)
    np.testing.assert_allclose(incr.sum(axis=1), np.sqrt(16) * z[:, 0], atol=1e-12)


def test_bridge_is_a_linear_isometry():
    """The bridge map is a fixed matrix A with A A^T = I (an orthogonal change of basis of
    Gaussian space), so the output increments are *exactly* i.i.d. N(0, 1) in law."""
    m = 12
    basis = np.eye(m).reshape(m, m, 1)  # m "paths", each a unit vector
    a = brownian_bridge_increments(basis)[..., 0].T  # column j = image of e_j
    np.testing.assert_allclose(a @ a.T, np.eye(m), atol=1e-12)


@pytest.mark.statistical
def test_bridge_source_preserves_plain_mc_results():
    src = BrownianBridgeSource(PseudoRandomSource(11))
    res = mc_price(GBM_MODEL, GeometricAsian(100), 1.0, 12, 200_000, rng=src)
    ref = geometric_asian_price(100, 100, 0.05, 0.2, time_grid(1.0, 12)[1:])
    assert_close_to_reference(res, ref)


def test_bridge_source_passes_uniforms_through():
    a = BrownianBridgeSource(PseudoRandomSource(12)).uniforms(5, 3, 2)
    b = PseudoRandomSource(12).uniforms(5, 3, 2)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------- Sobol


def test_sobol_source_shapes_reproducibility_and_checks():
    u = SobolSource(13).uniforms(64, 4, 2)
    assert u.shape == (64, 4, 2) and np.all((u > 0) & (u < 1))
    np.testing.assert_array_equal(u, SobolSource(13).uniforms(64, 4, 2))
    assert not np.allclose(u, SobolSource(14).uniforms(64, 4, 2))
    src = SobolSource(15)
    src.uniforms(32, 4, 1)
    with pytest.raises(ValueError, match="dimension"):
        src.uniforms(32, 5, 1)
    with pytest.raises(ValueError, match="power of two"):
        src.uniforms(30, 4, 1)


def test_sobol_points_are_stratified():
    """A (t, m, s)-net: 2^k points put exactly one point in each of 2^k equal 1-D cells."""
    u = SobolSource(16).uniforms(1024, 3, 1)[..., 0]
    for d in range(3):
        counts = np.bincount((u[:, d] * 1024).astype(int), minlength=1024)
        assert np.all(counts == 1)


@pytest.mark.statistical
@pytest.mark.parametrize("bb", [True, False])
def test_rqmc_unbiased_and_much_better_than_mc(bb):
    rq = rqmc_price(GBM_MODEL, European(100), 1.0, 8, 2**13, 16, seed=17, brownian_bridge=bb)
    mc = mc_price(GBM_MODEL, European(100), 1.0, 8, 2**17, rng=17)
    assert rq.n_paths == mc.n_paths
    assert_close_to_reference(rq, BS_ATM)
    assert rq.std_error < mc.std_error / 5


@pytest.mark.statistical
def test_rqmc_ci_coverage():
    """Coverage of the replication-based CI. With R = 16 the t-distribution would be exact;
    the normal quantile slightly undercovers (~94.4% instead of 95%), well inside the band."""
    n_runs = 200
    hits = 0
    for seed in range(n_runs):
        lo, hi = rqmc_price(GBM_MODEL, European(100), 1.0, 4, 2**8, 16, seed=seed).ci95
        hits += lo <= BS_ATM <= hi
    assert binom.ppf(0.0005, n_runs, 0.95) <= hits <= binom.ppf(0.9995, n_runs, 0.95)


@pytest.mark.statistical
def test_rqmc_works_for_heston_qe():
    res = rqmc_price(HESTON, European(100), 1.0, 32, 2**12, 16, seed=18, brownian_bridge=False)
    assert_close_to_reference(res, heston_price(HESTON, 100, 1.0))


def test_rqmc_argument_checks():
    with pytest.raises(ValueError):
        rqmc_price(GBM_MODEL, European(100), 1.0, 1, 1000, 16)
    with pytest.raises(ValueError):
        rqmc_price(GBM_MODEL, European(100), 1.0, 1, 1024, 1)


# ----------------------------------------------------------------------- comparison


def test_efficiency_table_columns():
    a = mc_price(GBM_MODEL, European(100), 1.0, 1, 20_000, rng=19)
    b = mc_price(GBM_MODEL, European(100), 1.0, 1, 20_000, rng=19, antithetic=True)
    df = efficiency_table({"plain": a, "antithetic": b})
    assert df.loc["plain", "vr_ratio"] == pytest.approx(1.0)
    assert df.loc["antithetic", "vr_ratio"] > 1.0
    assert {"var_per_path", "work_norm_var", "efficiency_gain"} <= set(df.columns)
