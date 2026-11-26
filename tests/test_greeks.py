"""Greeks: each method against analytic values, and the documented failure modes."""

import numpy as np
import pytest

from mc_engine import GBM, Heston
from mc_engine.analytic import (
    bs_delta,
    bs_digital_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_vega,
    geometric_asian_price,
)
from mc_engine.greeks import (
    GREEKS,
    GreekEstimate,
    fd_greeks,
    likelihood_ratio_greeks,
    pathwise_greeks,
)
from mc_engine.payoffs import (
    ArithmeticAsian,
    Digital,
    European,
    FixedLookbackCall,
    FloatingLookbackCall,
    GeometricAsian,
    SmoothedDigital,
    UpAndOutCall,
)
from mc_engine.pricer import time_grid
from mc_engine.variance_reduction import TerminalSpot

S0, K, T, R, SIG, Q = 100.0, 105.0, 1.0, 0.05, 0.2, 0.01
MODEL = GBM(S0=S0, r=R, sigma=SIG, q=Q)
N = 400_000


def bs_greeks(call: bool) -> dict[str, float]:
    a = (S0, K, T, R, SIG, Q)
    return {
        "delta": float(bs_delta(*a, call=call)),
        "gamma": float(bs_gamma(*a)),
        "vega": float(bs_vega(*a)),
        "rho": float(bs_rho(*a, call=call)),
    }


def assert_within(est: GreekEstimate, ref: float, n_se: float = 4.0, label: str = "") -> None:
    z = (est.value - ref) / est.std_error
    assert abs(z) < n_se, f"{label}: {est.value:.6f} vs {ref:.6f} ({z:+.2f} se)"


# ------------------------------------------------------------ European vs closed form


@pytest.mark.statistical
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize(
    "method", [pathwise_greeks, likelihood_ratio_greeks], ids=["pathwise", "lr"]
)
def test_unbiased_methods_match_black_scholes(method, call):
    est = method(MODEL, European(K, call), T, 1, N, seed=10 + call)
    for g, ref in bs_greeks(call).items():
        assert_within(est[g], ref, label=f"{method.__name__} {g}")


@pytest.mark.statistical
@pytest.mark.parametrize("call", [True, False])
def test_finite_differences_match_finite_difference_of_closed_form(call):
    """E[FD estimator] is exactly the FD of the true price (CRN changes variance, not mean),
    so the right reference is the FD of the BS formula with the *same* bumps; comparing to
    the true Greek would conflate MC error with the O(h^2) bump bias."""
    hs, hv, hr = 1.0, 0.005, 1e-3
    est = fd_greeks(
        MODEL, European(K, call), T, 1, N, seed=20, bump_spot=hs, bump_vol=hv, bump_rate=hr
    )

    def p(**kw):
        a = dict(S=S0, K=K, T=T, r=R, sigma=SIG, q=Q)
        a.update(kw)
        return float(bs_price(call=call, **a))

    refs = {
        "delta": (p(S=S0 + hs) - p(S=S0 - hs)) / (2 * hs),
        "gamma": (p(S=S0 + hs) - 2 * p() + p(S=S0 - hs)) / hs**2,
        "vega": (p(sigma=SIG + hv) - p(sigma=SIG - hv)) / (2 * hv),
        "rho": (p(r=R + hr) - p(r=R - hr)) / (2 * hr),
    }
    for g in GREEKS:
        assert_within(est[g], refs[g], label=f"fd {g}")
    # And the bump bias itself is small relative to the Greek at these bump sizes.
    true = bs_greeks(call)
    for g in GREEKS:
        assert abs(refs[g] - true[g]) < 2e-3 * max(abs(true[g]), 1e-2)


@pytest.mark.statistical
def test_variance_ranking_for_a_call():
    """Pathwise <= CRN finite differences < likelihood ratio, for delta and vega."""
    pw = pathwise_greeks(MODEL, European(K), T, 1, 100_000, seed=30)
    lr = likelihood_ratio_greeks(MODEL, European(K), T, 1, 100_000, seed=30)
    fd = fd_greeks(MODEL, European(K), T, 1, 100_000, seed=30)
    for g in ("delta", "vega"):
        assert pw[g].std_error <= 1.05 * fd[g].std_error
        assert fd[g].std_error < lr[g].std_error
    # Mixed pathwise-LR gamma beats pure LR gamma: one LR weight instead of a squared one.
    assert pw["gamma"].std_error < lr["gamma"].std_error


@pytest.mark.statistical
def test_crn_is_what_makes_finite_differences_work():
    """Without common random numbers the FD delta variance blows up like 1/h^2."""
    h = 0.5
    crn = fd_greeks(MODEL, European(K), T, 1, 50_000, seed=31, bump_spot=h)["delta"]
    import dataclasses

    from mc_engine.pricer import discounted_payoffs

    up = discounted_payoffs(dataclasses.replace(MODEL, S0=S0 + h), European(K), T, 1, 50_000, 1)
    dn = discounted_payoffs(dataclasses.replace(MODEL, S0=S0 - h), European(K), T, 1, 50_000, 2)
    indep_se = np.sqrt(up.var(ddof=1) + dn.var(ddof=1)) / (2 * h) / np.sqrt(50_000)
    assert indep_se > 20 * crn.std_error


# ----------------------------------------------------------- discontinuous payoffs


@pytest.mark.statistical
def test_pathwise_delta_of_a_digital_is_identically_zero_and_lr_fixes_it():
    true = float(bs_digital_delta(S0, K, T, R, SIG, Q))
    pw = pathwise_greeks(MODEL, Digital(K), T, 1, N, seed=40)["delta"]
    assert pw.value == 0.0 and pw.std_error == 0.0  # confidently, completely wrong
    assert true > 0.01
    lr = likelihood_ratio_greeks(MODEL, Digital(K), T, 1, N, seed=41)
    assert_within(lr["delta"], true, label="lr digital delta")


@pytest.mark.statistical
@pytest.mark.parametrize("eps", [0.5, 2.0])
def test_smoothed_digital_pathwise_is_unbiased_for_the_call_spread(eps):
    """Pathwise is valid for the Lipschitz call spread. Its exact delta is
    (Delta_C(K - eps) - Delta_C(K + eps)) / (2 eps)."""
    est = pathwise_greeks(MODEL, SmoothedDigital(K, eps), T, 1, N, seed=42)["delta"]
    ref = (bs_delta(S0, K - eps, T, R, SIG, Q) - bs_delta(S0, K + eps, T, R, SIG, Q)) / (2 * eps)
    assert_within(est, float(ref))


def test_smoothing_bias_is_second_order_in_eps():
    true = float(bs_digital_delta(S0, K, T, R, SIG, Q))

    def smoothed(eps):
        d = bs_delta(S0, K - eps, T, R, SIG, Q) - bs_delta(S0, K + eps, T, R, SIG, Q)
        return float(d) / (2 * eps)

    b1, b2 = smoothed(2.0) - true, smoothed(1.0) - true
    assert b1 / b2 == pytest.approx(4.0, rel=0.05)  # halving eps quarters the bias


@pytest.mark.statistical
def test_smoothing_trades_variance_for_bias():
    se = [
        pathwise_greeks(MODEL, SmoothedDigital(K, e), T, 1, 100_000, seed=43)["delta"].std_error
        for e in (4.0, 1.0)
    ]
    # Variance ~ 1/eps, so SE ~ eps^-1/2: a 4x smaller eps doubles the SE.
    assert se[1] / se[0] == pytest.approx(2.0, rel=0.15)


@pytest.mark.statistical
def test_pathwise_barrier_delta_is_biased_lr_and_fd_agree():
    """Up-and-out call: pathwise ignores paths pushed over the barrier by the bump, so it
    reports roughly the delta of a call that *happens* to survive. LR and CRN finite
    differences, computed on independent seeds, agree with each other instead."""
    p = UpAndOutCall(100, 130)
    m = GBM(S0=100, r=0.05, sigma=0.2)
    pw = pathwise_greeks(m, p, 1.0, 50, N, seed=44)["delta"]
    lr = likelihood_ratio_greeks(m, p, 1.0, 50, N, seed=45)["delta"]
    fd = fd_greeks(m, p, 1.0, 50, N, seed=46, bump_spot=0.5)["delta"]
    assert abs(lr.value - fd.value) < 4 * np.hypot(lr.std_error, fd.std_error)
    assert pw.value - lr.value > 20 * np.hypot(pw.std_error, lr.std_error)


# ----------------------------------------------------------------- path-dependent


def _geometric_asian_greeks(n_fix: int) -> dict[str, float]:
    fix = time_grid(T, n_fix)[1:]

    def p(**kw):
        a = dict(S0=S0, K=K, r=R, sigma=SIG, q=Q)
        a.update(kw)
        return geometric_asian_price(fixing_times=fix, **a)

    def d(name, x, h):  # Richardson-extrapolated central difference
        def c(hh):
            return (p(**{name: x + hh}) - p(**{name: x - hh})) / (2 * hh)

        return (4 * c(h / 2) - c(h)) / 3

    return {"delta": d("S0", S0, 0.1), "vega": d("sigma", SIG, 1e-4), "rho": d("r", R, 1e-4)}


@pytest.mark.statistical
@pytest.mark.parametrize(
    "method", [pathwise_greeks, likelihood_ratio_greeks], ids=["pathwise", "lr"]
)
def test_geometric_asian_greeks_match_closed_form(method):
    """A genuinely path-dependent check: 12 fixings, analytic Greeks from the closed form."""
    est = method(MODEL, GeometricAsian(K), T, 12, N, seed=50)
    for g, ref in _geometric_asian_greeks(12).items():
        assert_within(est[g], ref, label=f"{method.__name__} {g}")


@pytest.mark.statistical
def test_lr_delta_degrades_with_finer_grids_pathwise_does_not():
    """LR delta weight is Z_1 / (S0 sigma sqrt(dt_1)): more steps, smaller dt_1, more noise.

    Asymptotically (dt_1 -> 0, so Y becomes nearly independent of Z_1)
    Var(Y w) ~ E[Y^2] / (S0^2 sigma^2 dt_1), so the SE grows like sqrt(n_steps): x2 per 4x
    steps. On coarse grids Y and Z_1 are strongly correlated and the growth is slower
    (measured ratios 1.34, 1.70, 1.92 for 4->16->64->256 steps), so the test checks the
    asymptotic regime.
    """
    se = {
        n: likelihood_ratio_greeks(MODEL, ArithmeticAsian(K), T, n, 100_000, seed=51)[
            "delta"
        ].std_error
        for n in (64, 256)
    }
    assert se[256] / se[64] == pytest.approx(2.0, rel=0.1)
    pw_c = pathwise_greeks(MODEL, ArithmeticAsian(K), T, 4, 100_000, seed=51)["delta"]
    pw_f = pathwise_greeks(MODEL, ArithmeticAsian(K), T, 256, 100_000, seed=51)["delta"]
    assert pw_f.std_error < 1.5 * pw_c.std_error


def test_floating_lookback_pathwise_delta_is_price_over_spot():
    """The payoff is homogeneous of degree 1 in the path and S scales with S0 under GBM, so
    by Euler's theorem the per-path pathwise delta equals Y / S0 exactly."""
    from mc_engine.pricer import discounted_payoffs

    pw = pathwise_greeks(MODEL, FloatingLookbackCall(), T, 20, 5000, seed=52)["delta"]
    y = discounted_payoffs(MODEL, FloatingLookbackCall(), T, 20, 5000, rng=52)
    assert pw.value == pytest.approx(y.mean() / S0, rel=1e-12)


@pytest.mark.statistical
@pytest.mark.parametrize("payoff", [FixedLookbackCall(110), ArithmeticAsian(100, call=False)])
def test_pathwise_and_lr_agree_on_lipschitz_exotics(payoff):
    pw = pathwise_greeks(MODEL, payoff, T, 12, N, seed=53)
    lr = likelihood_ratio_greeks(MODEL, payoff, T, 12, N, seed=54)
    for g in ("delta", "vega", "rho"):
        diff = pw[g].value - lr[g].value
        assert abs(diff) < 4 * np.hypot(pw[g].std_error, lr[g].std_error), g


# ---------------------------------------------------------------------- plumbing


def test_fd_greeks_skip_vega_for_models_without_sigma():
    h = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03)
    est = fd_greeks(h, European(100), 1.0, 10, 2000, seed=60)
    assert set(est) == {"delta", "gamma", "rho"}


def test_pathwise_and_lr_require_gbm():
    h = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03)
    for fn in (pathwise_greeks, likelihood_ratio_greeks):
        with pytest.raises(TypeError):
            fn(h, European(100), 1.0, 10, 100, seed=0)


def test_payoff_without_gradient_raises():
    with pytest.raises(NotImplementedError):
        pathwise_greeks(MODEL, TerminalSpot(), 1.0, 1, 100, seed=0)


def test_smoothed_digital_validation_and_put_gradient():
    with pytest.raises(ValueError):
        SmoothedDigital(100, 0.0)
    paths = np.array([[100.0, 99.5], [100.0, 120.0]])
    g = SmoothedDigital(100, 1.0, call=False).path_gradient(paths, np.array([0.0, 1.0]))
    np.testing.assert_allclose(g[:, -1], [-0.5, 0.0])
    assert str(GreekEstimate(0.5, 0.01, 10)).startswith("+0.5")
