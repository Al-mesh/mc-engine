"""Heston characteristic-function prices and the Merton series, checked independently."""

import numpy as np
import pytest

from mc_engine import GBM, Heston, Merton
from mc_engine.analytic import bs_price, heston_char_func, heston_price, merton_price

# Feller-violating (2*kappa*theta = 0.12 < xi^2 = 0.36): the hard regime.
H = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03, q=0.01)
# Andersen (2008) test case I: very strong vol-of-vol, long maturity.
ANDERSEN_I = Heston(S0=100, v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, r=0.0)


def test_char_func_basic_identities():
    T = 2.0
    assert heston_char_func(0.0, T, H) == pytest.approx(1.0)
    # phi(-i) = E[S_T] = forward (martingale property, model-independent check).
    assert heston_char_func(-1j, T, H).real == pytest.approx(100 * np.exp(0.02 * T), rel=1e-12)
    # Hermitian symmetry of a real random variable's CF.
    u = 1.7
    assert heston_char_func(-u, T, H) == pytest.approx(np.conj(heston_char_func(u, T, H)))


@pytest.mark.parametrize("model,T", [(H, 1.0), (H, 0.1), (ANDERSEN_I, 10.0)])
@pytest.mark.parametrize("K", [60, 80, 100, 120, 140])
def test_two_inversion_formulas_agree(model, T, K):
    lewis = heston_price(model, K, T, method="lewis")
    gp = heston_price(model, K, T, method="gil_pelaez")
    assert lewis == pytest.approx(gp, abs=1e-7)


@pytest.mark.parametrize("K", [70, 100, 130])
@pytest.mark.parametrize("method", ["lewis", "gil_pelaez"])
def test_reduces_to_black_scholes_when_vol_of_vol_vanishes(K, method):
    """xi -> 0 with v0 = theta: variance is constant, so Heston = BS(sigma = sqrt(theta)).

    rho = 0 here on purpose: the leading correction to BS is O(rho * xi) (the skew term),
    while for rho = 0 it is O(xi^2), so xi = 1e-4 gives ~1e-8 agreement. Going much below
    xi ~ 1e-5 is pointless: the kappa*theta/xi^2 terms in the CF cancel catastrophically.
    """
    m = Heston(S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0, r=0.05, q=0.02)
    bs = float(bs_price(100, K, 1.0, 0.05, 0.2, 0.02))
    assert heston_price(m, K, 1.0, method=method) == pytest.approx(bs, abs=1e-6)


def test_correlation_correction_is_first_order_in_vol_of_vol():
    """With rho != 0 the deviation from BS is linear in xi: shrinking xi 10x shrinks it 10x."""
    bs = float(bs_price(100, 130, 1.0, 0.05, 0.2, 0.02))

    def gap(xi):
        m = Heston(S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=xi, rho=0.3, r=0.05, q=0.02)
        return heston_price(m, 130, 1.0) - bs

    assert gap(1e-3) / gap(1e-4) == pytest.approx(10.0, rel=0.01)


def test_put_call_parity_and_bounds():
    T = 1.5
    for K in (80, 100, 120):
        c = heston_price(H, K, T)
        p = heston_price(H, K, T, call=False)
        fwd_pv = 100 * np.exp(-0.01 * T)
        assert c - p == pytest.approx(fwd_pv - K * np.exp(-0.03 * T), abs=1e-10)
        assert max(fwd_pv - K * np.exp(-0.03 * T), 0) <= c <= fwd_pv
        assert p > 0


def test_negative_correlation_produces_downside_skew():
    # rho < 0 fattens the left tail: OTM puts are dearer than in BS at the ATM-matched vol.
    sym = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=0.0, r=0.0)
    neg = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.0)
    assert heston_price(neg, 80, 1.0, call=False) > heston_price(sym, 80, 1.0, call=False)


def test_heston_price_input_validation():
    with pytest.raises(ValueError):
        heston_price(H, 100, 0.0)
    with pytest.raises(ValueError):
        heston_price(H, 100, 1.0, method="fft")


def test_merton_without_jumps_is_black_scholes():
    m = Merton(S0=100, r=0.05, sigma=0.2, lam=0.0, mu_j=-0.1, sigma_j=0.1)
    assert merton_price(m, 100, 1.0) == pytest.approx(
        float(bs_price(100, 100, 1, 0.05, 0.2)), abs=1e-12
    )


def test_merton_with_deterministic_zero_mean_jump_size_matches_bs():
    # If ln J ~ N(0, 0) the jumps are trivial (J = 1): price must equal BS for any lambda.
    m = Merton(S0=100, r=0.05, sigma=0.2, lam=3.0, mu_j=0.0, sigma_j=0.0)
    assert merton_price(m, 110, 0.5) == pytest.approx(
        float(bs_price(100, 110, 0.5, 0.05, 0.2)), abs=1e-12
    )


def test_merton_parity_and_jump_premium():
    m = Merton(S0=100, r=0.05, sigma=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15, q=0.01)
    c, p = merton_price(m, 100, 1.0), merton_price(m, 100, 1.0, call=False)
    assert c - p == pytest.approx(100 * np.exp(-0.01) - 100 * np.exp(-0.05), abs=1e-10)
    # Jumps add variance: price exceeds BS with the diffusive vol alone.
    assert c > float(bs_price(100, 100, 1, 0.05, 0.2, 0.01))


def test_gbm_and_heston_reject_bad_parameters():
    with pytest.raises(ValueError):
        GBM(S0=-1, r=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        Heston(S0=100, v0=0.04, kappa=1, theta=0.04, xi=0.5, rho=1.5, r=0)
    with pytest.raises(ValueError):
        Heston(S0=100, v0=0.04, kappa=0, theta=0.04, xi=0.5, rho=0.0, r=0)
    with pytest.raises(ValueError):
        Heston(S0=100, v0=-0.1, kappa=1, theta=0.04, xi=0.5, rho=0.0, r=0)
    with pytest.raises(ValueError):
        Heston(S0=100, v0=0.04, kappa=1, theta=0.04, xi=0.5, rho=0.0, r=0, scheme="milstein")
    with pytest.raises(ValueError):
        Merton(S0=100, r=0.0, sigma=0.2, lam=-1, mu_j=0, sigma_j=0.1)
    with pytest.raises(ValueError):
        Merton(S0=100, r=0.0, sigma=0.0, lam=1, mu_j=0, sigma_j=0.1)


def test_feller_flag():
    assert not H.feller_satisfied
    assert Heston(S0=100, v0=0.04, kappa=3, theta=0.04, xi=0.3, rho=0, r=0).feller_satisfied
