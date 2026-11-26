"""Closed-form benchmarks must themselves be right before they can validate MC."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mc_engine.analytic import (
    bgk_shifted_barrier,
    bs_delta,
    bs_digital_price,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_vega,
    geometric_asian_price,
    up_and_out_call_price,
)

spot = st.floats(50, 150)
strike = st.floats(50, 150)
mat = st.floats(0.05, 3.0)
rate = st.floats(-0.02, 0.10)
vol = st.floats(0.05, 0.8)
div = st.floats(0.0, 0.05)


def test_textbook_value():
    # Hull, OFOD: S=100, K=100, T=1, r=5%, sigma=20% -> call 10.4506, put 5.5735.
    assert bs_price(100, 100, 1, 0.05, 0.2) == pytest.approx(10.450583572185565, abs=1e-10)
    assert bs_price(100, 100, 1, 0.05, 0.2, call=False) == pytest.approx(
        5.573526022256971, abs=1e-10
    )


@given(spot, strike, mat, rate, vol, div)
def test_put_call_parity(S, K, T, r, sigma, q):
    c = bs_price(S, K, T, r, sigma, q)
    p = bs_price(S, K, T, r, sigma, q, call=False)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-9)


@given(spot, strike, mat, rate, vol, div)
def test_no_arbitrage_bounds(S, K, T, r, sigma, q):
    c = bs_price(S, K, T, r, sigma, q)
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    assert lower - 1e-9 <= c <= S * np.exp(-q * T) + 1e-9


@settings(max_examples=50)
@given(spot, strike, mat, rate, vol, div, st.booleans())
def test_greeks_match_finite_differences(S, K, T, r, sigma, q, call):
    def price(**kw):
        args = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        args.update(kw)
        return float(bs_price(call=call, **args))

    # A plain central difference has truncation error h^2 f'''/6, which is large at low
    # vol / long maturity. Richardson extrapolation (4 D(h/2) - D(h)) / 3 cancels the h^2
    # term exactly, leaving O(h^4), so the comparison tests the formula, not the FD scheme.
    def d1(name, h):
        x = {"S": S, "sigma": sigma, "r": r}[name]

        def central(hh):
            return (price(**{name: x + hh}) - price(**{name: x - hh})) / (2 * hh)

        return (4 * central(h / 2) - central(h)) / 3

    def second(hh):
        return (price(S=S + hh) - 2 * price() + price(S=S - hh)) / hh**2

    fd_gamma = (4 * second(0.05) - second(0.1)) / 3
    assert bs_delta(S, K, T, r, sigma, q, call) == pytest.approx(d1("S", 1e-2), rel=1e-7, abs=1e-9)
    assert bs_gamma(S, K, T, r, sigma, q) == pytest.approx(fd_gamma, rel=1e-5, abs=1e-8)
    # Bumps must be small relative to the price's natural scale in that variable, which
    # for sigma and r is ~ sigma*sqrt(T)/d1^2 (tiny at low vol, deep ITM/OTM). h = 1e-4 is
    # safely below it while roundoff (~eps * price / h ~ 1e-11) stays negligible.
    assert bs_vega(S, K, T, r, sigma, q) == pytest.approx(d1("sigma", 1e-4), rel=1e-7, abs=1e-8)
    assert bs_rho(S, K, T, r, sigma, q, call) == pytest.approx(d1("r", 1e-4), rel=1e-7, abs=1e-8)


@given(spot, strike, mat, rate, vol, div)
def test_digital_call_plus_put_is_zero_coupon_bond(S, K, T, r, sigma, q):
    total = bs_digital_price(S, K, T, r, sigma, q) + bs_digital_price(
        S, K, T, r, sigma, q, call=False
    )
    assert total == pytest.approx(np.exp(-r * T), abs=1e-12)


def test_digital_is_minus_strike_derivative_of_call():
    # d C / d K = -e^{-rT} Q(S_T > K) = -digital price.
    S, K, T, r, sigma, h = 100, 105, 0.7, 0.03, 0.25, 1e-4
    dc_dk = (bs_price(S, K + h, T, r, sigma) - bs_price(S, K - h, T, r, sigma)) / (2 * h)
    assert -dc_dk == pytest.approx(bs_digital_price(S, K, T, r, sigma), abs=1e-7)


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        bs_price(100, 100, 1.0, 0.05, 0.0)


@pytest.mark.parametrize("call", [True, False])
def test_geometric_asian_single_fixing_is_european(call):
    g = geometric_asian_price(100, 95, 0.04, 0.3, [1.5], q=0.01, call=call)
    assert g == pytest.approx(float(bs_price(100, 95, 1.5, 0.04, 0.3, 0.01, call)), abs=1e-12)


def test_geometric_asian_matches_textbook_moments():
    # Equally spaced fixings: var = sigma^2 T (n+1)(2n+1) / (6 n^2).
    n, T, sigma = 12, 1.0, 0.2
    t = np.arange(1, n + 1) * T / n
    var_closed = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    var_code = sigma**2 * np.minimum.outer(t, t).sum() / n**2
    assert var_code == pytest.approx(var_closed, rel=1e-12)


def test_geometric_asian_cheaper_than_european_and_parity():
    t = np.linspace(0, 1, 13)[1:]
    c = geometric_asian_price(100, 100, 0.05, 0.2, t)
    p = geometric_asian_price(100, 100, 0.05, 0.2, t, call=False)
    assert 0 < c < float(bs_price(100, 100, 1, 0.05, 0.2))
    # Parity for the geometric average: C - P = e^{-rT}(E[G] - K).
    mu = np.log(100) + (0.05 - 0.02) * t.mean()
    var = 0.04 * np.minimum.outer(t, t).sum() / 144
    assert c - p == pytest.approx(np.exp(-0.05) * (np.exp(mu + var / 2) - 100), abs=1e-10)


def test_geometric_asian_rejects_bad_fixings():
    with pytest.raises(ValueError):
        geometric_asian_price(100, 100, 0.05, 0.2, [0.0, 1.0])


def test_barrier_limits():
    vanilla = float(bs_price(100, 100, 1, 0.05, 0.2))
    assert up_and_out_call_price(100, 100, 1e6, 1, 0.05, 0.2) == pytest.approx(vanilla, abs=1e-8)
    assert up_and_out_call_price(100, 100, 95, 1, 0.05, 0.2) == 0.0  # already knocked out
    assert up_and_out_call_price(100, 110, 105, 1, 0.05, 0.2) == 0.0  # B <= K
    mid = up_and_out_call_price(100, 100, 130, 1, 0.05, 0.2)
    assert 0 < mid < vanilla


def test_barrier_increasing_in_barrier_level():
    prices = [up_and_out_call_price(100, 100, b, 1, 0.05, 0.2) for b in (110, 120, 140, 200)]
    assert np.all(np.diff(prices) > 0)


def test_bgk_shift_direction():
    assert bgk_shifted_barrier(120, 0.2, 1 / 252) > 120
    assert bgk_shifted_barrier(80, 0.2, 1 / 252, up=False) < 80
