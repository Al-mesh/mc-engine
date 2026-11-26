"""Property-based no-arbitrage tests (hypothesis).

Most of these hold *pathwise*, i.e. exactly for every simulated path when the payoffs are
evaluated on common random numbers. Testing them pathwise is much stronger than testing
them in expectation: there is no statistical slack at all.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mc_engine import GBM, Heston, Merton
from mc_engine.payoffs import (
    ArithmeticAsian,
    Digital,
    European,
    FixedLookbackCall,
    FloatingLookbackCall,
    GeometricAsian,
    UpAndOutCall,
)

N_PATHS = 2000
SETTINGS = settings(max_examples=40, deadline=None)

gbm_models = st.builds(
    GBM,
    S0=st.floats(50, 150),
    r=st.floats(-0.02, 0.1),
    sigma=st.floats(0.05, 0.8),
    q=st.floats(0, 0.05),
)
heston_models = st.builds(
    Heston,
    S0=st.floats(50, 150),
    v0=st.floats(0.01, 0.2),
    kappa=st.floats(0.2, 5),
    theta=st.floats(0.01, 0.2),
    xi=st.floats(0.1, 1.2),
    rho=st.floats(-0.95, 0.5),
    r=st.floats(0, 0.08),
    scheme=st.sampled_from(["qe", "euler"]),
)
merton_models = st.builds(
    Merton,
    S0=st.floats(50, 150),
    r=st.floats(0, 0.08),
    sigma=st.floats(0.05, 0.5),
    lam=st.floats(0, 3),
    mu_j=st.floats(-0.3, 0.2),
    sigma_j=st.floats(0, 0.4),
)
any_model = st.one_of(gbm_models, heston_models, merton_models)
strikes = st.floats(40, 200)
seeds = st.integers(0, 2**32 - 1)


def _paths(model, seed, n_steps=12, T=1.0):
    times = np.linspace(0, T, n_steps + 1)
    return model.simulate(N_PATHS, times, rng=seed).paths, times


@SETTINGS
@given(any_model, strikes, seeds)
def test_put_call_parity_holds_pathwise(model, K, seed):
    """(S-K)^+ - (K-S)^+ = S - K on every path, so MC call - MC put = mean(S_T) - K."""
    paths, t = _paths(model, seed)
    c, p = European(K)(paths, t), European(K, call=False)(paths, t)
    np.testing.assert_allclose(c - p, paths[:, -1] - K, rtol=1e-12, atol=1e-9)


@SETTINGS
@given(any_model, strikes, st.floats(1, 50), seeds)
def test_call_decreasing_put_increasing_in_strike(model, K, dK, seed):
    paths, t = _paths(model, seed)
    assert np.all(European(K + dK)(paths, t) <= European(K)(paths, t))
    assert np.all(European(K + dK, call=False)(paths, t) >= European(K, call=False)(paths, t))


@SETTINGS
@given(gbm_models, strikes, st.floats(1.01, 1.5), seeds)
def test_gbm_call_increasing_in_spot_with_common_random_numbers(model, K, bump, seed):
    """Under GBM, S_T is linear in S0 for fixed shocks, so call value is monotone in S0."""
    import dataclasses

    up = dataclasses.replace(model, S0=model.S0 * bump)
    p0, t = _paths(model, seed)
    p1, _ = _paths(up, seed)
    assert np.all(European(K)(p1, t) >= European(K)(p0, t))
    assert np.all(UpAndOutCall(K, 1e9)(p1, t) >= UpAndOutCall(K, 1e9)(p0, t))


@SETTINGS
@given(any_model, strikes, st.floats(1.0, 2.0), seeds)
def test_barrier_never_exceeds_vanilla_and_is_monotone_in_barrier(model, K, mult, seed):
    paths, t = _paths(model, seed)
    B = model.S0 * mult
    uo = UpAndOutCall(K, B)(paths, t)
    uo_higher = UpAndOutCall(K, B * 1.1)(paths, t)
    vanilla = European(K)(paths, t)
    assert np.all(uo <= vanilla)
    assert np.all(uo <= uo_higher)


@SETTINGS
@given(any_model, strikes, seeds)
def test_geometric_asian_below_arithmetic_asian(model, K, seed):
    """AM-GM: the geometric mean never exceeds the arithmetic mean of positive numbers."""
    paths, t = _paths(model, seed)
    g, a = GeometricAsian(K)(paths, t), ArithmeticAsian(K)(paths, t)
    assert np.all(g <= a + 1e-9)
    gp, ap = GeometricAsian(K, call=False)(paths, t), ArithmeticAsian(K, call=False)(paths, t)
    assert np.all(gp >= ap - 1e-9)


@SETTINGS
@given(any_model, strikes, seeds)
def test_lookback_bounds(model, K, seed):
    paths, t = _paths(model, seed)
    floating = FloatingLookbackCall()(paths, t)
    assert np.all(floating >= 0)
    # Floating lookback >= ATM call: min_t S <= S0.
    assert np.all(floating >= European(model.S0)(paths, t) - 1e-9)
    # Fixed lookback >= European with same strike: max_t S >= S_T.
    assert np.all(FixedLookbackCall(K)(paths, t) >= European(K)(paths, t))


@SETTINGS
@given(any_model, strikes, st.floats(0.1, 100), seeds)
def test_digital_bounds_and_complement(model, K, payout, seed):
    paths, t = _paths(model, seed)
    dc = Digital(K, payout=payout)(paths, t)
    dp = Digital(K, call=False, payout=payout)(paths, t)
    assert np.all((dc == 0) | (dc == payout))
    # Complement up to the measure-zero event S_T == K.
    tie = paths[:, -1] == K
    np.testing.assert_allclose((dc + dp)[~tie], payout)


@SETTINGS
@given(any_model, seeds)
def test_simulated_paths_finite_positive(model, seed):
    paths, _ = _paths(model, seed)
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)
