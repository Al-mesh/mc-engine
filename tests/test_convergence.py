"""Empirical convergence rates: MC ~ N^-1/2, RQMC faster.

The target is a geometric Asian with 16 fixings: genuinely 16-dimensional, with an exact
closed form, so RMSE (bias + variance) can be measured rather than estimated.
"""

import numpy as np
import pytest

from mc_engine import GBM
from mc_engine.analytic import geometric_asian_price
from mc_engine.payoffs import GeometricAsian
from mc_engine.pricer import time_grid
from mc_engine.variance_reduction import convergence_study, loglog_slope

pytestmark = pytest.mark.statistical

MODEL = GBM(S0=100, r=0.05, sigma=0.2)
N_STEPS = 16
EXACT = geometric_asian_price(100, 100, 0.05, 0.2, time_grid(1.0, N_STEPS)[1:])
NS = [2**k for k in range(8, 15)]


def _slope(method: str) -> float:
    df = convergence_study(
        MODEL, GeometricAsian(100), 1.0, N_STEPS, EXACT, NS, n_reps=64, method=method, seed=1
    )
    return loglog_slope(df.n_paths.to_numpy(), df.rmse.to_numpy())


def test_mc_rate_is_one_half():
    # 64 replications give log-RMSE noise ~ 1/sqrt(128) ~ 9%, so the slope estimate has
    # SE ~ 0.025 over this range of N; [-0.6, -0.4] is a ~4 SE band around the CLT rate.
    assert -0.6 < _slope("mc") < -0.4


def test_rqmc_with_brownian_bridge_beats_root_n():
    # Theory: O(N^-1 (log N)^d) for smooth integrands of bounded variation; the call kink
    # breaks that, so empirically ~ -0.8 to -0.9 is expected rather than a clean -1.
    assert _slope("qmc_bb") < -0.7


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        convergence_study(MODEL, GeometricAsian(100), 1.0, 2, 1.0, [4], 2, method="lhs")


def test_loglog_slope_exact_power_law():
    n = np.array([10, 100, 1000])
    assert loglog_slope(n, 3.0 * n**-0.5) == pytest.approx(-0.5)
