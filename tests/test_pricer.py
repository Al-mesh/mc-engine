import numpy as np
import pytest

from mc_engine import GBM, mc_price
from mc_engine.payoffs import European
from mc_engine.pricer import Z_95, RunningMoments, discounted_payoffs, time_grid


@pytest.mark.parametrize("chunks", [[1000], [1, 999], [300, 300, 400], [7] * 50])
def test_running_moments_match_numpy(chunks):
    x = np.random.default_rng(0).lognormal(3, 1.5, sum(chunks))  # heavy-ish tail
    rm = RunningMoments()
    i = 0
    for c in chunks:
        rm.update(x[i : i + c])
        i += c
    rm.update(np.array([]))  # empty batch is a no-op
    assert rm.n == x.size
    assert rm.mean == pytest.approx(x.mean(), rel=1e-12)
    assert rm.variance == pytest.approx(x.var(ddof=1), rel=1e-10)


def test_running_moments_single_observation_has_undefined_variance():
    rm = RunningMoments()
    rm.update(np.array([1.0]))
    assert np.isnan(rm.variance)


def test_price_independent_of_batch_size_for_gbm():
    """One normals() call per batch and C-order filling => identical streams."""
    m, p = GBM(100, 0.05, 0.2), European(100)
    a = mc_price(m, p, 1.0, 4, 10_000, rng=1, batch_size=10_000)
    b = mc_price(m, p, 1.0, 4, 10_000, rng=1, batch_size=1_234)
    assert a.price == pytest.approx(b.price, rel=1e-12)
    assert a.std_error == pytest.approx(b.std_error, rel=1e-9)


def test_result_fields_and_ci():
    r = mc_price(GBM(100, 0.05, 0.2), European(100), 1.0, 1, 5000, rng=2)
    lo, hi = r.ci95
    assert lo < r.price < hi
    assert hi - lo == pytest.approx(2 * Z_95 * r.std_error)
    assert r.n_paths == 5000 and r.runtime_s >= 0
    assert "95% CI" in str(r)


def test_se_matches_sample_std():
    m, p = GBM(100, 0.05, 0.2), European(100)
    x = discounted_payoffs(m, p, 1.0, 1, 20_000, rng=3)
    r = mc_price(m, p, 1.0, 1, 20_000, rng=3)
    assert r.price == pytest.approx(x.mean(), rel=1e-12)
    assert r.std_error == pytest.approx(x.std(ddof=1) / np.sqrt(x.size), rel=1e-9)


def test_single_path_se_is_nan():
    r = mc_price(GBM(100, 0.05, 0.2), European(100), 1.0, 1, 1, rng=0)
    assert np.isnan(r.std_error)


@pytest.mark.parametrize("T,n", [(0.0, 1), (1.0, 0), (1.0, 2.5)])
def test_time_grid_validation(T, n):
    with pytest.raises(ValueError):
        time_grid(T, n)
