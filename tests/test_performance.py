"""Performance code must be *exactly* as correct as the reference code it replaces."""

import dataclasses

import numpy as np
import pytest

from mc_engine import GBM, Heston
from mc_engine._numba_kernels import NUMBA_AVAILABLE
from mc_engine.analytic import bs_price
from mc_engine.parallel import parallel_mc_price
from mc_engine.payoffs import European
from mc_engine.pricer import RunningMoments
from tests.conftest import assert_close_to_reference

needs_numba = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
HARSH = Heston(S0=100, v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, r=0.02, q=0.01)


@needs_numba
@pytest.mark.parametrize("scheme", ["qe", "euler"])
@pytest.mark.parametrize("correction", [True, False])
def test_numba_kernel_matches_numpy_reference(scheme, correction):
    """Same random numbers in, same paths out, to round-off. Uses a Feller-violating case
    (both QE branches exercised) on an irregular grid."""
    base = dataclasses.replace(HARSH, scheme=scheme, martingale_correction=correction)
    fast = dataclasses.replace(base, backend="numba")
    times = np.array([0.0, 0.01, 0.05, 0.3, 0.31, 1.0, 2.5])
    a = base.simulate(20_000, times, rng=1)
    b = fast.simulate(20_000, times, rng=1)
    np.testing.assert_allclose(b.paths, a.paths, rtol=1e-12)
    np.testing.assert_allclose(b.variance, a.variance, rtol=1e-12, atol=1e-15)
    if scheme == "qe":
        # Both QE branches must actually have been exercised for the test to mean much.
        assert np.mean(a.variance[:, -1] == 0.0) > 0.01


@needs_numba
def test_numba_inverse_normal_matches_scipy_to_machine_precision():
    """Acklam + one Halley step vs scipy.special.ndtri, across the body and both tails,
    including the region boundaries (0.02425) and extreme uniforms."""
    from scipy.special import ndtri

    from mc_engine._numba_kernels import norm_ppf

    p = np.concatenate(
        [
            np.logspace(-300, -1, 400),
            np.linspace(0.001, 0.999, 2001),
            [0.02425, 0.02425 + 1e-12, 1 - 0.02425, 0.5],
            1 - np.logspace(-15, -1, 200),
        ]
    )
    ours = np.array([norm_ppf(x) for x in p])
    np.testing.assert_allclose(ours, ndtri(p), rtol=2e-14, atol=2e-15)


@needs_numba
def test_uncompiled_kernel_source_agrees_with_compiled():
    """numba's .py_func runs the kernel as plain Python (prange degrades to range). This
    checks the compiled code against its own source, and gives coverage a view of it."""
    from mc_engine import _numba_kernels as k

    rng = np.random.default_rng(3)
    u = rng.random((40, 6, 2))
    z = rng.standard_normal((40, 6, 2))
    dt = np.full(6, 0.25)
    qe_args = (np.log(100.0), 0.04, 0.5, 0.04, 1.0, -0.9, 0.01, 1.5, True, dt, u)
    eu_args = (np.log(100.0), 0.04, 0.5, 0.04, 1.0, -0.9, 0.01, dt, z)
    for kernel, args in ((k.heston_qe_kernel, qe_args), (k.heston_euler_kernel, eu_args)):
        for compiled, python in zip(kernel(*args), kernel.py_func(*args), strict=True):
            np.testing.assert_allclose(compiled, python, rtol=1e-13)
    assert k.norm_ppf.py_func(0.975) == pytest.approx(1.959963984540054, rel=1e-15)
    from scipy.special import ndtri

    for p in (1e-200, 1e-10, 0.01, 0.3, 0.5):  # both Acklam regions, uncompiled
        assert k.ppf_lower.py_func(p) == pytest.approx(ndtri(p), rel=1e-14, abs=1e-15)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        dataclasses.replace(HARSH, backend="cuda")


def test_running_moments_merge_equals_update():
    x = np.random.default_rng(2).normal(size=1000)
    a, b = RunningMoments(), RunningMoments()
    a.update(x)
    b.merge(600, x[:600].mean(), ((x[:600] - x[:600].mean()) ** 2).sum())
    b.merge(400, x[600:].mean(), ((x[600:] - x[600:].mean()) ** 2).sum())
    b.merge(0, 0.0, 0.0)
    assert b.n == a.n and b.mean == pytest.approx(a.mean) and b.m2 == pytest.approx(a.m2)


def test_parallel_result_does_not_depend_on_worker_count():
    """Chunking is fixed before scheduling, so 1 process and 2 processes agree exactly."""
    g, p = GBM(100, 0.05, 0.2), European(100)
    one = parallel_mc_price(g, p, 1.0, 1, 40_000, seed=3, n_chunks=4, n_workers=1)
    two = parallel_mc_price(g, p, 1.0, 1, 40_000, seed=3, n_chunks=4, n_workers=2)
    assert one.price == two.price and one.std_error == two.std_error
    other_seed = parallel_mc_price(g, p, 1.0, 1, 40_000, seed=4, n_chunks=4, n_workers=1)
    assert other_seed.price != one.price


@pytest.mark.statistical
def test_parallel_estimator_is_unbiased_with_correct_se():
    g = GBM(100, 0.05, 0.2)
    res = parallel_mc_price(g, European(100), 1.0, 1, 400_001, seed=5, n_chunks=7, n_workers=1)
    assert res.n_paths == 400_001  # uneven split handled
    assert_close_to_reference(res, float(bs_price(100, 100, 1.0, 0.05, 0.2)))
    assert res.std_error == pytest.approx(14.7 / np.sqrt(400_001), rel=0.05)


def test_parallel_argument_checks():
    with pytest.raises(ValueError):
        parallel_mc_price(GBM(100, 0.05, 0.2), European(100), 1.0, 1, 10, n_chunks=20)
