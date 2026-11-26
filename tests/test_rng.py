import numpy as np
import pytest

from mc_engine.rng import PseudoRandomSource, RandomSource, as_random_source


def test_same_seed_reproduces_draws():
    a = PseudoRandomSource(42).normals(100, 5, 2)
    b = PseudoRandomSource(42).normals(100, 5, 2)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = PseudoRandomSource(1).normals(100, 5)
    b = PseudoRandomSource(2).normals(100, 5)
    assert not np.allclose(a, b)


def test_shapes_and_ranges():
    src = PseudoRandomSource(0)
    u = src.uniforms(1000, 3, 2)
    z = src.normals(1000, 3, 2)
    assert u.shape == z.shape == (1000, 3, 2)
    assert np.all((u > 0) & (u < 1))


def test_uniform_zero_is_nudged_inside_open_interval():
    class ZeroGen:
        def random(self, size):
            return np.zeros(size)

    src = PseudoRandomSource(0)
    src.generator = ZeroGen()
    assert np.all(src.uniforms(4, 1) > 0)


def test_moments_of_normals():
    z = PseudoRandomSource(7).normals(200_000, 1)[..., 0].ravel()
    # Mean ~ N(0, 1/n): 4 SE bound is 4/sqrt(2e5) ~ 0.009.
    assert abs(z.mean()) < 0.009
    assert abs(z.var() - 1) < 0.013  # Var of sample variance ~ 2/n


def test_default_normals_use_inverse_cdf():
    class FixedUniform(RandomSource):
        def uniforms(self, n_paths, n_steps, n_factors=1):
            return np.full((n_paths, n_steps, n_factors), 0.975)

    z = FixedUniform().normals(2, 1)
    np.testing.assert_allclose(z, 1.959963984540054)


@pytest.mark.parametrize("src", [PseudoRandomSource(20), None], ids=["direct", "inversion"])
def test_chisquare_moments(src):
    """Both paths (direct sampler, and inverse CDF via the base class) give chi2(df):
    mean df, variance 2 df."""
    if src is None:

        class Inv(RandomSource):
            def __init__(self):
                self.g = np.random.default_rng(21)

            def uniforms(self, n_paths, n_steps, n_factors=1):
                return self.g.random((n_paths, n_steps, n_factors))

        src = Inv()
    x = src.chisquare(5.0, 200_000, 1).ravel()
    assert x.mean() == pytest.approx(5.0, abs=4 * np.sqrt(10 / 2e5))
    assert x.var() == pytest.approx(10.0, rel=0.03)


def test_as_random_source_passthrough_and_generator():
    src = PseudoRandomSource(3)
    assert as_random_source(src) is src
    gen = np.random.default_rng(3)
    wrapped = as_random_source(gen)
    assert wrapped.generator is gen  # caller keeps control of the stream
    np.testing.assert_array_equal(
        as_random_source(5).normals(3, 1), PseudoRandomSource(5).normals(3, 1)
    )


def test_no_global_state_is_touched():
    # The legacy global API is used here deliberately, to prove we never consume from it.
    np.random.seed(123)  # noqa: NPY002
    before = np.random.get_state()[1].copy()  # noqa: NPY002
    PseudoRandomSource(0).normals(10, 10)
    after = np.random.get_state()[1]  # noqa: NPY002
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("seed", [None])
def test_none_seed_allowed(seed):
    assert PseudoRandomSource(seed).normals(1, 1).shape == (1, 1, 1)
