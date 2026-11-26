"""Random number sources.

Every model draws *all* of its randomness through a :class:`RandomSource`. Nothing
in the package touches numpy's legacy global state (``np.random.seed`` etc.).

Why a separate abstraction instead of passing a ``np.random.Generator`` around?
Because several techniques are just different ways of producing the same block of
standard normals / uniforms, and the model code should not care which one it gets:

* plain Monte Carlo            -> i.i.d. pseudo-random draws (this module)
* antithetic variates          -> the block and its reflection Z -> -Z, U -> 1-U
* common random numbers (CRN)  -> replay the same seed for bumped revaluations
* randomised quasi-Monte Carlo -> scrambled Sobol points mapped through Phi^{-1}

Shapes are always ``(n_paths, n_steps, n_factors)`` so that a model with, say, two
Brownian drivers per step asks for ``n_factors=2`` and gets one coherent block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc

FloatArray: TypeAlias = NDArray[np.float64]


class RandomSource(ABC):
    """Supplier of uniform and standard-normal blocks of shape ``(n_paths, n_steps, n_factors)``."""

    @abstractmethod
    def uniforms(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        """Return draws that are (marginally) U(0, 1), strictly inside the open interval."""

    def normals(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        """Return N(0, 1) draws. Default: inverse-CDF transform ``Z = Phi^{-1}(U)``.

        Inversion (rather than Box-Muller or ziggurat) is the only transform that maps
        one uniform to one normal monotonically, which is what QMC and antithetic
        sampling need. Pseudo-random sources override this with a faster sampler.
        """
        return ndtri(self.uniforms(n_paths, n_steps, n_factors))


class PseudoRandomSource(RandomSource):
    """I.i.d. draws from a seeded :class:`numpy.random.Generator` (PCG64 by default).

    Parameters
    ----------
    seed
        Anything accepted by :func:`numpy.random.default_rng`: an int, a
        :class:`~numpy.random.SeedSequence`, or an existing ``Generator`` (used as-is,
        so the caller keeps control of its stream). ``None`` draws OS entropy, which
        is non-reproducible; tests and benchmarks always pass an explicit seed.
    """

    def __init__(self, seed: int | np.random.SeedSequence | np.random.Generator | None = None):
        self.generator = np.random.default_rng(seed)

    def uniforms(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        u = self.generator.random((n_paths, n_steps, n_factors))
        # Generator.random samples [0, 1). An exact 0 (probability 2^-53 per draw)
        # would map to -inf under Phi^{-1} or log(); nudge it to the smallest
        # positive double instead of resampling so the stream stays aligned.
        return np.where(u == 0.0, np.finfo(np.float64).tiny, u)

    def normals(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        # Ziggurat sampler: roughly 2-3x faster than inverting uniforms, and the
        # monotone-map property is irrelevant for i.i.d. pseudo-random draws.
        return self.generator.standard_normal((n_paths, n_steps, n_factors))


class AntitheticSource(RandomSource):
    r"""Antithetic variates: returns the block ``[X; reflect(X)]`` stacked along the path axis.

    For ``n_paths = 2m`` it draws ``m`` rows from ``base`` and appends their reflections
    (:math:`Z \mapsto -Z`, :math:`U \mapsto 1 - U`), so path ``i`` and path ``i + m`` form an
    antithetic pair. Both maps preserve the marginal law, so each half is a valid sample.

    Why it helps: for a pair :math:`(Y, Y')`,
    :math:`\mathrm{Var}(\tfrac{Y + Y'}2) = \tfrac{\sigma^2}{2}(1 + \rho_{YY'})`. Compared with
    two independent draws (variance :math:`\sigma^2/2`) this is a gain iff
    :math:`\rho_{YY'} < 0`, which is guaranteed when the payoff is a monotone function of
    the driving noise (e.g. a call on GBM). For a payoff that is *even* in the noise, such as
    a straddle-like :math:`(\ln S_T - m)^2`, :math:`\rho = 1` and antithetics double the
    variance per unit of cost.

    The pricer must average each pair *before* estimating the variance, since the two
    halves are dependent; see :func:`mc_engine.pricer.mc_price`.
    """

    def __init__(self, base: RandomSource):
        self.base = base

    @staticmethod
    def _half(n_paths: int) -> int:
        if n_paths % 2:
            raise ValueError("antithetic sampling needs an even number of paths")
        return n_paths // 2

    def uniforms(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        u = self.base.uniforms(self._half(n_paths), n_steps, n_factors)
        return np.concatenate([u, 1.0 - u], axis=0)

    def normals(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        z = self.base.normals(self._half(n_paths), n_steps, n_factors)
        return np.concatenate([z, -z], axis=0)


class SobolSource(RandomSource):
    r"""Randomised quasi-Monte Carlo: scrambled Sobol points (Owen-type LMS + digital shift).

    One path consumes one point of dimension ``d = n_steps * n_factors``. The point's
    coordinates are laid out step-major, i.e. coordinate ``j * n_factors + f`` drives step
    ``j`` of factor ``f``. Sobol coordinates are best equidistributed in the *first*
    dimensions, so the most important randomness should come first; combined with the
    Brownian bridge (which puts :math:`W_T` on step 0) that is the terminal value.

    Successive calls continue the same sequence, so batching is fine as long as every
    batch size is a power of two (the (t, m, s)-net balance property only holds for
    :math:`2^k` consecutive points). Every call must request the same dimension: a model
    must draw all of its randomness in a single call per simulate.

    A single scrambled sequence gives an unbiased estimate but *no* usable error estimate,
    because the points are not independent. Independent scramblings (different ``seed``)
    give i.i.d. replicates; see :func:`mc_engine.variance_reduction.rqmc_price`.
    """

    def __init__(self, seed: int | np.random.SeedSequence | np.random.Generator | None = None):
        self._seed = seed
        self._engine: qmc.Sobol | None = None
        self._dim: int | None = None

    def uniforms(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        d = n_steps * n_factors
        if self._engine is None:
            self._engine = qmc.Sobol(d, scramble=True, rng=self._seed)
            self._dim = d
        elif d != self._dim:
            raise ValueError(
                f"SobolSource was initialised with dimension {self._dim}, got a request for "
                f"{d}; models must draw all randomness in one call per simulate"
            )
        if n_paths & (n_paths - 1):
            raise ValueError("Sobol batches must be a power of two to keep the net property")
        u = self._engine.random(n_paths)
        # Scrambled points are a.s. in (0, 1), but guard the endpoints for Phi^{-1} anyway.
        u = np.clip(u, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).epsneg)
        return u.reshape(n_paths, n_steps, n_factors)


SeedLike: TypeAlias = int | np.random.SeedSequence | np.random.Generator | RandomSource | None


def as_random_source(seed: SeedLike) -> RandomSource:
    """Coerce a seed / ``Generator`` / ``RandomSource`` into a ``RandomSource``.

    An existing ``RandomSource`` is returned unchanged (not copied), so a pricer that
    simulates in batches keeps consuming one continuous stream.
    """
    if isinstance(seed, RandomSource):
        return seed
    return PseudoRandomSource(seed)
