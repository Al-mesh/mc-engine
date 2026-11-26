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


SeedLike: TypeAlias = int | np.random.SeedSequence | np.random.Generator | RandomSource | None


def as_random_source(seed: SeedLike) -> RandomSource:
    """Coerce a seed / ``Generator`` / ``RandomSource`` into a ``RandomSource``.

    An existing ``RandomSource`` is returned unchanged (not copied), so a pricer that
    simulates in batches keeps consuming one continuous stream.
    """
    if isinstance(seed, RandomSource):
        return seed
    return PseudoRandomSource(seed)
