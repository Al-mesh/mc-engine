r"""Brownian bridge path construction, used to make quasi-Monte Carlo effective.

The standard ("forward") construction builds :math:`W` step by step, so every Sobol
coordinate is equally important. The bridge instead uses the first normal for the
terminal value, :math:`W_T = \sqrt T\,Z_1`, then fills in midpoints conditionally. For
:math:`l < i < r` with :math:`W_l, W_r` known,

.. math::
    W_i \mid W_l, W_r \sim N\!\Big(\frac{(t_r - t_i)W_l + (t_i - t_l)W_r}{t_r - t_l},\
    \frac{(t_i - t_l)(t_r - t_i)}{t_r - t_l}\Big).

The joint law of the path is unchanged (it is just a different square root of the
Brownian covariance matrix), so plain MC results are statistically identical. What
changes is *which inputs matter*: the variance explained by the first few coordinates is
much larger (for an Asian option most of it sits in :math:`W_T` and :math:`W_{T/2}`), so
the "effective dimension" falls, and that is where Sobol points are best distributed.

Here the bridge is applied on an equally spaced grid and returns standardised
increments :math:`(W_{j+1} - W_j)/\sqrt{\Delta}`, which are i.i.d. N(0,1) in law. Models
therefore consume them exactly as they would i.i.d. normals.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from mc_engine.rng import FloatArray, RandomSource


@lru_cache(maxsize=64)
def bridge_schedule(n_steps: int) -> tuple[tuple[int, int, int], ...]:
    """Coarse-to-fine construction order ``(i, left, right)`` on integer times ``0..n_steps``.

    The first entry is ``(n_steps, 0, -1)``: the terminal point, built from W_0 = 0 alone.
    Breadth-first bisection then fills midpoints level by level.
    """
    order: list[tuple[int, int, int]] = [(n_steps, 0, -1)]
    queue = deque([(0, n_steps)])
    while queue:
        left, right = queue.popleft()
        if right - left < 2:
            continue
        mid = (left + right) // 2
        order.append((mid, left, right))
        queue.append((left, mid))
        queue.append((mid, right))
    return tuple(order)


def brownian_bridge_increments(z: FloatArray) -> FloatArray:
    """Map normals ``z`` of shape ``(n_paths, n_steps, n_factors)`` (in bridge order) to
    standardised Brownian increments of the same shape, on a unit-spaced grid."""
    n_paths, n_steps, n_factors = z.shape
    w = np.zeros((n_paths, n_steps + 1, n_factors))
    for c, (i, left, right) in enumerate(bridge_schedule(n_steps)):
        if right < 0:  # terminal point: W_m ~ N(0, m)
            w[:, i] = np.sqrt(n_steps) * z[:, c]
            continue
        span = right - left
        mean = ((right - i) * w[:, left] + (i - left) * w[:, right]) / span
        w[:, i] = mean + np.sqrt((i - left) * (right - i) / span) * z[:, c]
    return np.diff(w, axis=1)


class BrownianBridgeSource(RandomSource):
    """Wraps a source so that ``normals()`` come out of a Brownian bridge.

    Only Gaussian drivers are re-ordered. ``uniforms()`` pass through unchanged, because a
    uniform consumed by an inverse-CDF (QE variance step, Poisson count) is not a Brownian
    increment and has no bridge interpretation. The bridge therefore benefits GBM and
    Euler-type schemes, not QE or Merton's jump counts.
    """

    def __init__(self, base: RandomSource):
        self.base = base

    def uniforms(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        return self.base.uniforms(n_paths, n_steps, n_factors)

    def normals(self, n_paths: int, n_steps: int, n_factors: int = 1) -> FloatArray:
        return brownian_bridge_increments(self.base.normals(n_paths, n_steps, n_factors))
