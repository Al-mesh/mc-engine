"""Multi-process Monte Carlo: independent streams on separate cores.

Monte Carlo is embarrassingly parallel, *provided the streams are independent*. Seeding
worker k with ``seed + k`` is a classic bug: nothing guarantees those streams don't
overlap. :class:`numpy.random.SeedSequence` ``.spawn`` derives children by hashing, with
statistical independence guarantees, which is what we use.

Reproducibility design: the work is cut into ``n_chunks`` pieces *before* scheduling, and
chunk i always gets child seed i and the same number of paths. Results are merged in
chunk order. Hence the answer depends on (seed, n_chunks, n_paths) but **not** on
``n_workers``: a 1-process debugging run and a 16-process production run agree to the
last bit, which makes parallel results testable.

When it pays off: each process must import numpy/scipy and unpickle the model (~0.5-1 s
on Windows, which spawns rather than forks), so it only helps for runs of seconds or more.
Do not combine with ``backend="numba"`` Heston, which is already multi-threaded: two
levels of parallelism oversubscribe the cores.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mc_engine.models.base import Model, validate_n_paths
from mc_engine.payoffs import Payoff
from mc_engine.pricer import PriceResult, RunningMoments, mc_price


def _run_chunk(args) -> tuple[int, float, float]:
    model, payoff, maturity, n_steps, n_paths, seed_seq, batch_size = args
    res = mc_price(model, payoff, maturity, n_steps, n_paths, rng=seed_seq, batch_size=batch_size)
    # Return sufficient statistics (n, mean, M2) so chunks merge exactly.
    m2 = res.std_error**2 * n_paths * (n_paths - 1)
    return n_paths, res.price, m2


def parallel_mc_price(
    model: Model,
    payoff: Payoff,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.SeedSequence | None = None,
    n_chunks: int = 16,
    n_workers: int | None = None,
    batch_size: int = 65_536,
) -> PriceResult:
    """Plain MC price computed over ``n_chunks`` independent streams in worker processes.

    ``n_workers=1`` runs the chunks in-process (no pool), which is useful for debugging
    and gives the identical result.
    """
    n_paths = validate_n_paths(n_paths)
    if n_chunks < 1 or n_chunks > n_paths:
        raise ValueError("need 1 <= n_chunks <= n_paths")
    sizes = np.full(n_chunks, n_paths // n_chunks)
    sizes[: n_paths % n_chunks] += 1
    children = np.random.SeedSequence(seed).spawn(n_chunks)
    tasks = [
        (model, payoff, maturity, n_steps, int(m), child, batch_size)
        for m, child in zip(sizes, children, strict=True)
    ]
    n_workers = min(n_workers or os.cpu_count() or 1, n_chunks)

    start = time.perf_counter()
    if n_workers == 1:
        parts = [_run_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            parts = list(pool.map(_run_chunk, tasks))  # map preserves chunk order
    elapsed = time.perf_counter() - start

    stats = RunningMoments()
    for n, mean, m2 in parts:
        stats.merge(n, mean, m2)
    se = float(np.sqrt(stats.variance / stats.n))
    return PriceResult(price=stats.mean, std_error=se, n_paths=n_paths, runtime_s=elapsed)
