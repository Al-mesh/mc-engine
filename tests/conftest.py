"""Shared helpers for statistical tests.

Threshold choice: statistical comparisons use |MC - ref| < 4 SE. Under the null that
the estimator is unbiased this fails with probability ~6e-5 per test, so across the
whole suite (a few dozen such checks) a false alarm is ~0.1% likely. All seeds are fixed,
so CI is deterministic; the threshold matters only for choosing seeds honestly (seeds are
never cherry-picked: they are 0, 1, 2, ... in order of appearance).
"""

from __future__ import annotations

from mc_engine.pricer import PriceResult

N_SE = 4.0


def assert_close_to_reference(result: PriceResult, reference: float, n_se: float = N_SE) -> None:
    z = (result.price - reference) / result.std_error
    assert abs(z) < n_se, (
        f"MC {result.price:.6f} vs reference {reference:.6f}: "
        f"{z:+.2f} standard errors (se={result.std_error:.2e})"
    )
