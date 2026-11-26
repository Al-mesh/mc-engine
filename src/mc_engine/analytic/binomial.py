r"""Cox-Ross-Rubinstein binomial tree: the benchmark for early-exercise prices.

Over a step :math:`\Delta = T/n` the stock moves up by :math:`u = e^{\sigma\sqrt\Delta}` or down
by :math:`d = 1/u`, with risk-neutral up-probability
:math:`p = (e^{(r-q)\Delta} - d)/(u - d)`, chosen so the one-step mean is right. The variance
matches to :math:`O(\Delta^2)`, so the tree converges to Black-Scholes as :math:`n \to \infty`.
Backward induction is

.. math::
    V_i(j) = \max\big(
        \underbrace{e^{-r\Delta}[pV_{i+1}(j+1) + (1-p)V_{i+1}(j)]}_{\text{continuation}},
    \ \underbrace{h(S_i(j))}_{\text{exercise}}\big)

at dates where exercise is allowed, and just the continuation value elsewhere.

Convergence is O(1/n) but **oscillates** with the parity of n, because the strike falls
at different positions relative to the terminal nodes. Averaging the prices for n and
n+1 cancels most of the oscillation, which is what :func:`crr_price` does by default for
European and American exercise. For Bermudan exercise the step count must be a multiple
of the number of exercise dates, so the average uses n and n + n_exercise_dates instead.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def _crr_single(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_steps: int,
    call: bool,
    exercise_every: int | None,
) -> float:
    """One tree. ``exercise_every``: None = European, k = exercise allowed every k steps."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("CRR probability outside (0, 1): increase n_steps")
    disc = np.exp(-r * dt)
    sign = 1.0 if call else -1.0

    j = np.arange(n_steps + 1)
    s = S0 * u ** (2 * j - n_steps)  # terminal nodes, j = number of up moves
    v = np.maximum(sign * (s - K), 0.0)
    for i in range(n_steps - 1, -1, -1):
        v = disc * (p * v[1:] + (1.0 - p) * v[:-1])
        if exercise_every is not None and i % exercise_every == 0:
            s_i = S0 * u ** (2 * np.arange(i + 1) - i)
            np.maximum(v, sign * (s_i - K), out=v)
    return float(v[0])


def crr_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_steps: int = 2000,
    call: bool = False,
    exercise: Literal["european", "american", "bermudan"] = "american",
    n_exercise_dates: int | None = None,
    smooth: bool = True,
) -> float:
    """CRR price of a European, American or Bermudan option (put by default).

    Bermudan exercise is allowed at ``t_k = kT/M`` for ``k = 0..M`` with
    ``M = n_exercise_dates``, which is the same schedule Longstaff-Schwartz uses, so the two
    can be compared like for like, separating the LSM algorithm's error from the
    (economic) Bermudan-vs-American gap.
    """
    if exercise == "european":
        every = None
        step = 1
    elif exercise == "american":
        every = 1
        step = 1
    elif exercise == "bermudan":
        if not n_exercise_dates or n_steps % n_exercise_dates:
            raise ValueError("bermudan needs n_steps to be a multiple of n_exercise_dates")
        every = n_steps // n_exercise_dates
        step = n_exercise_dates
    else:
        raise ValueError(f"unknown exercise style {exercise!r}")

    price = _crr_single(S0, K, T, r, sigma, q, n_steps, call, every)
    if not smooth:
        return price
    n2 = n_steps + step
    every2 = None if every is None else (1 if exercise == "american" else n2 // n_exercise_dates)
    return 0.5 * (price + _crr_single(S0, K, T, r, sigma, q, n2, call, every2))
