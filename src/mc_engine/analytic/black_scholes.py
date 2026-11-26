r"""Closed-form Black-Scholes(-Merton) results used as validation benchmarks.

Model: under the risk-neutral measure, :math:`dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t`,
so :math:`\ln S_T \sim N(\ln S_0 + (r - q - \tfrac12\sigma^2)T,\ \sigma^2 T)`.

With :math:`d_{1,2} = [\ln(S/K) + (r - q \pm \tfrac12\sigma^2)T] / (\sigma\sqrt T)`:

.. math::
    C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
    P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).

All functions broadcast over numpy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr
from scipy.stats import norm

# Broadie-Glasserman-Kou (1997) constant: beta = -zeta(1/2) / sqrt(2*pi).
BGK_BETA = 0.5825971579390106


def _d1_d2(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("T and sigma must be strictly positive")
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (np.asarray(r) - np.asarray(q) + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: bool = True,
) -> np.ndarray:
    """Black-Scholes price of a European call (``call=True``) or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-np.asarray(r) * T), np.exp(-np.asarray(q) * T)
    if call:
        return S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    return K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)


def bs_delta(S, K, T, r, sigma, q=0.0, call: bool = True) -> np.ndarray:
    r"""Delta: :math:`e^{-qT}N(d_1)` (call), :math:`-e^{-qT}N(-d_1)` (put)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    df_q = np.exp(-np.asarray(q) * T)
    return df_q * ndtr(d1) if call else -df_q * ndtr(-d1)


def bs_gamma(S, K, T, r, sigma, q=0.0) -> np.ndarray:
    r"""Gamma :math:`e^{-qT}\varphi(d_1)/(S\sigma\sqrt T)` (same for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-np.asarray(q) * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0) -> np.ndarray:
    r"""Vega :math:`S e^{-qT}\varphi(d_1)\sqrt T` per unit of sigma (not per 1%)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-np.asarray(q) * T) * norm.pdf(d1) * np.sqrt(T)


def bs_rho(S, K, T, r, sigma, q=0.0, call: bool = True) -> np.ndarray:
    r"""Rho :math:`KTe^{-rT}N(d_2)` (call), :math:`-KTe^{-rT}N(-d_2)` (put), per unit of r."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc = K * T * np.exp(-np.asarray(r) * T)
    return disc * ndtr(d2) if call else -disc * ndtr(-d2)


def bs_digital_price(
    S, K, T, r, sigma, q=0.0, call: bool = True, payout: float = 1.0
) -> np.ndarray:
    r"""Cash-or-nothing digital: pays ``payout`` if :math:`S_T > K` (call) or :math:`S_T < K`.

    Price :math:`= \text{payout}\cdot e^{-rT} N(\pm d_2)`, since :math:`N(d_2)` is the
    risk-neutral probability :math:`Q(S_T > K)`.
    """
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    return payout * np.exp(-np.asarray(r) * T) * ndtr(d2 if call else -d2)


def geometric_asian_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    fixing_times: ArrayLike,
    q: float = 0.0,
    call: bool = True,
) -> float:
    r"""Discretely monitored geometric-average Asian option (fixed strike).

    Payoff :math:`(G - K)^+` with :math:`G = (\prod_{i=1}^n S_{t_i})^{1/n}`, paid at
    :math:`T = t_n`. Because :math:`\ln G = \frac1n\sum_i \ln S_{t_i}` is a linear
    functional of Brownian motion, it is exactly Gaussian with

    .. math::
        \mu_G = \ln S_0 + (r - q - \tfrac12\sigma^2)\,\bar t, \qquad
        \sigma_G^2 = \frac{\sigma^2}{n^2}\sum_{i,j}\min(t_i, t_j),

    where :math:`\bar t = \frac1n \sum_i t_i`. Then
    :math:`E[(G-K)^+] = e^{\mu_G + \sigma_G^2/2}N(d_1) - K N(d_2)` with
    :math:`d_1 = (\mu_G - \ln K + \sigma_G^2)/\sigma_G`, :math:`d_2 = d_1 - \sigma_G`.

    This is the control variate for the arithmetic Asian in Phase 2: the two payoffs
    are almost perfectly correlated and this one has a known mean.
    """
    t = np.asarray(fixing_times, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(t <= 0):
        raise ValueError("fixing_times must be a non-empty 1-D array of positive times")
    T = t[-1]
    n = t.size
    mu_g = np.log(S0) + (r - q - 0.5 * sigma**2) * t.mean()
    var_g = sigma**2 * np.minimum.outer(t, t).sum() / n**2
    sd_g = np.sqrt(var_g)
    d1 = (mu_g - np.log(K) + var_g) / sd_g
    d2 = d1 - sd_g
    fwd_g = np.exp(mu_g + 0.5 * var_g)  # E[G]
    if call:
        undiscounted = fwd_g * ndtr(d1) - K * ndtr(d2)
    else:
        undiscounted = K * ndtr(-d2) - fwd_g * ndtr(-d1)
    return float(np.exp(-r * T) * undiscounted)


def up_and_out_call_price(
    S0: float, K: float, B: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    r"""Continuously monitored up-and-out call (Merton 1973; Reiner-Rubinstein 1991).

    For :math:`B > K`, :math:`C_{uo} = C_{BS} - C_{ui}`, with (Hull, *OFOD*, ch. 26)

    .. math::
        C_{ui} = S_0e^{-qT}N(x_1) - Ke^{-rT}N(x_1 - \sigma\sqrt T)
          - S_0e^{-qT}(B/S_0)^{2\lambda}[N(-y) - N(-y_1)]
          + Ke^{-rT}(B/S_0)^{2\lambda-2}[N(-y+\sigma\sqrt T) - N(-y_1+\sigma\sqrt T)],

    :math:`\lambda = (r - q + \sigma^2/2)/\sigma^2`,
    :math:`x_1 = \ln(S_0/B)/(\sigma\sqrt T) + \lambda\sigma\sqrt T`,
    :math:`y = \ln(B^2/(S_0K))/(\sigma\sqrt T) + \lambda\sigma\sqrt T`,
    :math:`y_1 = \ln(B/S_0)/(\sigma\sqrt T) + \lambda\sigma\sqrt T`.

    If :math:`B \le K` the option is worthless (finishing above K requires crossing B),
    and if :math:`S_0 \ge B` it is knocked out at inception.
    """
    if S0 >= B or B <= K:
        return 0.0
    vol = sigma * np.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    x1 = np.log(S0 / B) / vol + lam * vol
    y = np.log(B**2 / (S0 * K)) / vol + lam * vol
    y1 = np.log(B / S0) / vol + lam * vol
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    c_ui = (
        S0 * df_q * ndtr(x1)
        - K * df_r * ndtr(x1 - vol)
        - S0 * df_q * (B / S0) ** (2 * lam) * (ndtr(-y) - ndtr(-y1))
        + K * df_r * (B / S0) ** (2 * lam - 2) * (ndtr(-y + vol) - ndtr(-y1 + vol))
    )
    c = bs_price(S0, K, T, r, sigma, q, call=True)
    # Clip tiny negative values produced by cancellation when the option is nearly worthless.
    return float(max(c - c_ui, 0.0))


def bgk_shifted_barrier(B: float, sigma: float, dt: float, up: bool = True) -> float:
    r"""Broadie-Glasserman-Kou continuity correction.

    A barrier monitored discretely every :math:`\Delta t` behaves, to
    :math:`o(\sqrt{\Delta t})`, like a continuous barrier shifted *away* from the spot:
    :math:`B e^{\pm\beta\sigma\sqrt{\Delta t}}` with :math:`\beta = -\zeta(1/2)/\sqrt{2\pi}
    \approx 0.5826` (+ for up barriers). Intuition: between monitoring dates the path
    can cross and come back unobserved, so the discrete barrier is effectively further away.
    """
    shift = BGK_BETA * sigma * np.sqrt(dt)
    return float(B * np.exp(shift if up else -shift))
