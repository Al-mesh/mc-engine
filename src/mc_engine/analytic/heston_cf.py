r"""Semi-analytic Heston prices via the characteristic function.

Characteristic function of :math:`x_T = \ln S_T` in the "little Heston trap" form of
Albrecher, Mayer, Schoutens & Tistaert (2007):

.. math::
    \varphi(u) = \exp\!\Big(iu(\ln S_0 + (r-q)T)
      + \frac{\kappa\theta}{\xi^2}\big[(\kappa - \rho\xi iu - d)T
        - 2\ln\tfrac{1 - g e^{-dT}}{1 - g}\big]
      + \frac{v_0}{\xi^2}(\kappa - \rho\xi iu - d)\frac{1 - e^{-dT}}{1 - g e^{-dT}}\Big),

    d = \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)}, \qquad
    g = \frac{\kappa - \rho\xi iu - d}{\kappa - \rho\xi iu + d}.

Heston's original form uses :math:`1/g` and :math:`e^{+dT}`; with the principal branch of
the complex log it jumps discontinuously for long maturities, which silently corrupts
the integral. This form stays continuous on the principal branch.

Two independent inversion formulas are implemented so that each validates the other:

* **Gil-Pelaez** (``method="gil_pelaez"``): :math:`C = S_0e^{-qT}P_1 - Ke^{-rT}P_2`, combined
  into one integral
  :math:`C = \tfrac12(S_0e^{-qT} - Ke^{-rT}) + \frac{e^{-rT}}{\pi}\int_0^\infty
  \mathrm{Re}\big[e^{-iu\ln K}\,\frac{\varphi(u - i) - K\varphi(u)}{iu}\big]du`.
* **Lewis (2001)** (``method="lewis"``): with forward :math:`F` and
  :math:`\varphi_Y(w) = \varphi(w)e^{-iw\ln F}` the CF of :math:`\ln(S_T/F)`,
  :math:`C = e^{-rT}\big(F - \frac{\sqrt{FK}}{\pi}\int_0^\infty
  \frac{\mathrm{Re}[e^{iu\ln(F/K)}\varphi_Y(u - i/2)]}{u^2 + 1/4}du\big)`.
  The integrand decays like :math:`u^{-2}` even before the CF's own decay, which makes it
  the better-conditioned of the two for short maturities / deep OTM strikes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import quad

from mc_engine.models.heston import Heston


def heston_char_func(u: complex | np.ndarray, T: float, model: Heston) -> np.ndarray:
    """Characteristic function :math:`E[e^{iu\\ln S_T}]`; ``u`` may be complex."""
    u = np.asarray(u, dtype=complex)
    kap, th, xi, rho, v0 = model.kappa, model.theta, model.xi, model.rho, model.v0
    iu = 1j * u
    beta = kap - rho * xi * iu
    d = np.sqrt(beta**2 + xi**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)
    c_term = kap * th / xi**2 * ((beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    d_term = v0 / xi**2 * (beta - d) * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(iu * (np.log(model.S0) + (model.r - model.q) * T) + c_term + d_term)


def heston_price(
    model: Heston,
    K: float,
    T: float,
    call: bool = True,
    method: Literal["lewis", "gil_pelaez"] = "lewis",
) -> float:
    """European option price under Heston by numerical Fourier inversion.

    Integration uses adaptive Gauss-Kronrod (``scipy.integrate.quad``) on
    :math:`[0, \\infty)` with tight tolerances; the result is accurate to roughly 1e-8
    for reasonable parameters, far below any Monte Carlo standard error used against it.
    Puts are obtained from put-call parity, which holds model-independently.
    """
    if T <= 0 or K <= 0:
        raise ValueError("T and K must be strictly positive")
    disc_r, disc_q = np.exp(-model.r * T), np.exp(-model.q * T)
    log_k = np.log(K)

    if method == "gil_pelaez":

        def integrand(u: float) -> float:
            num = heston_char_func(u - 1j, T, model) - K * heston_char_func(u, T, model)
            return float((np.exp(-1j * u * log_k) * num / (1j * u)).real)

        integral, _ = quad(integrand, 0.0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-10)
        call_price = 0.5 * (model.S0 * disc_q - K * disc_r) + disc_r * integral / np.pi
    elif method == "lewis":
        log_f = np.log(model.S0) + (model.r - model.q) * T
        fwd = np.exp(log_f)
        log_fk = log_f - log_k

        def integrand(u: float) -> float:
            w = u - 0.5j
            phi_y = heston_char_func(w, T, model) * np.exp(-1j * w * log_f)
            return float((np.exp(1j * u * log_fk) * phi_y).real / (u * u + 0.25))

        integral, _ = quad(integrand, 0.0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-10)
        call_price = disc_r * (fwd - np.sqrt(fwd * K) * integral / np.pi)
    else:
        raise ValueError(f"unknown method {method!r}")

    if call:
        return float(call_price)
    return float(call_price - model.S0 * disc_q + K * disc_r)
