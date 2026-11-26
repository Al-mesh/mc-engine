r"""Heston (1993) stochastic volatility.

.. math::
    dS_t = (r - q)S_t\,dt + \sqrt{v_t}\,S_t\,dW^S_t, \qquad
    dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t, \qquad
    d\langle W^S, W^v\rangle_t = \rho\,dt.

The variance is a CIR process. It stays strictly positive only if the Feller
condition :math:`2\kappa\theta \ge \xi^2` holds; equity calibrations routinely violate
it, so the variance hits zero with positive probability. That is exactly where naive
schemes break, and why two schemes are offered:

``scheme="euler"`` -- full-truncation Euler (Lord, Koekkoek & van Dijk 2010)
    Let :math:`v^+ = \max(v, 0)`. Step
    :math:`v_{n+1} = v_n + \kappa(\theta - v_n^+)\Delta + \xi\sqrt{v_n^+\Delta}\,Z^v` and
    :math:`\ln S_{n+1} = \ln S_n + (r - q - \tfrac12 v_n^+)\Delta + \sqrt{v_n^+\Delta}\,Z^S`,
    :math:`Z^S = \rho Z^v + \sqrt{1-\rho^2}Z^\perp`. The discretised v may go negative;
    it is only *used* through :math:`v^+`. LKvD found this the least-biased of the Euler
    fixes, but the weak error is still first order in :math:`\Delta` and large when
    Feller is badly violated, because the Gaussian increment is a poor model of a
    non-central chi-squared near zero.

``scheme="qe"`` -- Quadratic-Exponential (Andersen 2008)
    Matches the first two conditional moments of :math:`v_{n+1}\mid v_n` (known in closed
    form for CIR) with either a scaled squared Gaussian (when the distribution is far
    from zero, :math:`\psi = s^2/m^2 \le 1.5`) or a mixture of a point mass at zero and an
    exponential (when it is close to zero). Variance is non-negative by construction.
    The log-price uses Andersen's central discretisation of
    :math:`\int v\,dt` (:math:`\gamma_1 = \gamma_2 = \tfrac12`) and, optionally, his
    martingale correction so that :math:`E[S_{n+1}\mid S_n, v_n] = S_n e^{(r-q)\Delta}`
    holds exactly in the discrete scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from mc_engine.models.base import Model, SimResult, validate_n_paths, validate_times
from mc_engine.rng import SeedLike, as_random_source

PSI_CRIT = 1.5  # Andersen's switching threshold; any value in [1, 2] is valid.


@dataclass(frozen=True)
class Heston(Model):
    S0: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    r: float
    q: float = 0.0
    scheme: Literal["qe", "euler"] = "qe"
    martingale_correction: bool = True

    def __post_init__(self) -> None:
        if self.S0 <= 0 or self.v0 < 0:
            raise ValueError("S0 must be positive and v0 non-negative")
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
            raise ValueError("kappa, theta and xi must be strictly positive")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        if self.scheme not in ("qe", "euler"):
            raise ValueError(f"unknown scheme {self.scheme!r}")

    @property
    def feller_satisfied(self) -> bool:
        """``True`` if :math:`2\\kappa\\theta \\ge \\xi^2` (variance never reaches zero)."""
        return 2 * self.kappa * self.theta >= self.xi**2

    def simulate(self, n_paths: int, times: ArrayLike, rng: SeedLike = None) -> SimResult:
        n_paths = validate_n_paths(n_paths)
        t = validate_times(times)
        source = as_random_source(rng)
        if self.scheme == "euler":
            log_s, var = self._simulate_euler(n_paths, np.diff(t), source)
        else:
            log_s, var = self._simulate_qe(n_paths, np.diff(t), source)
        return SimResult(times=t, paths=np.exp(log_s), variance=var)

    # ------------------------------------------------------------------ Euler
    def _simulate_euler(self, n_paths, dt, source):
        n_steps = dt.size
        z = source.normals(n_paths, n_steps, 2)
        z_v = z[..., 0]
        z_s = self.rho * z_v + np.sqrt(1.0 - self.rho**2) * z[..., 1]

        log_s = np.empty((n_paths, n_steps + 1))
        var = np.empty((n_paths, n_steps + 1))
        log_s[:, 0] = np.log(self.S0)
        var[:, 0] = self.v0
        v = np.full(n_paths, float(self.v0))  # the raw (possibly negative) Euler state
        mu = self.r - self.q
        for n in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            sq = np.sqrt(v_pos * dt[n])
            log_s[:, n + 1] = log_s[:, n] + (mu - 0.5 * v_pos) * dt[n] + sq * z_s[:, n]
            v = v + self.kappa * (self.theta - v_pos) * dt[n] + self.xi * sq * z_v[:, n]
            # Store the truncated variance: it is the quantity the scheme actually uses.
            var[:, n + 1] = np.maximum(v, 0.0)
        return log_s, var

    # --------------------------------------------------------------------- QE
    def _simulate_qe(self, n_paths, dt, source):
        n_steps = dt.size
        u_v = source.uniforms(n_paths, n_steps, 1)[..., 0]  # drives the variance step
        z_s = source.normals(n_paths, n_steps, 1)[..., 0]  # independent driver for ln S

        log_s = np.empty((n_paths, n_steps + 1))
        var = np.empty((n_paths, n_steps + 1))
        log_s[:, 0] = np.log(self.S0)
        var[:, 0] = self.v0
        for n in range(n_steps):
            v = var[:, n]
            v_next, k0_star = self._qe_variance_step(v, dt[n], u_v[:, n])
            k1, k2, k3, k4 = self._qe_k_coefficients(dt[n])
            k0 = (
                k0_star
                if self.martingale_correction
                else -self.rho * self.kappa * self.theta * dt[n] / self.xi
            )
            # Andersen eq. (33): ln S_{n+1} = ln S_n + K0 + K1 v_n + K2 v_{n+1}
            #                                 + sqrt(K3 v_n + K4 v_{n+1}) Z.
            # The rho-correlated part of dW^S is recovered *exactly* from the variance
            # increment via int sqrt(v) dW^v = (v_{n+1} - v_n - kappa*theta*dt
            # + kappa*int v dt) / xi, so only the orthogonal part needs Z.
            diffusion_var = np.maximum(k3 * v + k4 * v_next, 0.0)
            log_s[:, n + 1] = (
                log_s[:, n]
                + (self.r - self.q) * dt[n]
                + k0
                + k1 * v
                + k2 * v_next
                + np.sqrt(diffusion_var) * z_s[:, n]
            )
            var[:, n + 1] = v_next
        return log_s, var

    def _qe_k_coefficients(self, dt: float) -> tuple[float, float, float, float]:
        g1 = g2 = 0.5  # central (trapezoidal) weights for int_t^{t+dt} v_s ds
        a = self.kappa * self.rho / self.xi - 0.5
        k1 = g1 * dt * a - self.rho / self.xi
        k2 = g2 * dt * a + self.rho / self.xi
        k3 = g1 * dt * (1.0 - self.rho**2)
        k4 = g2 * dt * (1.0 - self.rho**2)
        return k1, k2, k3, k4

    def _qe_variance_step(
        self, v: np.ndarray, dt: float, u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """One QE step for the variance. Returns ``(v_next, K0*)``.

        ``K0*`` is the martingale-corrected drift constant (Andersen section 4.2.2): it is
        chosen so that :math:`E[\\exp(K_0^* + (K_1 + \\tfrac12K_3)v_n + A v_{n+1})] = 1` with
        :math:`A = K_2 + \\tfrac12 K_4`, using the moment generating function of whichever
        distribution (shifted squared Gaussian, or zero/exponential mixture) was sampled.
        """
        kap, th, xi = self.kappa, self.theta, self.xi
        e = np.exp(-kap * dt)
        # Exact conditional mean and variance of the CIR process.
        m = th + (v - th) * e
        s2 = v * xi**2 * e / kap * (1.0 - e) + th * xi**2 / (2.0 * kap) * (1.0 - e) ** 2
        psi = s2 / m**2  # m >= theta*(1-e) > 0, so this is always finite

        quad = psi <= PSI_CRIT
        # np.where evaluates both branches on every element, so feed each branch dummy
        # values where it is not selected to avoid spurious sqrt/log warnings.
        psi_q = np.where(quad, psi, 1.0)
        inv = 2.0 / psi_q
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
        a = m / (1.0 + b2)
        v_quad = a * (np.sqrt(b2) + ndtri(u)) ** 2

        psi_e = np.where(quad, 2.0, psi)
        p = (psi_e - 1.0) / (psi_e + 1.0)  # probability mass at zero
        beta = (1.0 - p) / m
        # Inverse CDF of the mixture: 0 if u <= p, else exponential tail.
        v_exp = np.where(u <= p, 0.0, np.log((1.0 - p) / (1.0 - np.maximum(u, p))) / beta)

        v_next = np.where(quad, v_quad, v_exp)

        k1, k2, k3, k4 = self._qe_k_coefficients(dt)
        big_a = k2 + 0.5 * k4
        base = -(k1 + 0.5 * k3) * v
        # The MGFs only exist for A < 1/(2a) (quadratic) and A < beta (exponential).
        # Andersen notes this holds for all practical parameters; where it fails we fall
        # back to the uncorrected constant rather than produce NaNs.
        ok_q = quad & (2.0 * big_a * a < 1.0)
        ok_e = ~quad & (big_a < beta)
        one_m = np.where(ok_q, 1.0 - 2.0 * big_a * a, 1.0)
        k0_quad = -big_a * b2 * a / one_m + 0.5 * np.log(one_m)
        denom = np.where(ok_e, beta - big_a, 1.0)
        k0_exp = -np.log(np.where(ok_e, p + beta * (1.0 - p) / denom, 1.0))
        k0_plain = -self.rho * kap * th * dt / xi
        k0_star = np.where(ok_q, k0_quad + base, np.where(ok_e, k0_exp + base, k0_plain))
        return v_next, k0_star
