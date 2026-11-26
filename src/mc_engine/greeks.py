r"""Monte Carlo Greeks: finite differences (CRN), pathwise, and likelihood ratio.

We want :math:`\partial_\theta V(\theta)` with :math:`V(\theta) = E[Y(\theta)]` and
:math:`Y = e^{-rT}h(S(\theta))`. There are three ways to move the derivative inside the
expectation:

**Finite differences (bump and revalue).** :math:`\frac{V(\theta+h) - V(\theta-h)}{2h}`,
estimated by re-running the simulation. Bias :math:`O(h^2)` (central), but the variance
depends on how the two runs are coupled. With independent draws it is
:math:`O(h^{-2})` and explodes. With **common random numbers** (same seed, hence the
same normals) the two runs differ only through :math:`\theta`, and for a Lipschitz payoff
the per-path difference is :math:`O(h)`, so the variance is :math:`O(1)`. For a
discontinuous payoff a fraction :math:`O(h)` of paths flip across the jump, giving
variance :math:`O(h^{-1})`. For gamma (second difference) these become :math:`O(h^{-1})`
for a call and :math:`O(h^{-3})` for a digital.

**Pathwise (infinitesimal perturbation analysis).** Differentiate each path:
:math:`\partial_\theta Y = e^{-rT}\sum_i
\frac{\partial h}{\partial S_{t_i}}\frac{\partial S_{t_i}}{\partial\theta}`.
Valid (interchange of :math:`\partial` and :math:`E` justified) when :math:`Y` is a.s.
differentiable and Lipschitz in :math:`\theta`. That holds for calls, puts and Asians,
**fails** for digitals and barriers, where the jump's contribution is lost. Usually the
lowest-variance method when valid. Under GBM, :math:`\partial S_t/\partial S_0 = S_t/S_0`,
:math:`\partial S_t/\partial\sigma = S_t(W_t - \sigma t)`, :math:`\partial S_t/\partial r = tS_t`.

**Likelihood ratio (score function).** Differentiate the *density* instead of the payoff:
:math:`\partial_\theta E[h(X)] = E[h(X)\,\partial_\theta\log p_\theta(X)]`. Only needs the
density to be smooth in :math:`\theta`, so it works for **any** payoff, digitals
included, but the score weights are noisy and the variance is typically much larger.
For GBM with standardised increments :math:`Z_j` on steps :math:`\Delta_j`:

* delta: :math:`\frac{Z_1}{S_0\sigma\sqrt{\Delta_1}}` (only step 1 depends on :math:`S_0`),
* gamma: :math:`\frac{Z_1^2 - 1}{S_0^2\sigma^2\Delta_1} - \frac{Z_1}{S_0^2\sigma\sqrt{\Delta_1}}`,
* vega: :math:`\sum_j\big(\frac{Z_j^2 - 1}{\sigma} - Z_j\sqrt{\Delta_j}\big)`,
* rho: :math:`\sum_j \frac{Z_j\sqrt{\Delta_j}}{\sigma} - T` (:math:`-T` from discounting).

Note the :math:`1/\sqrt{\Delta_1}` in delta: with a fine grid the LR delta of a
path-dependent option gets *worse*, a known weakness of the method.

**Gamma by "pathwise" is a mixed estimator.** A second pathwise derivative of a call
hits the kink (a Dirac delta), so instead the LR method is applied to the pathwise
delta :math:`G = e^{-rT}\sum_i \partial_i h\cdot S_{t_i}/S_0`:
:math:`\Gamma = E[G\,(\text{score}_{S_0} - 1/S_0)]` (Glasserman 2004, section 7.3).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from mc_engine.models.base import Model
from mc_engine.models.gbm import GBM
from mc_engine.payoffs import Payoff
from mc_engine.pricer import discounted_payoffs, time_grid

GREEKS = ("delta", "gamma", "vega", "rho")


@dataclass(frozen=True)
class GreekEstimate:
    value: float
    std_error: float
    n_paths: int

    def __str__(self) -> str:
        return f"{self.value:+.6f} (se {self.std_error:.2e})"


def _estimate(samples: np.ndarray) -> GreekEstimate:
    n = samples.size
    return GreekEstimate(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n)), n)


# --------------------------------------------------------------- finite differences


def fd_greeks(
    model: Model,
    payoff: Payoff,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    bump_spot: float | None = None,
    bump_vol: float = 0.005,
    bump_rate: float = 1e-3,
) -> dict[str, GreekEstimate]:
    """Central bump-and-revalue Greeks with common random numbers.

    Every revaluation re-seeds with the same ``seed``, so path i sees identical normals in
    all runs and the per-path differences are formed *before* averaging. That pairing is
    what keeps the variance bounded; the SE reported is the SE of the per-path difference
    quotients, which is the correct one under CRN.

    ``bump_spot`` defaults to 1% of spot. Vega is computed only for models with a ``sigma``
    parameter. Note that the *expectation* of these estimators is the finite difference of
    the true price, which differs from the true Greek by O(h^2); the tests compare against
    the finite difference of the analytic price for exactly that reason.
    """
    h_s = 0.01 * model.S0 if bump_spot is None else bump_spot

    def y(m: Model) -> np.ndarray:
        return discounted_payoffs(m, payoff, maturity, n_steps, n_paths, rng=seed)

    base = y(model)
    up = y(dataclasses.replace(model, S0=model.S0 + h_s))
    down = y(dataclasses.replace(model, S0=model.S0 - h_s))
    out = {
        "delta": _estimate((up - down) / (2 * h_s)),
        "gamma": _estimate((up - 2 * base + down) / h_s**2),
    }
    if hasattr(model, "sigma"):
        v_up = y(dataclasses.replace(model, sigma=model.sigma + bump_vol))
        v_dn = y(dataclasses.replace(model, sigma=model.sigma - bump_vol))
        out["vega"] = _estimate((v_up - v_dn) / (2 * bump_vol))
    r_up = y(dataclasses.replace(model, r=model.r + bump_rate))
    r_dn = y(dataclasses.replace(model, r=model.r - bump_rate))
    out["rho"] = _estimate((r_up - r_dn) / (2 * bump_rate))
    return out


# ------------------------------------------------------------- GBM path quantities


def _require_gbm(model: Model) -> GBM:
    if not isinstance(model, GBM):
        raise TypeError("pathwise and likelihood-ratio Greeks are implemented for GBM")
    return model


def _gbm_paths(model: GBM, maturity: float, n_steps: int, n_paths: int, seed):
    """Simulate, then recover W_t and the standardised increments Z_j from the path.

    Recovering them from S (rather than asking the source) keeps these estimators
    independent of how the normals were produced (pseudo-random, antithetic, bridge).
    """
    times = time_grid(maturity, n_steps)
    sim = model.simulate(n_paths, times, rng=seed)
    log_ret = np.log(sim.paths / model.S0)
    drift = model.r - model.q - 0.5 * model.sigma**2
    w = (log_ret - drift * times) / model.sigma  # Brownian path W_t, W_0 = 0
    dt = np.diff(times)
    z = np.diff(w, axis=1) / np.sqrt(dt)  # standardised increments
    return sim.paths, times, w, z, dt


# ---------------------------------------------------------------------- pathwise


def pathwise_greeks(
    model: Model, payoff: Payoff, maturity: float, n_steps: int, n_paths: int, seed=None
) -> dict[str, GreekEstimate]:
    """Pathwise delta, vega, rho and the mixed pathwise-LR gamma (GBM only)."""
    m = _require_gbm(model)
    paths, times, w, z, dt = _gbm_paths(m, maturity, n_steps, n_paths, seed)
    disc = np.exp(-m.r * maturity)
    grad = payoff.path_gradient(paths, times)  # dh/dS_{t_i}

    delta = disc * (grad * paths).sum(axis=1) / m.S0
    vega = disc * (grad * paths * (w - m.sigma * times)).sum(axis=1)
    # d/dr of e^{-rT} h(S): the discount factor contributes -T * Y.
    rho = disc * (grad * paths * times).sum(axis=1) - maturity * disc * payoff(paths, times)
    score_s0 = z[:, 0] / (m.S0 * m.sigma * np.sqrt(dt[0]))
    gamma = delta * (score_s0 - 1.0 / m.S0)
    return {
        "delta": _estimate(delta),
        "gamma": _estimate(gamma),
        "vega": _estimate(vega),
        "rho": _estimate(rho),
    }


# -------------------------------------------------------------- likelihood ratio


def likelihood_ratio_greeks(
    model: Model, payoff: Payoff, maturity: float, n_steps: int, n_paths: int, seed=None
) -> dict[str, GreekEstimate]:
    """Score-function Greeks (GBM only). Works for any payoff, including discontinuous ones."""
    m = _require_gbm(model)
    paths, times, _, z, dt = _gbm_paths(m, maturity, n_steps, n_paths, seed)
    y = np.exp(-m.r * maturity) * payoff(paths, times)
    s0, sig = m.S0, m.sigma
    z1, dt1 = z[:, 0], dt[0]

    w_delta = z1 / (s0 * sig * np.sqrt(dt1))
    w_gamma = (z1**2 - 1.0) / (s0**2 * sig**2 * dt1) - z1 / (s0**2 * sig * np.sqrt(dt1))
    w_vega = ((z**2 - 1.0) / sig - z * np.sqrt(dt)).sum(axis=1)
    w_rho = (z * np.sqrt(dt)).sum(axis=1) / sig - maturity
    return {
        "delta": _estimate(y * w_delta),
        "gamma": _estimate(y * w_gamma),
        "vega": _estimate(y * w_vega),
        "rho": _estimate(y * w_rho),
    }
