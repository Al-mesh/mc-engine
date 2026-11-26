# The mathematics behind mc-engine

Short derivations of the ideas the code relies on. Each section names the module that
implements it. Notation: $r$ is the risk-free rate, $q$ the dividend yield, $T$ maturity,
$Z \sim N(0,1)$, $\varphi$ and $N$ the standard normal density and CDF.

---

## 1. Risk-neutral pricing and the Monte Carlo estimator

**Why we can price by expectation.** In an arbitrage-free market with a bank account
$B_t = e^{rt}$, the fundamental theorem of asset pricing gives an equivalent measure $Q$
under which every traded price divided by $B_t$ is a martingale. A claim paying $h(S)$ at
$T$ must therefore be worth

$$V_0 = E^Q\!\left[e^{-rT}h(S)\right].$$

Under $Q$ the stock drifts at $r - q$, not at its real-world expected return $\mu$.
Hedging removes $\mu$ from the price, which is why every model here is simulated with
drift $r-q$. The tests check this martingale property directly:
$E[S_t] = S_0e^{(r-q)t}$ for every model (`tests/test_models.py`).

**The estimator.** With i.i.d. discounted payoffs $Y_i = e^{-rT}h(S^{(i)})$,
$\hat V_N = \frac1N\sum_i Y_i$ is unbiased. If $\sigma^2 = \mathrm{Var}(Y) < \infty$, the CLT gives

$$\sqrt N(\hat V_N - V_0) \Rightarrow N(0,\sigma^2), \qquad
\mathrm{SE} = \hat\sigma/\sqrt N, \qquad \text{95\% CI} = \hat V_N \pm 1.96\,\mathrm{SE}.$$

Three consequences drive the whole project:

1. The error decays like $N^{-1/2}$, independent of dimension. 100x more paths buy only
   10x more accuracy, which is why variance reduction pays off.
2. The SE is itself estimated, so the CI is only asymptotically valid. The test suite
   checks *coverage*, not just the point estimate: over 500 seeds the CI must contain the
   Black-Scholes price a Binomial(500, 0.95)-consistent number of times.
3. Discretisation bias (e.g. Euler for Heston) does *not* shrink with $N$. It must be
   controlled separately, through the time step.

**Exact simulation of GBM** (`models/gbm.py`). $S_T = S_0\exp((r-q-\tfrac12\sigma^2)T +
\sigma\sqrt T Z)$ holds exactly, so European prices under GBM need one step and carry no
discretisation bias. The $-\tfrac12\sigma^2$ is the Itô correction that makes
$E[S_T] = S_0e^{(r-q)T}$, since $E[e^{\sigma\sqrt T Z}] = e^{\sigma^2T/2}$.

---

## 2. Antithetic variates

Pair each draw $Z$ with $-Z$, which has the same distribution, and average:
$\bar Y = \tfrac12(Y(Z) + Y(-Z))$. Then

$$\mathrm{Var}(\bar Y) = \tfrac{\sigma^2}{2}(1 + \rho), \qquad \rho = \mathrm{Corr}(Y(Z), Y(-Z)).$$

Two independent draws would give $\sigma^2/2$ at the same cost, so antithetics help iff
$\rho < 0$.

* **When it helps.** If $Y = f(Z)$ with $f$ monotone (a call under GBM), $f(Z)$ and $f(-Z)$
  are negatively correlated (Chebyshev's association inequality), so $\rho<0$. Measured: about a 2x
  variance reduction for calls and Asians.
* **When it hurts.** Decompose $f = f_{\text{even}} + f_{\text{odd}}$. Antithetic averaging
  cancels the odd part exactly and *keeps* the even part. A payoff that is even in $Z$
  (a straddle-like $(\ln S_T - m)^2$) has $\rho = 1$, so the variance per unit of cost
  *doubles*. The test suite checks this.
* **The SE must come from pair averages.** The $2m$ paths are not independent.
  (`pricer.py`, `rng.AntitheticSource`)

## 3. Control variates

Let $X$ be a control with known mean $\mu_X$. For any fixed $\beta$,
$Y_\beta = Y - \beta(X - \mu_X)$ is unbiased, with

$$\mathrm{Var}(Y_\beta) = \sigma_Y^2 - 2\beta\sigma_{XY} + \beta^2\sigma_X^2.$$

This is minimised at $\beta^* = \sigma_{XY}/\sigma_X^2$, where it equals $\sigma_Y^2(1-\rho_{XY}^2)$.
The variance reduction factor is $1/(1-\rho^2)$: $\rho = 0.9$ gives 5x, $\rho=0.99$ gives
50x, and $\rho = 0.9996$ gives about 1200x.

**Geometric Asian as control for the arithmetic Asian** (Kemna-Vorst). $\ln G =
\frac1n\sum\ln S_{t_i}$ is a linear functional of Brownian motion, so it is exactly Gaussian with

$$\mu_G = \ln S_0 + (r-q-\tfrac12\sigma^2)\bar t, \qquad
\sigma_G^2 = \frac{\sigma^2}{n^2}\sum_{i,j}\min(t_i,t_j),$$

and $E[(G-K)^+]$ has a Black-Scholes-type closed form. Since $G \le A$ path by path and the
two averages move almost in lockstep, $\rho \approx 0.9996$ (measured).

**Implementation choice.** Estimating $\beta$ from the same paths makes
$\hat V$ biased by $O(1/N)$, because $\hat\beta$ and $\bar X$ are correlated. Here
$\beta$ comes from an independent pilot run, which keeps the estimator exactly
unbiased (`variance_reduction.control_variate_price`).

## 4. Quasi-Monte Carlo, randomisation and the Brownian bridge

**Koksma-Hlawka.** For an integrand of bounded Hardy-Krause variation $V(f)$ on
$[0,1]^d$,

$$\Big|\tfrac1N\textstyle\sum f(u_i) - \int f\Big| \le V(f)\,D_N^*,$$

and low-discrepancy (Sobol) points achieve $D_N^* = O(N^{-1}(\log N)^d)$, against MC's
$N^{-1/2}$. In practice the gain depends on the *effective dimension*: Sobol coordinates
are best spread out in the first few dimensions.

**Randomisation (RQMC).** A single Sobol sequence is deterministic, so it gives no
variance estimate. Owen scrambling with a random digital shift makes each point marginally
$U(0,1)^d$ while keeping the net structure. Then $R$ independent scramblings give i.i.d.
unbiased estimates, and $\mathrm{SE} = \mathrm{sd}(\hat V_r)/\sqrt R$ is valid.

**Brownian bridge.** Build $W$ by its terminal value first, $W_T = \sqrt T Z_1$, then
midpoints conditionally:

$$W_{t} \mid W_{s}, W_{u} \sim N\!\left(\tfrac{(u-t)W_s + (t-s)W_u}{u-s},\
\tfrac{(t-s)(u-t)}{u-s}\right), \qquad s < t < u.$$

This is just another square root of the Brownian covariance matrix, so the law of the path
is unchanged (tested: the map is an exact isometry). But the first coordinate now carries
most of the variance of path functionals, which cuts the effective dimension. Measured
convergence slopes for a 16-fixing Asian: MC $-0.50$, Sobol forward construction $-0.68$,
Sobol + bridge $-0.81$.

---

## 5. Greeks: pathwise vs likelihood ratio

We want $\partial_\theta E[Y(\theta)]$. There are two ways to move the derivative inside the
expectation.

**Pathwise (differentiate the payoff).** If $Y$ is a.s. differentiable in $\theta$ and
Lipschitz, with an integrable constant, dominated convergence gives

$$\partial_\theta E[Y] = E[\partial_\theta Y] = e^{-rT}E\Big[\textstyle\sum_i
\frac{\partial h}{\partial S_{t_i}}\frac{\partial S_{t_i}}{\partial\theta}\Big].$$

Under GBM, $\partial S_t/\partial S_0 = S_t/S_0$ and $\partial S_t/\partial\sigma =
S_t(W_t - \sigma t)$. For a call, delta $= e^{-rT}E[\mathbb 1\{S_T>K\}S_T/S_0]$.

**Why pathwise fails for a digital.** $h(S) = \mathbb 1\{S > K\}$ has derivative $0$
almost everywhere, so the pathwise estimator is identically zero. Yet the true delta,
$e^{-rT}\varphi(d_2)/(S_0\sigma\sqrt T)$, is positive. The interchange of $\partial$
and $E$ needs Lipschitz continuity, and a jump violates it: the whole sensitivity sits in
the probability mass crossing $K$, a boundary term the a.s. derivative never sees. A
knock-out barrier fails the same way. A small bump in $S_0$ pushes paths over the
barrier and destroys payoffs of size about $B-K$. Measured: pathwise barrier delta 0.38,
true value about 0.05.

**Likelihood ratio (differentiate the density).** If $S$ has density $p_\theta$,

$$\partial_\theta E[h(S)] = \int h\,\partial_\theta p_\theta = E\big[h(S)\,\partial_\theta\log p_\theta(S)\big].$$

Only the density needs to be smooth, so it works for any payoff, digitals included. For
GBM with first standardised increment $Z_1$ on step $\Delta_1$, the delta weight is
$\frac{Z_1}{S_0\sigma\sqrt{\Delta_1}}$. The price is variance: the score weight is
uncorrelated with the payoff structure. Also, the $1/\sqrt{\Delta_1}$ means the LR delta of a
path-dependent option gets *noisier* on finer grids (measured: SE $\propto\sqrt{n_\text{steps}}$
asymptotically).

**Fixes for discontinuous payoffs.** Use LR. Or use a mixed estimator, LR applied to the
pathwise delta, which gives the lowest-variance gamma. Or smooth the payoff: a call
spread of half-width $\varepsilon$ is Lipschitz, with bias $O(\varepsilon^2)$ and pathwise
variance $O(1/\varepsilon)$.

**Finite differences need common random numbers.**
$\mathrm{Var}\big(\frac{\hat V(\theta+h) - \hat V(\theta-h)}{2h}\big)$ is $O(h^{-2})$ with
independent draws but $O(1)$ with the same draws for a Lipschitz payoff. For discontinuous
payoffs it is $O(h^{-1})$, since only a fraction $O(h)$ of paths flip.

---

## 6. Longstaff-Schwartz: the regression step

An American option is an optimal stopping problem, $V_0 = \sup_\tau E[e^{-r\tau}h(S_\tau)]$.
On exercise dates $t_1<\dots<t_M$ the optimal rule is to stop at $t_j$ iff
$h(S_{t_j}) \ge C_j(S_{t_j})$, where

$$C_j(s) = E\big[e^{-r\Delta}V_{j+1} \mid S_{t_j} = s\big]$$

is a conditional expectation. A conditional expectation is the $L^2$ projection onto
functions of $S_{t_j}$. LSM approximates it by projecting onto a basis $\psi_0,\dots,\psi_d$:

$$\hat\beta_j = \arg\min_\beta\sum_{i\in\text{ITM}}\Big(Y_i - \sum_k\beta_k\psi_k(S^{(i)}_{t_j})\Big)^2,
\qquad \hat C_j = \sum_k\hat\beta_{j,k}\psi_k,$$

where $Y_i$ is the *realised* discounted cash flow of path $i$ under the already-estimated
future policy. The algorithm works backwards: at each date, exercise where
$h > \hat C_j$, and replace that path's cash flow with $h$.

* **Why regress on realised cash flows.** Errors in $\hat C$ only affect *decisions*, never
  the value credited to a path. And wrong decisions happen where $h \approx C$, where they
  cost almost nothing (measured: spurious exercises of an American call cost
  $0.004\pm0.004$).
* **Why in-the-money paths only.** Exercise is only ever considered where $h>0$, so the
  fit's accuracy is spent where it matters.
* **Bias in both directions.** Applying a *fixed* estimated rule to independent paths
  values a feasible stopping time, which is at most $V_0$: a lower bound. Fitting and
  pricing on the same paths lets the regression overfit each path's future, a foresight
  bias that pushes the estimate up. The two bracket the true value and meet as $N$ grows.
* **Basis.** Polynomial families of equal degree span the same space, so they give the
  same fit (tested). Only the span (degree, weighting, state variables) changes the answer.

---

## 7. Heston discretisation: why QE beats Euler

The variance follows a CIR process. Over a step it has a non-central $\chi^2$ law with
known mean $m$ and variance $s^2$, and point mass near zero when the Feller condition
$2\kappa\theta \ge \xi^2$ fails. Euler replaces it with a Gaussian increment. That goes
negative, needs truncation, and has $O(\Delta)$ weak error, which in practice is large in
Feller-violating regimes.

Andersen's QE matches $m$ and $s^2$ exactly with a sampleable distribution. For
$\psi = s^2/m^2 \le 1.5$ it uses $a(b+Z)^2$. Otherwise it uses a point mass at 0 plus an
exponential. A martingale correction to the drift constant makes $E[S_{n+1}\mid S_n] = S_ne^{(r-q)\Delta}$
hold exactly in the discrete scheme. Measured on Andersen's case I ($\xi=1$, $\rho=-0.9$,
$T=10$): at $\Delta = 1/4$ the Euler bias is +2.10 (55 SE), while QE's is not
significant.

## 8. VaR, ES and the Kupiec test

$\mathrm{VaR}_\alpha$ is the $\alpha$-quantile of the loss.
$\mathrm{ES}_\alpha = \frac1{1-\alpha}\int_\alpha^1\mathrm{VaR}_u\,du$. ES is coherent
(subadditive) and sees the size of tail losses. VaR is neither.

If a VaR model is correct, exceedances are i.i.d. Bernoulli($p = 1-\alpha$). With $x$
exceedances in $n$ days, Kupiec's statistic

$$LR = -2\ln\frac{(1-p)^{n-x}p^x}{(1-x/n)^{n-x}(x/n)^x} \;\xrightarrow{d}\; \chi^2_1$$

tests the *rate*. Christoffersen's Markov test checks *clustering*. A static-volatility
model can pass the first and fail the second, which is exactly what the 2012-2024
backtest shows.
