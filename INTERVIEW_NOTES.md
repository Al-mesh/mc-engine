# Interview notes

Likely questions about this project, with concise answers. Each answer points to the
evidence in the repo, because "I measured it" beats "I read it".

---

**1. How do you know your Monte Carlo prices are right, rather than just plausible?**

Three layers of checks.

* **Point estimates.** Every estimate is compared with an independent reference within
  4 SE: Black-Scholes, the Merton series, the Heston characteristic function (two
  inversion formulas that agree with each other to 1e-7), the geometric-Asian closed
  form, and a binomial tree for American options.
* **The error bar.** Over 500 seeds, the 95% CI must contain the true price a number of
  times consistent with Binomial(500, 0.95). A wrong SE formula, or a biased estimator,
  fails this even when individual prices look fine.
* **Pathwise properties.** Hypothesis property tests check identities that hold exactly
  on every path: put-call parity, monotonicity, barrier ≤ vanilla, geometric ≤ arithmetic.

Seeds are fixed, so CI is deterministic.

**2. Why does antithetic sampling help for a call but not for every payoff?**

For a pair, $\mathrm{Var}(\bar Y) = \tfrac{\sigma^2}{2}(1+\rho)$, so it helps iff the payoffs at $Z$
and $-Z$ are negatively correlated. A monotone payoff (a call) guarantees this. Splitting
$f$ into even and odd parts: averaging cancels the odd part and keeps the even part. For
a payoff that is even in $Z$, such as a straddle or $(\ln S_T - m)^2$, $\rho = 1$, and you
pay for two paths to get one. The variance per unit of cost doubles (tested: ratio 2.0).
Measured gains: 2.0–2.4x for calls and Asians, 1.6x for a barrier, because the knock-out
breaks monotonicity.

**3. Why is the geometric Asian such a good control variate for the arithmetic Asian?**

$\ln G$ is a linear functional of Brownian motion, so it is exactly Gaussian and
$E[(G-K)^+]$ has a closed form. And $G \le A$ with the two moving almost in lockstep:
the measured correlation is 0.99957. The variance reduction is $1/(1-\rho^2) \approx$ 1200x,
and I measured 1,283x. I estimate $\beta$ on an independent pilot run, because using the
same paths introduces an $O(1/N)$ bias.

**4. Why is the pathwise delta of a digital option wrong?**

Pathwise moves the derivative inside the expectation, which needs the payoff to be
Lipschitz in the parameter. A digital is a step function: its derivative is 0 almost
everywhere, so the estimator is identically 0, with SE 0. It is confidently wrong. All
the sensitivity comes from probability mass crossing the strike, a boundary term no
almost-sure derivative sees. A knock-out barrier fails the same way: measured pathwise
delta 0.38 against 0.05 from LR and FD. The fixes are likelihood ratio, a smoothed payoff
(a call spread, with bias $O(\varepsilon^2)$ and variance $O(1/\varepsilon)$), or
conditional Monte Carlo.

**5. If likelihood ratio works for any payoff, why not always use it?**

Its variance is much higher. The weight, e.g. $Z_1/(S_0\sigma\sqrt{\Delta_1})$ for delta,
ignores the payoff's structure. Measured: 2–3.5x the SE of pathwise for a call. Worse,
for path-dependent options the delta weight uses only the first step, so its variance
grows like $1/\Delta_1$. Monitoring an Asian daily instead of monthly makes LR delta much
noisier (measured SE growth tends to $\sqrt{4} = 2$ per 4x more steps). Use pathwise where
the payoff is Lipschitz and LR (or a mix) where it isn't.

**6. Why do finite-difference Greeks need common random numbers?**

With independent draws, the variance of $(\hat V_+ - \hat V_-)/2h$ is $O(h^{-2})$: you divide
the noise of two independent estimates by a small number. With the same draws, the two
revaluations differ only through $h$. The per-path difference is $O(h)$ for a Lipschitz
payoff, so the variance is $O(1)$. Measured: over 20x smaller SE. My implementation rejects
stateful generators as seeds, because they would silently give each bump different draws.

**7. What is the bias of Euler for Heston, and how does QE fix it?**

The CIR variance has a non-central $\chi^2$ transition with mass near zero when
$2\kappa\theta < \xi^2$. Euler uses a Gaussian increment, which goes negative and needs
truncation, and its weak error is first order in $\Delta$. On Andersen's case I
($\xi = 1$, $\rho=-0.9$, $T=10$) I measured an Euler bias of +6.4 at $\Delta=1$ and +0.30
at $\Delta = 1/32$, still 10 SE. QE moment-matches the exact conditional mean and variance:
a squared Gaussian far from zero, a point mass plus an exponential near zero. A
martingale correction makes $E[S_{n+1}\mid S_n]$ exact. Its bias is statistically zero from
$\Delta = 1/4$. Crucially, this bias does not fall with more paths.

**8. How did you validate the Heston semi-analytic price itself?**

I validated it four ways:

* Two independent inversions (Gil-Pelaez and Lewis) agree to 1e-7, including on the
  10-year Andersen case.
* $\varphi(-i) = $ forward and $\varphi(0)=1$.
* As $\xi\to0$ with $\rho = 0$ it reduces to Black-Scholes. With $\rho \ne 0$ the gap is
  first order in $\xi$, which a test also checks.
* QE Monte Carlo agrees with it within SE.

I used the "little trap" form of the characteristic function, because Heston's original
form has a complex-log branch cut that corrupts long maturities.

**9. Why does quasi-Monte Carlo need a Brownian bridge?**

Sobol points are best equidistributed in their first coordinates, and QMC's advantage
depends on the effective dimension. Forward construction spreads importance evenly over all
steps. The bridge puts $W_T$ on the first coordinate and refines midpoints after that. The
law of the path is unchanged (it is an exact isometry, tested), but most of the variance now
sits in the first few dimensions. Measured slopes for a 16-fixing Asian: MC −0.50,
Sobol −0.68, Sobol + bridge −0.81. I randomise by scrambling and use 16 independent
replicates to get a valid SE.

**10. Explain the regression step in Longstaff-Schwartz.**

The continuation value is a conditional expectation, and I approximate it by projecting
the realised discounted future cash flows onto a polynomial basis in $S_t$. I regress on
in-the-money paths only, because that is where exercise is decided. The key subtlety:
the regression only drives the *decision*. Each path is credited its realised cash flow,
never the fitted value. So regression error matters only near the exercise boundary,
where it costs little. I measured spurious exercises of an American call costing
0.004 ± 0.004.

**11. Is your LSM price biased? In which direction?**

The out-of-sample estimate (rule fitted on training paths, applied to fresh ones) values
a feasible stopping rule, so it is a lower bound. The in-sample estimate is biased upward,
because the regression overfits the paths it prices. Measured: with 256 training paths,
in-sample 4.77 vs out-of-sample 4.40. They converge to 4.472 against the tree's 4.478.
Over 20 parameter sets, LSM minus tree averages −0.39 SE. A dual (Andersen-Broadie)
upper bound would tighten the bracket. That is a listed limitation.

**12. Does the choice of basis functions matter?**

Less than people think. Monomials, Chebyshev and plain Laguerre polynomials of the same
degree span the same space, so they give identical fits. I verified that to 1e-9. Only the
span matters: degree, weighting, or extra state variables. Measured: degree 1 is clearly
sub-optimal (−0.07), and from degree 3 the curve is flat. For Heston, a spot-only basis
ignores the variance state, which is a real limitation.

**13. Your VaR models fail Kupiec at 99%. Isn't that a problem?**

It's the finding. The Gaussian static-window model gets 71 exceedances where 33 are
expected, and they cluster in 2015, 2018 and 2020. EWMA volatility removes the clustering
(Christoffersen p 0.25–0.58). Student-t shocks cut exceedances to 51, but they still fail.
The diagnosis: devolatilised portfolio returns are negatively skewed (−0.47; 1% quantile
−2.88 vs the t's −2.62), and a symmetric distribution cannot fit both tails. I deliberately
didn't add a skewed model after seeing the result, because that would be fitting to the
test set. Filtered historical simulation or a skewed-t would be the next step.

**14. Why is ES preferred to VaR, and why is it harder to backtest?**

VaR ignores how bad losses are beyond the quantile, and it is not subadditive:
diversification can appear to increase it. ES averages the tail, is coherent, and is what
Basel FRTB uses (at 97.5%). But VaR is *elicitable* and ES is not: there's no scoring function
whose expected value ES minimises. So you can't backtest ES by simply counting exceedances.
You need joint VaR/ES tests (e.g. Acerbi-Szekely).

**15. What did numba buy you, and what limited it?**

Profiling first: Heston QE spent its time in about 30 temporary numpy arrays per step. A
fused parallel kernel gave only 2.3x at first. The kernel was 8% of the run time, because
`ndtri` and `exp` still ran single-threaded outside it (Amdahl's law). Moving them inside,
with my own inverse normal accurate to about 1e-14 (scipy's isn't callable from numba),
gave 5.5x (100k to 550k paths/s). The rest is random-number generation (45%), which I keep
serial on purpose: both backends use identical draws, and that is what lets the tests prove
they produce the same paths to round-off. Multiprocessing gave only 1.8x on Windows:
spawning workers costs 2.3 s, and they contend for memory bandwidth.

---

## Quick facts to have ready

| | |
|---|---|
| Tests | 260+, ~99% line coverage, fixed seeds, deterministic |
| CI coverage check | 500 seeds, Binomial(500, 0.95) acceptance region |
| Best variance reduction (Asian) | RQMC + bridge ≈ 1,400–1,500x efficiency; geometric CV ≈ 1,000x |
| Heston numba speed-up | 5.5x (550k paths/s, 100 steps, 16 threads) |
| LSM vs tree, 20 cases | mean −0.39 SE, sd 0.91 |
| VaR backtest | 3,273 days, 5 methods; none pass Kupiec at 99%; residual skew −0.47 |
