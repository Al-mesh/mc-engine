# mc-engine — Plan

A Monte Carlo derivatives pricing and risk engine. Correctness first: every
estimator ships with a closed-form or independent numerical benchmark and a
statistical test against it.

## 1. Architecture

Four layers. Each layer depends only on the one below it.

```
 cli.py / notebooks / benchmarks           (presentation)
        │
 pricer.py  greeks/  american/  risk/      (estimators)
        │
 models/   payoffs/   variance_reduction/  (building blocks)
        │
 rng.py   analytic/   _numba_kernels.py    (primitives + reference solutions)
```

### Core contracts

```python
class Model(ABC):
    def simulate(self, n_paths: int, times: NDArray, rng: RandomSource) -> SimResult
    # SimResult.paths: (n_paths, n_steps+1) underlying; optional aux (variance, jump counts)

class Payoff(ABC):
    path_dependent: bool
    def __call__(self, paths: NDArray, times: NDArray) -> NDArray   # (n_paths,) undiscounted

@dataclass(frozen=True)
class PriceResult:
    price: float; std_error: float; ci95: tuple[float, float]; n_paths: int; runtime_s: float
```

Design decisions I'll defend in the code comments:

- **Models consume normals, not seeds.** `RandomSource` wraps either a seeded
  `np.random.Generator` (pseudo-random) or a scrambled Sobol sequence mapped
  through the inverse normal CDF. Models ask it for a `(n_paths, n_steps, d)`
  block of standard normals. That one seam gives us antithetics (negate the
  block), common random numbers for finite-difference Greeks (reuse the block),
  and QMC (swap the source) without touching any model code.
- **No global RNG state.** Every entry point takes `seed` or a `Generator`.
  Sub-streams use `SeedSequence.spawn` so parallel workers are independent and
  reproducible.
- **Payoffs are pure functions of the path matrix.** Discounting lives in the
  pricer. This keeps payoffs trivially testable and lets pathwise Greeks
  differentiate them.
- **Monitoring is explicit.** Barrier and lookback payoffs are discretely
  monitored on the simulation grid, and the docs say so. The continuous-
  monitoring bias, and the Broadie–Glasserman–Kou correction for it, gets
  documented and tested rather than hidden.
- **The QMC standard error comes from randomisation.** A single Sobol run has
  no valid sample variance, so the QMC pricer runs R independent scramblings
  and reports the SE across those replicates.

## 2. Module layout

```
mc-engine/
├── pyproject.toml            # hatchling build; ruff, pytest, coverage config
├── .github/workflows/ci.yml  # ruff + pytest (+ coverage) on 3.11/3.12, ubuntu + windows
├── src/mc_engine/
│   ├── rng.py                # RandomSource: PseudoRandom, SobolQMC (scrambled, Brownian bridge)
│   ├── brownian_bridge.py    # BB path construction (dimension reordering for QMC)
│   ├── models/
│   │   ├── base.py           # Model ABC, SimResult
│   │   ├── gbm.py            # exact log-normal stepping, multi-asset with Cholesky
│   │   ├── heston.py         # full-truncation Euler + Andersen QE (with martingale correction)
│   │   └── merton.py         # exact: Poisson jump counts + lognormal jump sizes
│   ├── payoffs/              # european, digital, asian (arith + geom), barrier (up-and-out), lookback
│   ├── analytic/
│   │   ├── black_scholes.py  # prices + delta/gamma/vega/rho, digital, geometric Asian (discrete)
│   │   ├── heston_cf.py      # semi-analytic via characteristic function ("little trap" form) + quad
│   │   ├── merton.py         # Merton series solution
│   │   └── binomial.py       # CRR tree (European + American), used as the LSM benchmark
│   ├── pricer.py             # mc_price(model, payoff, ...) -> PriceResult
│   ├── variance_reduction/   # antithetic, control variates (optimal beta), qmc driver, efficiency report
│   ├── greeks/               # finite_difference (CRN), pathwise, likelihood_ratio
│   ├── american/lsm.py       # Longstaff–Schwartz with pluggable basis (monomial, Laguerre, Hermite)
│   ├── risk/
│   │   ├── portfolio.py      # positions, revaluation (full or delta-gamma)
│   │   ├── scenarios.py      # correlated GBM (Gaussian copula) + Student-t copula
│   │   ├── var.py            # VaR / ES at 95% and 99%, 1-day and 10-day horizons
│   │   ├── backtest.py       # Kupiec POF likelihood-ratio test (+ Christoffersen independence if time allows)
│   │   └── data.py           # yfinance download with a parquet cache in data/
│   ├── _numba_kernels.py     # njit hot loops (Heston QE stepping, path-dependent payoffs)
│   └── cli.py                # `mc-engine price|greeks|var ...` (argparse)
├── tests/                    # mirrors src/; statistical tests marked @pytest.mark.statistical
├── benchmarks/               # convergence, variance reduction, numba vs numpy, scaling scripts -> CSV + PNG
├── notebooks/walkthrough.ipynb
├── docs/maths.md
├── data/                     # cached price history committed so CI never hits the network
├── README.md
└── INTERVIEW_NOTES.md
```

## 3. Validation strategy

The tests are as much the deliverable as the engine.

| Claim | Test |
|---|---|
| European MC is unbiased | Over 200 seeds, the 95% CI contains the BS price in about 95% of runs. I check the coverage count against a binomial(200, 0.95) acceptance region, which is stricter and fairer than a fixed tolerance. |
| MC error ~ N^-1/2 | Log-log regression of RMSE over seeds vs N. The fitted slope must lie in [-0.6, -0.4]. |
| RQMC error close to N^-1 | Same regression on a smooth payoff. Slope expected around -0.8 to -1.0; I report whatever comes out. |
| Heston pricing | MC vs CF price across a strike grid, plus a bias-vs-Δt study comparing Euler and QE in a Feller-violating regime. |
| Merton | MC vs series solution. |
| No-arbitrage properties (hypothesis) | Put-call parity on common paths, call monotone in S0 and decreasing in K, price bounds max(S−Ke^{−rT},0) ≤ C ≤ S, barrier ≤ vanilla, geometric Asian ≤ arithmetic Asian. |
| Greeks | All three methods vs analytic BS Greeks. Demonstrate pathwise failing on a digital (its derivative is zero almost surely) and show LR or a smoothed payoff recovering it. |
| LSM | American put vs CRR with 5000+ steps. Show the low bias (in-sample vs out-of-sample) and sensitivity to basis and path count. |
| Kupiec | Unit tests on synthetic exceedance sequences with known answers, then the real backtest on cached data. |

Statistical tests use fixed seeds so they are deterministic in CI. A failing
test means investigating the cause; tolerances do not get loosened to make it
pass.

Target: >85% line coverage via `pytest-cov`.

## 4. Build order (one commit per phase; tests run and shown after each)

0. **Scaffold.** pyproject, ruff, CI, package skeleton, `rng.py`, `analytic/black_scholes.py` with tests.
1. **Core engine.** Model ABC; GBM, Heston (Euler + QE), Merton; all payoffs; `mc_price`; Heston CF and Merton series benchmarks; CI-coverage and property tests.
2. **Variance reduction.** Antithetic, geometric-Asian control variate (closed form for discrete monitoring), Sobol + scrambling + Brownian bridge with randomised SE; efficiency table; convergence slope tests.
3. **Greeks.** FD with CRN, pathwise, LR; comparison table; the digital/barrier failure demo.
4. **American.** CRR tree and LSM; basis-function and path-count studies.
5. **Risk.** Correlated GBM and t-copula scenarios, VaR/ES, yfinance cache, Kupiec backtest.
6. **Performance.** Numba kernels with numpy fallbacks (kept as the reference implementation and cross-checked in tests); multiprocessing over `SeedSequence.spawn` if it actually pays off; paths/sec and scaling table.
7. **Deliverables.** CLI, notebook, README with plots generated by `benchmarks/`, `docs/maths.md`, `INTERVIEW_NOTES.md`.

Every number in the README is produced by a script in `benchmarks/` that writes
to `docs/results/`. The README embeds only those outputs.

## 5. Known scope limits (to be stated honestly in the README)

- Barriers and lookbacks are discretely monitored.
- Heston calibration is out of scope; parameters are taken from the literature (e.g. Andersen 2008 test cases).
- LSM gives a lower bound. A dual upper bound (Andersen–Broadie) is a stretch goal only.
- VaR uses a static portfolio and ignores liquidity and intraday effects.
