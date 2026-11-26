"""Command-line interface.

Examples::

    mc-engine price --model gbm --payoff european --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2
    mc-engine price --model heston --payoff asian --S0 100 --K 100 --T 1 --r 0.03 \\
        --v0 0.04 --kappa 1.5 --theta 0.04 --xi 0.6 --rho -0.7 --steps 100 --paths 200000
    mc-engine price --model gbm --payoff american --put --S0 36 --K 40 --T 1 --r 0.06 \\
        --sigma 0.2 --steps 50
    mc-engine greeks --S0 100 --K 105 --T 1 --r 0.05 --sigma 0.2 --method all

Whenever a closed-form or semi-analytic reference exists (Black-Scholes, digital, geometric
Asian, Heston characteristic function, Merton series, CRR tree for American options), it is
printed next to the Monte Carlo estimate, with the gap measured in standard errors.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mc_engine.analytic import (
    bs_delta,
    bs_digital_price,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_vega,
    geometric_asian_price,
    heston_price,
    merton_price,
)
from mc_engine.analytic.binomial import crr_price
from mc_engine.models import GBM, Heston, Merton, Model
from mc_engine.payoffs import (
    ArithmeticAsian,
    Digital,
    European,
    FixedLookbackCall,
    FloatingLookbackCall,
    GeometricAsian,
    Payoff,
    UpAndOutCall,
)
from mc_engine.pricer import PriceResult, mc_price, time_grid

PAYOFFS = (
    "european",
    "digital",
    "asian",
    "geometric-asian",
    "barrier",
    "lookback-floating",
    "lookback-fixed",
    "american",
)


def _model(a: argparse.Namespace) -> Model:
    if a.model == "gbm":
        return GBM(S0=a.S0, r=a.r, sigma=a.sigma, q=a.q)
    if a.model == "merton":
        return Merton(
            S0=a.S0, r=a.r, sigma=a.sigma, lam=a.lam, mu_j=a.mu_j, sigma_j=a.sigma_j, q=a.q
        )
    return Heston(
        S0=a.S0,
        v0=a.v0,
        kappa=a.kappa,
        theta=a.theta,
        xi=a.xi,
        rho=a.rho,
        r=a.r,
        q=a.q,
        scheme=a.scheme,
        backend=a.backend,
    )


def _payoff(a: argparse.Namespace) -> Payoff:
    call = not a.put
    if a.payoff == "european":
        return European(a.K, call)
    if a.payoff == "digital":
        return Digital(a.K, call)
    if a.payoff == "asian":
        return ArithmeticAsian(a.K, call)
    if a.payoff == "geometric-asian":
        return GeometricAsian(a.K, call)
    if a.payoff == "barrier":
        if a.barrier is None:
            raise SystemExit("--barrier is required for the up-and-out barrier payoff")
        if a.put:
            raise SystemExit("only the up-and-out *call* is implemented")
        return UpAndOutCall(a.K, a.barrier)
    if a.payoff == "lookback-floating":
        return FloatingLookbackCall()
    return FixedLookbackCall(a.K)


def _reference(a: argparse.Namespace, model: Model) -> tuple[str, float] | None:
    """Closed-form / semi-analytic benchmark for this (model, payoff), if one exists."""
    call = not a.put
    if a.payoff == "european":
        if isinstance(model, GBM):
            return "Black-Scholes", float(bs_price(a.S0, a.K, a.T, a.r, a.sigma, a.q, call))
        if isinstance(model, Heston):
            return "Heston CF (Lewis)", heston_price(model, a.K, a.T, call)
        return "Merton series", merton_price(model, a.K, a.T, call)
    if isinstance(model, GBM) and a.payoff == "digital":
        return "BS digital", float(bs_digital_price(a.S0, a.K, a.T, a.r, a.sigma, a.q, call))
    if isinstance(model, GBM) and a.payoff == "geometric-asian":
        fix = time_grid(a.T, a.steps)[1:]
        return "closed form", geometric_asian_price(a.S0, a.K, a.r, a.sigma, fix, a.q, call)
    return None


def _print_result(label: str, res: PriceResult, ref: tuple[str, float] | None) -> None:
    lo, hi = res.ci95
    print(f"{label}")
    print(f"  price      {res.price:.6f}")
    print(f"  std error  {res.std_error:.6f}")
    print(f"  95% CI     [{lo:.6f}, {hi:.6f}]")
    print(f"  paths      {res.n_paths:,}   time {res.runtime_s:.3f}s")
    if ref is not None:
        name, value = ref
        gap = (res.price - value) / res.std_error if res.std_error > 0 else float("nan")
        print(f"  reference  {value:.6f} ({name}); MC - reference = {gap:+.2f} SE")


def cmd_price(a: argparse.Namespace) -> int:
    model = _model(a)
    if a.payoff == "american":
        from mc_engine.american import lsm_price

        res = lsm_price(
            model,
            a.K,
            a.T,
            a.steps,
            a.paths,
            call=not a.put,
            seed=a.seed,
            degree=a.degree,
            basis=a.basis,
        )
        ref = None
        if isinstance(model, GBM):
            ref = (
                "CRR Bermudan, same dates",
                crr_price(
                    a.S0,
                    a.K,
                    a.T,
                    a.r,
                    a.sigma,
                    a.q,
                    n_steps=100 * a.steps,
                    call=not a.put,
                    exercise="bermudan",
                    n_exercise_dates=a.steps,
                ),
            )
        _print_result(
            f"Longstaff-Schwartz ({a.steps} exercise dates, {a.basis} degree "
            f"{a.degree}); in-sample {res.in_sample_price:.6f}",
            res,
            ref,
        )
        return 0

    payoff = _payoff(a)
    if a.qmc:
        from mc_engine.variance_reduction import rqmc_price

        n_rep = 16
        per_rep = 1 << max(1, (a.paths // n_rep).bit_length() - 1)  # power of two
        res = rqmc_price(
            model,
            payoff,
            a.T,
            a.steps,
            per_rep,
            n_rep,
            seed=a.seed,
            brownian_bridge=not isinstance(model, Heston) or a.scheme == "euler",
        )
        method = f"randomised QMC ({n_rep} scramblings x {per_rep:,} Sobol points)"
    else:
        paths = a.paths + (a.paths % 2 if a.antithetic else 0)
        res = mc_price(model, payoff, a.T, a.steps, paths, rng=a.seed, antithetic=a.antithetic)
        method = "Monte Carlo" + (" with antithetic variates" if a.antithetic else "")
    _print_result(
        f"{a.model} / {a.payoff} ({'put' if a.put else 'call'}): {method}, {a.steps} steps",
        res,
        _reference(a, model),
    )
    return 0


def cmd_greeks(a: argparse.Namespace) -> int:
    from mc_engine.greeks import fd_greeks, likelihood_ratio_greeks, pathwise_greeks

    model = GBM(S0=a.S0, r=a.r, sigma=a.sigma, q=a.q)
    payoff = European(a.K, not a.put)
    args = (a.S0, a.K, a.T, a.r, a.sigma, a.q)
    exact = {
        "delta": float(bs_delta(*args, call=not a.put)),
        "gamma": float(bs_gamma(*args)),
        "vega": float(bs_vega(*args)),
        "rho": float(bs_rho(*args, call=not a.put)),
    }
    methods = {"fd": fd_greeks, "pathwise": pathwise_greeks, "lr": likelihood_ratio_greeks}
    chosen = methods if a.method == "all" else {a.method: methods[a.method]}
    print(f"European {'put' if a.put else 'call'} under GBM, {a.paths:,} paths")
    print(f"{'greek':<7}{'exact':>12}" + "".join(f"{m:>26}" for m in chosen))
    results = {m: fn(model, payoff, a.T, 1, a.paths, seed=a.seed) for m, fn in chosen.items()}
    for g, ref in exact.items():
        cells = "".join(
            f"{results[m][g].value:>12.5f} ± {results[m][g].std_error:<11.2e}" for m in chosen
        )
        print(f"{g:<7}{ref:>12.5f}{cells}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mc-engine", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--S0", type=float, default=100.0)
        sp.add_argument("--K", type=float, default=100.0)
        sp.add_argument("--T", type=float, default=1.0)
        sp.add_argument("--r", type=float, default=0.05)
        sp.add_argument("--q", type=float, default=0.0, help="dividend yield")
        sp.add_argument("--sigma", type=float, default=0.2, help="GBM / Merton diffusive vol")
        sp.add_argument("--put", action="store_true", help="put instead of call")
        sp.add_argument("--paths", type=int, default=100_000)
        sp.add_argument("--seed", type=int, default=0)

    pr = sub.add_parser("price", help="price an option by Monte Carlo")
    common(pr)
    pr.add_argument("--model", choices=("gbm", "heston", "merton"), default="gbm")
    pr.add_argument("--payoff", choices=PAYOFFS, default="european")
    pr.add_argument(
        "--steps",
        type=int,
        default=None,
        help="time steps / fixings / exercise dates (default 1, or 50 if path-dependent or Heston)",
    )
    pr.add_argument("--barrier", type=float, default=None)
    pr.add_argument("--antithetic", action="store_true")
    pr.add_argument("--qmc", action="store_true", help="randomised Sobol QMC")
    h = pr.add_argument_group("Heston")
    h.add_argument("--v0", type=float, default=0.04)
    h.add_argument("--kappa", type=float, default=1.5)
    h.add_argument("--theta", type=float, default=0.04)
    h.add_argument("--xi", type=float, default=0.6, help="vol of vol")
    h.add_argument("--rho", type=float, default=-0.7)
    h.add_argument("--scheme", choices=("qe", "euler"), default="qe")
    h.add_argument("--backend", choices=("numpy", "numba"), default="numpy")
    m = pr.add_argument_group("Merton")
    m.add_argument("--lam", type=float, default=0.5, help="jump intensity")
    m.add_argument("--mu-j", dest="mu_j", type=float, default=-0.1)
    m.add_argument("--sigma-j", dest="sigma_j", type=float, default=0.15)
    am = pr.add_argument_group("American (Longstaff-Schwartz)")
    am.add_argument("--basis", choices=("laguerre", "monomial", "chebyshev"), default="laguerre")
    am.add_argument("--degree", type=int, default=3)
    pr.set_defaults(func=cmd_price)

    gr = sub.add_parser("greeks", help="European Greeks under GBM by FD / pathwise / LR")
    common(gr)
    gr.add_argument("--method", choices=("all", "fd", "pathwise", "lr"), default="all")
    gr.set_defaults(func=cmd_greeks)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.command == "price" and a.steps is None:
        needs_grid = a.payoff != "european" and a.payoff != "digital" or a.model == "heston"
        a.steps = 50 if needs_grid else 1
    return a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
