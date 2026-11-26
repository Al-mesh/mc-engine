"""Discretisation bias of full-truncation Euler vs Andersen QE for Heston.

Bias(dt) = E[MC price with step dt] - exact price (characteristic function). Each MC
estimate carries a standard error, so a bias is only *resolved* if it is several SEs from
zero; the plot shows +/- 2 SE bars to make that visible.

Test case: Andersen (2008) case I (kappa=0.5, theta=0.04, xi=1, rho=-0.9, v0=0.04, r=0,
T=10). 2*kappa*theta = 0.04 << xi^2 = 1, so the variance spends a lot of time near zero.
This is where Euler-type schemes are known to struggle.

Usage: python benchmarks/heston_bias.py [--paths N] [--out docs/results]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mc_engine import Heston, mc_price  # noqa: E402
from mc_engine.analytic import heston_price  # noqa: E402
from mc_engine.payoffs import European  # noqa: E402

CASE_I = Heston(S0=100, v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, r=0.0)
T = 10.0
STEPS_PER_YEAR = [1, 2, 4, 8, 16, 32]
STRIKES = [70, 100, 140]


def run(n_paths: int) -> pd.DataFrame:
    rows = []
    for K in STRIKES:
        exact = heston_price(CASE_I, K, T)
        for spy in STEPS_PER_YEAR:
            for scheme in ("euler", "qe"):
                model = dataclasses.replace(CASE_I, scheme=scheme)
                # Seed depends only on (K, spy), so both schemes see comparable noise levels;
                # they consume randomness differently, so this is not CRN across schemes.
                res = mc_price(model, European(K), T, int(spy * T), n_paths, rng=1000 * K + spy)
                rows.append(
                    dict(
                        K=K,
                        scheme=scheme,
                        dt=1 / spy,
                        exact=exact,
                        mc=res.price,
                        se=res.std_error,
                        bias=res.price - exact,
                        bias_in_se=(res.price - exact) / res.std_error,
                        runtime_s=res.runtime_s,
                    )
                )
                print(
                    f"K={K:>3} dt=1/{spy:<2} {scheme:>5}: bias={res.price - exact:+.4f} "
                    f"({(res.price - exact) / res.std_error:+.1f} se)  {res.runtime_s:.1f}s"
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(STRIKES), figsize=(13, 4), sharex=True)
    colors = {"euler": "#c0392b", "qe": "#2471a3"}
    labels = {"euler": "Full-truncation Euler", "qe": "Andersen QE (martingale-corrected)"}
    for ax, K in zip(axes, STRIKES, strict=True):
        sub = df[df.K == K]
        for scheme in ("euler", "qe"):
            s = sub[sub.scheme == scheme]
            ax.errorbar(
                s.dt,
                s.bias,
                yerr=2 * s.se,
                marker="o",
                capsize=3,
                color=colors[scheme],
                label=labels[scheme],
            )
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_title(f"K = {K} (exact {sub.exact.iloc[0]:.4f})")
        ax.set_xlabel("time step dt (years)")
    axes[0].set_ylabel("price bias  (MC - exact), ±2 SE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Heston discretisation bias, Andersen case I (T = 10, ξ = 1, ρ = -0.9)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = run(args.paths)
    df.to_csv(args.out / "heston_bias.csv", index=False)
    plot(df, args.out / "heston_bias.png")
    print(f"wrote {args.out / 'heston_bias.csv'} and heston_bias.png")


if __name__ == "__main__":
    main()
