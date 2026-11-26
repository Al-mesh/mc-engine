"""Greeks accuracy: FD (CRN) vs pathwise vs likelihood ratio, and where pathwise fails.

Panels:
1. European call: standard error of each method for each Greek at equal N (all are
   checked against Black-Scholes; the table reports the error in SE units).
2. Digital delta: pathwise is identically 0; LR and FD recover the true value; a smoothed
   (call-spread) payoff makes pathwise usable at the cost of an O(eps^2) bias.
3. Up-and-out call delta: pathwise is badly biased; LR and FD agree.

Usage: python benchmarks/greeks.py [--log2n 18] [--out docs/results]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mc_engine import GBM  # noqa: E402
from mc_engine.analytic import (  # noqa: E402
    bs_delta,
    bs_digital_delta,
    bs_gamma,
    bs_rho,
    bs_vega,
)
from mc_engine.greeks import (  # noqa: E402
    GREEKS,
    fd_greeks,
    likelihood_ratio_greeks,
    pathwise_greeks,
)
from mc_engine.payoffs import Digital, European, SmoothedDigital, UpAndOutCall  # noqa: E402

S0, K, T, R, SIG, Q = 100.0, 105.0, 1.0, 0.05, 0.2, 0.01
MODEL = GBM(S0=S0, r=R, sigma=SIG, q=Q)
METHODS = {
    "FD + CRN": fd_greeks,
    "Pathwise": pathwise_greeks,
    "Likelihood ratio": likelihood_ratio_greeks,
}
COLORS = {"FD + CRN": "#7d3c98", "Pathwise": "#2471a3", "Likelihood ratio": "#c0392b"}


def european_table(n: int) -> pd.DataFrame:
    a = (S0, K, T, R, SIG, Q)
    exact = {
        "delta": float(bs_delta(*a)),
        "gamma": float(bs_gamma(*a)),
        "vega": float(bs_vega(*a)),
        "rho": float(bs_rho(*a)),
    }
    rows = []
    for name, fn in METHODS.items():
        t0 = time.perf_counter()
        est = fn(MODEL, European(K), T, 1, n, seed=1)
        elapsed = time.perf_counter() - t0
        for g in GREEKS:
            e = est[g]
            rows.append(
                dict(
                    method=name,
                    greek=g,
                    estimate=e.value,
                    std_error=e.std_error,
                    exact=exact[g],
                    error_in_se=(e.value - exact[g]) / e.std_error,
                    runtime_s=elapsed,
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2n", type=int, default=18)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 2**args.log2n

    eu = european_table(n)
    eu.to_csv(args.out / "greeks_european.csv", index=False)
    lines = [
        f"European call S0={S0:g} K={K:g} T={T:g} r={R} sigma={SIG} q={Q}, N = 2^{args.log2n}",
        "",
        "| Greek | Exact | FD + CRN | Pathwise | Likelihood ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in GREEKS:
        sub = eu[eu.greek == g].set_index("method")
        cells = [
            f"{sub.loc[m, 'estimate']:.4f} ± {sub.loc[m, 'std_error']:.1e} "
            f"({sub.loc[m, 'error_in_se']:+.1f}σ)"
            for m in METHODS
        ]
        lines.append(f"| {g} | {sub.exact.iloc[0]:.4f} | " + " | ".join(cells) + " |")
    print("\n".join(lines))

    # Digital delta.
    true_dd = float(bs_digital_delta(S0, K, T, R, SIG, Q))
    dig = {
        "Pathwise": pathwise_greeks(MODEL, Digital(K), T, 1, n, seed=2)["delta"],
        "Likelihood ratio": likelihood_ratio_greeks(MODEL, Digital(K), T, 1, n, seed=2)["delta"],
        "FD + CRN": fd_greeks(MODEL, Digital(K), T, 1, n, seed=2)["delta"],
    }
    eps_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    smooth = [
        pathwise_greeks(MODEL, SmoothedDigital(K, e), T, 1, n, seed=3)["delta"] for e in eps_grid
    ]

    # Barrier delta.
    bm = GBM(S0=100, r=0.05, sigma=0.2)
    bp = UpAndOutCall(100, 130)
    bar = {
        "Pathwise": pathwise_greeks(bm, bp, 1.0, 50, n, seed=4)["delta"],
        "Likelihood ratio": likelihood_ratio_greeks(bm, bp, 1.0, 50, n, seed=5)["delta"],
        "FD + CRN": fd_greeks(bm, bp, 1.0, 50, n, seed=6, bump_spot=0.5)["delta"],
    }
    extra = [
        "",
        f"Digital delta (exact {true_dd:.5f}): "
        + ", ".join(f"{k} {v.value:.5f} ± {v.std_error:.1e}" for k, v in dig.items()),
        "Smoothed-digital pathwise delta: "
        + ", ".join(
            f"eps={e:g}: {s.value:.5f} ± {s.std_error:.1e}"
            for e, s in zip(eps_grid, smooth, strict=True)
        ),
        "Up-and-out (B=130, 50 dates) delta: "
        + ", ".join(f"{k} {v.value:.4f} ± {v.std_error:.1e}" for k, v in bar.items()),
    ]
    print("\n".join(extra))
    (args.out / "greeks.md").write_text("\n".join(lines + extra) + "\n", encoding="utf-8")

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    width = 0.27
    x = np.arange(len(GREEKS))
    for i, m in enumerate(METHODS):
        sub = eu[eu.method == m].set_index("greek").loc[list(GREEKS)]
        rel = sub.std_error / sub.exact.abs()
        ax.bar(x + (i - 1) * width, rel, width, label=m, color=COLORS[m])
    ax.set_xticks(x, GREEKS)
    ax.set_yscale("log")
    ax.set_ylabel("relative standard error (SE / |exact|)")
    worst = eu.error_in_se.abs().max()
    ax.set_title(
        f"European call: SE at equal N = 2^{args.log2n}\n"
        f"(largest deviation from Black-Scholes: {worst:.1f} SE)",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(true_dd, color="black", lw=1, label="exact")
    for i, (m, e) in enumerate(dig.items()):
        ax.errorbar(i, e.value, yerr=2 * e.std_error, fmt="o", color=COLORS[m], capsize=4)
    ax.set_xticks(range(len(dig)), list(dig), fontsize=8)
    ax2 = ax.inset_axes([0.52, 0.12, 0.45, 0.45])
    ax2.errorbar(
        eps_grid,
        [s.value for s in smooth],
        yerr=[2 * s.std_error for s in smooth],
        fmt="o-",
        color=COLORS["Pathwise"],
        capsize=2,
        ms=3,
    )
    ax2.axhline(true_dd, color="black", lw=0.8)
    ax2.set_xscale("log", base=2)
    ax2.set_title("pathwise on call spread vs ε", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax.set_title("Digital delta: pathwise = 0 a.s.", fontsize=10)
    ax.set_ylabel("delta (±2 SE)")

    ax = axes[2]
    for i, (m, e) in enumerate(bar.items()):
        ax.errorbar(i, e.value, yerr=2 * e.std_error, fmt="o", color=COLORS[m], capsize=4)
    ax.set_xticks(range(len(bar)), list(bar), fontsize=8)
    ax.set_title(
        "Up-and-out call delta (B = 130, 50 dates)\npathwise misses the knock-out term", fontsize=10
    )
    ax.set_ylabel("delta (±2 SE)")
    fig.tight_layout()
    fig.savefig(args.out / "greeks.png", dpi=130)
    print(f"\nwrote {args.out}/greeks.(csv|md|png)")


if __name__ == "__main__":
    main()
