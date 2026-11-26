"""Benchmark every variance-reduction technique against plain MC on four problems.

For each (problem, technique) the same total number of paths is used. Runtimes are the
minimum over ``--repeats`` runs (the minimum is the least noisy estimate of the
cost of the computation itself; noise from the OS only ever adds time). Standard errors
come from the first run (re-runs use the same seed and give identical numbers).

Usage: python benchmarks/variance_reduction.py [--log2n 20] [--repeats 3] [--out docs/results]
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from mc_engine import GBM, Heston, mc_price  # noqa: E402
from mc_engine.analytic import bs_price, heston_price  # noqa: E402
from mc_engine.payoffs import ArithmeticAsian, European, UpAndOutCall  # noqa: E402
from mc_engine.pricer import PriceResult  # noqa: E402
from mc_engine.variance_reduction import (  # noqa: E402
    control_variate_price,
    efficiency_table,
    geometric_asian_control,
    rqmc_price,
    terminal_spot_control,
)

GBM_M = GBM(S0=100, r=0.05, sigma=0.2)
HESTON_M = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03)
R = 16  # RQMC replications


def timed(fn: Callable[[], PriceResult], repeats: int) -> PriceResult:
    first = fn()
    best = min([first.runtime_s] + [fn().runtime_s for _ in range(repeats - 1)])
    return dataclasses.replace(first, runtime_s=best)


def problems(n: int):
    """(name, reference price or None, {technique: thunk})."""
    call = European(100)
    spot_ctrl = terminal_spot_control(GBM_M, 1.0)
    yield (
        "European call, GBM (1 step)",
        float(bs_price(100, 100, 1.0, 0.05, 0.2)),
        {
            "plain": lambda: mc_price(GBM_M, call, 1.0, 1, n, rng=1),
            "antithetic": lambda: mc_price(GBM_M, call, 1.0, 1, n, rng=1, antithetic=True),
            "control (S_T)": lambda: control_variate_price(
                GBM_M, call, *spot_ctrl, 1.0, 1, n, rng=1
            ),
            "RQMC": lambda: rqmc_price(GBM_M, call, 1.0, 1, n // R, R, seed=1),
        },
    )
    asian = ArithmeticAsian(100)
    geo_ctrl = geometric_asian_control(GBM_M, 100, 1.0, 12)
    yield (
        "Arithmetic Asian, GBM (12 fixings)",
        None,
        {
            "plain": lambda: mc_price(GBM_M, asian, 1.0, 12, n, rng=2),
            "antithetic": lambda: mc_price(GBM_M, asian, 1.0, 12, n, rng=2, antithetic=True),
            "control (geometric Asian)": lambda: control_variate_price(
                GBM_M, asian, *geo_ctrl, 1.0, 12, n, rng=2
            ),
            "RQMC (forward)": lambda: rqmc_price(
                GBM_M, asian, 1.0, 12, n // R, R, seed=2, brownian_bridge=False
            ),
            "RQMC + Brownian bridge": lambda: rqmc_price(GBM_M, asian, 1.0, 12, n // R, R, seed=2),
        },
    )
    barrier = UpAndOutCall(100, 130)
    yield (
        "Up-and-out call B=130, GBM (50 dates)",
        None,
        {
            "plain": lambda: mc_price(GBM_M, barrier, 1.0, 50, n, rng=3),
            "antithetic": lambda: mc_price(GBM_M, barrier, 1.0, 50, n, rng=3, antithetic=True),
            "RQMC + Brownian bridge": lambda: rqmc_price(
                GBM_M, barrier, 1.0, 50, n // R, R, seed=3
            ),
        },
    )
    nh = n // 4  # Heston is ~50x more work per path; keep the benchmark quick
    h_ctrl = terminal_spot_control(HESTON_M, 1.0)
    yield (
        "European call, Heston QE (50 steps)",
        heston_price(HESTON_M, 100, 1.0),
        {
            "plain": lambda: mc_price(HESTON_M, call, 1.0, 50, nh, rng=4),
            "antithetic": lambda: mc_price(HESTON_M, call, 1.0, 50, nh, rng=4, antithetic=True),
            "control (S_T)": lambda: control_variate_price(
                HESTON_M, call, *h_ctrl, 1.0, 50, nh, rng=4
            ),
            "RQMC": lambda: rqmc_price(
                HESTON_M, call, 1.0, 50, nh // R, R, seed=4, brownian_bridge=False
            ),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2n", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 2**args.log2n

    frames = []
    for name, ref, techniques in problems(n):
        results = {k: timed(fn, args.repeats) for k, fn in techniques.items()}
        df = efficiency_table(results)
        df.insert(0, "problem", name)
        df["reference"] = ref if ref is not None else np.nan
        frames.append(df)
        print(f"\n{name}" + (f"   (reference {ref:.6f})" if ref is not None else ""))
        print(
            df[["price", "std_error", "n_paths", "runtime_s", "vr_ratio", "efficiency_gain"]]
            .round(6)
            .to_string()
        )
    table = pd.concat(frames).rename_axis("technique").reset_index()
    table.to_csv(args.out / "variance_reduction.csv", index=False)

    # Markdown table for the README.
    lines = [
        "| Problem | Technique | Price | SE | VR ratio | Efficiency gain |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row.problem} | {row.technique} | {row.price:.4f} | {row.std_error:.2e} | "
            f"{row.vr_ratio:,.1f} | {row.efficiency_gain:,.1f} |"
        )
    (args.out / "variance_reduction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Bar chart of efficiency gains, log scale.
    probs = table.problem.unique()
    fig, axes = plt.subplots(1, len(probs), figsize=(16, 4.2))
    for ax, p in zip(axes, probs, strict=True):
        sub = table[table.problem == p]
        bars = ax.barh(sub.technique, sub.efficiency_gain, color="#2471a3")
        ax.bar_label(
            bars, labels=[f"{v:,.1f}x" for v in sub.efficiency_gain], fontsize=8, padding=2
        )
        ax.set_xscale("log")
        # Minor-tick labels on short log axes collide; major decades plus bar labels suffice.
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(0.5, sub.efficiency_gain.max() * 8)
        ax.axvline(1, color="grey", lw=0.8)
        ax.set_title(p, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("efficiency gain vs plain MC (log)")
    fig.suptitle(f"Variance reduction: work-normalised efficiency gain (N = 2^{args.log2n})")
    fig.tight_layout()
    fig.savefig(args.out / "variance_reduction.png", dpi=130)
    print(f"\nwrote {args.out}/variance_reduction.(csv|md|png)")


if __name__ == "__main__":
    main()
