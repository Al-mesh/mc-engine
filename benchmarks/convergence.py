"""Log-log convergence of MC vs randomised QMC, with fitted slopes.

Two targets with exact prices:
* European call under GBM simulated on 16 steps (the payoff only depends on W_T, so with
  a Brownian bridge it is effectively one-dimensional: QMC's best case);
* geometric Asian with 16 fixings (a genuinely 16-dimensional integrand).

Usage: python benchmarks/convergence.py [--reps 64] [--out docs/results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mc_engine import GBM  # noqa: E402
from mc_engine.analytic import bs_price, geometric_asian_price  # noqa: E402
from mc_engine.payoffs import European, GeometricAsian  # noqa: E402
from mc_engine.pricer import time_grid  # noqa: E402
from mc_engine.variance_reduction import convergence_study, loglog_slope  # noqa: E402

MODEL = GBM(S0=100, r=0.05, sigma=0.2)
STEPS = 16
TARGETS = {
    "European call (16 steps)": (European(100), float(bs_price(100, 100, 1.0, 0.05, 0.2))),
    "Geometric Asian (16 fixings)": (
        GeometricAsian(100),
        geometric_asian_price(100, 100, 0.05, 0.2, time_grid(1.0, STEPS)[1:]),
    ),
}
METHODS = {"mc": "Plain MC", "qmc": "Sobol (forward)", "qmc_bb": "Sobol + Brownian bridge"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ns = [2**k for k in range(6, 17)]

    frames = []
    for tname, (payoff, exact) in TARGETS.items():
        for method in METHODS:
            df = convergence_study(MODEL, payoff, 1.0, STEPS, exact, ns, args.reps, method, seed=7)
            df["target"] = tname
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(args.out / "convergence.csv", index=False)

    slopes = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {"mc": "#c0392b", "qmc": "#7d3c98", "qmc_bb": "#2471a3"}
    for ax, tname in zip(axes, TARGETS, strict=True):
        for method, label in METHODS.items():
            sub = data[(data.target == tname) & (data.method == method)]
            s = loglog_slope(sub.n_paths.to_numpy(), sub.rmse.to_numpy())
            slopes.append({"target": tname, "method": label, "slope": s})
            ax.loglog(sub.n_paths, sub.rmse, "o-", color=colors[method], label=f"{label}: {s:.2f}")
        # Reference lines anchored at the first MC point.
        mc0 = data[(data.target == tname) & (data.method == "mc")].rmse.iloc[0]
        n = np.array(ns, dtype=float)
        ax.loglog(n, mc0 * (n / n[0]) ** -0.5, "k--", lw=0.8, label="N^-1/2")
        ax.loglog(n, mc0 * (n / n[0]) ** -1.0, "k:", lw=0.8, label="N^-1")
        ax.set_title(tname)
        ax.set_xlabel("paths per estimate N")
        ax.legend(fontsize=8, title="fitted slope", title_fontsize=8)
    axes[0].set_ylabel(f"RMSE over {args.reps} independent estimates")
    fig.suptitle("Convergence: plain MC vs randomised QMC")
    fig.tight_layout()
    fig.savefig(args.out / "convergence.png", dpi=130)
    sl = pd.DataFrame(slopes)
    sl.to_csv(args.out / "convergence_slopes.csv", index=False)
    print(sl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
