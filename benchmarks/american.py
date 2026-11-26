"""Longstaff-Schwartz vs binomial trees, plus basis and path-count sensitivity.

1. Grid in the style of Longstaff-Schwartz (2001) Table 1 (K = 40, r = 6%): European
   (Black-Scholes), American and 50-dates-per-year Bermudan (CRR, 5000 steps/yr), and LSM
   (out-of-sample and in-sample).
2. Basis / degree study at S0 = 36, sigma = 0.2, T = 1, all on the same training and
   pricing paths (common random numbers), so differences between rows are sharp.
3. Training-set size: in-sample estimates come down from above, out-of-sample estimates
   rise from below; both meet the tree value. Averaged over independent seeds.

Usage: python benchmarks/american.py [--paths 100000] [--seeds 5] [--out docs/results]
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
from mc_engine.american import lsm_price  # noqa: E402
from mc_engine.analytic import bs_price  # noqa: E402
from mc_engine.analytic.binomial import crr_price  # noqa: E402

K, R = 40.0, 0.06


def table(n_paths: int) -> pd.DataFrame:
    rows = []
    for S0 in (36.0, 38.0, 40.0, 42.0, 44.0):
        for sigma in (0.2, 0.4):
            for T in (1.0, 2.0):
                m = int(50 * T)
                berm = crr_price(
                    S0, K, T, R, sigma, n_steps=100 * m, exercise="bermudan", n_exercise_dates=m
                )
                amer = crr_price(S0, K, T, R, sigma, n_steps=int(5000 * T))
                euro = float(bs_price(S0, K, T, R, sigma, call=False))
                res = lsm_price(
                    GBM(S0, R, sigma), K, T, m, n_paths, seed=int(S0 * 100 + sigma * 10 + T)
                )
                rows.append(
                    dict(
                        S0=S0,
                        sigma=sigma,
                        T=T,
                        european=euro,
                        crr_american=amer,
                        crr_bermudan=berm,
                        lsm=res.price,
                        lsm_se=res.std_error,
                        lsm_in_sample=res.in_sample_price,
                        z_vs_bermudan=(res.price - berm) / res.std_error,
                        early_exercise_premium=berm - euro,
                    )
                )
    return pd.DataFrame(rows)


def basis_study(n_paths: int) -> pd.DataFrame:
    g = GBM(36.0, R, 0.2)
    rows = []
    for basis in ("monomial", "laguerre"):
        for deg in range(0, 9):
            r = lsm_price(g, K, 1.0, 50, n_paths, basis=basis, degree=deg, seed=11)
            rows.append(
                dict(
                    basis=basis,
                    degree=deg,
                    oos=r.price,
                    se=r.std_error,
                    in_sample=r.in_sample_price,
                )
            )
    return pd.DataFrame(rows)


def paths_study(n_seeds: int) -> pd.DataFrame:
    g = GBM(36.0, R, 0.2)
    rows = []
    for k in range(8, 18, 1):
        n_train = 2**k
        oos, ins = [], []
        for s in range(n_seeds):
            r = lsm_price(g, K, 1.0, 50, 200_000, n_train=n_train, seed=1000 + s)
            oos.append(r.price)
            ins.append(r.in_sample_price)
        rows.append(
            dict(
                n_train=n_train,
                oos_mean=np.mean(oos),
                oos_se=np.std(oos, ddof=1) / np.sqrt(n_seeds),
                in_mean=np.mean(ins),
                in_se=np.std(ins, ddof=1) / np.sqrt(n_seeds),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tab = table(args.paths)
    tab.to_csv(args.out / "american_table.csv", index=False)
    md = [
        "| S0 | σ | T | European (BS) | American (CRR) | Bermudan-50/yr (CRR) | "
        "LSM out-of-sample | LSM in-sample | LSM − Bermudan (SE) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in tab.iterrows():
        md.append(
            # r["T"], not r.T: on a pandas Series .T is the transpose.
            f"| {r.S0:g} | {r.sigma:g} | {r['T']:g} | {r.european:.3f} | {r.crr_american:.3f} | "
            f"{r.crr_bermudan:.3f} | {r.lsm:.3f} ± {r.lsm_se:.3f} | {r.lsm_in_sample:.3f} | "
            f"{r.z_vs_bermudan:+.1f} |"
        )
    (args.out / "american_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    z = tab.z_vs_bermudan
    print(
        f"\nLSM - Bermudan in SE units: mean {z.mean():+.2f}, sd {z.std():.2f}, "
        f"max |z| {z.abs().max():.2f} over {len(z)} cases"
    )

    bs = basis_study(args.paths)
    bs.to_csv(args.out / "american_basis.csv", index=False)
    ps = paths_study(args.seeds)
    ps.to_csv(args.out / "american_paths.csv", index=False)
    print(bs.round(4).to_string(index=False))
    print(ps.round(4).to_string(index=False))

    berm = tab[(tab.S0 == 36) & (tab.sigma == 0.2) & (tab["T"] == 1)].crr_bermudan.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    colors = {"monomial": "#2471a3", "laguerre": "#c0392b"}
    for basis, sub in bs.groupby("basis"):
        off = -0.1 if basis == "monomial" else 0.1
        ax.errorbar(
            sub.degree + off,
            sub.oos,
            yerr=2 * sub.se,
            fmt="o-",
            capsize=3,
            color=colors[basis],
            label=f"{basis}: out-of-sample",
        )
        ax.plot(
            sub.degree + off,
            sub.in_sample,
            "x--",
            color=colors[basis],
            alpha=0.6,
            label=f"{basis}: in-sample",
        )
    ax.axhline(berm, color="black", lw=1, label=f"CRR Bermudan {berm:.4f}")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("price (±2 SE)")
    ax.set_title(
        "Basis choice (S0=36, σ=0.2, T=1)\nbars: absolute ±2 SE; all points share paths, "
        "so differences between them are far more precise",
        fontsize=9,
    )
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.errorbar(
        ps.n_train,
        ps.in_mean,
        yerr=2 * ps.in_se,
        fmt="x--",
        color="#c0392b",
        capsize=3,
        label="in-sample (foresight bias ↑)",
    )
    ax.errorbar(
        ps.n_train,
        ps.oos_mean,
        yerr=2 * ps.oos_se,
        fmt="o-",
        color="#2471a3",
        capsize=3,
        label="out-of-sample (sub-optimal policy ↓)",
    )
    ax.axhline(berm, color="black", lw=1, label="CRR Bermudan")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training paths")
    ax.set_ylabel(f"price, mean of {args.seeds} seeds (±2 SE)")
    ax.set_title("Training-set size: the two estimates bracket the true value", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "american.png", dpi=130)
    print(f"\nwrote {args.out}/american*.(csv|md|png)")


if __name__ == "__main__":
    main()
