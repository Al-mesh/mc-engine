"""Portfolio VaR/ES report and a 13-year VaR backtest on real equity data.

Part 1: a five-stock portfolio (plus a short call overlay) with parameters estimated from
the full history: 1-day and 10-day VaR and ES at 95% and 99%, Gaussian vs Student-t,
compared with the sqrt(10) rule.

Part 2: a rolling one-day 95%/99% VaR backtest of an equally weighted portfolio
(500-day estimation window, no look-ahead) for five methods: historical simulation,
Gaussian and Student-t MC each with window or EWMA volatility. Kupiec and Christoffersen
tests per method.

Prices come from data/ (downloaded once with yfinance, then cached and committed).

Usage: python benchmarks/var_backtest.py [--sims 20000] [--out docs/results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, skew  # noqa: E402

from mc_engine.risk import (  # noqa: E402
    OptionPosition,
    Portfolio,
    backtest_summary,
    risk_report,
    rolling_var_backtest,
)
from mc_engine.risk.backtest import _nu_from_kurtosis, ewma_covariances  # noqa: E402
from mc_engine.risk.data import DEFAULT_TICKERS, load_prices, log_returns  # noqa: E402

METHODS = [
    ("historical", "window", "Historical simulation"),
    ("normal", "window", "Gaussian, 500d window"),
    ("t", "window", "Student-t, 500d window"),
    ("normal", "ewma", "Gaussian, EWMA"),
    ("t", "ewma", "Student-t, EWMA"),
]


def portfolio_report(rets: pd.DataFrame, prices: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    vol = rets.std().to_numpy() * np.sqrt(252)
    corr = rets.corr().to_numpy()
    spot = prices.iloc[-1].to_numpy()
    shares = 200_000 / spot  # $200k in each of 5 stocks = $1m
    nu = _nu_from_kurtosis((rets / rets.std()).to_numpy())
    frames = []
    for label, opts in (
        ("stocks", ()),
        # Overlay: short 1-month ATM calls on the first stock, 2000 shares' worth.
        (
            "stocks + short calls",
            (OptionPosition(0, float(spot[0]), 21 / 252 + 10 / 252, True, -2000),),
        ),
    ):
        p = Portfolio(spot=spot, vol=vol, shares=shares, options=opts, r=0.0)
        for dist, nu_ in (("normal", None), ("t", nu)):
            df = risk_report(
                p, np.zeros(len(spot)), corr, dist=dist, nu=nu_, n_sims=n_sims * 10, seed=42
            )
            df.insert(0, "dist", dist if dist == "normal" else f"t (nu={nu:.1f})")
            df.insert(0, "portfolio", label)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    prices = load_prices(cache_dir=args.data)
    rets = log_returns(prices)
    print(
        f"{len(rets)} daily returns, {rets.index[0].date()} to {rets.index[-1].date()}, "
        f"tickers {', '.join(DEFAULT_TICKERS)}"
    )

    rep = portfolio_report(rets, prices, args.sims)
    rep.to_csv(args.out / "var_report.csv", index=False)
    print(rep.round({"VaR": 0, "ES": 0, "sqrt_time_var": 0}).to_string(index=False))

    w = np.ones(rets.shape[1]) / rets.shape[1]
    summaries, series = [], {}
    for method, vm, label in METHODS:
        bt = rolling_var_backtest(rets, w, method, vm, window=500, n_sims=args.sims, seed=1)
        s = backtest_summary(bt)
        s.insert(0, "method", label)
        summaries.append(s)
        series[label] = bt
        print(s.round(4).to_string(index=False))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.out / "var_backtest.csv", index=False)

    lines = [
        "| Method | α | Exceedances | Expected | Rate | Kupiec p | Christoffersen p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.method} | {r.alpha:.0%} | {r.exceedances} | {r.expected:.1f} | "
            f"{r.rate:.2%} | {r.kupiec_p:.3g} | {r.christoffersen_p:.3g} |"
        )

    # Diagnostic for the 99% failures: the shape of EWMA-devolatilised portfolio residuals
    # (out-of-sample period only). Symmetric t cannot match a skewed, sharply peaked tail.
    covs = ewma_covariances(rets.to_numpy(), 0.94, np.cov(rets.to_numpy()[:500].T))
    port = np.expm1(rets.to_numpy()) @ w
    z = (port / np.sqrt(np.einsum("i,tij,j->t", w, covs, w)))[500:]
    diag = (
        f"EWMA-standardised portfolio residuals: 1% quantile {np.quantile(z, 0.01):.3f} "
        f"(normal {norm.ppf(0.01):.3f}), 99% quantile {np.quantile(z, 0.99):.3f}, "
        f"5% quantile {np.quantile(z, 0.05):.3f} (normal {norm.ppf(0.05):.3f}), "
        f"skewness {skew(z):.2f}"
    )
    print(diag)
    lines += ["", diag]
    (args.out / "var_backtest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: realised returns vs 99% VaR for the window-Gaussian and EWMA-t models.
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, label in zip(axes, ["Gaussian, 500d window", "Student-t, EWMA"], strict=True):
        bt = series[label]
        ax.plot(bt.index, bt.realized, lw=0.4, color="grey", label="realised daily return")
        ax.plot(bt.index, -bt["var_0.99"], lw=0.9, color="#c0392b", label="−VaR 99%")
        ex = bt[bt["exceed_0.99"]]
        ax.scatter(
            ex.index, ex.realized, s=10, color="black", zorder=3, label=f"exceedances ({len(ex)})"
        )
        row = summary[(summary.method == label) & (summary.alpha == 0.99)].iloc[0]
        ax.set_title(
            f"{label}: Kupiec p = {row.kupiec_p:.3g}, "
            f"Christoffersen p = {row.christoffersen_p:.3g}",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="lower left")
        ax.set_ylabel("portfolio return")
    fig.tight_layout()
    fig.savefig(args.out / "var_backtest.png", dpi=130)
    print(f"\nwrote {args.out}/var_*.(csv|md|png)")


if __name__ == "__main__":
    main()
