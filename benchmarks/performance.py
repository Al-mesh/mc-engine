"""Performance: numpy vs numba vs multiprocessing, paths/sec and scaling with N.

All timings are wall-clock, best of ``--repeats`` runs, after a warm-up call (so numba JIT
compilation and first-touch effects are excluded; the one-off JIT cost is reported
separately). Machine details are printed and saved with the results, because absolute
numbers only mean something next to the hardware they were measured on.

Usage: python benchmarks/performance.py [--repeats 3] [--out docs/results]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mc_engine import GBM, Heston, mc_price  # noqa: E402
from mc_engine.parallel import parallel_mc_price  # noqa: E402
from mc_engine.payoffs import ArithmeticAsian, European  # noqa: E402
from mc_engine.rng import PseudoRandomSource  # noqa: E402

HESTON = Heston(S0=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7, r=0.03)
STEPS = 100


def best_of(fn, repeats: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def throughput_table(repeats: int) -> pd.DataFrame:
    n = 2**18
    rows = []
    for scheme in ("qe", "euler"):
        base = dataclasses.replace(HESTON, scheme=scheme)
        fast = dataclasses.replace(base, backend="numba")
        draw = PseudoRandomSource.uniforms if scheme == "qe" else PseudoRandomSource.normals
        rng_only = best_of(lambda draw=draw: draw(PseudoRandomSource(0), n, STEPS, 2), repeats)
        cfg = [
            ("numpy", base, None),
            ("numba, 1 thread", fast, 1),
            ("numba, all threads", fast, numba.config.NUMBA_NUM_THREADS),
        ]
        for label, model, threads in cfg:
            if threads is not None:
                numba.set_num_threads(threads)
            t = best_of(lambda m=model: mc_price(m, European(100), 1.0, STEPS, n, rng=1), repeats)
            rows.append(
                dict(
                    model=f"Heston {scheme.upper()}",
                    backend=label,
                    n_paths=n,
                    seconds=t,
                    paths_per_sec=n / t,
                    rng_share=rng_only / t,
                )
            )
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    df = pd.DataFrame(rows)
    df["speedup_vs_numpy"] = df.groupby("model").seconds.transform(lambda s: s.iloc[0] / s)
    return df


def scaling(repeats: int) -> pd.DataFrame:
    rows = []
    fast = dataclasses.replace(HESTON, backend="numba")
    gbm = GBM(100, 0.05, 0.2)
    for k in range(10, 21, 2):
        n = 2**k
        for label, fn in (
            ("Heston QE numpy", lambda n=n: mc_price(HESTON, European(100), 1, STEPS, n, rng=1)),
            ("Heston QE numba", lambda n=n: mc_price(fast, European(100), 1, STEPS, n, rng=1)),
            (
                "GBM European numpy (1 step)",
                lambda n=n: mc_price(gbm, European(100), 1, 1, n, rng=1),
            ),
        ):
            t = best_of(fn, repeats)
            rows.append(dict(config=label, n_paths=n, seconds=t, paths_per_sec=n / t))
    return pd.DataFrame(rows)


def multiprocessing_study() -> pd.DataFrame:
    gbm = GBM(100, 0.05, 0.2)
    n = 2**21
    rows = []
    serial = best_of(lambda: mc_price(gbm, ArithmeticAsian(100), 1, 252, n, rng=1), 1)
    rows.append(dict(workers="serial mc_price", seconds=serial))
    for w in (1, 2, 4, 8, 16):
        if w > (os.cpu_count() or 1):
            break
        t0 = time.perf_counter()
        parallel_mc_price(gbm, ArithmeticAsian(100), 1, 252, n, seed=1, n_chunks=16, n_workers=w)
        rows.append(dict(workers=w, seconds=time.perf_counter() - t0))
    # Fixed cost of spawning workers (Windows has no fork: each re-imports numpy/scipy).
    t0 = time.perf_counter()
    parallel_mc_price(gbm, ArithmeticAsian(100), 1, 252, 16 * 64, seed=1, n_chunks=16, n_workers=8)
    rows.append(dict(workers="8 (near-empty job: pure overhead)", seconds=time.perf_counter() - t0))
    df = pd.DataFrame(rows)
    df["speedup_vs_serial"] = serial / df.seconds
    df["paths_per_sec"] = n / df.seconds
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    machine = (
        f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical cores, "
        f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
        f"numpy {np.__version__}, numba {numba.__version__}"
    )
    print(machine)

    # One-off JIT cost, measured in a fresh state of this process (cache may make it small).
    t0 = time.perf_counter()
    dataclasses.replace(HESTON, backend="numba").simulate(10, [0, 0.5, 1], 0)
    jit = time.perf_counter() - t0
    print(f"numba first call (compile or load from cache): {jit:.2f}s")

    thr = throughput_table(args.repeats)
    sc = scaling(args.repeats)
    mp = multiprocessing_study()
    thr.to_csv(args.out / "perf_throughput.csv", index=False)
    sc.to_csv(args.out / "perf_scaling.csv", index=False)
    mp.to_csv(args.out / "perf_multiprocessing.csv", index=False)
    print(thr.round(3).to_string(index=False))
    print(sc.round(4).to_string(index=False))
    print(mp.round(3).to_string(index=False))

    md = [
        f"Machine: {machine}",
        "",
        f"Heston, {STEPS} steps, European call, N = 2^18 (best of {args.repeats}):",
        "",
        "| Model | Backend | Paths/sec | Speed-up | RNG share of time |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in thr.iterrows():
        md.append(
            f"| {r.model} | {r.backend} | {r.paths_per_sec:,.0f} | "
            f"{r.speedup_vs_numpy:.1f}x | {r.rng_share:.0%} |"
        )
    md += [
        "",
        "Arithmetic Asian (252 steps), GBM, N = 2^21, multiprocessing (16 chunks):",
        "",
        "| Workers | Seconds | Paths/sec | Speed-up |",
        "|---|---:|---:|---:|",
    ]
    for _, r in mp.iterrows():
        if "overhead" in str(r.workers):  # throughput is meaningless for the empty job
            md.append(f"| {r.workers} | {r.seconds:.2f} | – | – |")
            continue
        md.append(
            f"| {r.workers} | {r.seconds:.2f} | {r.paths_per_sec:,.0f} | "
            f"{r.speedup_vs_serial:.1f}x |"
        )
    md += ["", f"numba first-call cost (compile or cache load): {jit:.2f}s"]
    (args.out / "performance.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for cfg, sub in sc.groupby("config"):
        slope = np.polyfit(np.log(sub.n_paths), np.log(sub.seconds), 1)[0]
        ax.loglog(sub.n_paths, sub.seconds, "o-", label=f"{cfg} (slope {slope:.2f})")
    ax.set_xlabel("paths N")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Scaling with N: linear (slope 1) once fixed overheads are amortised", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    sub = mp[mp.workers.apply(lambda w: isinstance(w, int))]
    ax.plot(sub.workers.astype(int), sub.speedup_vs_serial, "o-", label="measured")
    ax.plot(sub.workers.astype(int), sub.workers.astype(int), "k--", lw=0.8, label="ideal")
    ax.set_xlabel("worker processes")
    ax.set_ylabel("speed-up vs serial")
    ax.set_title("Multiprocessing, GBM Asian (252 steps), N = 2^21", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "performance.png", dpi=130)
    print(f"\nwrote {args.out}/perf_*.csv, performance.(md|png)")


if __name__ == "__main__":
    main()
