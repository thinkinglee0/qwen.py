#!/usr/bin/env python3
"""
Concurrency-sweep visualizer for LLM serving benchmarks (vLLM / SGLang style).

Input: one row per max_concurrency level, with TTFT / TPOT / ITL / output throughput.
Output: a 6-panel figure + a derived-metrics table printed to stdout.

The point of the panel set is not "plot the numbers" but to separate three
regimes that a raw latency-vs-batch plot hides:
  (1) linear scaling      throughput x2 per doubling, TPOT flat
  (2) sublinear scaling   throughput gain > TPOT cost, still worth it
  (3) past the knee       TPOT cost > throughput gain, pure loss

Usage: python bench_viz.py [--out fig.png]
"""

import argparse

import numpy as np
import matplotlib
from pathlib import Path
from typing import Any
import orjson

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Hardware / model constants, used only for the roofline annotations.
# Set to None to disable that part of the report.
PEAK_HBM_BW_BPS = 1008e9  # RTX 4090, 24 GB GDDR6X
WEIGHT_BYTES = 15.2e9  # Qwen2.5-7B-Instruct, fp16/bf16
PEAK_DENSE_FLOPS = 165e12  # fp16 with fp32 accumulate
MODEL_PARAMS = 7.6e9

# Editorial Warm palette, kept consistent across panels.
BG = "#faf6ef"
INK = "#2f2a24"
GRID = "#d9d0c1"
C_TP = "#8c5a2b"  # throughput
C_TPOT = "#3f5f6b"  # per-token latency
C_TTFT = "#a4443a"  # time to first token
C_IDEAL = "#9a9284"  # reference / ideal lines
C_ITL = "#6b7f52"


def step_ratio(y):
    """Growth factor between consecutive sweep points (dimensionless, n-1 values)."""
    return y[1:] / y[:-1]


def interval_x(x):
    """Geometric midpoint of each batch pair: the correct x for an interval quantity
    on a log2 axis. Plotting a ratio at the right endpoint shifts the knee by a
    full doubling."""
    return np.sqrt(x[1:] * x[:-1])


def report(batch, ttft, tpot, throughput):
    """Derived per-point metrics. Every column answers a scheduling question."""
    per_req = 1000.0 / tpot  # tok/s seen by a single client
    ideal_tp = throughput[0] * batch  # perfect linear scaling from batch=1
    scaling_eff = throughput / ideal_tp
    # Little's law consistency: aggregate throughput should equal
    # concurrency x per-request rate if every slot is busy for the whole run.
    little = throughput / (batch * per_req)
    gain = np.concatenate([[np.nan], step_ratio(throughput)])
    cost = np.concatenate([[np.nan], step_ratio(tpot)])

    print("=" * 78)
    print(
        f"{'batch':>6} {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>10} "
        f"{'tok/s/req':>10} {'sc.eff':>7} {'Little':>7} {'gain':>6} {'cost':>6} {'g/c':>6}"
    )
    for i in range(len(batch)):
        g, c = gain[i], cost[i]
        gc = g / c if np.isfinite(g) else np.nan
        print(
            f"{batch[i]:6.0f} {ttft[i]:9.2f} {tpot[i]:9.2f} {throughput[i]:10.1f} "
            f"{per_req[i]:10.2f} {scaling_eff[i]:7.2f} {little[i]:7.3f} "
            f"{g:6.2f} {c:6.2f} {gc:6.2f}"
            if np.isfinite(g)
            else f"{batch[i]:6.0f} {ttft[i]:9.2f} {tpot[i]:9.2f} {throughput[i]:10.1f} "
            f"{per_req[i]:10.2f} {scaling_eff[i]:7.2f} {little[i]:7.3f} {'-':>6} {'-':>6} {'-':>6}"
        )
    print()

    ratio = step_ratio(throughput) / step_ratio(tpot)
    profitable = np.where(ratio > 1.0)[0]
    if len(profitable):
        knee = batch[profitable[-1] + 1]
        print(f"last profitable doubling ends at batch = {knee:.0f} (gain/cost > 1)")
    print(f"saturated aggregate throughput  ~ {throughput[-1]:.0f} tok/s")

    if PEAK_HBM_BW_BPS and WEIGHT_BYTES:
        mbu = WEIGHT_BYTES / (tpot[0] * 1e-3) / PEAK_HBM_BW_BPS
        mfu = 2 * MODEL_PARAMS * throughput[-1] / PEAK_DENSE_FLOPS
        print(f"MBU at batch={batch[0]} (weights only)   ~ {mbu * 100:.0f} %")
        print(f"MFU at batch={batch[-1]} (2*N*tok/s)    ~ {mfu * 100:.0f} %")
    print("=" * 78)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.8)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def make_figure(path, batch, ttft, tpot, itl, throughput):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Georgia", "Charter"]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.0), facecolor=BG)
    fig.suptitle(
        "Concurrency sweep: throughput / latency trade-off",
        fontsize=14,
        color=INK,
        y=0.985,
    )

    # --- 1. throughput vs concurrency, against perfect linear scaling ---------
    ax = axes[0, 0]
    ax.plot(batch, throughput, "o-", color=C_TP, lw=1.8, ms=5, label="measured")
    ax.plot(
        batch,
        throughput[0] * batch,
        "--",
        color=C_IDEAL,
        lw=1.2,
        label="linear scaling from batch=1",
    )
    ax.axhline(throughput[-1], color=C_IDEAL, ls=":", lw=1.2)
    ax.text(
        1.1,
        throughput[-1] * 1.08,
        f"saturation ~{throughput[-1]:.0f} tok/s",
        fontsize=8,
        color=INK,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batch)
    ax.set_xticklabels([f"{int(b)}" for b in batch])
    ax.legend(fontsize=8, frameon=False)
    style(ax, "Output throughput vs concurrency", "max_concurrency", "output tok/s")

    # --- 2. latency components ------------------------------------------------
    ax = axes[0, 1]
    ax.plot(batch, ttft, "o-", color=C_TTFT, lw=1.8, ms=5, label="TTFT (prefill)")
    ax.plot(batch, tpot, "s-", color=C_TPOT, lw=1.8, ms=5, label="TPOT")
    ax.plot(batch, itl, "^--", color=C_ITL, lw=1.2, ms=4, alpha=0.8, label="ITL")
    ax.annotate(
        "batch=1 TTFT above batch=2:\nwarmup / cudagraph capture artifact",
        xy=(1, ttft[0]),
        xytext=(3.2, 90),
        fontsize=8,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=C_TTFT, lw=1.0),
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batch)
    ax.set_xticklabels([f"{int(b)}" for b in batch])
    ax.legend(fontsize=8, frameon=False)
    style(ax, "Latency components vs concurrency", "max_concurrency", "latency (ms)")

    # --- 3. marginal gain vs marginal cost, per doubling ----------------------
    ax = axes[1, 0]
    xm = interval_x(batch)
    g, c = step_ratio(throughput), step_ratio(tpot)
    ax.plot(xm, g, "o-", color=C_TP, lw=1.8, ms=5, label="throughput gain x")
    ax.plot(xm, c, "s-", color=C_TPOT, lw=1.8, ms=5, label="TPOT cost x")
    ax.axhline(2.0, color=C_IDEAL, ls="--", lw=1.0)
    ax.axhline(1.0, color=C_IDEAL, ls=":", lw=1.0)
    cross = g > c
    ax.fill_between(xm, g, c, where=cross, color=C_TP, alpha=0.12)
    ax.fill_between(xm, g, c, where=~cross, color=C_TTFT, alpha=0.15)
    ax.text(xm[0], 2.06, "ideal 2x", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xm)
    ax.set_xticklabels([f"{int(a)}->{int(b)}" for a, b in zip(batch[:-1], batch[1:])], rotation=45, fontsize=7)
    ax.legend(fontsize=8, frameon=False)
    style(
        ax,
        "Marginal effect of each doubling (red = net loss)",
        "concurrency step",
        "ratio to previous point",
    )

    # --- 4. the actual SLO curve ---------------------------------------------
    ax = axes[1, 1]
    ax.plot(throughput, tpot, "o-", color=C_TP, lw=1.8, ms=5)
    for b, x, y in zip(batch, throughput, tpot):
        ax.annotate(f"{int(b)}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8, color=INK)
    for slo, label in ((30.0, "30 ms SLO"), (50.0, "50 ms SLO")):
        ax.axhline(slo, color=C_TTFT, ls=":", lw=1.0)
        feasible = throughput[tpot <= slo]
        if len(feasible):
            ax.annotate(
                f"{label}: <= {feasible.max():.0f} tok/s",
                (throughput[0], slo * 1.04),
                fontsize=8,
                color=C_TTFT,
            )
    ax.set_yscale("log")
    style(ax, "Throughput-latency frontier (labels = concurrency)", "output tok/s", "TPOT (ms)")

    # --- 5. per-request goodput ----------------------------------------------
    ax = axes[2, 0]
    ax.plot(batch, 1000.0 / tpot, "o-", color=C_TPOT, lw=1.8, ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch)
    ax.set_xticklabels([f"{int(b)}" for b in batch])
    ax.set_ylim(0, 50)
    ax2 = ax.twinx()
    ax2.plot(batch, throughput / batch, "s--", color=C_TP, lw=1.2, ms=4)
    ax2.set_ylabel("achieved tok/s per slot", fontsize=9, color=C_TP)
    ax2.tick_params(colors=C_TP, labelsize=8)
    ax2.set_ylim(0, 50)
    style(ax, "Per-request rate: nominal (1000/TPOT) vs achieved", "max_concurrency", "tok/s per request")

    # --- 6. Little's law residual --------------------------------------------
    ax = axes[2, 1]
    little = throughput / (batch * (1000.0 / tpot))
    ax.bar([str(int(b)) for b in batch], little, color=C_TP, alpha=0.75, width=0.6)
    ax.axhline(1.0, color=C_IDEAL, ls="--", lw=1.2)
    ax.set_ylim(0.75, 1.05)
    style(
        ax,
        "Slot occupancy = throughput / (batch x 1000/TPOT)",
        "max_concurrency",
        "fraction of ideal",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140, facecolor=BG)
    print(f"wrote {path}")


def parse_benchmark_result(args):
    dir_path = Path(args.log_dir)
    assert dir_path.exists()

    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = {int(x) for x in args.batch_sizes.split(",")}

    benchmark_files = [file for file in dir_path.glob("benchmark_metrics.*") if file.is_file()]
    # benchmark_files = [
    #     dir_path / "benchmark_metrics.cuda.256.2560.8192.8192.6144.20260901_063347.json",
    #     dir_path / "benchmark_metrics.cuda.320.3200.8192.8192.6144.20260901_063512.json",
    #     dir_path / "benchmark_metrics.cuda.384.3840.8192.8192.6144.20260901_063651.json",
    #     dir_path / "benchmark_metrics.cuda.448.4480.8192.8192.6144.20260901_063841.json",
    #     dir_path / "benchmark_metrics.cuda.512.5120.8192.8192.6144.20260901_064045.json",
    # ]

    benchmarks = {}
    for file in benchmark_files:
        fields = file.name.split(".")
        assert len(fields) > 3
        batch = int(fields[2])
        if batch_sizes is not None and batch not in batch_sizes:
            continue

        with open(file) as f:
            raw: dict[str, Any] = orjson.loads(f.read())
            benchmarks[batch] = raw

    if batch_sizes is not None:
        missing = batch_sizes - benchmarks.keys()
        assert not missing, f"no benchmark file found for batch sizes: {sorted(missing)}"

    sorted_batchs = sorted(benchmarks)
    benchmarks = sorted(benchmarks.items())

    ttft, tpot, itl, throughput = [], [], [], []
    for _, b in benchmarks:
        ttft.append(float(b["prefill"]["mean"]))
        tpot.append(float(b["tpot"]["mean"]))
        itl.append(float(b["itls"]["mean"]))
        throughput.append(float(b["basic"]["tok_throughput"]))

    print(f"batch: {sorted_batchs}")
    print(f"ttft: {ttft}")
    print(f"tpot: {tpot}")
    print(f"itl: {itl}")
    print(f"throughput: {throughput}")

    batch = np.array(sorted_batchs, dtype=int)
    ttft = np.array(ttft)
    tpot = np.array(tpot)
    itl = np.array(itl)
    throughput = np.array(throughput)
    return batch, ttft, tpot, itl, throughput


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="concurrency_sweep.png")
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes to include, e.g. 1,2,4,8. Defaults to all found in --log-dir.",
    )
    args = p.parse_args()
    dir_path = Path(args.log_dir)
    assert dir_path.exists()

    batch, ttft, tpot, itl, throughput = parse_benchmark_result(args)
    report(batch, ttft, tpot, throughput)
    out_path = str(dir_path / args.out)
    make_figure(out_path, batch, ttft, tpot, itl, throughput)
