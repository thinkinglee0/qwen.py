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

Usage: python bench_viz.py --log-dir DIR [--model qwen2.5-0.5b] [--out fig.png]

Both outputs land next to the logs in --log-dir: the figure at --out, and the
textual report at the same stem with a .txt suffix (concurrency_sweep.txt by
default). Pass --stdout to print the report to the terminal instead.

The roofline annotations (MBU / MFU) depend on which model was served. The
benchmark logs do not record that, so pass --model; it defaults to the 0.5B.
"""

import argparse
import contextlib

import numpy as np
import matplotlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import orjson

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Hardware constants, used only for the roofline annotations.
# Set PEAK_HBM_BW_BPS to None to disable that part of the report.
PEAK_HBM_BW_BPS = 1008e9  # RTX 4090, 24 GB GDDR6X
PEAK_DENSE_FLOPS = 165e12  # fp16/bf16 with fp32 accumulate, dense (non-sparse)
HARDWARE_NAME = "RTX 4090"


# ----------------------------------------------------------------------------
# Model registry. Derived from the architecture instead of hard-coded totals, so
# a new model is one row and the numbers stay auditable against config.json.
#
# Two different quantities are needed and they are NOT the same once embeddings
# are untied:
#   weight_bytes -- every distinct stored weight, streamed once per decode step
#                   -> denominator of MBU
#   flop_params  -- weights that are actually a GEMM: transformer body + one
#                   lm_head pass. The input embedding is a gather, not a matmul
#                   -> N in the 2*N*tok/s FLOP estimate, i.e. MFU
# With tied embeddings the two coincide; with untied ones flop_params drops the
# duplicated copy while weight_bytes keeps both.
@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    dtype_bytes: int = 2  # fp16 / bf16

    @property
    def body_params(self) -> int:
        """Transformer blocks + final norm; everything except the vocab matrices."""
        h, d = self.hidden_size, self.head_dim
        q, kv = self.num_attention_heads * d, self.num_key_value_heads * d
        attn = (h * q + q) + 2 * (h * kv + kv) + q * h  # q,k,v carry a bias; o does not
        mlp = 3 * h * self.intermediate_size
        norms = 2 * h  # input_layernorm + post_attention_layernorm
        return (attn + mlp + norms) * self.num_hidden_layers + h  # + model.norm

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def total_params(self) -> int:
        """What a checkpoint actually stores (one vocab matrix if tied, two if not)."""
        return self.body_params + self.embed_params * (1 if self.tie_word_embeddings else 2)

    @property
    def flop_params(self) -> int:
        """Params that do matmul work on a forward pass: body + a single lm_head."""
        return self.body_params + self.embed_params

    @property
    def weight_bytes(self) -> float:
        return self.total_params * self.dtype_bytes


MODELS: dict[str, ModelSpec] = {
    # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json
    "qwen2.5-0.5b": ModelSpec(
        name="Qwen2.5-0.5B-Instruct",
        hidden_size=896,
        num_hidden_layers=24,
        intermediate_size=4864,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=151936,
        tie_word_embeddings=True,
    ),
    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
    "qwen2.5-7b": ModelSpec(
        name="Qwen2.5-7B-Instruct",
        hidden_size=3584,
        num_hidden_layers=28,
        intermediate_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=152064,
        tie_word_embeddings=False,
    ),
}
DEFAULT_MODEL = "qwen2.5-0.5b"


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


def report(batch, ttft, tpot, throughput, model: ModelSpec):
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

    if PEAK_HBM_BW_BPS and PEAK_DENSE_FLOPS:
        # Roofline numbers are only as good as the two assumptions below, and a
        # model/hardware mismatch is silent -- so always print what was assumed.
        print(
            f"roofline assumes {model.name} "
            f"({model.total_params / 1e9:.3f} B params, {model.weight_bytes / 1e9:.2f} GB "
            f"@ {model.dtype_bytes} B/param) on {HARDWARE_NAME} "
            f"({PEAK_HBM_BW_BPS / 1e9:.0f} GB/s, {PEAK_DENSE_FLOPS / 1e12:.0f} TFLOP/s)"
        )
        mbu = model.weight_bytes / (tpot[0] * 1e-3) / PEAK_HBM_BW_BPS
        mfu = 2 * model.flop_params * throughput[-1] / PEAK_DENSE_FLOPS
        print(f"MBU at batch={batch[0]} (weights only)   ~ {mbu * 100:.1f} %")
        print(f"MFU at batch={batch[-1]} (2*N*tok/s)    ~ {mfu * 100:.1f} %")
    print("=" * 78)


# batch-1 TTFT must exceed batch-2 by this factor before it is worth annotating
TTFT_HEAD_ANOMALY = 1.05


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.8)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def make_figure(path, batch, ttft, tpot, itl, throughput, model: ModelSpec):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Georgia", "Charter"]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.0), facecolor=BG)
    fig.suptitle(
        f"Concurrency sweep: throughput / latency trade-off\n{model.name} on {HARDWARE_NAME}",
        fontsize=14,
        color=INK,
        y=0.99,
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
    # Annotate only when the phenomenon is in the data. This used to be drawn
    # unconditionally and blamed "cudagraph capture" -- on an engine that has no
    # cudagraph, and on sweeps where batch 1 sits *below* batch 2 (log915: 23.40
    # vs 23.43 ms). State the observation; do not name a cause we have not shown.
    if len(ttft) > 1 and ttft[0] > TTFT_HEAD_ANOMALY * ttft[1]:
        ax.annotate(
            f"batch=1 TTFT {100 * (ttft[0] / ttft[1] - 1):.0f}% above batch=2\n(warm-up not amortised?)",
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
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=sorted(MODELS),
        help=f"Model the sweep was run against; sets the roofline constants. Default: {DEFAULT_MODEL}. "
        "The log files do not record the model, so this must match the run.",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Print the report to the terminal instead of writing it beside the figure.",
    )
    args = p.parse_args()
    dir_path = Path(args.log_dir)
    assert dir_path.exists()

    model = MODELS[args.model]
    fig_path = dir_path / args.out
    # Same stem as the figure: the two always belong to one run, so one name
    # identifies both. with_suffix() also normalises a --out given without one.
    report_path = fig_path.with_suffix(".txt")
    assert report_path != fig_path, f"--out must not already be a .txt path: {args.out}"

    # The report is the artifact worth keeping; sending it to a file next to the
    # figure means a sweep no longer has to be reconstructed from a shell scrollback.
    with contextlib.ExitStack() as stack:
        if not args.stdout:
            sink = stack.enter_context(open(report_path, "w"))
            stack.enter_context(contextlib.redirect_stdout(sink))

        print(f"log-dir: {dir_path}")
        print(f"model:   {args.model} ({model.name})")
        batch, ttft, tpot, itl, throughput = parse_benchmark_result(args)
        report(batch, ttft, tpot, throughput, model)
        make_figure(str(fig_path), batch, ttft, tpot, itl, throughput, model)

    # back on the real stdout: say where everything went
    print(f"wrote {fig_path}")
    if not args.stdout:
        print(f"wrote {report_path}")
