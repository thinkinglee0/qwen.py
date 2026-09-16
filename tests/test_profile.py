'''This file is maintained by Claude'''

import pytest
import logging
import math
import time
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch._C._autograd import DeviceType
from torch._C._profiler import _ExperimentalConfig
import gc
import os
import random
import orjson
import numpy as np
from collections import Counter
from contextlib import contextmanager
from typing import NamedTuple
from datetime import datetime

from constants import *
from qwen.engine import LLMEngine
from qwen.config import ModelConfig
from qwen.scheduler import ModelRequest
from qwen.metrics import summarize
from utils import parse_env_list_value

logger = logging.getLogger(__name__)

STEP_MARKER = "decode_step"   # record_function marker: one per profiled decode step


class TraceStats(NamedTuple):
    gpu_busy_us: float          # union of kernel/memcpy/memset spans (overlap counted once)
    gpu_sum_us: float           # plain sum of the same spans
    wall_us: float              # first to last cpu_op
    gpu_names: set[str]         # names of the real device rows -- everything else in
                                # key_averages() with device_type CUDA is a wait marker


def parse_chrome_trace(trace_path: Path, profile_steps:int=0):

    with open(trace_path) as f:
        events = orjson.loads(f.read())["traceEvents"]

    if profile_steps > 0:
        # Count our own record_function marker, not an aten op: op counts per step
        # are an implementation detail (e.g. torch.multinomial(num_samples=1) runs
        # replacement=False -> the gumbel trick -> an *internal* aten::argmax, so
        # the sampler emits two aten::argmax rows per step, not one).
        marks = Counter(e["name"] for e in events
                        if e.get("ph") == "X" and e.get("cat", "").lower() == "user_annotation")
        assert marks[STEP_MARKER] == profile_steps, \
            f"expected {profile_steps} steps, trace has {marks[STEP_MARKER]}"

    GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
    # NOTE: cat "cuda_sync" (Stream/Context/Event Sync, emitted because of
    # enable_cuda_sync_events) is deliberately NOT a GPU cat: those spans are the
    # host waiting, not the device working, and they are as long as the idle gap.
    gpu_events = [e for e in events
        if e.get("ph") == "X" and e.get("cat", "").lower() in GPU_CATS]
    gpu = [(e["ts"], e["ts"] + e["dur"]) for e in gpu_events]
    cpu = [(e["ts"], e["ts"] + e["dur"]) for e in events
        if e.get("ph") == "X" and e.get("cat", "").lower() == "cpu_op"]


    def union_duration(spans):
        """Merge overlapping intervals; correct even with multiple CUDA streams."""
        if not spans:
            return 0.0
        
        spans = sorted(spans)
        total = 0.0
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            if s > cur_e:                 # gap -> close current interval
                total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        return total + (cur_e - cur_s)


    gpu_busy = union_duration(gpu)
    wall = max(e for _, e in cpu) - min(s for s, _ in cpu)
    return TraceStats(gpu_busy_us=gpu_busy,
                      gpu_sum_us=sum(e["dur"] for e in gpu_events),
                      wall_us=wall,
                      gpu_names={e["name"] for e in gpu_events})

CHROME_TRACE_FILE_NAME = "decode_tracing_chrome.{}.json"
DECODE_STACK_CUDA_FILE_NAME = "decode_stacks_cuda.{}.txt"
DECODE_STACK_CPU_FILE_NAME = "decode_stacks_cpu.{}.txt"

'''
excluded from execution from file, only allowed from specified execution.

Usage:
baseline:
for bz in 1 8 32 128 512; do
SWEEP_PROFILE_BATCH_SIZES=$bz pytest -x -s tests/test_profile.py::test_profile_decode_idle_fraction --compile-rope=False --use-sampling-param-table=false --pre-gather-cos-sin=false;
done

# in one line:
for bz in 1 8 32 128 512; do SWEEP_PROFILE_BATCH_SIZES=$bz pytest -x -s tests/test_profile.py::test_profile_decode_idle_fraction --compile-rope=False --use-sampling-param-table=false --pre-gather-cos-sin=false; done
'''
_DEFAULT_PROFILE_BATCH_SIZES = [512]
@pytest.mark.parametrize("batch_size", parse_env_list_value(env_name="SWEEP_PROFILE_BATCH_SIZES", default_value=_DEFAULT_PROFILE_BATCH_SIZES))
@pytest.mark.parametrize("use_d_first_schedule", parse_env_list_value(env_name="SWEEP_PROFILE_USE_D_FIRST_SCHEDULE", default_value=[False]))
def test_profile_decode_idle_fraction(tmp_target_config_for_sharegpt_benchmarking: ModelConfig, batch_size: int, use_d_first_schedule:bool):
    '''
    GPU idle fraction of a *pure decode* step, isolated from prefill.

    Same shape as test_benchmark_sweep_batch_size (fixed input_len,
    ignore EOS) so all `batch_size` requests prefill together and finish
    decode in lockstep -- that gives a steady-state decode window with no
    admissions/preemptions to sample from, instead of the mixed
    prefill+decode average that run_to_completion() would produce.
    '''
    logger.info(f"batch_size: {batch_size}, use_d_first_schedule: {use_d_first_schedule}")

    # MEASURE_STEPS (run A, clean) carries the timing statistics -- it is the only
    # window free of profiler overhead, and one step_metrics line is dumped per step
    # either way. PROFILE_STEPS (run B) only has to be long enough to hold a
    # representative trace, and every extra step there costs trace size and risks a
    # CUPTI buffer overflow (dropped events would silently shrink gpu_busy).
    PROMPT_LEN, WARMUP_STEPS, MEASURE_STEPS, PROFILE_STEPS = 512, 64, 20, 5

    # KV grows by one token per step, so run B starts MEASURE_STEPS tokens behind
    # where run A started, and drifts PROFILE_STEPS more while being profiled. Both
    # runs have to stay the same workload for their numbers to be comparable, which
    # is what bounds MEASURE_STEPS (<= 28 at PROMPT_LEN=512, WARMUP_STEPS=64).
    kv_drift = (MEASURE_STEPS + PROFILE_STEPS) / (PROMPT_LEN + WARMUP_STEPS)
    logger.info(f"KV drift between runs: {kv_drift:.2%}")
    assert kv_drift < 0.05

    cfg = tmp_target_config_for_sharegpt_benchmarking
    cfg.ignore_eos()
    cfg.max_num_seqs = batch_size
    cfg.max_model_len = 1024
    cfg.max_num_batched_tokens = 8*1024
    cfg.long_prefill_token_threshold = 8*1024
    cfg.num_blocks = int(4*1024)     # 4 * 24 * 1024*2 * 256 * 2 * 64 * 2 B = 17716740096 B ≈ 12.8 GB
    cfg.max_waiting = batch_size    # exactly one wave, no refill needed
    cfg.use_d_first_schedule = use_d_first_schedule

    assert cfg.device is not None
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name_flag = f'{cfg.max_num_seqs}.{cfg.max_num_batched_tokens}.{cfg.long_prefill_token_threshold}.{cfg.use_d_first_schedule}.{time_str}'


    batch_input_ids = [[random.randrange(cfg.vocab_size) for _ in range(PROMPT_LEN)]
                       for _ in range(batch_size)]

    engine: LLMEngine | None = None
    try:
        engine = LLMEngine(cfg)
        engine.scheduler.log_name_flag = log_name_flag

        for input_ids in batch_input_ids:
            req = ModelRequest(cfg, loop=None, input_ids=input_ids)
            assert engine.scheduler.add_request(req), "request rejected: check num_blocks/max_waiting"

        # warmup: drain the initial prefill wave (CUDA context init, lazy
        # module loading and prefill compute are not representative of
        # steady-state decode) until every running request is decoding.

        # Warm up the compiled function first: inductor's own init allocates CUDA
        # tensors and would trip the sync debug guard.
        torch.cuda.set_sync_debug_mode("default")
        for _ in range(WARMUP_STEPS):
            engine.step()
        assert engine.scheduler.running, "no requests admitted -- check num_blocks / max_num_seqs"
        assert len(engine.scheduler.running) == batch_size, "sequences finished mid-measurement -- raise cfg.max_model_len"
        assert all(r.is_decoding for r in engine.scheduler.running), "still prefilling after warmup_steps -- increase warmup_steps"

        # run A: do not profile, only record wall time
        torch.cuda.set_sync_debug_mode("warn") # or error
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(MEASURE_STEPS):
            engine.step()
        torch.cuda.synchronize()
        wall_clean_us = (time.perf_counter() - t0) * 1e6
        torch.cuda.set_sync_debug_mode("default")

        # run B: interleave
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    #  with_stack=True,     # only needed for export_stacks() -- not needed for key_averages() or export_chrome_trace()
                     experimental_config=_ExperimentalConfig(verbose=True, enable_cuda_sync_events=True)) as prof:
            for _ in range(PROFILE_STEPS):
                with record_function(STEP_MARKER):
                    engine.step()
            torch.cuda.synchronize()

        logger.info("profile finished, start to export stacks")

        # Only device_type==CUDA rows are real kernel executions. CPU-side op
        # rows (aten::mm, flash_attn::...) also carry a nonzero self_cuda/
        # self_device_time_total -- the *same* GPU time attributed to their
        # launched kernel -- so including them here double-counts (this is
        # exactly how torch's own EventList.table() computes its "Self CUDA
        # time total" line, see torch.autograd.profiler_util._build_table).
        ka = prof.key_averages()
        _ATTR = "self_cuda_time_total" if hasattr(ka[0], "self_cuda_time_total") else "self_device_time_total"
        # export_chrome_trace / export_stacks / key_averages() all require the
        # profiler to have actually stopped (i.e. the `with` block exited) --
        # calling them while still inside raises "Profiler didn't finish
        # running" (or, on newer torch, a bare AttributeError on
        # kineto_results being None).
        trace_path = Path(cfg.log_dir) / CHROME_TRACE_FILE_NAME.format(log_name_flag)
        prof.export_chrome_trace(str(trace_path))
        # prof.export_stacks(str(Path(cfg.log_dir) / DECODE_STACK_CUDA_FILE_NAME.format(log_name_flag)), "self_cuda_time_total")
        # prof.export_stacks(str(Path(cfg.log_dir) / DECODE_STACK_CPU_FILE_NAME.format(log_name_flag)), "self_cpu_time_total")

        stats = parse_chrome_trace(trace_path=trace_path, profile_steps=PROFILE_STEPS)
        gpu_busy_from_trace_us, wall_from_trace_us = stats.gpu_busy_us, stats.wall_us

        # Keep only rows the trace itself calls a kernel/memcpy/memset. A name
        # blocklist does not work here: enable_cuda_sync_events adds device rows
        # ("Stream Sync", "Context Sync", ...) whose duration is *host wait*, i.e.
        # roughly the whole idle gap -- at batch_size=1 that alone pushed this sum
        # past the wall clock (177ms sum vs 16ms of real kernels in 167ms of wall).
        gpu_device_time_sum_us = sum(
            getattr(e, _ATTR) for e in ka
            if e.device_type == DeviceType.CUDA and e.key in stats.gpu_names)

        # run A and run B no longer cover the same number of steps, so nothing below
        # compares two raw totals: the trace numbers are per PROFILE_STEPS, the clean
        # wall is per MEASURE_STEPS, and both are scaled to one step before use.
        gpu_busy_per_step_us = gpu_busy_from_trace_us / PROFILE_STEPS
        gpu_sum_per_step_us = stats.gpu_sum_us / PROFILE_STEPS
        wall_trace_per_step_us = wall_from_trace_us / PROFILE_STEPS
        wall_clean_per_step_us = wall_clean_us / MEASURE_STEPS

        logger.info(f"batch_size: {batch_size}, measure_steps: {MEASURE_STEPS}, profile_steps: {PROFILE_STEPS}, "
                    f"wall_from_trace_us: {wall_from_trace_us:.1f}, wall_clean_us: {wall_clean_us:.1f}, "
                    f"gpu_busy_from_trace_us: {gpu_busy_from_trace_us:.1f}, "
                    f"gpu_sum_from_trace_us: {stats.gpu_sum_us:.1f}, "
                    f"gpu_device_time_sum_us: {gpu_device_time_sum_us:.1f}")

        assert gpu_device_time_sum_us > 0, \
            "no key_averages row matched a trace kernel name -- check DeviceType filtering"
        # same events counted two ways (kineto rows vs chrome trace) -- must agree
        ka_vs_trace_pct = (gpu_device_time_sum_us - stats.gpu_sum_us) / stats.gpu_sum_us
        logger.info(f"key_averages vs trace kernel sum: {ka_vs_trace_pct:+.2%}")
        assert abs(ka_vs_trace_pct) < 0.05, "sum vs sum mismatch -- check trace parsing"

        # union < sum only where kernels really overlap, i.e. multiple streams
        overlap_pct = (stats.gpu_sum_us - gpu_busy_from_trace_us) / stats.gpu_sum_us
        logger.info(f"kernel overlap: {overlap_pct:.2%} (near 0 expected on a single stream)")
        assert -0.01 < overlap_pct < 0.5, "sum vs union mismatch -- check trace parsing"

        # what profiling itself costs: the same step, traced vs clean
        profiler_overhead_pct = (wall_trace_per_step_us - wall_clean_per_step_us) / wall_clean_per_step_us
        logger.info(f"profiler overhead: {profiler_overhead_pct:+.1%} "
                    f"(trace wall {wall_trace_per_step_us:.1f} vs clean wall {wall_clean_per_step_us:.1f} us/step)")

        # Kernel durations barely move under profiling, the host side does -- so the
        # honest utilization is traced GPU busy over the *clean* wall. Dividing by the
        # trace wall instead would flatter it by exactly the overhead above.
        gpu_utilization = gpu_busy_per_step_us / wall_clean_per_step_us
        assert 0.0 <= gpu_utilization <= 1.05, f"nonsensical utilization: {gpu_utilization}"

        logger.info(f"for per step, wall_clean_us={wall_clean_per_step_us:.1f} us/step, "
                    f"gpu_busy_from_trace_us={gpu_busy_per_step_us:.1f} us/step, "
                    f"gpu_sum_from_trace_us={gpu_sum_per_step_us:.1f} us/step, "
                    f"gpu_idle_fraction={1 - gpu_utilization:.1%}")
        logger.info(f"key_averages len: {len(ka)} distinct rows")
        logger.info(ka.table(sort_by=_ATTR, row_limit=-1))
        # logger.info(ka.table(sort_by="self_cpu_time_total", row_limit=-1))
    finally:
        # explicitly release kv cache
        if engine is not None:
            engine.teardown()
            del engine

        gc.collect()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

STEP_METRICS_GLOB = "step_metrics.*.json"
REPORT_FILE_NAME = "mean_step_metrics.{}.log"


# An anomaly is a step that COST time, judged against its own neighbourhood:
# value >= ANOMALY_RATIO * local median AND at least ANOMALY_EXCESS_MS above it.
# Both conditions are needed. The ratio alone fires constantly on the microsecond
# fields (sched_wait at 0.003 ms doubling is not an event), and a distribution-based
# cut (mean/std, or even median/MAD at 3.5) is useless here: these series are so
# fat-tailed that a Gaussian-calibrated threshold flags a third of the run.
ANOMALY_RATIO = 1.5
ANOMALY_EXCESS_MS = 0.5
# ...and the local median is rolling, because a run is not stationary: the KV keeps
# growing, so every field drifts slowly from the first step to the last.
ANOMALY_WINDOW = 51
ANOMALY_SUFFIX = ".anomaly"


class RunStats(NamedTuple):
    label: str
    path: Path                          # the dump these came from
    rows: list[tuple[int, dict]]        # (1-based line number in the dump, row)
    stats: dict[str, dict]              # field -> summarize() dict
    series: dict[str, list[float]]      # field -> the raw per-step values


def _batch_of(name: str) -> str | None:
    """Batch size out of a dump name: the first all-digit dot-token. Covers both
    'step_metrics.cuda.512.5120...' (test_benchmark) and 'step_metrics.512.8192...'
    (test_profile) without hard-coding a position."""
    return next((tok for tok in name.split(".")[1:] if tok.isdigit()), None)


def _pick_dump(run_dir: Path, batch: str | None) -> Path:
    dumps = list(run_dir.glob(STEP_METRICS_GLOB))
    assert dumps, f"no dump matching {run_dir / STEP_METRICS_GLOB}"
    if batch is None:
        return max(dumps, key=lambda p: p.stat().st_mtime)

    hits = [p for p in dumps if _batch_of(p.name) == batch]
    assert hits, (f"no dump for batch {batch} in {run_dir}; "
                  f"available: {sorted({_batch_of(p.name) or '?' for p in dumps}, key=lambda t: int(t) if t.isdigit() else -1)}")
    if len(hits) > 1:      # same batch run more than once into one directory
        hits.sort(key=lambda p: p.stat().st_mtime)
        logger.warning(f"{len(hits)} dumps for batch {batch} in {run_dir}, taking the newest: {hits[-1].name}")
    return hits[-1]


def _effective_n(xs: list[float]) -> float:
    """AR(1) effective sample size. Consecutive steps are NOT independent -- the KV
    keeps growing and the batch composition shifts -- so std/sqrt(n) understates the
    error of the mean and inflates z. n_eff = n*(1-rho)/(1+rho) on the lag-1
    autocorrelation is the standard deflation."""
    n = len(xs)
    a = np.asarray(xs, dtype=np.float64)
    a = a - a.mean()
    denom = float(a @ a)
    if n < 3 or denom == 0.:
        return float(n)

    rho = float(a[:-1] @ a[1:] / denom)
    rho = min(max(rho, 0.), 0.99)       # anti-correlated steps get no bonus; cap the blow-up
    return max(n * (1. - rho) / (1. + rho), 2.)


def _parse_line_range(spec: str | None, n: int) -> tuple[int, int]:
    """'first:last' -> [start, stop). 1-based and inclusive (as an editor shows it),
    either side optional: '33:40', '33:' (to EOF), ':40' (from the top)."""
    if not spec:
        return 0, n

    first, sep, last = spec.partition(":")
    assert sep, f"bad line range {spec!r} -- expected 'first:last', e.g. '33:40', '33:', ':40'"
    start = int(first) - 1 if first.strip() else 0
    stop = int(last) if last.strip() else n
    assert 0 <= start < stop <= n, f"line range {spec!r} out of bounds: the dump has {n} lines"
    return start, stop


@contextmanager
def _tee_logs(path: Path, level: int = logging.INFO):
    """Mirror every record at `level`+ into `path` while the block runs. Attached to
    the ROOT logger so qwen.* modules land in it too, and removed afterwards so the
    handler does not leak into the next test."""
    handler = logging.FileHandler(path, mode="w")       # one file per invocation, not a growing log
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s - %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    # a record is dropped before reaching any handler if the logger level is higher
    prev_level = root.level
    if root.getEffectiveLevel() > level:
        root.setLevel(level)

    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_level)


def _load_step_metrics(run_dir: Path, batch: str | None, lines_spec: str | None) -> RunStats:
    """One dump, one line range -> label, per-field stats, per-field raw series."""
    path = _pick_dump(run_dir, batch)

    lines = [line for line in path.read_bytes().splitlines() if line.strip()]
    assert lines, f"empty dump: {path}"
    start, stop = _parse_line_range(lines_spec, len(lines))
    # carry the 1-based line number along, so an anomaly can be pointed at in the dump
    numbered = [(start + 1 + i, orjson.loads(line)) for i, line in enumerate(lines[start:stop])]
    assert all(r.keys() == numbered[0][1].keys() for _, r in numbered), \
        f"{path.name}: rows do not share the same fields -- concatenated dumps from different runs?"

    # keep only pure-decode steps: a step that carries prefill tokens is a different
    # workload and its timings are not comparable to steady-state decode
    numbered = [(ln, r) for ln, r in numbered if r.get("n_p") == 0]
    rows = [r for _, r in numbered]

    # summarize() uses a ddof=1 sample std, which needs two points
    assert len(rows) > 1, f"{path.name}: need at least 2 steps for a sample std, got {len(rows)}"

    # A field is a float field iff it shows up as a float in at least one row: the
    # int counters (bz, n_p, run, ...) never do, while a timing that happens to
    # round to a whole number still serializes as e.g. 19.0.
    float_keys = [k for k in rows[0] if any(isinstance(r[k], float) for r in rows)]
    assert float_keys, f"no float fields in {path}"

    series = {k: [float(r[k]) for r in rows] for k in float_keys}
    label = (f"{path.parent.name}/{path.name}: lines {start + 1}-{stop} of {len(lines)}, "
             f"{len(rows)} decode steps")
    # scale=1.0: summarize() defaults to s -> ms, but a step metrics dump is already ms
    return RunStats(label, path, numbered, {k: summarize(v, scale=1.0) for k, v in series.items()}, series)


def _rolling_median(a: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling median, edge-padded so the result keeps the input length."""
    if w < 3 or len(a) <= w:
        return np.full_like(a, np.median(a))

    half = w // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(a, half, mode="edge"), w)
    return np.median(windows, axis=-1)


def _write_anomalies(run: RunStats) -> Path:
    """Flag the individual steps that spike on any field, next to the dump as
    <dump>.anomaly (JSONL: the original row plus the fields that tripped)."""
    local = {k: _rolling_median(np.asarray(xs, dtype=np.float64), ANOMALY_WINDOW)
             for k, xs in run.series.items()}

    hits, per_field = [], Counter()
    for i, (line_no, row) in enumerate(run.rows):
        flagged = {}
        for k in run.series:
            value, norm = float(row[k]), float(local[k][i])
            if value < ANOMALY_RATIO * norm or value - norm < ANOMALY_EXCESS_MS:
                continue
            flagged[k] = {"value": value, "local_median": round(norm, 3),
                          "ratio": round(value / norm, 2) if norm else None,
                          "excess_ms": round(value - norm, 3)}
        if flagged:
            hits.append({"line": line_no, "anomaly": flagged, "row": row})
            per_field.update(flagged.keys())      # count fields, not their values

    out_path = run.path.with_name(run.path.name + ANOMALY_SUFFIX)
    out_path.write_bytes(b"".join(orjson.dumps(h) + b"\n" for h in hits))
    logger.info(f"{len(hits)}/{len(run.rows)} anomalous steps -> {out_path}")
    logger.info(f"  by field: {per_field.most_common()}") if hits else None
    return out_path


def test_mean_step_metrics(log_dir: str):
    '''
    Per-field stats over a step_metrics dump (JSONL, one object per step, written by
    Scheduler.log_step_metrics), for one run or as a diff between two runs:

        STEP_METRICS_DIR=log_vast/log914 \
        STEP_METRICS_RUNS=benchmark_baseline,benchmark_baseline2 \
        STEP_METRICS_BATCH=512 \
        pytest -x -s tests/test_profile.py::test_mean_step_metrics

    STEP_METRICS_DIR   common path holding the run directories (default: --log-dir)
    STEP_METRICS_RUNS  one or two directories under it (default: the path itself)
    STEP_METRICS_BATCH batch size, matched against the dump name (default: newest dump)
    STEP_METRICS_LINES 1-based inclusive line range over non-empty lines, one for both
                       runs or one each ('65:69', '65:', ':69', '65:69,33:40')
    '''
    base = Path(os.environ.get("STEP_METRICS_DIR") or log_dir)
    runs = [r.strip() for r in os.environ.get("STEP_METRICS_RUNS", "").split(",") if r.strip()]
    specs = [r.strip() for r in os.environ.get("STEP_METRICS_LINES", "").split(",") if r.strip()] or [None]
    batch = os.environ.get("STEP_METRICS_BATCH") or None

    run_dirs = [base / r for r in runs] if runs else [base]
    assert len(run_dirs) <= 2, f"pass one run, or two to compare, got {len(run_dirs)}"
    for d in run_dirs:
        assert d.is_dir(), f"no such directory: {d}"
    if len(specs) == 1:
        specs *= len(run_dirs)          # one range, applied to both runs
    assert len(specs) == len(run_dirs), \
        f"STEP_METRICS_LINES: give one range or {len(run_dirs)}, got {len(specs)}"

    report_path = base / REPORT_FILE_NAME.format(batch or "latest")
    with _tee_logs(report_path):
        logger.info(f"report -> {report_path}")
        _report_step_metrics(run_dirs, batch, specs)


def _report_step_metrics(run_dirs: list[Path], batch: str | None, specs: list[str | None]):
    loaded = [_load_step_metrics(d, batch, spec) for d, spec in zip(run_dirs, specs)]
    for run in loaded:
        _write_anomalies(run)

    if len(loaded) == 1:
        run = loaded[0]
        logger.info(f"{run.label}, {len(run.stats)} float fields")
        w = max(len(k) for k in run.stats)
        # one logger call for the whole table: a per-row call would prefix every line
        # with a timestamp and break the columns (same reason ka.table() is logged whole)
        table = [f"{'field':<{w}}  {'mean':>9}  {'std':>8}  {'cv':>7}  {'p50':>9}"
                 f"  {'p90':>9}  {'max':>9}  {'n_eff':>7}"]
        for k, st in run.stats.items():
            # cv = std/mean: the scale-free one, and the only column that stays
            # comparable across fields spanning three orders of magnitude
            cv = st["std"] / st["mean"] if st["mean"] else float("nan")
            table.append(f"{k:<{w}}  {st['mean']:9.3f}  {st['std']:8.3f}  {cv:7.1%}  {st['p50']:9.3f}"
                         f"  {st['p90']:9.3f}  {st['max']:9.3f}  {_effective_n(run.series[k]):7.0f}")
        logger.info("\n" + "\n".join(table))
        return

    a, b = loaded
    logger.info(f"base: {a.label}")
    logger.info(f"cand: {b.label}")
    keys = [k for k in a.stats if k in b.stats]
    assert keys, "the two runs share no float field"
    dropped = [k for k in (a.stats | b.stats) if k not in keys]
    logger.warning(f"fields in only one run, skipped: {dropped}") if dropped else None

    w = max(len(k) for k in keys)
    table = [f"{'field':<{w}}  {'base':>9}  {'std':>8}  {'cv':>6}  {'cand':>9}  {'std':>8}  {'cv':>6}"
             f"  {'delta':>8}  {'delta%':>7}  {'n_eff':>7}  {'z':>6}  verdict"]
    shifted = []
    for k in keys:
        x, y = a.stats[k], b.stats[k]
        cv_x = x["std"] / x["mean"] if x["mean"] else float("nan")
        cv_y = y["std"] / y["mean"] if y["mean"] else float("nan")
        delta = y["mean"] - x["mean"]
        pct = delta / x["mean"] if x["mean"] else float("nan")

        # Standard error of the difference of two means (Welch, unequal variances),
        # on n_eff rather than n. With ~1000 autocorrelated steps the raw-n version
        # calls a 0.5% gap significant, which is how every field ends up "SHIFT".
        n_x, n_y = _effective_n(a.series[k]), _effective_n(b.series[k])
        se = math.sqrt(x["std"] ** 2 / n_x + y["std"] ** 2 / n_y)
        z = delta / se if se else (0. if delta == 0 else math.inf)
        verdict = "SHIFT" if abs(z) >= 2 else "noise"
        shifted.append(k) if verdict == "SHIFT" else None
        table.append(f"{k:<{w}}  {x['mean']:9.3f}  {x['std']:8.3f}  {cv_x:6.1%}"
                     f"  {y['mean']:9.3f}  {y['std']:8.3f}  {cv_y:6.1%}"
                     f"  {delta:+8.3f}  {pct:+7.2%}  {min(n_x, n_y):7.0f}  {z:+6.1f}  {verdict}")

    logger.info("\n" + "\n".join(table))
    logger.info(f"{len(shifted)}/{len(keys)} fields beyond 2 sigma: {shifted}")


def test_parse_chrome_trace(log_dir: str):
    trace_path = Path(log_dir) / CHROME_TRACE_FILE_NAME
    gpu_busy, _, wall, _ = parse_chrome_trace(trace_path=trace_path)
    logger.info(f"wall={wall/1e3:.1f}ms  gpu_busy={gpu_busy/1e3:.1f}ms  "
        f"util={100*gpu_busy/wall:.1f}%  idle={(wall-gpu_busy)/1e3:.1f}ms")

