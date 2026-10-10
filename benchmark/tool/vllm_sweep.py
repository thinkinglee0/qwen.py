"""vLLM counterpart of tests/test_benchmark.py::test_benchmark_sweep_batch_size.

Runs in the SEPARATE vLLM venv (section 6a of env/vastai/vast-evn-build.md), which
has only vllm and its dependencies -- so this file imports nothing from qwen.py and
nothing outside the stdlib + vllm. Do not add `from qwen...` here.

for arm in eager compile cudagraph; do
  "$VLLM_VENV/bin/python" benchmark/tool/vllm_sweep.py --arm $arm --runs 3 --out ./log/vllm_$arm.jsonl > ./log/vllm_$arm.log 2>&1
done

Three arms, because A->B alone cannot tell you which half to build:

  eager     (A)   no inductor, no graphs        engine design alone
  compile   (A')  inductor, graphs OFF          A->A'  = inductor fusion
  cudagraph (B)   inductor + graphs (default)   A'->B  = launch elimination

log1003 ran only A and B, so its 7.28x at batch 1 and 1.17x at batch 512 are a bundle:
enforce_eager=True turns off compilation AND capture together.

--out holds only the JSON rows. Everything else -- vLLM's engine log plus this
script's progress lines -- goes to stdout/stderr, so redirect the whole process to a
file as above. Which stream vLLM logs to varies by version, and its V1 EngineCore
runs in a SEPARATE process whose output an in-process logging handler would miss, so
a shell redirect is the only capture that is complete.

What it mirrors, and why each line is load-bearing, is documented inline. The three
settings that silently destroy comparability if you get them wrong:

  1. Throughput is OUTPUT tokens only. vLLM's own harnesses also report
     "total token throughput" = (prompt + output) / s, which at this fixed
     512-in / 128-out shape is exactly 5x larger. Never quote that number against
     qwen.py's.
  2. The sampling params must match generation_config.json item for item. They are
     not an incidental knob -- after log1001 the sampler is 60 % of qwen.py's device
     work, so turning them off here deletes the very thing under test (and vLLM
     takes a different, cheaper code path when NO request needs penalties/top-k/
     top-p, so partial matching is worse than none).
  3. Offline closed loop, not `vllm serve`. The qwen.py test enqueues every request
     before starting the timer and has no HTTP and no tokenizer in the loop.

Latency: rows carry `latency` with queueing / ttft / prefill / tpot summarised exactly
as metrics.analyze_metrics() does it, including np.percentile's interpolation, so the
columns line up with concurrency_sweep.txt (whose "TTFT ms" is in fact `prefill`). The
pooled per-token ITL distribution is NOT reproducible here -- it needs per-token
timestamps and the offline path exposes only the first and last. If this vLLM build
leaves RequestOutput.metrics unpopulated, `latency` is {} and `latency_note` says so
rather than the row quietly carrying nothing.

vLLM 0.30.x does populate it, under `queued_ts` / `scheduled_ts` / `first_token_ts` /
`last_token_ts` / `first_token_latency` / `num_generation_tokens` / `num_preemptions`
(discovered from log1009/host2's latency_note, which is what that guard is for).
_SCHEMAS carries that naming and the older one; see the clock-domain warning there.
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import statistics
import sys
import time

VOCAB_SIZE = 151936          # Qwen2.5-0.5B-Instruct
IN_LEN, OUT_LEN = 512, 128   # the test's fixed-length input / output
MAX_MODEL_LEN = 1024
MAX_NUM_BATCHED_TOKENS = 8192
KV_TOKENS = 4096 * 256       # qwen.py num_blocks * block_size = 1 048 576 slots = 12.0 GiB
BLOCK_SIZE = 16              # vLLM has no 256; match TOTAL CAPACITY, not block size
SEED = 1234

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def log(msg: str) -> None:
    """Progress goes to stderr; --out stays machine-readable; stdout stays vLLM's."""
    print(f"[sweep] {msg}", file=sys.stderr, flush=True)


def build_prompts(count: int, dump: pathlib.Path | None = None) -> tuple[list[list[int]], str]:
    """Fixed-length random-token prompts, regenerated from SEED -- never transported.

    SEED + VOCAB_SIZE + IN_LEN fully determine the set, so both arms get identical
    input without a 37 MB file of random ints travelling anywhere. What IS worth
    keeping is the digest: it proves two runs used the same prompts in 64 bytes.

    The content does not move throughput either way -- output length is pinned by
    ignore_eos + max_tokens, and uniform-random ids carry no data dependence. Pass
    `dump` only when you want to feed the SAME ids to the qwen.py side, whose test
    seeds nothing and therefore draws fresh prompts on every run.
    """
    random.seed(SEED)
    ids = [[random.randrange(VOCAB_SIZE) for _ in range(IN_LEN)] for _ in range(count)]

    h = hashlib.sha256()
    for row in ids:
        h.update(b",".join(b"%d" % t for t in row))
    digest = h.hexdigest()[:16]

    if dump is not None:
        dump.write_text(json.dumps(ids))
    return ids, digest


# Fields that carry parity with the qwen.py test. If one of these is missing or was
# renamed, the arm is mis-configured in a way no downstream assertion would catch, so
# the run must stop. (An unknown kwarg to LLM() is a TypeError, but a RENAMED field
# would just be dropped.)
REQUIRED_ENGINE_FIELDS = [
    "dtype", "seed", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "block_size", "num_gpu_blocks_override", "enable_prefix_caching", "enforce_eager",
]

# Fields we pass only to pin a default we do not want drifting -- their ABSENCE is
# already the behaviour we are asking for, so a missing one is not an error.
#   swap_space=0: disables CPU<->GPU KV swapping, which qwen.py has no equivalent of.
#     vLLM V1 dropped CPU swapping altogether (preemption recomputes instead), so on
#     those versions the field is gone and there is nothing to disable.
#   disable_log_stats=False: puts vLLM's periodic scheduler stats in the engine log,
#     which is how you confirm num_preempted == 0 -- the qwen.py side asserts the same
#     thing from sch_metrics. The interval is seconds, so it does not perturb timing,
#     and qwen.py logs its own metrics every 60 s anyway (req_metrics_interval).
# Anything dropped here is recorded in the header so the run stays auditable.
OPTIONAL_ENGINE_FIELDS = {"swap_space": 0, "disable_log_stats": False}


# Arms -- section 6d of the runbook, and log1003 section 6.4.
#
#   eager     (A)   no inductor, no graphs        -> engine design alone
#   compile   (A')  inductor, graphs OFF          -> A->A' is fusion
#   cudagraph (B)   inductor + graphs (default)   -> A'->B is launch elimination
#
# log1003 could only price the BUNDLE (A->B: 7.28x at batch 1, 1.17x at 512) because
# enforce_eager=True turns off compilation AND capture together. A' splits it, and the
# split decides which half to build in qwen.py first.
ARMS = ("eager", "compile", "cudagraph")

# What to look for in the engine log to prove an arm actually took. Printed at startup
# so the expectation sits next to the evidence in the same file.
ARM_EVIDENCE = {
    "eager":     "expect  'mode': <CompilationMode.NONE  and  0 x 'Capturing CUDA graph'",
    "compile":   "expect  'mode': <CompilationMode.VLLM_COMPILE  and  0 x 'Capturing CUDA graph'",
    "cudagraph": "expect  'mode': <CompilationMode.VLLM_COMPILE  and  >=1 x 'Capturing CUDA graph'",
}


def resolve_arm(arm: str) -> dict:
    """LLM() kwargs for this arm, with the compile-no-graphs path pre-flighted.

    Arm A' asks for compilation WITHOUT capture, which is a CompilationConfig field
    rather than an EngineArgs one -- so it needs its own check. A renamed or removed
    `cudagraph_mode` would otherwise be swallowed by the dict and leave A' silently
    identical to B, which is the one failure that would make the whole split useless.
    """
    if arm == "eager":
        return {"enforce_eager": True}
    if arm == "cudagraph":
        return {"enforce_eager": False}          # vLLM's own default: compile + capture

    try:
        from vllm.config import CompilationConfig
    except ImportError as e:
        sys.exit(f"cannot import CompilationConfig ({e}) -- find where this vLLM version "
                 f"keeps it and update resolve_arm(); do not fall back to arm B")

    fields = set(getattr(CompilationConfig, "__dataclass_fields__", {})) or set(
        getattr(CompilationConfig, "model_fields", {}))
    if "cudagraph_mode" not in fields:
        hints = sorted(f for f in fields if "graph" in f or "cudagraph" in f)
        sys.exit(f"CompilationConfig has no 'cudagraph_mode' in this vLLM version.\n"
                 f"Graph-related fields present: {hints}\n"
                 f"Arm A' is 'compile on, capture off'. Find the new spelling -- running "
                 f"without it would silently produce arm B again.")

    return {"enforce_eager": False, "compilation_config": {"cudagraph_mode": "NONE"}}


def read_cudagraph_mode(llm) -> str:
    """Best-effort: what vLLM RESOLVED to, not what we asked for. Never fatal.

    vLLM may override a requested cudagraph_mode (unsupported backend, a conflicting
    flag), so the row records the resolved value where it is reachable. The attribute
    path is version-dependent, hence the walk and the broad except.
    """
    for path in (("llm_engine", "vllm_config", "compilation_config", "cudagraph_mode"),
                 ("llm_engine", "model_config", "compilation_config", "cudagraph_mode")):
        try:
            o = llm
            for attr in path:
                o = getattr(o, attr)
            return str(o)
        except Exception:
            continue
    return "unreadable"


def resolve_engine_fields() -> tuple[dict, list[str]]:
    """Hard-fail on missing parity fields; drop absent optional ones and report them."""
    from vllm import EngineArgs

    known = set(EngineArgs.__dataclass_fields__)

    missing = [n for n in REQUIRED_ENGINE_FIELDS if n not in known]
    if missing:
        hints = {n: sorted(k for k in known if n.split("_")[-1] in k) for n in missing}
        sys.exit(f"EngineArgs is missing parity field(s) {missing} in this vLLM version.\n"
                 f"Similar field names present: {hints}\n"
                 f"These carry comparability with the qwen.py test -- find the new "
                 f"spelling and update REQUIRED_ENGINE_FIELDS, do not just delete them.")

    optional = {k: v for k, v in OPTIONAL_ENGINE_FIELDS.items() if k in known}
    dropped = [k for k in OPTIONAL_ENGINE_FIELDS if k not in known]
    return optional, dropped


def sampling_params():
    """Must equal data/generation_config.json, which is what qwen.py's config resolves to.

    temperature>0 keeps vLLM on the random-sampling path; temperature=0 would make it
    greedy (argmax only) and no longer comparable. top_p/top_k/repetition_penalty are
    the three that cost real device time on BOTH engines -- they are the measurement.
    """
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,     # same formula as qwen.py: where(l>0, l/rep, l*rep)
        presence_penalty=0.0,
        frequency_penalty=0.0,
        max_tokens=OUT_LEN,
        min_tokens=OUT_LEN,         # belt-and-braces; ignore_eos already guarantees it
        ignore_eos=True,            # == cfg.ignore_eos()
        detokenize=False,           # qwen.py never detokenizes inside the loop
        seed=SEED,
    )


def _pct(a: list[float], q: float) -> float:
    """numpy's default 'linear' percentile, stdlib-only.

    qwen.py's metrics.summarize() uses np.percentile, whose default method
    interpolates; replicating it exactly is three lines and keeps the two sides'
    p90/p99 comparable instead of nearly-comparable.
    """
    if len(a) == 1:
        return a[0]
    pos = q / 100.0 * (len(a) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def summarize(samples: list[float], scale: float = 1e3) -> dict:
    """Same keys, same scale and the same percentile method as metrics.summarize()."""
    if not samples:
        return {"n": 0}
    a = sorted(v * scale for v in samples)
    mean = sum(a) / len(a)
    var = sum((v - mean) ** 2 for v in a) / (len(a) - 1) if len(a) > 1 else 0.0
    return {"n": len(a), "mean": round(mean, 3), "std": round(var ** 0.5, 3),
            "p50": round(_pct(a, 50), 3), "p90": round(_pct(a, 90), 3),
            "p99": round(_pct(a, 99), 3), "max": round(a[-1], 3)}


# qwen.py's definitions, from metrics.analyze_metrics() -- match them or the columns
# are not comparable:
#   queueing = first_scheduled - arrival
#   ttft     = first_token - arrival          (includes queueing)
#   prefill  = first_token - first_scheduled  (this is what concurrency_sweep.txt
#                                              labels "TTFT ms")
#   tpot     = (last_token - first_token) / (n_output - 1)
# NOT reproducible here: the `itls` DISTRIBUTION. qwen.py pools every inter-token
# interval, which needs per-token timestamps; the offline path exposes only the first
# and last. tpot is the per-request mean of those intervals and is comparable.
#
# vLLM renames these between releases, and -- the trap -- the two generations do not
# share a clock. `arrival_time` has historically been wall-clock while the `*_ts`
# fields are monotonic, so a ttft taken as `first_token_ts - arrival_time` is off by
# the epoch (~1.7e9 s) and looks like a plausible number of milliseconds only after
# the subtraction silently overflows your expectations. Each schema below therefore
# names its own fields, and every quantity is computed inside ONE domain.
_SCHEMAS = (
    {   # vLLM 0.30.x -- field names confirmed from log1009/host2's latency_note
        "name": "ts",
        "need": ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"),
        "queued": "queued_ts", "scheduled": "scheduled_ts",
        "first": "first_token_ts", "last": "last_token_ts",
        "n_out": "num_generation_tokens",   # optional
        "ttft": "first_token_latency",      # optional: vLLM already computed it
    },
    {   # older RequestMetrics
        "name": "legacy",
        "need": ("arrival_time", "first_scheduled_time", "first_token_time", "last_token_time"),
        "queued": "arrival_time", "scheduled": "first_scheduled_time",
        "first": "first_token_time", "last": "last_token_time",
        "n_out": None, "ttft": None,
    },
)


def _attrs(obj) -> set:
    """Field names, whether RequestMetrics uses __dict__ or __slots__."""
    try:
        return set(vars(obj))
    except TypeError:
        return {a for a in dir(obj) if not a.startswith("_")}


def request_latencies(outs) -> tuple[dict, str]:
    """Per-request latency summaries, or an empty dict and the reason why.

    vLLM's V1 engine does not necessarily populate RequestOutput.metrics in the
    offline path. When it does not, say so -- a latency column derived from the
    aggregate would be a fabrication, and a silently absent one is worse.
    """
    metrics = [getattr(o, "metrics", None) for o in outs]
    missing = sum(1 for m in metrics if m is None)
    if missing:
        return {}, (f"RequestOutput.metrics is None for {missing}/{len(metrics)} requests -- "
                    f"this vLLM build does not expose per-request timing from LLM.generate(). "
                    f"Latency needs the streaming AsyncLLM path; see log1003 section 6.5.")

    present = _attrs(metrics[0])
    schema = next((sc for sc in _SCHEMAS if set(sc["need"]) <= present), None)
    if schema is None:
        return {}, (f"RequestMetrics matches no schema in _SCHEMAS; present: {sorted(present)}. "
                    f"Add one -- and keep each quantity inside a single clock domain.")

    g = lambda m, key: getattr(m, schema[key])
    use_ttft_field = schema["ttft"] in present if schema["ttft"] else False
    use_n_out_field = schema["n_out"] in present if schema["n_out"] else False

    queueing, ttft, prefill, tpot = [], [], [], []
    preempted = short = 0
    for o, m in zip(outs, metrics):
        if g(m, "first") is None or g(m, "queued") is None:
            continue
        queueing.append(g(m, "scheduled") - g(m, "queued"))
        prefill.append(g(m, "first") - g(m, "scheduled"))
        # Prefer vLLM's own TTFT: it is the one number guaranteed to be a duration
        # rather than a difference of two timestamps that may not share a clock.
        ttft.append(getattr(m, schema["ttft"]) if use_ttft_field
                    else g(m, "first") - g(m, "queued"))

        n_out = getattr(m, schema["n_out"]) if use_n_out_field else len(o.outputs[0].token_ids)
        if n_out != OUT_LEN:
            short += 1
        if n_out > 1 and g(m, "last") is not None:
            tpot.append((g(m, "last") - g(m, "first")) / (n_out - 1))

        preempted += getattr(m, "num_preemptions", 0) or 0

    if not ttft:
        return {}, f"RequestOutput.metrics present ({schema['name']}) but first/queued unset on every request"

    out = {"queueing": summarize(queueing), "ttft": summarize(ttft),
           "prefill": summarize(prefill), "tpot": summarize(tpot),
           "schema": schema["name"], "preemptions": preempted}

    # ttft should equal queueing + prefill. When it does not, the three were not read
    # from one clock -- which is the failure this split is designed to catch, and it
    # would otherwise produce numbers that look fine and are not.
    want = out["queueing"]["mean"] + out["prefill"]["mean"]
    got = out["ttft"]["mean"]
    note = ""
    if want and abs(got - want) / want > 0.05:
        note = (f"ttft mean {got:.1f} ms != queueing + prefill {want:.1f} ms -- the three are "
                f"probably not on one clock, or '{schema['ttft']}' means something else. "
                f"Treat the latency columns as suspect.")
    if short:
        note += f" {short}/{len(outs)} requests did not generate exactly {OUT_LEN} tokens."
    return out, note


def run_one(model: str, bz: int, prompts_ids: list[list[int]], arm: str, runs: int,
            optional: dict, arm_kwargs: dict) -> dict:
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    req_num = max(64, 10 * bz)          # == the test's req_num
    prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts_ids[:req_num]]
    sp = sampling_params()

    # Fresh engine per batch size: the test rebuilds LLMEngine (and the KV cache) for
    # every parametrised bz, so an engine reused across bz would not be the same thing.
    llm = LLM(
        model=model,
        dtype="bfloat16",
        seed=SEED,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=bz,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        block_size=BLOCK_SIZE,
        num_gpu_blocks_override=KV_TOKENS // BLOCK_SIZE,
        enable_prefix_caching=False,    # random prompts share no prefix anyway; keep it clean
        **arm_kwargs,                   # enforce_eager, and cudagraph_mode for arm A'
        **optional,                     # swap_space=0 where the version still has it
    )
    try:
        llm.generate(prompts[:bz], sp, use_tqdm=False)      # warmup is not optional (section 1b)

        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()                        # same boundary as engine.benchmark()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            elapsed = time.perf_counter() - t0

            o_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            # ignore_eos + max_tokens must give EXACTLY req_num * OUT_LEN. A short count
            # means EOS leaked through and the arms are no longer doing equal work.
            assert o_tok == req_num * OUT_LEN, f"bz={bz}: got {o_tok}, want {req_num * OUT_LEN}"
            # summarise per run and keep only the summary: holding three runs' worth of
            # RequestOutputs to post-process later is pointless memory.
            lat, lat_note = request_latencies(outs)
            samples.append((elapsed, o_tok, lat, lat_note))

        # median, not mean: section 6e -- one neighbour-induced outlier drags a mean
        elapsed = statistics.median(e for e, *_ in samples)
        # The reported latency must describe the SAME run as the reported elapsed. With an
        # even number of runs the median is not one of the samples, so take the nearest.
        _, o_tok, lat, lat_note = min(samples, key=lambda r: abs(r[0] - elapsed))
        return {
            "bz": bz,
            "req_num": req_num,
            "runs": runs,
            "elapsed": round(elapsed, 3),
            "elapsed_all": [round(e, 3) for e, *_ in samples],
            "i_tok": req_num * IN_LEN,
            "o_tok": o_tok,
            "tok_s": round(o_tok / elapsed, 2),             # OUTPUT ONLY -- see module docstring
            "arm": arm,
            "cudagraph_mode": read_cudagraph_mode(llm),     # resolved, not requested
            "latency": lat,                                 # {} when vLLM exposes none
            "latency_note": lat_note,                       # why, when it is {}
        }
    finally:
        del llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL_DIR", "") + "/qwen2.5-0.5b-instruct")
    ap.add_argument("--arm", choices=list(ARMS), default="eager",
                    help="eager = A (no inductor, no graphs); compile = A' (inductor, "
                         "graphs off); cudagraph = B (vLLM default). A->A' is fusion, "
                         "A'->B is launch elimination; A->B alone cannot separate them")
    ap.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    ap.add_argument("--runs", type=int, default=3, help="per point; the median is reported")
    ap.add_argument("--out", default="vllm_sweep.jsonl",
                    help="machine-readable results; nothing else is written here")
    ap.add_argument("--dump-prompts", default=None, metavar="PATH",
                    help="also write the prompt ids (~37 MB) -- only needed to feed "
                         "the same ids to the qwen.py side; regenerable from SEED")
    args = ap.parse_args()

    optional, dropped = resolve_engine_fields()
    arm_kwargs = resolve_arm(args.arm)

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    dump = pathlib.Path(args.dump_prompts) if args.dump_prompts else None
    prompts_ids, prompts_digest = build_prompts(max(64, 10 * max(batch_sizes)), dump)

    import vllm
    import torch

    header = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "arm": args.arm,
        "arm_kwargs": dict(arm_kwargs),     # what was requested; the rows carry what resolved
        "engine_fields_dropped": dropped,   # absent in this version; see OPTIONAL_ENGINE_FIELDS
        "prompts_seed": SEED,
        "prompts_sha256_16": prompts_digest,   # same digest == same input set
        "kv_tokens": KV_TOKENS,
        "in_len": IN_LEN,
        "out_len": OUT_LEN,
        "note": "tok_s is OUTPUT tokens / s, matching qwen.py's o_tok_num/elapsed",
    }
    # Everything section 7's checklist asks you to record, emitted with the data.
    # stdout is left to vLLM's own engine log (which stream it uses varies by version),
    # so redirect the whole process to a file and keep --out purely machine-readable:
    #   vllm_sweep.py --out results.jsonl > engine.log 2>&1
    log(f"header {json.dumps(header)}")
    log(f"arm={args.arm}: {ARM_EVIDENCE[args.arm]}")

    with open(args.out, "a") as f:
        f.write(json.dumps(header) + "\n")
        for bz in batch_sizes:
            log(f"bz={bz} starting ({args.runs} runs)")
            row = run_one(args.model, bz, prompts_ids, args.arm, args.runs, optional, arm_kwargs)
            f.write(json.dumps(row) + "\n")
            f.flush()
            lat = row["latency"]
            lat_s = (f"   prefill p50 {lat['prefill']['p50']:.1f} ms"
                     f"   tpot p50 {lat['tpot']['p50']:.2f} ms"
                     f"   preempt {lat['preemptions']}") if lat else "   latency: n/a"
            log(f"bz={bz:<5} {row['tok_s']:>10.2f} output tok/s   median {row['elapsed']}s"
                f"   runs {row['elapsed_all']}   cudagraph_mode={row['cudagraph_mode']}{lat_s}")
            if row["latency_note"]:
                kind = "latency WARNING" if lat else "latency unavailable"
                log(f"bz={bz} {kind}: {row['latency_note']}")


if __name__ == "__main__":
    main()
