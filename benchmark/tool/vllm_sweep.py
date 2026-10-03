"""vLLM counterpart of tests/test_benchmark.py::test_benchmark_sweep_batch_size.

Runs in the SEPARATE vLLM venv (section 6a of env/vastai/vast-evn-build.md), which
has only vllm and its dependencies -- so this file imports nothing from qwen.py and
nothing outside the stdlib + vllm. Do not add `from qwen...` here.

"$VLLM_VENV/bin/python" benchmark/tool/vllm_sweep.py --arm eager --runs 3 --out ./log/vllm_eager.jsonl > ./log/vllm_eager.log 2>&1

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


def run_one(model: str, bz: int, prompts_ids: list[list[int]], eager: bool, runs: int,
            optional: dict) -> dict:
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
        enforce_eager=eager,            # arm A / arm B, section 6d
        **optional,                      # swap_space=0 where the version still has it
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
            samples.append((elapsed, o_tok))

        # median, not mean: section 6e -- one neighbour-induced outlier drags a mean
        elapsed = statistics.median(e for e, _ in samples)
        o_tok = samples[0][1]
        return {
            "bz": bz,
            "req_num": req_num,
            "runs": runs,
            "elapsed": round(elapsed, 3),
            "elapsed_all": [round(e, 3) for e, _ in samples],
            "i_tok": req_num * IN_LEN,
            "o_tok": o_tok,
            "tok_s": round(o_tok / elapsed, 2),             # OUTPUT ONLY -- see module docstring
            "arm": "eager" if eager else "cudagraph",
        }
    finally:
        del llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL_DIR", "") + "/qwen2.5-0.5b-instruct")
    ap.add_argument("--arm", choices=["eager", "cudagraph"], default="eager",
                    help="eager = arm A (your scheduler/sampler); cudagraph = arm B")
    ap.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    ap.add_argument("--runs", type=int, default=3, help="per point; the median is reported")
    ap.add_argument("--out", default="vllm_sweep.jsonl",
                    help="machine-readable results; nothing else is written here")
    ap.add_argument("--dump-prompts", default=None, metavar="PATH",
                    help="also write the prompt ids (~37 MB) -- only needed to feed "
                         "the same ids to the qwen.py side; regenerable from SEED")
    args = ap.parse_args()

    optional, dropped = resolve_engine_fields()

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

    with open(args.out, "a") as f:
        f.write(json.dumps(header) + "\n")
        for bz in batch_sizes:
            log(f"bz={bz} starting ({args.runs} runs)")
            row = run_one(args.model, bz, prompts_ids, args.arm == "eager", args.runs, optional)
            f.write(json.dumps(row) + "\n")
            f.flush()
            log(f"bz={bz:<5} {row['tok_s']:>10.2f} output tok/s   median {row['elapsed']}s"
                f"   runs {row['elapsed_all']}")


if __name__ == "__main__":
    main()
