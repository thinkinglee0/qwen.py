# Performance Analysis — Continuous-Batching Decode on RTX 4090 (log910 baseline)

**Artifacts analysed**

| Artifact | Path |
| --- | --- |
| Concurrency sweep | `log_vast/log910/benchmark_baseline/` |
| Decode-step profile | `log_vast/log910/profile_baseline/` |
| Cross-config sweeps | `log_vast/log910/benchmark_default/`, `…_pre_gather_cos_sin_plus_staging_sampling/` |
| Cross-config profiles | `log_vast/log910/profile_{default,compile_rope,pre_gather_cos_sin,staging_sampling,pre_gather_cos_sin_plus_staging_sampling}/` |
| Aggregated console output | `log_vast/log910/note` |

**Code under test** — current `src/qwen/` with all optimisation switches off:
`compile_rope = false`, `make_sampling_tensor_strategy = 0`, `pre_gather_cos_sin = false`.

**Executive summary**

The engine is **not** limited by the GPU. Across the whole concurrency sweep the GPU is idle
**70–88 %** of every decode step. Two distinct bottlenecks produce that idle time:

1. **Low concurrency (batch ≤ 64): CPU launch/dispatch bound.** A decode step issues a fixed
   **1 210 CUDA kernel launches** regardless of batch size, and roughly 3 800 ATen op invocations
   behind them. That costs ~22 ms of CPU per step while the GPU does only **3.1 ms** of real work.
   TPOT is flat at ~23–29 ms from batch 1 to batch 64 purely because of this.
2. **High concurrency (batch ≥ 128): the penalty/sampling tail.** `apply_penalties` is
   `O(batch × vocab)` in both GPU memory traffic and *Python* work, re-derived from scratch every
   step. At batch 1024 it consumes **77.6 ms of the 124 ms decode step (63 %)** on the CPU and
   **57 % of all GPU-busy time** at batch 512.

Corrected roofline numbers (see [§4.2](#42-roofline-position)) put the decode path at **4.3 % MBU**
at batch 1 and **4.2 % MFU** at batch 1024 — not the 66 % / 65 % originally printed by
`bench_viz.py`, whose roofline constants were hard-coded for Qwen2.5-**7B** while the model under
test is **0.5B**.
Prefill, by contrast, reaches **57 % MFU** at 8 192 tokens/step, which confirms the kernels
themselves are healthy and the deficit is orchestration overhead.

The clearest demonstration is `compile_rope`: it reduces GPU-busy time by 13 % and makes the step
**slower** (§4.5). While the GPU idles 86 % of the time, only CPU-side work counts.

---

## 1. System under test

### 1.1 Hardware

NVIDIA RTX 4090 (24 GB GDDR6X) on a vast.ai instance, per `README.md` §1.2.

| Roofline constant | Value |
| --- | --- |
| Peak HBM bandwidth | 1 008 GB/s |
| Peak BF16 dense (FP32 accumulate) | 165.2 TFLOP/s |

> The run logs do not record `torch.cuda.get_device_name()`; the hardware identity is taken from the
> project README. Adding the device name, driver and torch version to the metrics header would make
> these logs self-describing.

### 1.2 Model

Qwen2.5-0.5B-Instruct, `torch.bfloat16`, weights from `../models/qwen2.5-0.5b-instruct`.

| Property | Value |
| --- | --- |
| Layers | 24 |
| Hidden size | 896 |
| Intermediate size | 4 864 |
| Q heads / KV heads | 14 / 2 (GQA), `head_dim` 64 |
| Vocab | 151 936 (tied embeddings) |
| Parameters | 494.03 M total = 357.9 M body + 136.1 M embedding/LM head |
| BF16 weight footprint | **0.988 GB** (0.716 GB body + 0.272 GB LM head) |
| KV cache per token | 24 × 2 × 2 × 64 × 2 B = **12 KiB** |

### 1.3 Engine configuration

| Setting | Sweep | Profile |
| --- | --- | --- |
| `max_model_len` | 1 024 | 1 024 |
| `max_num_batched_tokens` | 8 192 | 8 192 |
| `long_prefill_token_threshold` | 8 192 (chunking effectively off) | 8 192 |
| `num_blocks` × `block_size` | 5 632 × 256 → **16.5 GiB** KV | 2 048 × 256 → 6.0 GiB KV |
| `max_num_seqs` | swept 1 → 1 024 | swept 1 → 512 |
| Sampling path | `do_sample=false`, `do_penalities=true` → penalties + `argmax` | same |
| Attention | `flash_attn_varlen` (splitkv) | same |
| RoPE | eager `apply_rotary` (`compile_rope=false`) | same |

---

## 2. Benchmark methodology

### 2.1 Concurrency sweep — [`test_benchmark.py:test_benchmark_sweep_batch_size`](tests/test_benchmark.py#L74)

```bash
pytest -x tests/test_benchmark.py::test_benchmark_sweep_batch_size
# SWEEP_BATCH_SIZES / SWEEP_BLOCKS override the defaults
python benchmark/tool/bench_viz.py --log-dir=log_vast/log910/benchmark_baseline/
```

* **Workload** — synthetic, fixed shape: every request is **512 random token ids in, 128 tokens
  out**, EOS ignored (`cfg.ignore_eos()`), so all sequences have identical cost and the decode
  batch never ragged-ends mid-measurement. Random ids avoid tokenizer/dataset skew; they do change
  the penalty mask density versus real text, which matters for §5.2.
* **Load shape** — closed-loop, saturating. All `req_num = max(64, 10 × batch)` requests are
  enqueued up front (`max_waiting = req_num`) and `run_to_completion()` drains them. There is no
  arrival process, so `queueing` and the reported `ttft` are admission-queue artefacts; the
  meaningful prefill figure is the `prefill` field (`first_token_time − first_schedule_time`), which
  is what `bench_viz.py` labels TTFT.
* **Warm-up** — batch 512 is run first and discarded, absorbing CUDA context init and cuBLAS
  autotune.
* **Isolation** — one `LLMEngine` per point, torn down with `gc.collect()` +
  `torch.cuda.empty_cache()` between points.
* **Emitted per point** — `benchmark_metrics.*.json` (mean/std/p50/p90/p99/max for queueing,
  prefill, TTFT, TPOT, ITL), `sch_metrics.*.json` (periodic snapshots), `step_metrics.*.json`
  (one JSON line **per scheduler step**).

**Metric definitions** (from [`metrics.py`](src/qwen/metrics.py)):

| Metric | Definition |
| --- | --- |
| `prefill` (reported as TTFT) | `first_token_time − first_schedule_time` — the step in which the prompt is prefilled |
| `tpot` | `(last_token_time − first_token_time) / len(itls)` per request, then averaged |
| `itls` | every inter-token gap, pooled across all requests — carries the tail |
| `tok_throughput` | `Σ output tokens / (last token − first schedule)` |

Because output length is fixed, `mean(ITL) == mean(TPOT)` by construction; their **percentiles**
differ and that difference is informative (§4.1).

### 2.2 Per-step instrumentation

Every step writes one line with paired CPU and GPU timings. Semantics matter for reading §4.3:

* `X` (e.g. `fwd`, `sample`) — **CPU wall time** of that section. Since launches are asynchronous,
  this measures Python + ATen dispatch + `cudaLaunchKernel`, *not* GPU execution.
* `X_gpu` — `cuda.Event.elapsed_time` between events bracketing the section, i.e. **stream-elapsed**
  time. On a single stream with a CPU that is the bottleneck, this includes stream idle gaps, so
  `fwd_gpu ≈ fwd` is itself a signal that the GPU is starved.
* `rope`/`rope_gpu` are **summed over all 24 layers** (`SchedulerStepMetrics.merge`).
* `dth` — `next_tokens.tolist()`, the one mandatory device→host sync per step; it is the amount of
  GPU backlog the CPU has to wait out.
* `ci` — commit: freeing blocks, appending tokens, updating request state.

### 2.3 GPU idle fraction — [`test_profile.py:test_profile_decode_idle_fraction`](tests/test_profile.py#L69)

```bash
SWEEP_PROFILE_BATCH_SIZES=1,8,32,128,512 pytest -x -s tests/test_profile.py::test_profile_decode_idle_fraction
```

Measures a **pure decode window**, isolated from prefill:

1. 256-token prompts, `max_waiting = batch_size` → exactly one prefill wave, no admissions or
   preemptions during measurement.
2. **32 warm-up steps**, then asserts every running request `is_decoding` and that none finished.
3. **Run A (clean)** — 5 steps between `cuda.synchronize()` calls → `wall_clean_us`. This is the
   number to quote; it carries no profiler overhead.
4. **Run B (traced)** — the same 5 steps under `torch.profiler` with `with_stack=True`; the Chrome
   trace gives `gpu_busy` as the **union** of all `kernel`/`gpu_memcpy`/`gpu_memset` intervals
   (union, not sum, so it stays correct with multiple streams).
5. `gpu_idle_fraction = 1 − gpu_busy_from_trace / wall_clean` — deliberately mixes the traced
   numerator with the clean denominator, which is the conservative direction: the GPU work is
   real, the CPU wall excludes profiler cost.
6. Guards: trace must contain exactly `PROFILE_STEPS` `aten::argmax` ops; kernel sum-vs-union
   overlap must be ≈0 on a single stream; KV drift between run A and run B is asserted < 5 %.

**Known distortion:** `with_stack=True` inflates the traced wall by **+46 % (batch 512) to +106 %
(batch 1)**, and inflates `fwd` from ~30 ms to ~48 ms in the `step_metrics` of steps 38–42. All
per-step numbers quoted from the profile directory below are taken from **steps 34–37 (run A,
untraced)**; kernel-level attributions come from run B, where only the *ratios* are used.

---

## 3. How to read the two harnesses together

The sweep and the profile use different prompt lengths (512 vs 256) and different KV pools, so their
absolute step times are not interchangeable. They are used for different things:

* **Sweep** → absolute latency/throughput, and the CPU-side breakdown per batch size.
* **Profile** → the GPU-busy fraction and the kernel-level attribution.

Cross-check: at batch 512, profile `wall_clean = 62.06 ms/step` versus the sum of that run's own
step-metric fields, 60.5 ms — consistent to 2.5 %.

---

## 4. Results

### 4.1 Concurrency sweep (baseline)

| batch | TTFT ms | TPOT ms | tok/s | tok/s/req | scaling eff. | gain/cost | ITL p50 | ITL p99 | ITL max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23.07 | 22.82 | 43.8 | 43.82 | 1.00 | — | 22.87 | 23.87 | 50.31 |
| 2 | 23.47 | 24.48 | 81.7 | 40.86 | 0.93 | 1.74 | 24.46 | 25.52 | 47.82 |
| 4 | 23.58 | 24.54 | 163.0 | 40.75 | 0.93 | 1.99 | 24.64 | 25.28 | 48.92 |
| 8 | 33.66 | 24.86 | 320.9 | 40.23 | 0.92 | 1.94 | 24.93 | 25.85 | 48.65 |
| 16 | 63.82 | 24.51 | 644.7 | 40.81 | 0.92 | 2.04 | 24.33 | 25.84 | 49.48 |
| 32 | 60.37 | 26.61 | 1 188.3 | 37.59 | 0.85 | 1.70 | 26.13 | 29.38 | 65.80 |
| 64 | 62.46 | 29.43 | 2 146.3 | 33.98 | 0.77 | 1.63 | 28.60 | 64.72 | 67.65 |
| 128 | 68.24 | 37.35 | 3 379.8 | 26.77 | 0.60 | 1.24 | 35.63 | 70.47 | 72.26 |
| 256 | 77.70 | 52.52 | 4 798.3 | 19.04 | 0.43 | 1.01 | 49.28 | 80.74 | 93.33 |
| 512 | 94.25 | 79.23 | 6 328.0 | 12.62 | 0.28 | 0.87 | 74.38 | 102.07 | 293.29 |
| 1024 | 151.11 | 140.93 | 7 028.1 | 7.10 | 0.16 | 0.62 | 150.11 | 159.43 | 476.47 |

* **Last profitable doubling ends at batch 256** (`throughput gain / TPOT cost > 1`).
* **Saturated throughput ≈ 7 028 tok/s.**
* No preemptions and no cache exhaustion at any point; Little's-law consistency is 0.967–1.000, so
  the scheduler keeps every slot busy and none of this is a queueing artefact.
* **TPOT is flat from batch 1 to 16** (22.8 → 24.5 ms) — adding 16× the work costs 7 % more time.
  That is the signature of a fixed per-step overhead, not of a loaded GPU.
* **Tail jitter from prefill/decode co-scheduling:** at batch 64, TPOT p99 is 29.8 ms but ITL p99 is
  **64.7 ms** — a decode token that lands in a step which also carries a prefill chunk costs ~2×.
  TPOT averages that away per request; ITL does not. At batch 512 the ITL max reaches 293 ms.

### 4.2 Roofline position

> **Correction to `bench_viz.py`.** At the time these sweeps were reduced, its roofline constants
> were `WEIGHT_BYTES = 15.2e9` / `MODEL_PARAMS = 7.6e9` — Qwen2.5-**7B**. The model actually
> benchmarked is **0.5B / 0.988 GB**, so the "MBU ~66 %" and "MFU ~65 %" in `log_vast/log910/note`
> are inflated by ~15×. The tool now carries an architecture-derived model registry and a `--model`
> flag (`--model qwen2.5-0.5b` / `qwen2.5-7b`) and prints the assumption it used; the values below
> are what it reports for the 0.5B.

**Decode** (bytes/step = 0.988 GB weights + batch × context × 12 KiB of KV):

| Point | bytes/step | step | achieved BW | MBU | GPU-busy only | MBU while busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sweep batch 1 | 0.995 GB | 22.82 ms | 43.6 GB/s | **4.3 %** | 3.15 ms | 31.4 % |
| profile batch 512 | 2.831 GB | 62.06 ms | 45.6 GB/s | **4.5 %** | 18.41 ms | 15.3 % |
| profile batch 512, transformer body only | 2.559 GB | — | — | — | 7.90 ms | **32.1 %** |
| sweep batch 1024 | 8.236 GB | 140.93 ms | 58.4 GB/s | **5.8 %** | — | — |

MFU at batch 1024 = `2 × 494.0M × 7028 tok/s / 165.2 TFLOP/s` = **4.2 %**, where N counts every
weight that does matmul work per decoded token: the 357.9 M transformer body plus one 136.1 M
LM-head pass. (Counting the body alone gives 3.1 %; the input embedding is a gather, not a GEMM,
and is excluded either way.)

**Prefill** (one step, 512-token sequences, `prefill_chunk = 1.0` for batch ≤ 16, so each row is a
single un-chunked step):

| batch | tokens/step | step | TFLOP/s | MFU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 23.07 ms | 16.4 | 9.9 % |
| 2 | 1 024 | 23.47 ms | 32.2 | 19.5 % |
| 4 | 2 048 | 23.58 ms | 64.1 | 38.8 % |
| 8 | 4 096 | 33.66 ms | 89.9 | 54.4 % |
| 16 | 8 192 | 63.82 ms | 94.8 | **57.4 %** |

This table is the cleanest statement of the problem. **A 2 048-token prefill costs the same 23.6 ms
as a single 1-token decode.** The step time is constant until ~2 k tokens and only then starts
tracking the work — i.e. the GPU becomes the limiter only above roughly 2–4 k tokens per step. At
8 192 tokens the same kernels hit 57 % MFU, so the kernels are fine; everything below that point is
paying a fixed ~22 ms orchestration tax.

### 4.3 Per-step CPU breakdown (sweep, pure-decode steps only)

Steps with `n_p == 0`, middle 80 % of each run, milliseconds:

| batch | sched | bld_meta | fwd | *(of which rope)* | logits | **sample** | dth | ci | **total** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.18 | 0.29 | 21.27 | 5.36 | 0.13 | 0.49 | 0.03 | 0.01 | **22.40** |
| 8 | 0.20 | 0.30 | 22.81 | 5.38 | 0.14 | 0.95 | 0.03 | 0.02 | **24.45** |
| 32 | 0.26 | 0.34 | 22.61 | 5.20 | 0.13 | 2.42 | 0.08 | 0.03 | **25.87** |
| 64 | 0.33 | 0.42 | 22.13 | 5.15 | 0.13 | 4.43 | 0.63 | 0.06 | **28.14** |
| 128 | 0.50 | 0.53 | 22.54 | 5.16 | 0.13 | 9.42 | 1.70 | 0.13 | **34.96** |
| 256 | 0.84 | 0.79 | 22.72 | 5.21 | 0.14 | 19.75 | 3.76 | 0.29 | **48.27** |
| 512 | 1.77 | 1.24 | 22.34 | 5.20 | 0.14 | 38.67 | 7.84 | 0.59 | **72.59** |
| 1024 | 3.88 | 2.12 | 23.34 | 5.25 | 0.14 | 77.59 | 15.88 | 1.09 | **124.04** |

Three facts fall out:

1. **`fwd` is a constant ~22.3 ms across a 1024× range of batch size.** It is not model compute; it
   is the CPU walking the graph.
2. **`rope` is a constant ~5.2 ms, i.e. 23 % of `fwd`** — for an operation whose arithmetic content
   is a handful of multiply-adds.
3. **`sample` is perfectly linear in batch**, ~75.7 µs per sequence, and overtakes `fwd` between
   batch 256 and 512. At batch 1024 it is **63 % of the whole step**.

### 4.4 GPU idle fraction and kernel attribution (profile)

| batch | wall_clean/step | GPU busy/step | **GPU idle** | traced CPU/step | CPU:GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23.15 ms | 3.15 ms | **86.4 %** | 31.2 ms | 9.7× |
| 8 | 31.29 ms | 3.84 ms | **87.7 %** | 38.6 ms | 9.9× |
| 32 | 34.93 ms | 4.08 ms | **88.3 %** | 40.6 ms | 9.8× |
| 128 | 40.44 ms | 6.59 ms | **83.7 %** | 38.7 ms | 5.8× |
| 512 | 62.06 ms | 18.41 ms | **70.3 %** | 43.1 ms | 2.3× |

`cudaLaunchKernel` is called **6 050 times per 5 steps = 1 210 launches per step at every batch
size** — batch-invariant, as expected for a graph that is walked once per step regardless of shape.
Average kernel duration at batch 1 is 3.15 ms / 1 210 ≈ **2.6 µs**, against ~6.5 µs of traced CPU
per launch. The GPU finishes each kernel long before the next one is submitted.

Splitting GPU-busy time by call frequency cleanly separates the transformer body (per-layer
kernels, >4 calls/step) from the logits+sampler tail (batch-level kernels, ≤4 calls/step):

| batch | transformer body | logits + sampler | total | sampler share |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.10 ms | 0.04 ms | 3.14 ms | 1 % |
| 8 | 3.75 ms | 0.08 ms | 3.83 ms | 2 % |
| 32 | 3.47 ms | 0.57 ms | 4.04 ms | 14 % |
| 128 | 4.00 ms | 2.54 ms | 6.54 ms | 39 % |
| 512 | 7.90 ms | 10.41 ms | 18.31 ms | **57 %** |

(Totals reconstruct the trace-union `gpu_busy` of the table above to within 0.5 %, which
validates the split.)

The body grows only **2.5× while batch grows 512×** — exactly the behaviour of a weight-bandwidth
bound decode with enormous spare capacity. Meanwhile the sampler tail grows 260× and takes over.
Of the 10.41 ms batch-level total at batch 512, the LM-head GEMM is only 0.86 ms; **~9.5 ms is the
penalty path.**

Top batch-level kernels at batch 512 (per step): 2.02 ms (×2 elementwise), 1.10 ms
(`torch.where`), 1.10 ms (×2 elementwise), 0.90 ms (×2), 0.86 ms (LM-head GEMM), 0.84 ms
(`aten::div`), 0.84 ms (`aten::sub`), 0.76 ms (`masked_fill_`), 0.59 ms (×4 `fill_`), 0.38 ms
(`argmax`). Every one of these is a full `[512, 151936]` pass.

### 4.5 Effect of the optimisation switches

**In the sweeps.** Three sweeps exist in `log910`, but their config headers show that
`benchmark_default` and `benchmark_pre_gather_cos_sin_plus_staging_sampling` were run with
**identical** settings (`compile_rope=false, make_sampling_tensor_strategy=1,
pre_gather_cos_sin=true`; both logs print *"Using eager path for apply_rotary"*). They therefore
serve as a **run-to-run repeatability estimate**, not as two configurations.

| Sweep | compile_rope | sampling strategy | pre_gather | tok/s @1024 | TPOT @1 | rope/step |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `benchmark_baseline` | false | 0 | false | 7 028.1 | 22.82 ms | 5.36 ms |
| `benchmark_default` | false | 1 | true | 7 013.9 | 22.63 ms | 5.29 ms |
| `benchmark_pre_gather…` | false | 1 | true | 7 077.9 | 22.31 ms | 5.21 ms |

The two identical runs differ by **0.9 %** and the baseline sits between them: **the staged sampling
tensor plus `pre_gather_cos_sin` produce no improvement above run-to-run noise.** That 0.9 % is the
significance floor for everything else in this document.

**In the profile.** All six switch combinations *were* profiled, and the result is the sharpest
single piece of evidence in this report. Untraced wall per decode step (ms):

| Config | bs 1 | bs 8 | bs 32 | bs 128 | bs 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 23.15 | 31.29 | 34.93 | 40.44 | 62.06 |
| `compile_rope` | 24.98 | 32.96 | 35.49 | 41.81 | 65.11 |
| `pre_gather_cos_sin` | **21.95** | **29.55** | 31.72 | 38.51 | **59.17** |
| `staging_sampling` | 23.35 | 31.48 | 33.51 | 39.78 | 61.91 |
| `pre_gather` + `staging` | 27.16 | 30.55 | **31.72** | **37.15** | 60.77 |
| all three (`default`) | 23.54 | 31.16 | 32.41 | 41.03 | 61.20 |

GPU-busy per decode step (ms) for the same runs:

| Config | bs 1 | bs 8 | bs 32 | bs 128 | bs 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.147 | 3.843 | 4.077 | 6.595 | 18.411 |
| `compile_rope` | 2.726 | 3.369 | 3.631 | 6.182 | 17.962 |
| `pre_gather_cos_sin` | 3.005 | 3.730 | 3.865 | 6.489 | 18.258 |
| `staging_sampling` | 3.144 | 3.842 | 4.075 | 6.589 | 18.415 |
| `pre_gather` + `staging` | 3.005 | 3.733 | 3.960 | 6.477 | 18.241 |
| all three (`default`) | **2.584** | **3.253** | **3.396** | **6.060** | **17.802** |

Read the two tables together:

* **`compile_rope` works, and it buys nothing.** It cuts GPU-busy time by **13 %** at batch 1 and
  **11 %** at batch 32; with all three switches on it is **18 %** and **17 %**.
  And yet `compile_rope` alone makes the step **slower** (23.15 → 24.98 ms at batch 1, 62.06 → 65.11
  at batch 512), because the Inductor-generated wrapper costs more CPU per call than it saves in GPU
  time. When the GPU is idle 86 % of the step, removing GPU work is worth exactly zero; adding CPU
  work is not.
* **`pre_gather_cos_sin` is the only switch that helps**, and it helps for the expected reason: it
  removes CPU launches (two index gathers per layer): **4–9 % off wall at every batch size**, with
  GPU-busy time essentially unchanged. That win then vanished in the full sweep, i.e. it sits at the
  edge of the noise floor.
* **`staging_sampling` is neutral** at every batch size — consistent with §5.2: it replaces six
  small H2D copies while the `flat_idx` copies and the penalty kernels are untouched.

Caveat: `wall_clean` is a five-step measurement, so single cells are noisy (the 27.16 ms at
`pre_gather`+`staging`, batch 1, is an outlier against its own 21.95/31.72 neighbours). `gpu_busy`
is far more stable and the trends there are consistent across all five batch sizes.

---

## 5. Bottleneck analysis

### 5.1 Bottleneck 1 — the decode step is CPU dispatch bound

**Evidence:** 1 210 launches/step independent of batch; `fwd` flat at 22.3 ms from batch 1 to 1024;
`fwd_gpu ≈ fwd` (the CUDA-event span covering the forward equals its CPU launch time, meaning the
span is almost entirely stream idle); `dth = 0.03 ms` at batch 1 (by the time the CPU asks for the
tokens, the GPU has been finished for ~19 ms); 86 % GPU idle; a 2 048-token prefill costing the same
as a 1-token decode.

**Where the launches come from.** Per decode step the trace shows ~317 `aten::mul`, 169
`aten::linear`, 164 `aten::copy_`, 163 `aten::_to_copy`, 145 `aten::add`, 49 each of
`aten::index`/`mean`/`pow`/`rsqrt`, 48 each of `aten::cat`/`index_copy_`, plus ~1 440 pure-view ops
(`as_strided` 495, `view` 291, `reshape` 241, `transpose` 241, `t` 169). Two constructs dominate:

* **Eager RoPE** — [`apply_rotary`](src/qwen/rope.py#L9) decomposes into slice ×2, mul ×4, sub, add,
  cat ≈ 8 kernels, called twice per layer (q and k) plus 2 index gathers ⇒ **~18 kernels × 24 layers
  ≈ 430 launches/step, ~35 % of all launches** for ~22 % of forward stream time (`rope_gpu` 4.5 ms
  of `fwd_gpu` 22.3 ms).
* **Dtype churn** — 163 `aten::_to_copy` per step. `cos_cached`/`sin_cached` are registered as
  fp32 buffers while q/k are bf16, so every RoPE multiply type-promotes and materialises a copy.

**Why the switches didn't help.** `compile_rope` is the switch aimed at this, and §4.5 shows it does
what it claims on the GPU (−13 % busy time at batch 1) while making the wall clock *worse*, because
the Inductor wrapper adds CPU per call. `pre_gather_cos_sin` removes only 2 of ~18 RoPE launches per
layer. Fusing RoPE helps here only if the fused version also **costs less CPU per call** — which
means a hand-written kernel or a CUDA-graph capture, not `torch.compile` on a hot Python path.

### 5.2 Bottleneck 2 — `apply_penalties` is O(batch × vocab) on both sides

The benchmark runs `do_sample=false, do_penalities=true`, so
[`model.sampler`](src/qwen/model.py#L72) takes the **penalties + argmax** path — no top-k, no
top-p, no multinomial. All of the sampler cost measured above comes from
[`apply_penalties`](src/qwen/sampling.py#L157).

**GPU side.** Each step materialises several full `[batch, 151936]` tensors — 78 MB at batch 128,
**311 MB at batch 512, 622 MB at batch 1024** *per fp32 tensor*:

```python
counts = torch.zeros(bsz * vocab_size, ...)        # ×2 (prompt + output), int32
rep = rep_pen[:, None].repeat(1, vocab_size)       # fp32, full materialisation
rep[~(prompt_mask | output_mask)] = 1.0            # boolean index_put over the whole thing
logits = torch.where(logits > 0, logits / rep, logits * rep)   # 3 more full passes
logits = logits - freq_pen[:, None] * out_counts   # 2 more
logits = logits - pres_pen[:, None] * output_mask  # 2 more
```

That is >4 GB of traffic per step at batch 512, in ~15 unfused elementwise kernels — matching the
measured 9.5 ms.

**CPU side — the larger problem.** `bin_counts_and_mask` rebuilds the index list in pure Python
*every step*:

```python
flat_idx = [row * vocab_size + t for row, seq in enumerate(token_ids) for t in seq]
```

called twice, over **prompt tokens and all output tokens so far**. At batch 1024 with 512-token
prompts that is ~650 k Python-level integers materialised per step, which is precisely the
**77.6 ms** of `sample` CPU time observed. The cost is `O(Σ context length)` per step even though
exactly **one** new token per sequence changes between steps.

**Blocking pageable copies.** The profile shows **16 `cudaMemcpyAsync` per step averaging 487 µs**
at batch 512 — 7.8 ms/step, **18 % of traced CPU time** — against 7.4 µs each at batch 1. Fifteen of
them are `Memcpy HtoD (Pageable → Device)`: the two `flat_idx` lists above (~1 MB of int64 at batch
512), the six per-step `torch.tensor([...], device=cuda)` sampling-parameter tensors of
[`from_sampling_list_0`](src/qwen/sampling.py#L83), and the attention-metadata tensors. A pageable
H2D copy is synchronous, so each one drains the stream; the 487 µs is not transfer cost, it is the
CPU waiting. `make_sampling_tensor_strategy=1` already fixes six of these by staging through pinned
memory — the remaining ones, especially `flat_idx`, are why that switch alone changes nothing.

**Additional syncs on the sampling path (not exercised here).** With `do_sample=true`,
`apply_top_k` calls `disabled.all()` and `int(top_k[~disabled].max().item())`, and `sample` calls
`dead.any()` — three more blocking device→host round trips per step, plus a `torch.sort` over
`[batch, 151936]`.

### 5.3 Bottleneck 3 — per-request Python in the scheduler path

`sched + bld_meta + ci` grows from 0.48 ms at batch 1 to **7.1 ms at batch 1024** — small in
absolute terms, but it is per-request Python that will dominate once (1) and (2) are fixed.
`dth` (15.9 ms at batch 1024) is mostly legitimate waiting for the penalty kernels, and should
shrink with them.

### 5.4 What is *not* a bottleneck

* **The scheduler's policy** — zero preemptions, zero cache exhaustions, Little's law 0.967–1.000
  across the entire sweep.
* **Attention** — `flash_fwd_splitkv` is 2.6 ms/step at batch 512, 14 % of GPU busy.
* **KV cache capacity** — 16.5 GiB pool, peak usage well under it at every point.
* **The GEMMs** — cuBLAS/CUTLASS bf16 kernels reach 57 % MFU in prefill.

---

## 6. Headroom and prioritised recommendations

A decode step at batch 512 needs **7.9 ms of transformer-body GPU work + ~1 ms of LM head**. It
currently takes **62–79 ms**. Even allowing generously for a fused penalty path and graph-replay
overhead, a target of ~10–12 ms/step is defensible, i.e. **5–6× on throughput** and a similar factor
on TPOT. At batch 1, GPU busy is 3.15 ms against a 22.8 ms step — **~6× on single-stream latency**.

| # | Change | Attacks | Expected effect | Effort |
| --- | --- | --- | --- | --- |
| 1 | **CUDA Graphs for the decode step** (capture per bucketed batch size, replay) | §5.1 | Collapses 1 210 launches to one replay. Directly targets the 70–88 % idle. Biggest single win at batch ≤ 256. | High |
| 2 | **Incremental penalty state** — keep a persistent `[batch, vocab]` count buffer (or a per-sequence sparse token set) on GPU and scatter only the *one* new token per step | §5.2 CPU | Removes the O(Σ context) Python rebuild: −77 ms/step at batch 1024 | Medium |
| 3 | **Fuse the penalty math** into one Triton/`torch.compile` kernel over `[batch, vocab]`, skip `rep.repeat` entirely (broadcast instead) | §5.2 GPU | ~15 full-vocab passes → 1–2; −6 to −8 ms/step at batch 512 | Medium |
| 4 | **Fuse RoPE** into one kernel per layer applying q and k together — a hand-written/Triton kernel, **not** `torch.compile` (§4.5: the Inductor wrapper's CPU cost exceeds its GPU saving here) | §5.1 | ~430 launches/step → ~48; up to −5 ms/step at every batch size, but only if per-call CPU drops too | Medium |
| 5 | **Store `cos_cached`/`sin_cached` in bf16** | §5.1 | Removes ~160 `_to_copy` per step | Trivial |
| 6 | **Fuse RMSNorm** (`mean`/`pow`/`rsqrt`/`mul`/`add` ≈ 5 kernels × 49) | §5.1 | ~245 launches/step → 49 | Low |
| 7 | **Pin the remaining H2D staging** — `flat_idx` in [`bin_counts_and_mask`](src/qwen/sampling.py#L128) and the attention-metadata tensors — so no per-step copy is pageable | §5.2 | Removes up to 15 synchronous stream drains/step (7.8 ms at batch 512) | Low |
| 7b | **Remove `.item()`/`.any()`/`.all()` syncs** from `apply_top_k` / `sample`; make disabled-row and dead-row handling branch-free | §5.2 | 3 fewer stream drains/step — only matters once `do_sample=true` | Low |
| 8 | ~~Fix `bench_viz.py` roofline constants~~ **done** — model registry + `--model` flag. Still to do: record device name / driver / torch version in the metrics header | reporting | Correct MBU/MFU, self-describing logs | Trivial |

**Suggested order:** 5 → 7 → 6 (cheap, immediately measurable), then 2 → 3 (unblocks high
concurrency), then 1 (the structural fix). Re-run the sweep after each; the 0.9 % run-to-run noise
established in §4.5 is the significance floor — anything smaller is not a result.

---

## 7. Caveats

1. **Synthetic workload.** Uniform 512-in/128-out with random token ids. Real traffic has variable
   lengths (so ragged decode batches and more scheduler churn) and real text (so denser penalty
   masks, but the same `O(batch × vocab)` shape). The `sharegpt` variants of both tests exist for
   this and should be used to confirm §5.2 on realistic token distributions.
2. **Closed-loop, fully pre-enqueued.** The reported `ttft`/`queueing` fields are admission-queue
   artefacts of a saturating load and should not be read as user-facing TTFT; only the `prefill`
   field is meaningful.
3. **`do_sample = false`.** The full top-k/top-p/multinomial path is *not* exercised. It adds a
   `torch.sort` over `[batch, 151936]`, which will be considerably more expensive than the penalty
   path measured here, so §5.2 is a lower bound for a sampling deployment.
4. **Profiler distortion.** `with_stack=True` adds +46 % to +106 % wall. Mitigated as described in
   §2.3, but kernel-level percentages carry that bias.
5. **Different KV pool between harnesses** (16.5 GiB sweep vs 6.0 GiB profile) and different prompt
   length (512 vs 256). Absolute step times are not directly comparable across the two; ratios are.
6. **Hardware identity is inferred** from `README.md`, not recorded in the logs (see §1.1).
7. **`benchmark_default` and `benchmark_pre_gather_cos_sin_plus_staging_sampling` share one
   configuration**, so no *sweep* in `log910` exercises `compile_rope=true`; the throughput effect of
   that switch is inferred from the profile runs (§4.5), which measure 5 steps each and are
   correspondingly noisy on wall time.
