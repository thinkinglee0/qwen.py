# Performance Analysis — Sampling-Enabled Decode on RTX 4090 (log914 baseline)

**Artifacts analysed**

| Artifact | Path |
| --- | --- |
| Concurrency sweep, run 1 | `log_vast/log914/benchmark_baseline/` |
| Concurrency sweep, run 2 | `log_vast/log914/benchmark_baseline2/` |
| Decode-step profile, run 1 / run 2 | `log_vast/log914/profile_baseline/`, `…/profile_baseline2/` |
| Per-field run diff | `log_vast/log914/mean_step_metrics.{256,512}.log` |
| Flagged steps | `…/step_metrics.*.json.anomaly` (written beside each dump) |

**Code under test** — `src/qwen/` with all optimisation switches off: `compile_rope = false`,
`make_sampling_tensor_strategy = 0`, `pre_gather_cos_sin = false`.

**Predecessor** — [`performance_analysis_log910.md`](./performance_analysis_log910.md). The engine
configuration is **not** identical; see [§6](#6-what-changed-since-log910) before comparing any
number across the two.

---

## Executive summary

The sweep was run **twice, twenty minutes apart, with no code change**, so the noise floor is
measured rather than assumed: throughput reproduces within **1.1 %** on the four largest batches.
Every finding below is between 5× and 100×.

1. **Sampling is now the engine.** At batch 512 it is **61.5 % of total wall-clock time** and 75 %
   of every decode step; at batch 1024, 83.5 % of a decode step. It scales **linearly with batch**
   — 1.0 ms at batch 1, 131.0 ms at batch 1024 — and it is device-side work, not launch overhead.
2. **The model forward still costs ~18.5 ms whatever you put in it.** `fwd_gpu` moves from 16.6 ms
   at batch 1 to 18.9 ms at batch 1024: +14 % for 1024× the tokens. A mixed step carrying 7 500
   prefill tokens spends 17.4 ms of device time in it. The GPU is idle **82.8 %** of a batch-1
   decode step and still 45.5 % idle at batch 512.
3. **Peak throughput is 5 300 tok/s**, reached at batch 1024, and the last profitable doubling ends
   at 512. At that peak the model's weight traffic is ~11 GB/s against the card's ~1 008 GB/s.

log910's prediction is confirmed and quantified: enabling `do_sample` added the full-vocabulary
`torch.sort` it warned about, costing **~25 % of peak throughput**, while the model path itself got
~20 % faster over the same period.

---

## 1. System under test

### 1.1 Hardware

NVIDIA RTX 4090 (24 GB GDDR6X) on a vast.ai instance. Peak HBM bandwidth 1 008 GB/s; peak BF16
dense (FP32 accumulate) 165.2 TFLOP/s.

### 1.2 Model

Qwen2.5-0.5B-Instruct, `torch.bfloat16`. 494.03 M parameters = 357.9 M body + 136.1 M embedding /
LM head; **0.988 GB** of BF16 weights (0.716 GB body + 0.272 GB LM head); 12 KiB of KV per token.

### 1.3 Engine configuration

| Setting | Sweep | Profile |
| --- | --- | --- |
| `max_model_len` | 1 024 | 1 024 |
| `max_num_batched_tokens` | 8 192 | 8 192 |
| `long_prefill_token_threshold` | 8 192 | 8 192 |
| `num_blocks` × `block_size` | 4 096 × 256 → **12.0 GiB** KV | 5 632 × 256 → 16.5 GiB KV |
| `max_num_seqs` | swept 1 → 1 024 | swept 1 → 512 |
| `use_d_first_schedule` | true | false |
| **Sampling path** | **`do_sample=true`**, `do_penalities=true`, `temperature=0.7`, `top_k=20`, `top_p=0.8` | same |
| Attention | `flash_attn_varlen` (paged) | same |
| RoPE | eager `apply_rotary` | same |

The sampling row is the one that matters: unlike log910, the full
[`sample()`](../src/qwen/sampling.py#L218) path runs — temperature, `apply_top_k`, `apply_top_p`,
`multinomial`, `argmax`, on top of the penalties.

---

## 2. Methodology

### 2.1 Workload

Synthetic and fixed-shape, as in log910: every request is **512 random token ids in, 128 tokens
out**, EOS ignored, `req_num = max(64, 10 × batch)` enqueued up front and drained by
`run_to_completion()`. Closed-loop and saturating, so the reported `queueing`/`ttft` are
admission-queue artefacts, not latency.

### 2.2 Two runs, so the noise floor is measured

The whole sweep ran twice (12:04 and 12:24) with no change in between. That gives a **direct
reproducibility bound** instead of a guess:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Δ tok/s, run 2 vs run 1 | +1.1 % | +0.8 % | −2.8 % | +1.5 % | −3.3 % | −0.2 % | −5.5 % | −0.7 % | −0.8 % | −0.5 % | −0.6 % |

The four largest batches — the long runs, 1 280 to 10 240 requests — agree within **1.1 %**. The
small batches drift up to 5.5 %, but those runs finish in 30–40 s on 64–640 requests.

**Working rule: treat anything under 3 % as noise.**

### 2.3 Per-field run diff — [`test_profile.py:test_mean_step_metrics`](../tests/test_profile.py#L448)

```bash
STEP_METRICS_DIR=log_vast/log914 \
STEP_METRICS_RUNS=benchmark_baseline,benchmark_baseline2 \
STEP_METRICS_BATCH=512 \
pytest -x -s tests/test_profile.py::test_mean_step_metrics
```

Prints mean / std / cv / Δ / z per float field for one run, or a field-by-field diff of two, and
writes the whole report to `<common dir>/mean_step_metrics.<batch>.log`.

Two corrections make its verdicts usable:

* **Decode-only.** Rows with `n_p > 0` are dropped — a step carrying prefill tokens is a different
  workload and averaging it in is meaningless.
* **`n_eff`, not `n`.** Consecutive decode steps are *not* independent: the KV grows every step, so
  each series is autocorrelated. With `se = std/√n` over ~1 000 steps, a 0.5 % gap "passes" a 2σ
  test and 15 of 17 fields come back significant. The test deflates to an AR(1) effective sample
  size, `n_eff = n·(1−ρ)/(1+ρ)`. At batch 512 `sample` has `n_eff = 31` out of 965 (ρ ≈ 0.94 — the
  series is almost entirely KV-growth trend), while `sched` keeps `n_eff = 964` (ρ ≈ 0 — pure
  spikes). After the correction, **nothing at batch 512 is outside noise**; at batch 256 a uniform
  +1–3 % lands on host-side and device-side fields alike, which is the signature of machine drift.

### 2.4 Anomalous-step detection

The same test writes `<dump>.anomaly` (JSONL) beside each dump: every step where a field is
**≥ 1.5× its local median and at least 0.5 ms above it**. Both conditions are needed — the ratio
alone fires constantly on the microsecond fields, and a distribution-based cut does not work here
(these series are fat-tailed enough that a Gaussian-calibrated 3.5 modified-z flags a third of the
run). The local median is a 51-step rolling median, because the run is not stationary.

### 2.6 Notation — `sample`, `sample_gpu`, and "sampling"

Three things carry almost the same name and mean different things. Throughout this report:

| Term | What it is | Clock |
| --- | --- | --- |
| **sampling** | the *work*: penalties → top-k → top-p → multinomial → argmax, i.e. everything [`model.sampler`](../src/qwen/model.py#L73) does. Used in prose and in aggregate columns. | — |
| **`sample`** | a field in `step_metrics`: host wall time inside the `timed(…, "sample")` block of [`engine.forward`](../src/qwen/engine.py#L67). | CPU |
| **`sample_gpu`** | a field in `step_metrics`: the CUDA-event window around that same block — the span the *stream* takes to get from the start event to the stop event. | GPU |

The two fields measure the same code and are **not** interchangeable:

* In a **pure decode** step they agree closely (69.4 vs 68.4 ms at batch 512), because the host is
  the thing feeding the GPU and neither runs far ahead of the other.
* In a **mixed prefill/decode** step they diverge hard — `sample` 101.4 ms against `sample_gpu`
  40.8 ms — because the host spends most of that block *blocked* on the prefill work queued
  earlier in the step, not sampling (§7.1).
* `sample_gpu` is a **window, not busy time**: stream gaps inside it are counted (§7.2).

Every aggregate in §3.3 and §4 is built on `sample_gpu`, because it is the only one of the two that
stays meaningful when prefill steps are in the mix. `sample` appears in the CPU breakdown (§3.2)
and nowhere else.

### 2.5 GPU idle fraction — [`test_profile.py:test_profile_decode_idle_fraction`](../tests/test_profile.py#L101)

Unchanged in principle from log910 — warm up, run a clean window, then re-run the same window under
`torch.profiler` and export a chrome trace — with three fixes made while producing this report:

* The step count is verified against a `record_function("decode_step")` marker instead of counting
  `aten::argmax`. Op counts per step are an implementation detail: `torch.multinomial(num_samples=1)`
  defaults to `replacement=False`, takes the Gumbel path, and emits an **internal** `aten::argmax`,
  so the sampler produces two per step, not one.
* GPU-busy time is the union of `cat ∈ {kernel, gpu_memcpy, gpu_memset}` spans, and the
  `key_averages()` cross-check now keeps only rows whose name appears in that set. A name blocklist
  does not work: with `enable_cuda_sync_events`, `key_averages()` also carries device rows such as
  `Context Sync` whose duration is *host wait*. At batch 1 that alone summed to 177 ms of "GPU
  time" against 16 ms of real kernels in a 167 ms window.
* The clean window (`MEASURE_STEPS`) and the profiled window (`PROFILE_STEPS`) no longer have to be
  the same length, so every comparison is scaled per step first.

---

## 3. Results

### 3.1 Concurrency sweep (run 1)

| batch | tok/s | Δ run 2 | step ms | sample ms | fwd ms | other ms | sample % step | sampling % of run | TPOT ms | ITL p99 | elapsed s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.8 | +1.1 % | 18.71 | 1.03 | 16.62 | 1.06 | 5.5 % | 5.5 % | 19.0 | 20.5 | 155.3 |
| 2 | 95.4 | +0.8 % | 20.69 | 1.32 | 18.24 | 1.13 | 6.4 % | 6.4 % | 21.0 | 24.8 | 85.8 |
| 4 | 198.5 | −2.8 % | 19.92 | 1.34 | 17.50 | 1.08 | 6.7 % | 6.7 % | 20.1 | 21.7 | 41.3 |
| 8 | 374.5 | +1.5 % | 20.98 | 1.74 | 18.07 | 1.17 | 8.3 % | 8.2 % | 21.3 | 27.4 | 27.3 |
| 16 | 738.2 | −3.3 % | 21.06 | 2.22 | 17.67 | 1.17 | 10.5 % | 10.3 % | 21.3 | 23.9 | 27.7 |
| 32 | 1 323.5 | −0.2 % | 23.24 | 3.72 | 18.25 | 1.27 | 16.0 % | 15.5 % | 23.8 | 32.5 | 30.9 |
| 64 | 2 277.9 | −5.5 % | 26.51 | 7.38 | 17.75 | 1.38 | 27.8 % | 26.2 % | 27.7 | 67.0 | 36.0 |
| 128 | 3 184.6 | −0.7 % | 37.36 | 16.88 | 18.67 | 1.81 | 45.2 % | 41.3 % | 39.7 | 74.8 | 51.4 |
| 256 | 4 165.9 | −0.8 % | 56.06 | 35.14 | 18.43 | 2.49 | 62.7 % | 54.9 % | 60.6 | 91.8 | 78.7 |
| 512 | 4 975.5 | −0.5 % | 92.35 | 69.37 | 18.72 | 4.26 | 75.1 % | 61.5 % | 101.0 | 127.2 | 131.7 |
| 1024 | 5 300.2 | −0.6 % | 156.93 | 131.04 | 18.93 | 6.96 | 83.5 % | 58.9 % | 187.7 | 233.3 | 247.3 |

`step`/`sample`/`fwd` are means over pure-decode steps; `sample` and `fwd` are the device-side
windows (`sample_gpu`, `fwd_gpu` — see §2.6), `other` is the remainder of the step.

The last two ratios answer **different questions** and must not be read as one number:

```
sample % step      = mean(sample_gpu | decode steps) / mean(step | decode steps)
sampling % of run  = Σ sample_gpu (every step) / Σ step (every step)
```

* **`sample % step`** — *inside one decode step, how much is sampling?* Decode steps only, so it is
  the number to quote when reasoning about a single step's composition. 75.1 % at batch 512.
* **`sampling % of run`** — *across the whole benchmark, how much of the wall clock went to
  sampling?* Prefill and mixed steps are included in both numerator and denominator, which is what
  makes it the number to prioritise work by. 61.5 % at batch 512 — lower than 75.1 % because the
  prefill steps in the denominator carry 26.3 s of genuine model compute (§3.3).

* **Saturated throughput ≈ 5 300 tok/s**; 512 → 1024 buys 6.5 % of throughput for 86 ms of TPOT.
* **TPOT is flat from batch 1 to 16** (19.0 → 21.3 ms) — the signature of fixed per-step overhead.
* No preemptions, no cache exhaustions, no rescheduled requests at any point in either run.

### 3.2 Per-step CPU breakdown (middle 80 % of each run, pure-decode steps)

| batch | sched | bld_meta | fwd | *(of which rope)* | logits | **sample** | dth | ci | **total** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.14 | 0.24 | 16.64 | 4.13 | 0.11 | **1.28** | 0.03 | 0.01 | **18.71** |
| 8 | 0.17 | 0.27 | 18.07 | 4.19 | 0.12 | **2.01** | 0.03 | 0.02 | **20.95** |
| 32 | 0.21 | 0.30 | 18.24 | 4.12 | 0.12 | **3.99** | 0.03 | 0.04 | **23.19** |
| 64 | 0.27 | 0.34 | 17.70 | 4.05 | 0.11 | **7.64** | 0.05 | 0.06 | **26.43** |
| 128 | 0.44 | 0.47 | 18.74 | 4.22 | 0.13 | **17.02** | 0.33 | 0.13 | **37.56** |
| 256 | 0.68 | 0.64 | 18.35 | 4.20 | 0.12 | **34.55** | 1.24 | 0.25 | **56.13** |
| 512 | 1.43 | 1.05 | 18.57 | 4.31 | 0.13 | **68.43** | 2.73 | 0.50 | **93.16** |
| 1024 | 2.57 | 1.75 | 18.71 | 4.23 | 0.12 | **134.86** | 5.72 | 0.98 | **165.02** |

Same three facts as log910, with the second one worse and the third one much worse:
`fwd` is constant across a 1024× range of batch size, `rope` is a constant ~4.2 ms (23 % of `fwd`),
and `sample` is linear in batch — ~131 µs per sequence — overtaking `fwd` between batch 128 and 256.

### 3.3 Where the whole run goes

Summing every step of each run and attributing by device-side window:

| batch | prefill fwd | prefill sampling | decode fwd | decode sampling | host | **sampling % of run** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5.1 s | 0.8 s | 22.4 s | 20.2 s | 2.0 s | **41.3 %** |
| 256 | 11.0 s | 3.3 s | 20.8 s | 39.6 s | 2.8 s | **54.9 %** |
| 512 | 26.3 s | 13.7 s | 18.1 s | 66.9 s | 4.9 s | **61.5 %** |
| 1024 | 78.8 s | 63.9 s | 11.7 s | 81.1 s | 8.6 s | **58.9 %** |

```
sampling % of run = Σ sample_gpu (every step, prefill and decode) / Σ step (every step)
```

At batch 512 that is `(13.7 + 66.9) / 131.1 = 61.5 %`. The denominator is the summed step time of
the whole run — 131.1 s against the 131.7 s the benchmark reports as `elapsed`, so it accounts for
essentially all of it. **The five time columns do not add up to that denominator**: `logits_gpu`
(1.2 s at batch 512) is left out of the table, and it is the residual.

The prefill-sampling column is the surprise: a chunked-prefill step samples **every running decode
row** as well as its own new sequences, so the batch-sized sampling cost lands on prefill steps too.
At batch 1024, prefill steps are 60.6 % of the run and 63.9 s of that 149 s is sampling.

### 3.4 GPU idle fraction (profile runs)

| batch | clean wall / step | GPU busy / step | idle | run 2 idle |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.08 ms | 3.28 ms | **82.8 %** | 84.9 % |
| 8 | 28.42 ms | 4.33 ms | **84.7 %** | 83.4 % |
| 32 | 30.64 ms | 5.53 ms | **81.9 %** | 82.8 % |
| 128 | 39.92 ms | 14.08 ms | **64.7 %** | 66.2 % |
| 512 | 93.17 ms | 50.79 ms | **45.5 %** | 46.7 % |

Both profile runs reproduce every point within 2 points. Kernel overlap is 0.00 % at every batch
size, so the union and the sum of kernel spans agree — single stream, as expected.

---

## 4. Bottlenecks, ranked by what they cost

### 4.1 Sampling — 61.5 % of the run at batch 512

**Evidence.** `sample_gpu` is 69.4 ms of a 92.4 ms decode step at batch 512 and 131.0 ms of 156.9 ms
at batch 1024. Device-side, so it is real kernel work. Including mixed steps, sampling totals
**80.6 s of the 131.1 s run**.

**Root cause — two full-vocabulary passes per step.**

* [`apply_top_p`](../src/qwen/sampling.py#L202) sorts the complete 151 936-wide row for every
  sequence: `torch.sort(logits, descending=True, dim=-1)`. At batch 512 that is a **78 M-element
  sort every step**; at batch 1024, 156 M.
* [`apply_penalties`](../src/qwen/sampling.py#L159) materialises `[batch, vocab]` fp32 tensors —
  `rep_pen[:, None].repeat(1, vocab_size)` alone is 311 MB at batch 512 and **622 MB at batch
  1024** — then boolean-index-writes over the whole thing and makes ~7 more full passes.

**Fix.** Cut to the candidate set before doing anything expensive. With `top_k = 20`, top-p only
needs the 20 survivors ranked, not 151 936: `torch.topk(logits, k)` → sort/softmax/`multinomial`
over *k*, then scatter the chosen id back. Apply penalties on gathered candidates, or fuse them
into the logits pass, instead of building a vocabulary-wide tensor. Both changes are algorithmic,
not kernel-level.

**Upside.** Sampling is 61.5 % of the run at batch 512; a 5× reduction there is ≈ **2× end-to-end
throughput**, and it pays a second time through §3.3.

### 4.2 The model forward costs 18.5 ms whatever you put in it

**Evidence.** `fwd_gpu` is 16.6 ms at batch 1 and 18.9 ms at batch 1024 — **+14 % for 1024× the
tokens**. A mixed step with 7 500 prefill tokens spends 17.4 ms of device time in the same window.
Inside it the card is idle 82.8 % of the time at batch 1 (3.3 ms of kernels in a 19.1 ms step) and
45.5 % idle at batch 512.

**Root cause.** Eager per-layer dispatch: 24 layers × ~12 kernels, each paying Python, dispatch and
launch, with ~50 µs of gap between kernels. `rope` alone is 4.2 ms — 23 % of `fwd` — and is equally
flat across the sweep, for an operation whose arithmetic content is a handful of multiply-adds.

**Fix.** Capture the decode path in a **CUDA graph** (shapes are static once the batch is fixed), or
`torch.compile(mode="reduce-overhead")`. This is the only finding that improves single-stream
latency: it is what sets the 19 ms TPOT floor and the 53 tok/s at batch 1.

### 4.3 Prefill steps pay the sampling bill twice

Not a separate defect — §4.1 again, in the half of the steps one would not think to look at. At
batch 1024, prefill steps hold 60.6 % of the run and **63.9 s of it is sampling**, against 78.8 s of
genuine prefill compute. Step 641 of the batch-512 run is typical: 15 prefill requests + 497
decodes → `fwd_gpu` 80.6 ms (8 192 tokens at ~88 TFLOP/s, healthy) plus `sample_gpu` 40.8 ms on top.

It is also why the 512 → 1024 doubling returns only 6.5 %.

### 4.4 The token read-back blocks, and grows faster than the batch

`dth` — the `next_tokens.tolist()` in [`engine.forward`](../src/qwen/engine.py#L71) — goes from
0.03 ms at batch 1 to **5.4 ms at batch 1024**, a 196× rise for a 1024× batch, and it is a hard
synchronisation point. Two more blocking syncs sit inside sampling itself: the `.item()` in
`apply_top_k` and `dead.any()` in `sample`.

Fix after §4.1 and §4.2, not before: copy into pinned memory on a side stream and consume it one
step later, or keep the ids on device until the scheduler genuinely needs them.

### 4.5 The host path is small, but it owns the tail

`sched + bld_meta + ci` is a steady **3.5–3.8 % of wall time** at every batch size. But every
multi-millisecond spike comes from there: **43 of 965** decode steps at batch 512 carry one
(`ci` 23, `bld_meta` 17, `sched` 10), against **3 of 1 126** at batch 256. `sched` at batch 512 has
mean 1.39 ms and std 4.15 ms — a cv of 298 %, i.e. the variance is almost entirely rare spikes,
and it is what surfaces as the 127 ms ITL p99. The flagged steps are enumerated in the `.anomaly`
files; the fix is to read those 43 steps, not to optimise the mean.

---

## 5. What is *not* a bottleneck

* **The LM head.** `logits_gpu` is 0.93 ms at batch 512 for a 139 GFLOP matmul — ~150 TFLOP/s,
  near the card's BF16 ceiling.
* **Prefill compute.** 8 192 tokens through the model in 61–80 ms is ~88 TFLOP/s, ~53 % of peak.
  The kernels are healthy; the deficit is orchestration.
* **Scheduler policy and the KV cache.** Zero preemptions, zero cache exhaustions, zero
  rescheduled requests across 11 batch sizes × 2 runs.
* **The 60 s TTFT.** A closed-loop benchmark artefact — all requests are enqueued at t=0 and only
  `batch` of them run at once. Not a latency number.

---

## 6. What changed since log910

Both the code **and** the configuration moved, so the two reports are only partly comparable.

| | log910 | log914 |
| --- | --- | --- |
| `do_sample` | **false** (penalties + `argmax`) | **true** (penalties + top-k + top-p + multinomial) |
| `num_blocks` | 5 632 | 4 096 |
| RoPE cos/sin | per layer | gathered once per step (`1b67a1c`) |
| Per-layer timing | double-counted | fixed (`1b67a1c`) |

Decode step, CPU side, middle 80 %:

| batch | log910 | log914 | Δ | `fwd` 910 → 914 | `sample` 910 → 914 |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 22.40 | 18.71 | **−16.5 %** | 21.27 → 16.64 | 0.49 → 1.28 |
| 32 | 25.87 | 23.19 | −10.4 % | 22.61 → 18.24 | 2.42 → 3.99 |
| 128 | 34.96 | 37.56 | +7.4 % | 22.54 → 18.74 | 9.42 → 17.02 |
| 512 | 72.59 | 93.16 | +28.3 % | 22.34 → 18.57 | 38.67 → 68.43 |
| 1024 | 124.04 | 165.02 | **+33.0 %** | 23.34 → 18.71 | 77.59 → 134.86 |

Two independent movements, in opposite directions:

* **The model path got ~20 % cheaper.** `fwd` fell 21.3 → 16.6 ms at batch 1 and `rope` 5.36 → 4.13
  ms, from the per-step cos/sin gather. Throughput at batch 1 rose 43.8 → 52.8 tok/s (**+21 %**).
* **The sampling path got 1.7× more expensive**, because it was switched on. Peak throughput fell
  7 028 → 5 300 tok/s (**−25 %**) and TPOT at batch 1024 rose 140.9 → 187.7 ms.

log910 §5.2 called this in advance — *"with `do_sample=true` … plus a `torch.sort` over
`[batch, 151936]`"* — as a note on a path that was not being exercised. It now is, and it costs a
quarter of peak throughput. **Any future comparison must state `do_sample` alongside the batch
size**, or it is comparing two different engines.

---

## 7. Three traps in this instrumentation

1. **CPU-side timings lie in mixed steps.** In a prefill step the host `sample` reads 95–101 ms
   while the device does 41 ms; the difference is the host blocking on the prefill backlog queued
   during `fwd`. Step 641, batch 512: `fwd` 18.2 ms CPU / **80.6 ms GPU**, `sample` 101.4 ms CPU /
   40.8 ms GPU. Attribute mixed steps with the `*_gpu` fields only.
2. **A `*_gpu` number is a window, not busy time.** It is the span between two CUDA events on the
   stream, gaps included. `fwd_gpu` reads 18.7 ms at batch 1 while only 3.3 ms of kernels run inside
   it — window minus busy is the bubble, and only the trace gives you busy.
3. **`rope_gpu` at batch 1024 is not believable.** It reads 1.28 ms (median 0.96) against 3.5–3.7 ms
   at every other batch size, while the host-side `rope` is unchanged at 4.24 ms. Both runs agree,
   so it is systematic — most likely a bug in the per-layer event merge. Do not draw a rope
   conclusion from the 1024 column until it is explained.

---

## 8. Recommendations, in order

1. **Rewrite the sampling path to work on the candidate set** (§4.1). The only change that moves
   throughput by a multiple, and it pays twice (§4.3). Estimated ≈2× at batch 512.
2. **CUDA-graph the decode step** (§4.2). The only change that improves latency; also lifts
   small-batch throughput, where `fwd` is 89 % of the step.
3. **Un-block the token read-back** (§4.4) — worth doing once the step is short enough for 3 ms to
   matter.
4. **Open the 43 flagged steps** (§4.5) before touching the host path's mean.
5. **Fix `rope_gpu` at batch 1024** (§7.3) — a metric that is wrong is worse than a metric that is
   missing.

---

## 9. Caveats

* Both sweeps ran on the same vast.ai instance within 20 minutes; no cross-machine reproduction.
* `sample % run` attributes by device-side window, which counts stream gaps inside the window as
  sampling. At batch ≥ 256 the GPU is well fed and the error is small; at batch ≤ 32 it is not, so
  the `sample % run` column should only be read from 128 up.
* The workload is synthetic (random ids, fixed 512/128). Random ids change the penalty mask density
  versus real text, which affects §4.1's GPU traffic but not its asymptotics.
* GPU-busy figures come from the profile harness (`max_num_seqs` ≤ 512, `num_blocks` 5 632), not
  from the sweep runs. Step times agree to within 1 % at batch 512 (93.17 vs 92.35 ms), so the two
  harnesses are measuring the same steady state.
