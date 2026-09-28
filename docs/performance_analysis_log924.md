# Performance Analysis — Decode Engine on RTX 4090 (log924 baseline)

**Artifacts analysed**

| Artifact | Path |
| --- | --- |
| Concurrency sweep, run 1 | `log_vast/log924/benchmark_baseline/` |
| Concurrency sweep, run 2 | `log_vast/log924/benchmark_baseline2/` |
| Idle-fraction profile, run 1 | `log_vast/log924/profile_baseline/` |
| Idle-fraction profile, run 2 | `log_vast/log924/profile_baseline2/` |
| Step-metric A/B reports | `log_vast/log924/mean_step_metrics.{1,8,32,128,256,512}.log` |
| GPU static inventory | `log_vast/log924/gpu.static.csv` |

Two independent instruments, each run **twice** with no change in between: the concurrency sweep
(11 batch sizes, 1 → 1 024) and the pure-decode idle-fraction profile (6 batch sizes). Everything
below that says "device-side" comes from the profile; everything that says "host-side" comes from the
sweep's `step_metrics` dumps, which carry no profiler contamination.

**Code under test** — `src/qwen/` on branch `token_id_table` at `c03c61b` plus uncommitted changes to
`constants.py`, `engine.py`, `metrics.py`, `sampling.py`, `scheduler.py`. Switches:
**`compile_rope = false`** (rope runs eager), **`pre_gather_cos_sin = true`**,
`use_d_first_schedule = false`.

> **Do not read the switch state off the `config.py:170 - default config:` line in `pytest.log`.**
> It prints `compile_rope=True` in all four runs and is wrong about what ran: that line is emitted
> inside `ModelConfig.from_pretrained`, *before*
> [`target_config`](../tests/conftest.py#L235) sets `compile_rope = False` and
> [`tmp_target_config_for_sharegpt_benchmarking`](../tests/conftest.py#L348) sets it to the
> `--compile-rope` option, whose default is `False`. The authoritative line is
> `rope.py:23 - Using eager path for apply_rotary`, which fires **6/6 times in each profile log and
> 12/12 in each benchmark log**. §4.3 corroborates it from the op counts.
>
> Versus [`performance_analysis_log915.md`](./performance_analysis_log915.md), whose baseline had
> both switches off, **only `pre_gather_cos_sin` differs** (false → true); `compile_rope` was false
> in both. §7 compares them.

---

## Executive summary

1. **Sampling is still the engine, and it is still almost entirely waste.** At batch 512 it is
   **40.6 ms of device work inside a 51.8 ms decode step — 80 % of everything the GPU does**, and
   **54.3 % of the whole run**. A single `aten::sort` over the full 151 936-wide vocabulary is 15.4 ms/step of that, and it runs
   *after* top-k has already reduced the live candidates to 20. The penalty pass costs another
   ~10 ms/step computing a mathematical identity (`rep_pen = 1.0`, `freq_pen = pres_pen = 0`).
2. **~40 % of the run at batch 512 is Python writing host tensors one element at a time.**
   `bld_meta` is 24.3 % of the run and `sched` 16.7 %, and both are dominated by scalar
   `tensor[i] = value` loops in [`RequestBuffer`](../src/qwen/scheduler.py#L57) and
   [`TokenIdTable.add_req`](../src/qwen/scheduler.py#L279). A local microbenchmark of the same
   pattern: **3.35 ms scalar loop → 0.023 ms vectorised, 144×**.
3. **The GPU is idle 85 % of a batch-1 decode step** and 78 % at batch 32. The host needs 21.7 ms to
   launch a step whose kernels occupy 3.3 ms. A 512-token prefill and a 1-token decode both cost
   ~22 ms — the step price is host overhead, essentially independent of the work in it.
4. **Peak delivered throughput 6 752 tok/s at batch 512**; the last profitable doubling ends at
   **batch 256**. Pure-decode capability is higher — 9 890 tok/s at batch 512 — and the gap is lost
   to prefill/decode co-scheduling, which also drives ITL p99 to 2.7× p50.
5. **Above batch 256 the card is power-capped, not compute-limited.** `enforced.power.limit` is
   **250 W against a 450 W card**; `sw_power_cap` is active 99 % of the time at batch ≥ 512 and SM
   clocks fall from 2 610 to 2 169 MHz (**−17 %**). Large-batch numbers here are clock-limited.
6. **Reproducibility is excellent.** Throughput agrees within **1.27 %** across the two sweeps
   (8 of 11 points within 0.7 %); device-side GPU-busy time agrees within **0.2 %**. Every finding
   below is 5× to 300× the noise floor.

---

## 1. System under test

### 1.1 Hardware

NVIDIA GeForce RTX 4090, 24 564 MiB GDDR6X, driver 580.142, PCIe 4.0 ×16, on a vast.ai instance.
Peak HBM bandwidth 1 008 GB/s; peak BF16 dense (FP32 accumulate) 165.2 TFLOP/s.

**Power and clock ceiling** — from `gpu.static.csv`:

| | |
| --- | --- |
| `clocks.max.sm` | 3 150 MHz |
| `enforced.power.limit` | **250.00 W** |
| `power.max_limit` | 450.00 W |

The card is administratively capped at **56 % of its rated power**. §4.5 shows what that costs.

### 1.2 Model

Qwen2.5-0.5B-Instruct, `torch.bfloat16`. 494.03 M parameters = 357.9 M body + 136.1 M embedding /
LM head; **0.988 GB** of BF16 weights; 12 KiB of KV per token (24 layers × 2 KV heads × 64 dim × 2 B × 2).

### 1.3 Engine configuration

| Setting | Sweep | Profile |
| --- | --- | --- |
| `max_model_len` | 4 096 | 1 024 |
| `max_num_batched_tokens` | 8 192 | 8 192 |
| `long_prefill_token_threshold` | 8 192 | 8 192 |
| `max_num_seqs` | swept 1 → 1 024 | swept 1 → 512 |
| `num_blocks` × `block_size` | — | 4 096 × 256 → 12.0 GiB KV |
| `use_d_first_schedule` | false | false |
| `compile_rope` / `pre_gather_cos_sin` | **false / true** | **false / true** |
| Attention | `flash_attn_varlen` (paged) | same |

**Sampling** — merged from `generation_config.json`, so the defaults in
[`config.py`](../src/qwen/config.py#L68) are *not* what runs:

| | value | effect |
| --- | --- | --- |
| `do_sample` | `true` | full sampling path |
| `temperature` | 0.7 | active |
| `top_k` | 20 | active, ≤ `MAX_EFFECTIVE_TOP_K` (1 024) |
| `top_p` | 0.8 | **active → the full-vocabulary sort runs** |
| `do_penalities` | `true` | penalty pass runs |
| `rep_pen` / `freq_pen` / `pres_pen` | 1.0 / 0.0 / 0.0 | **all three are no-ops** |

`generation_config.json` carries `repetition_penalty: 1.1`, but the field was named `rep_pen` on
`ModelConfig` at measurement time, so `from_pretrained` silently dropped it. For this report that
means **the entire penalty pass is an identity transform that still costs ~10 ms/step at batch 512**
(§4.1).

> **Already fixed in the working tree** (`ModelConfig.repetition_penalty`,
> `SamplingParamTable.FLOAT32_FIELDS`), so the next run will have `repetition_penalty = 1.1` **live**
> and the penalty pass will no longer be an identity — it will do the same ~10 ms of work and the
> result will matter. Every number in §4.1 was measured with it inert. See §6 #2/#8 for the ordering:
> the short-circuit has to key on the *values*, and with 1.1 live it will no longer fire, so the
> ~10 ms/step does not come back as a saving — it becomes a real cost that only the candidate-set
> rewrite (§6 #3) removes.

---

## 2. Methodology

### 2.1 Workload

Synthetic fixed shape: every request is **512 random token ids in, 128 tokens out**, EOS ignored,
`req_num = max(64, 10 × batch)` enqueued up front and drained by `run_to_completion()`. Closed-loop
and saturating, so the reported `queueing` and `ttft` are admission-queue artefacts, not latency —
quote `prefill` instead.

The profile harness ([`test_profile.py::test_profile_decode_idle_fraction`](../tests/test_profile.py#L111))
isolates **pure decode**: 64 warm-up steps until every one of `batch_size` requests is decoding in
lockstep, then 20 clean measured steps (run A, wall clock only) followed by 5 profiled steps (run B).
GPU-busy time is the *union* of kernel/memcpy/memset spans from the Chrome trace, divided by the
**clean** wall — `cuda_sync` rows are excluded because their duration is host waiting, not device work.

### 2.2 Two traps in these logs

**(a) `step` is not a per-step wall time.** `step_metrics.start("step")` runs at the top of
[`LLMEngine.step()`](../src/qwen/engine.py#L34), but `stop("step")` runs in
[`Scheduler.commit_step`](../src/qwen/scheduler.py#L669), which fires **one step later** — the engine
launches batch *N* and commits batch *N−1* in the same call. So `step` measures the
launch→commit span of one batch, ≈ **2× the step wall** whenever the in-flight pipeline is at depth 1.
This is visible directly in the dumps: at batch 512, `step` = 104.4 ms while the sub-timers sum to
51.8 ms.

Consequently the `step` rows in `mean_step_metrics.*.log` (45.9 ms at batch 1, 107.9 ms at 512)
are **pipeline latencies, not step times**, and the apparent 2× jump between batch 16 (24.3 ms) and
batch 32 (51.3 ms) in the sweep dumps is the pipeline depth changing, not the engine slowing down.
Every step time in this report is the **sum of the sub-timers**
(`sched + bld_meta + fwd + logits + sample + dth + ci`), which is verified against three independent
references: the profile's `wall_clean_us/step`, the benchmark's `itls` p50, and `elapsed / steps`.

**(b) The in-flight pipeline is a one-way latch.** `in_flight_steps` (named `in_flight_batchs` at
measurement time) can only shrink: [`forward`](../src/qwen/engine.py#L88) appends at most one entry and
[`sample_in_flight_step`](../src/qwen/engine.py#L92) pops exactly one, so the first step that
schedules nothing collapses depth 1 → 0 permanently. Measured: exactly one `pend` transition per run,
at step 128 for batch ≤ 16 (when the first request hits its 128-token limit and the only slot is
still held by the in-flight batch), and at the final tail step for batch ≥ 32. So batch ≤ 16 ran
**98 % of the run with no launch/commit overlap at all**. It costs nothing today — the GPU finishes
3.3 ms of work inside a 21.7 ms launch window either way — see §4.6.

### 2.3 Measured noise floor

Throughput, run 2 vs run 1:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Δ | −0.20 % | −0.19 % | −0.97 % | −0.52 % | +1.13 % | −0.31 % | +0.50 % | +1.27 % | +0.67 % | −0.03 % | +0.04 % |

Device-side, run 2 vs run 1 (GPU busy per decode step):

| batch | 1 | 8 | 32 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU busy µs/step | 3 261 / 3 256 | 4 291 / 4 285 | 5 464 / 5 461 | 14 046 / 14 046 | 26 694 / 26 697 | 50 977 / 50 998 |
| Δ | −0.14 % | −0.15 % | −0.05 % | 0.00 % | +0.01 % | +0.04 % |

**Anything below ~1.5 % is noise.** The `mean_step_metrics` comparator flags 1–9 of 19 fields as
`SHIFT` at every batch size, but every one of those is a sub-1 % delta on a field whose CV is 0.2–0.6 %
— the z statistic is correct and the effect is irrelevant. Treat `|delta%| < 2 %` as
noise-equivalent for decisions, whatever the verdict column says.

The two flagged anomalies per profile run are **measurement-window edges, not engine events**:

* line 65 (first measured step) — `fwd_gpu` 20.8 ms against a 10.6 ms local median at batch 512.
  The preceding `torch.cuda.synchronize()` drains the launch queue, so this one step's forward span
  reflects an empty pipeline. It is also the honest measure of the model's own span, §4.3.
* line 84 (last measured step) — `step` 207 ms against a 104 ms median. Its `stop("step")` lands
  inside the *profiled* run B, per trap (a).

---

## 3. Results

### 3.1 Concurrency sweep (run 1)

| batch | req | tok/s | Δ run 2 | tok/s/req | sc. eff. | gain/cost | TPOT ms | ITL p50 | ITL p99 | ITL max | prefill ms | elapsed s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 45.1 | −0.20 % | 45.15 | 1.00 | — | 22.15 | 22.1 | 22.6 | 29.6 | 22.8 | 181.7 |
| 2 | 64 | 84.3 | −0.19 % | 42.22 | 0.93 | 1.75 | 23.68 | 23.7 | 24.4 | 31.8 | 23.7 | 97.2 |
| 4 | 64 | 168.9 | −0.97 % | 42.38 | 0.94 | 2.01 | 23.60 | 23.6 | 24.2 | 29.6 | 24.1 | 48.5 |
| 8 | 80 | 332.1 | −0.52 % | 41.94 | 0.92 | 1.95 | 23.85 | 23.8 | 24.8 | 29.2 | 34.6 | 30.8 |
| 16 | 160 | 637.3 | +1.13 % | 40.89 | 0.88 | 1.87 | 24.46 | 24.5 | 24.7 | 29.2 | 67.0 | 32.1 |
| 32 | 320 | 1 203.4 | −0.31 % | 38.72 | 0.83 | 1.79 | 25.83 | 25.5 | 28.7 | 69.4 | 82.4 | 34.0 |
| 64 | 640 | 2 249.4 | +0.50 % | 36.30 | 0.78 | 1.75 | 27.55 | 26.6 | 67.3 | 70.9 | 91.4 | 36.4 |
| 128 | 1 280 | 3 891.6 | +1.27 % | 31.53 | 0.67 | 1.50 | 31.71 | 29.2 | 79.6 | 80.7 | 110.4 | 42.1 |
| 256 | 2 560 | 5 970.7 | +0.67 % | 24.34 | 0.52 | **1.18** | 41.08 | 33.6 | 103.2 | 107.2 | 154.0 | 54.9 |
| 512 | 5 120 | **6 752.3** | −0.03 % | 13.74 | 0.29 | **0.64** | 72.78 | 52.4 | 141.8 | 248.6 | 223.7 | 97.1 |
| 1024 | 10 240 | 6 180.9 | +0.04 % | 6.29 | 0.13 | 0.42 | 158.90 | 204.2 | 274.1 | 372.4 | 373.5 | 212.1 |

*Figures: `benchmark_baseline/concurrency_sweep.png` and `benchmark_baseline2/concurrency_sweep.png`.*

* **The last profitable doubling ends at batch 256.** 256 → 512 buys 13 % more throughput for 77 %
  more latency; 512 → 1 024 *loses* 8 % throughput for 118 % more latency.
* **TPOT is flat from batch 1 to 16** (22.15 → 24.46 ms): 16× the work for 10 % more time. That is a
  fixed host cost, not a loaded GPU (§4.3 confirms: 85 % → 82 % idle over the same range).
* **The tail is prefill/decode co-scheduling.** At batch 64, TPOT is 27.6 ms but ITL p99 is 67.3 ms.
  At batch 512, p50 52.4 ms vs p99 141.8 ms — **2.7×** (§4.4).
* **Zero preemptions, zero cache exhaustions** at every point in both runs.

### 3.2 Per-step host breakdown — pure-decode steps only (run 1, ms)

| batch | sched | **bld_meta** | fwd | *(of which rope)* | logits | sample | dth | ci | **step wall** | ITL p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.07 | 0.55 | 19.43 | *4.94* | 0.12 | 1.44 | 0.07 | 0.02 | **21.70** | 22.1 |
| 8 | 0.08 | 0.83 | 20.80 | *4.98* | 0.13 | 1.47 | 0.07 | 0.03 | **23.40** | 23.8 |
| 16 | 0.10 | 1.11 | 21.08 | *5.06* | 0.13 | 1.50 | 0.07 | 0.04 | **24.02** | 24.5 |
| 32 | 0.12 | 1.66 | 21.54 | *5.01* | 0.13 | 1.53 | 0.07 | 0.05 | **25.09** | 25.5 |
| 64 | 0.18 | 2.75 | 21.44 | *5.04* | 0.13 | 1.53 | 0.07 | 0.08 | **26.18** | 26.6 |
| 128 | 0.29 | 4.95 | 21.67 | *5.08* | 0.13 | 1.52 | 0.07 | 0.15 | **28.76** | 29.2 |
| 256 | 0.49 | 9.28 | 21.57 | *5.21* | 0.13 | 1.52 | 0.07 | 0.26 | **33.31** | 33.6 |
| 512 | 1.45 | **17.83** | 30.24 | *5.26* | 0.13 | 1.53 | 0.07 | 0.53 | **51.77** | 52.4 |
| 1024 | 2.96 | **33.71** | 56.87 | *14.62* | 0.13 | 1.57 | 0.07 | 1.03 | **96.34** | 204.2¹ |

¹ At batch 1 024 only 46 % of steps are pure decode, so ITL p50 is set by mixed steps (§3.4).

* `fwd` is **flat at ~21.5 ms from batch 1 to 256** — a 256× range. It is host launch time, not
  compute. The rise at 512/1 024 is launch-queue backpressure, not more launching: the same
  measurement at 256 is 21.57 ms.
* `rope` is **flat at ~5.0 ms**, i.e. **23 % of `fwd`**, for an operation whose arithmetic content is
  a handful of multiply-adds (§4.3).
* `bld_meta` is the one host cost that scales: **0.55 → 33.71 ms, 61× for a 1 024× batch**, and by
  batch 512 it is 34 % of the step (§4.2).
* `sample` (host) is **flat at ~1.5 ms** and `dth` at **0.07 ms** — both fixed since log915 by the
  resident-table work on this branch. The sampling cost has moved entirely to the device.

### 3.3 Per-step device breakdown (profile run 1, ms/step)

| batch | GPU busy | `fwd_gpu` span | `sample_gpu` | `logits_gpu` | clean wall | **GPU idle** | span density² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.26 | 19.39 | 1.18 | 0.37 | 22.13 | **85.3 %** | 16 % |
| 8 | 4.29 | 20.77 | 1.21 | 0.38 | 23.83 | **82.0 %** | 19 % |
| 32 | 5.46 | 21.47 | 1.83 | 0.40 | 25.44 | **78.5 %** | 23 % |
| 128 | 14.05 | 19.17 | 9.37 | 0.43 | 29.38 | **52.2 %** | 48 % |
| 256 | 26.69 | 12.72 | 20.17 | 0.54 | 33.62 | **20.6 %** | 80 % |
| 512 | 50.98 | 10.79 | 40.58 | 0.87 | 53.54 | **4.8 %** | 98 % |

² `GPU busy ÷ (fwd_gpu + sample_gpu + logits_gpu)`. The `*_gpu` fields are CUDA-event **spans**, so
they include device idle inside the window; density says how much of the span is real work. Below
batch 128 the spans are mostly gaps — which is why `fwd_gpu` *shrinks* from 19.4 ms to 10.8 ms as the
batch grows 512×: at batch 1 the span is the host's launch rate, at batch 512 it is the kernels.

**The crossover is between batch 32 and 128.** Below it the engine is host-bound with an 80 %-idle
GPU; above it the GPU saturates — and what saturates it is the sampler, not the model:

| batch | sampling share of GPU busy | model forward share |
| ---: | ---: | ---: |
| 32 | 34 % | 66 % |
| 128 | 67 % | 33 % |
| 256 | 76 % | 24 % |
| 512 | **80 %** | 20 % |

Profiler overhead on the host was +64 % to +157 %; the device-side numbers are unaffected
(`key_averages` vs trace kernel sum agree to ±0.00 %, kernel overlap 0.00 % — single stream).

### 3.4 Mixed (prefill-carrying) steps vs pure-decode steps (run 1)

| batch | mixed steps | decode steps | mixed wall ms | decode wall ms | ratio | mixed `sched_wait` ms | mixed n_p_tok |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 90 (7 %) | 1 207 | 75.2 | 28.8 | 2.6× | — | 7 282 |
| 256 | 170 (13 %) | 1 135 | 96.5 | 33.3 | 2.9× | — | 7 710 |
| 512 | 348 (26 %) | 975 | 131.4 | 51.8 | 2.5× | **41.1** | 7 533 |
| 1024 | 732 (54 %) | 630 | 205.2 | 96.3 | 2.1× | **39.3** | 7 162 |

`long_prefill_token_threshold` equals `max_num_batched_tokens` (8 192), so an admitting step takes a
~7 500-token prefill chunk **on top of** sampling every running decode row. Two consequences:

* **Pure-decode capability is well above delivered throughput.** At batch 512, `512 / 51.8 ms` =
  **9 890 tok/s** against 6 752 delivered — **32 % lost** to mixed steps. At batch 1 024, 10 629 vs
  6 181 — **42 % lost**.
* On mixed steps, `sched_wait` alone is **~40 ms of host time** (§4.2).

### 3.5 Where the whole run goes (run 1, seconds)

| batch | run wall | Σ `sample_gpu` | % run | Σ `bld_meta` | % run | Σ `sched` | % run | Σ `ci` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 178.0 | 9.7 | 5.5 % | 4.5 | 2.5 % | 0.7 | 0.4 % | 0.2 |
| 32 | 33.5 | 2.4 | 7.0 % | 2.1 | 6.4 % | 1.1 | 3.4 % | 0.1 |
| 128 | 41.5 | 12.0 | 28.9 % | 6.4 | 15.5 % | 3.9 | 9.3 % | 0.2 |
| 256 | 54.2 | 25.9 | 47.7 % | 12.1 | 22.2 % | 7.8 | 14.4 % | 0.3 |
| 512 | 96.2 | 52.2 | **54.3 %** | 23.4 | **24.3 %** | 16.1 | **16.7 %** | 0.7 |
| 1024 | 210.9 | 104.5 | 49.6 % | 46.1 | 21.9 % | 32.7 | 15.5 % | 1.4 |

At batch 512, sampling and per-request host bookkeeping between them account for **95 % of the run**.
(`sample_gpu` and the host columns overlap in time — the point is the size of each, not a partition.)

### 3.6 GPU telemetry, correlated per sweep point (run 1)

| batch | SM clock MHz | `utilization.gpu` | power avg W | temp °C | mem peak MiB | `sw_power_cap` active |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 610 | 22.6 % | 81.0 | 56.9 | 14 039 | 0.0 % |
| 16 | 2 626 | 22.0 % | 90.7 | 58.2 | 14 193 | 1.6 % |
| 32 | 2 682 | 27.3 % | 112.7 | 60.2 | 14 195 | 4.4 % |
| 64 | 2 675 | 36.6 % | 136.6 | 62.3 | 14 513 | 8.4 % |
| 128 | 2 702 | 57.1 % | 195.0 | 67.4 | 15 461 | 15.5 % |
| 256 | 2 556 | 85.7 % | 242.0 | 71.3 | 18 197 | 79.1 % |
| 512 | **2 292** | 98.3 % | 245.9 | 71.6 | 23 623 | **98.9 %** |
| 1024 | **2 169** | 98.9 % | 247.0 | 71.5 | **24 063** | **99.1 %** |

`hw_slowdown` and both thermal-slowdown flags are **never** active — 71 °C is nowhere near thermal
limits. The clock loss is purely the 250 W cap. Run 2 reproduces every row within 1 %.

Note also `memory.used` at batch 1 024: **24 063 of 24 564 MiB, 98 %**. §4.1's transient
`[batch, vocab]` allocations are ~1.5 GB per step at that batch.

### 3.7 Roofline position

| | batch 1 | batch 512 |
| --- | ---: | ---: |
| Decode step wall | 21.70 ms | 51.77 ms |
| Weight traffic / step | 0.988 GB | 0.988 GB |
| KV traffic / step (≈596 tok/seq) | 0.007 GB | 3.75 GB |
| Achieved bandwidth | 45.5 GB/s | 91.6 GB/s |
| **MBU** (vs 1 008 GB/s) | **4.5 %** | **9.1 %** |
| **MFU** (2·N·tokens vs 165 TFLOP/s) | 0.03 % | **5.9 %** |
| Same step if the host vanished (GPU busy only) | 3.26 ms → 303 GB/s, **30 % MBU** | 50.98 ms → 9.3 % MBU |

`bench_viz.py` reports 4.4 % MBU at batch 1 and 3.7 % MFU at batch 1 024 for the same run — it counts
all 494 M parameters against the run-average throughput where the table above counts the decode step
alone. Both are right about what they measure; quote the definition with the number.

**The individual kernels that matter are not the problem** (batch 512):

| kernel | measured | roofline position |
| --- | ---: | --- |
| `flash_fwd_splitkv` (paged decode attention) | 4.86 ms/step | 3.75 GB of KV → **772 GB/s, 77 % MBU** |
| LM head GEMM (`ampere_bf16_s16816gemm_128x64`) | 0.82 ms/step | 139 GFLOP → **~160 TFLOP/s, ~97 % MFU** |
| Prefill forward, mixed step (8 030 tokens) | 92.6 ms | ≥ 5.89 TFLOP → **≥ 64 TFLOP/s, ≥ 39 % MFU**³ |

³ Attention FLOPs excluded, so this is a lower bound.

---

## 4. Bottlenecks, ranked by what they cost

### 4.1 Sampling — 80 % of device work, 54 % of the run, and almost all of it avoidable

**Evidence.** `sample_gpu` is **40.58 ms of the 50.98 ms of GPU-busy time** in a batch-512 decode
step, and 75.9 ms at batch 1 024. It scales **linearly** — 1.18 / 1.21 / 1.83 / 9.37 / 20.17 / 40.58 ms
for batch 1 / 8 / 32 / 128 / 256 / 512 in the profile, 75.9 ms at batch 1 024 in the sweep. Per-kernel, from `key_averages` at batch 512
(self CUDA, ÷ 5 profiled steps):

| aten op | ms/step | share of the 51.0 ms GPU step |
| --- | ---: | ---: |
| `aten::sort` | **15.35** | **30.1 %** |
| `aten::sub` | 3.07 | 6.0 % |
| `aten::div` | 2.50 | 4.9 % |
| `aten::_softmax` (×2 full-vocab) | 2.45 | 4.8 % |
| `aten::where` | 2.24 | 4.4 % |
| `aten::masked_fill_` | 2.18 | 4.3 % |
| `Memcpy DtoD` (the `repeat`) | 1.91 | 3.8 % |
| `aten::topk` | 1.86 | 3.6 % |
| `aten::scatter_` | 0.92 | 1.8 % |
| `aten::cumsum` | 0.81 | 1.6 % |
| `aten::fill_` | 0.79 | 1.5 % |
| `argmax` / `gt` / `sum` / `lt` / `min` / `max` / `exponential_` | 3.54 | 6.9 % |

For contrast, the model's own attention kernel is 4.86 ms and the LM head 0.82 ms.

**Root cause 1 — top-p sorts the whole vocabulary after top-k already threw it away.**
[`apply_top_p`](../src/qwen/sampling.py#L143) runs `torch.sort(logits, descending=True)` over the
complete 151 936-wide row for every sequence: a **78 M-element sort at batch 512**, 156 M at 1 024,
every step. But it runs *after* [`apply_top_k`](../src/qwen/sampling.py#L119), which has already set
all but ≤ 20 entries to `-inf`. The sort, the softmax over it, the `cumsum`, the `scatter_` back and
the `masked_fill` are all full-vocab passes computing a result determined by 20 values — and `topk`
has already returned those 20 **in descending order**.

**Root cause 2 — the penalty pass is an identity transform at full vocabulary width.**
`rep_pen = 1.0`, `freq_pen = pres_pen = 0.0` (§1.3), so
[`apply_penalties`](../src/qwen/sampling.py#L110) is arithmetically `logits → logits`. It still
materialises `rep_pen[:, None].repeat(1, vocab_size)` — **311 MB of fp32 at batch 512, 622 MB at
1 024** — masks it, and makes five more full-vocab passes. Feeding it,
[`bin_count_and_mask`](../src/qwen/scheduler.py#L354) allocates and zeroes a `[B, V]` int32
`output_counts` (311 MB) and a `[B, 2V+16]` bool `masks` (155 MB) **every step**. Grouped estimate of
the 40.6 ms: **~20 ms top-p machinery, ~10 ms penalties + mask build, ~2.7 ms top-k, ~5 ms final
softmax / dead-row guard / multinomial / argmax.**

**Fix.**
1. Short-circuit `do_penalities` on the actual parameter values, not just the flag — compute one
   host-side boolean when a slot is filled in
   [`SamplingParamTable.set_slot`](../src/qwen/sampling.py#L63) and skip the whole pass (and
   `bin_count_and_mask`) when no live row asks for a penalty. Removes ~10 ms/step at batch 512 and
   ~1.5 GB of transient allocation at batch 1 024, for a few lines.
2. Do top-p on the candidate set. `topk` already returns `[n, max_k]` descending; `softmax` →
   `cumsum` → threshold on that, then one `masked_fill` against the kth value. Removes the
   151 936-wide sort/softmax/cumsum/scatter. Removes ~20 ms/step at batch 512.
3. Then sample from the `[n, max_k]` candidates and scatter the chosen index back through
   `topk`'s indices, which also shrinks the final softmax and `multinomial`.

**Upside.** Device work at batch 512 goes from ~51 ms to ~15 ms. The step becomes host-bound again,
so it pays only together with §4.2 — combined, ≈**2× delivered throughput at batch 512**, and it pays
a second time on mixed steps, which sample every running decode row too.

### 4.2 Per-request Python — 41 % of the run at batch 512, and a 144× fix

**Evidence.** `bld_meta` 17.83 ms/step and `sched` 1.45 ms/step on decode steps, `sched_wait`
**41.1 ms/step** on mixed steps; over the whole batch-512 run, 23.4 s + 16.1 s of 96.2 s.

**Root cause — scalar `tensor[i] = value` in three loops.** Each such write is a full PyTorch
dispatch, ~2–6 µs:

* [`RequestBuffer._build_block_table`](../src/qwen/scheduler.py#L57) — nested loop,
  `block_table_host[i, j] = b`. At batch 512 × 3 blocks/request that is ~1 536 dispatches/step.
* [`RequestBuffer.set`](../src/qwen/scheduler.py#L68) — three scalar writes plus a `torch.as_tensor`
  per request, ~2 560 dispatches/step at batch 512.
* [`TokenIdTable.add_req`](../src/qwen/scheduler.py#L279) —
  `for i, tok in enumerate(req.input_ids): tok_id_host[slot, i] = tok`. **512 dispatches per
  admitted request.** ~15 admissions per mixed step → this is what `sched_wait` is.

Microbenchmarked locally (512 int32 writes into a host tensor row):

| pattern | time |
| --- | ---: |
| scalar loop, 512 × `t[0,i] = tok` | **3.351 ms** |
| `t[0,:512].copy_(torch.as_tensor(ids))` | 0.289 ms (11.6×) |
| numpy view, `t_np[0,:512] = np.asarray(ids)` | **0.023 ms (144×)** |

Simulating the full `bld_meta` pattern (block table + three per-request arrays at batch 512):
**13.02 ms scalar → 0.010 ms vectorised**. The measured 17.8 ms on the vast box is the same number.

**Fix.** Keep a `numpy` view of each pinned host tensor (`self.x_np = self.x_host.numpy()`) and write
whole slices: one `np.asarray(req.input_ids)` assignment in `add_req`, one padded
`(B, max_blocks)` int32 array built from the block tables, one slice assignment each for
`slot_idx` / `want` / `needs_sample` / `cache_slot`. The pinned buffers, the async H2D copies and the
`non_blocking` semantics are all unchanged — only the host-side fill changes.

**Upside.** `bld_meta` 17.8 → < 1 ms and `sched_wait` ~41 → < 1 ms. That is **~35 s off the 96 s
batch-512 run on its own**, before any device-side change, and it is the cheapest item on this list
by a wide margin.

### 4.3 Eager per-layer dispatch — the GPU is idle 85 % of a batch-1 step

**Evidence.** `fwd` (host) is **flat at ~21.5 ms from batch 1 to batch 256** while GPU busy over the
same range goes 3.26 → 26.69 ms. At batch 1 the GPU is idle **85.3 %** of the step, and the model's
real kernel time is ~2.8 ms inside a 19.4 ms window. Corroboration from three directions: the trace's
kernel union, `nvidia-smi utilization.gpu` at 22 % for batch 4–16, and the fact that a **512-token
prefill (22.8 ms) and a 1-token decode (22.1 ms) cost the same** — the step price is independent of
the work in it.

Per-step aten counts at batch 1 (from `key_averages`, ÷ 5 steps): 321 `aten::mul`, 97 `aten::mm`,
152 `aten::add`, 122 `aten::copy_`, 49 each `mean`/`pow`/`rsqrt`, 53 `sub`, 24
`flash_attn_varlen`, 48 `index_copy_`. Roughly **1 200 dispatches per decode step** for 24 layers,
each paying Python, dispatch and launch for ~2.5 µs of device work.

**`rope` is 23 % of `fwd`, and it is ~28 % of all the dispatches in the step.** `rope` is
4.94 ms/step host, flat across the sweep, while `rope_gpu` falls from 4.26 ms (span, batch 1) to
0.83 ms (batch 512) — i.e. the window is host time, not device time.

It ran **eager** ([`rope.py:23`](../src/qwen/rope.py#L16) logged `Using eager path` in every run), and
[`BaseRoPE.forward2`](../src/qwen/rope.py#L64) calls `apply_rotary` **twice per layer, 48 times per
step**. Each call is `2 slice + 4 mul + 1 sub + 1 add + 1 cat`, so rope alone contributes
**192 `mul` + 48 each of `sub`/`add`/`cat` + 96 `slice` per step**. The measured counts at batch 1 are
exactly that shape:

| op | calls/step | rope's share |
| --- | ---: | --- |
| `aten::cat` | 50 | **48** |
| `aten::sub` | 53 | **48** |
| `aten::add` | 152 | 48 (+48 residuals) |
| `aten::mul` | 321 | **192** |
| `aten::slice` | 120 | **96** |
| `aten::index_select` | 10 | 2 — confirms `pre_gather_cos_sin=true`; the per-layer path would show 48 |

So ~336 of the ~1 200 dispatches per step are rope, for 4.94 ms of the 19.43 ms `fwd` — 103 µs per
`apply_rotary` call, i.e. ~11 µs per eager pointwise op, which is exactly what the `Self CPU` column
shows for `aten::mul` (13.7 µs avg). The jump to 14.62 ms at batch 1 024 is launch-queue
backpressure, not rope getting slower: `fwd` doubles on the same step (30.24 → 56.87 ms).

**Fix.**
1. **CUDA-graph the decode path.** Shapes are static once `max_num_seqs` is fixed and the metadata
   buffers are already persistent (`RequestBuffer`, `TokenIdTable`, `SamplingParamTable` are all
   preallocated) — this engine is unusually well set up for capture. Alternatively
   `torch.compile(mode="reduce-overhead")` on the decode step.
2. Cheaper and independent: **fuse rope**. Nine eager pointwise ops × 48 calls is the single largest
   block of dispatches in the step. Two options, and they are not equivalent: fold q and k into one
   `apply_rotary` call (halves the call count, keeps eager), or turn `compile_rope` on so the whole
   chain becomes one kernel. **Measure, do not assume** — log915's switch report found compiled rope
   *removes* 0.4 ms of GPU work and *adds* 1–4 ms of wall time, because Dynamo's guard evaluation on
   a `dynamic=True` closure called 48×/step costs more than the kernels it saves. That measurement
   predates this branch and is worth redoing: `--compile-rope=true` is a one-flag sweep arm and log924
   is a clean `compile_rope=false` baseline to compare it against.

**Upside.** This is the only change that improves single-stream latency: batch-1 TPOT is 22.1 ms
against 3.3 ms of device work, so the ceiling is ~6× before the GPU is the limit. It is also the
prerequisite for §4.1's upside to show up at batch ≥ 256, because once the sampler is fixed the host
is the binding constraint again.

### 4.4 Prefill/decode co-scheduling — 32–42 % of decode throughput, and the whole ITL tail

**Evidence.** §3.4: mixed steps cost 2.1–2.9× a decode step and are 26 % of steps at batch 512, 54 %
at batch 1 024. Delivered throughput is 6 752 tok/s against a pure-decode capability of 9 890 tok/s
at batch 512 (10 629 at batch 1 024). ITL p50 52.4 ms vs p99 141.8 ms.

**Root cause.** `long_prefill_token_threshold = max_num_batched_tokens = 8 192`, so a single
admitting step can swallow the entire token budget: ~7 500 prefill tokens land in one step alongside
a full decode batch.

**Fix.** Lower `long_prefill_token_threshold` (e.g. 1 024–2 048) so prefill is chunked across more
steps, trading a slightly longer prefill for a much flatter ITL. This is a configuration experiment,
not a code change — worth one sweep arm before anything else on this list, because it costs nothing
to try.

### 4.5 The 250 W power cap — large-batch numbers are clock-limited

**Evidence.** §3.6: `sw_power_cap` active **98.9 %** of the batch-512 run and 99.1 % at batch 1 024,
SM clock down from 2 610 to 2 169 MHz (**−17 %**), power pinned at 246–247 W against a 250 W enforced
limit on a 450 W card. No thermal or hardware slowdown anywhere.

**Consequence.** The batch ≥ 256 rows of §3.1 and §3.3 understate what the silicon can do by roughly
the clock deficit. It does not change any conclusion — §4.1 and §4.2 are both order-of-magnitude
effects — but it belongs in the header of any cross-instance comparison. `enforced.power.limit` and
`clocks.max.sm` are already captured in `gpu.static.csv`; the per-run metrics header should carry
them too, alongside the CPU model, since §4.3 makes host speed the dominant variable at small batch.

### 4.6 The in-flight pipeline latch — latent, not yet costly

Per §2.2(b), `in_flight_batchs` depth can only decrease, and one step with nothing to schedule
collapses it permanently. Batch ≤ 16 spent 98 % of the run at depth 0. Measured cost today: **none**
— 21.94 ms/step at depth 0 vs 22.28 ms at depth 1 at batch 1, because a 3.3 ms GPU tail hides inside
a 21.7 ms launch window either way. It becomes real the moment §4.1 and §4.3 land and the device tail
stops being free. Fix when it matters: allow the deque to refill (don't pop on a step that appended
nothing), and record the depth in the metrics header so the `step` field's meaning is
self-describing.

---

## 5. What is *not* a bottleneck

* **Paged decode attention.** `flash_fwd_splitkv` moves 3.75 GB of KV in 4.86 ms at batch 512 —
  **772 GB/s, 77 % MBU**. Healthy.
* **The LM head.** 139 GFLOP in 0.82 ms — **~160 TFLOP/s, ~97 % of the card's BF16 peak**. The
  best-utilised kernel in the engine.
* **Prefill compute.** ≥ 64 TFLOP/s (≥ 39 % MFU) on 8 030-token steps. The deficit everywhere else
  is orchestration, not arithmetic.
* **The token read-back.** `dth` is **0.067 ms, flat across the entire sweep** — log915 measured
  5.24 ms at batch 1 024. Fixed by the on-device append + async D2H on this branch.
* **Host-side sampling.** `sample` (host) 1.44 → 1.57 ms across a 1 024× batch range, vs 71.5 ms at
  batch 512 in log915. Fixed by the resident `SamplingParamTable`.
* **Scheduler policy and the KV cache.** Zero preemptions, zero cache exhaustions across 11 batch
  sizes × 2 runs. `num_rescheduled == req_num` is an accounting artefact of
  `report_on_schedule` counting first admission, not a real re-schedule.
* **Thermals.** 71 °C peak, no slowdown flags. It is the power cap, not the cooler.
* **The reported TTFT / queueing.** Closed-loop artefacts — everything is enqueued at t=0. Quote
  `prefill` (22.8 ms at batch 1, 373.5 ms at 1 024).

---

## 6. Recommendations, in order

| # | Change | Cost to implement | Expected effect |
| --- | --- | --- | --- |
| 1 | **Vectorise the three host fill loops** (§4.2) — numpy views over the pinned buffers | hours | `bld_meta` 17.8 → <1 ms, `sched_wait` 41 → <1 ms; **~35 s off a 96 s run at batch 512** with no device change |
| 2 | **Short-circuit `do_penalities` on the parameter values** (§4.1 fix 1) | hours | −10 ms/step at batch 512, −1.5 GB transient at batch 1 024 |
| 3 | **Top-p on the candidate set** (§4.1 fix 2–3) | days | device work 51 → ~15 ms/step at batch 512; with #1, ≈**2× delivered throughput** |
| 4 | **Lower `long_prefill_token_threshold`** (§4.4) | one sweep arm | recovers part of the 32–42 % lost to mixed steps; flattens ITL p99 |
| 5 | **CUDA-graph the decode step** (§4.3) | days | the only fix for single-stream latency; up to ~6× headroom at batch 1; prerequisite for #3 to show at large batch |
| 6 | **Fuse rope** — one `apply_rotary` call per layer instead of two, and/or A/B `--compile-rope=true` against this baseline (§4.3) | hours | up to ~2.5 ms/step from halving the call count; the compiled arm is unmeasured on this branch and log915 found it *net negative* |
| 7 | **Record host CPU, driver, torch version, `enforced.power.limit` and pipeline depth in the metrics header** (§4.5, §4.6) | hours | makes these logs self-describing across instances |
| 8 | ~~**Fix the `repetition_penalty` → `rep_pen` field mapping**~~ — **done in the working tree** (§1.3) | — | correctness fix, but it makes the penalty pass *live*: the §6 #2 short-circuit will no longer fire on this model, so the ~10 ms/step is now a real cost that only #3 removes. #3 rises in priority accordingly |

Items 1 and 2 are a day of work between them and cover the largest measured effects. Item 8 has
already landed, which *raises* the price of the penalty pass on this model — so #2's value is now
limited to callers that genuinely leave all three penalties at their neutral values, and #3 moves up
to carry the penalty cost as well as top-p.

---

## 7. What changed since log915

The switch configuration differs (log915: both switches off; log924: both on) **and** the engine
internals changed, so treat this as directional.

| batch 512, pure-decode step | log915 | log924 | Δ |
| --- | ---: | ---: | ---: |
| `sample` (host) | 71.47 ms | **1.53 ms** | −98 % |
| `dth` (host) | 2.63 ms | **0.07 ms** | −97 % |
| `sample_gpu` (device) | 73.2 ms | **39.85 ms** | −46 % |
| `rope` (host) | 6.29 ms | 5.26 ms | −16 %⁵ |
| `bld_meta` (host) | 1.09 ms | **17.83 ms** | **+1 536 %** |
| `fwd` (host) | 22.50 ms | 30.24 ms⁴ | +34 % |
| **step wall** | 100.17 ms | **51.8 ms** | **−48 %** |
| Peak throughput | 4 681.7 tok/s @ 512 | **6 752.3 tok/s @ 512** | **+44 %** |
| Last profitable doubling | batch 128 | **batch 256** | — |

⁴ Launch-queue backpressure at batch 512 only; the comparable flat value is 21.57 ms at batch 256.
⁵ Both runs are eager rope, so this row is apples-to-apples and the gain is `pre_gather_cos_sin`
alone — consistent with log915's switch report pricing that switch at ~1.6 % of the step at batch 512.

* **The resident-table work paid off, twice.** Host sampling and the token read-back are gone as
  costs, and the device-side sampler halved as a side effect of building masks and counts on device.
* **`bld_meta` is the regression.** Per-request bookkeeping moved out of `sample` into
  `RequestBuffer` and picked up scalar host writes on the way (§4.2). Net still strongly positive,
  but this is a new 24 %-of-run item that did not exist in log915.
* **log915's recommendations #1 (candidate-set sampling) and #2 (CUDA graphs) are still open**, and
  still the two largest items. #3 (turn `pre_gather_cos_sin` on) is done — and it is the only switch
  that moved, `compile_rope` being false in both.
* **The GPU is no longer idle at large batch.** log915's profile measured 50–85 % idle; log924
  measures **4.8 % at batch 512**. The engine has crossed from host-bound to device-bound above
  batch ~128 — which is why §4.1 is now the top item and §4.2 the cheapest.

---

## 8. Caveats

* Both sweeps and both profiles ran on the same vast.ai instance within ~40 minutes. No cross-machine
  reproduction of this configuration.
* Five source files were uncommitted at measurement time (`constants.py`, `engine.py`, `metrics.py`,
  `sampling.py`, `scheduler.py`), so the exact code under test is not addressable by a commit hash.
* The workload is synthetic (random token ids, fixed 512 in / 128 out). Random ids change the penalty
  mask density versus real text, which affects §4.1's memory traffic but not its asymptotics.
* These runs are the `compile_rope = false` arm. The compiled arm is **unmeasured on this branch**;
  §4.3 fix 2 is the experiment that would close it. `pre_gather_cos_sin = true` is confirmed
  indirectly, from `aten::index_select` at 10 calls/step rather than the ~48 the per-layer gather
  path would produce.
* Per-layer CUDA-event instrumentation (24 `rope` event pairs + 5 more per step ≈ 59 event records)
  is present in *every* number here, benchmark and profile alike. ~0.1–0.3 ms/step — below the noise
  floor at every batch size, but it is not zero, and it is why `Event Sync` / `cudaEventQuery` appear
  in the traces.
* Grouped attribution inside §4.1's 40.6 ms divides shared aten rows (`sub`, `div`, `where`,
  `masked_fill_`, `_softmax`) between the penalty and top-p stages by op count. The per-row numbers
  are measured; the grouping is an estimate.
* MBU/MFU treat the decode step as reading the full 0.988 GB of weights once, exact for the body,
  plus 12 KiB per cached token. KV length is taken as ~596 tokens/sequence from `blk_used` (1 536
  blocks × 256 ÷ 512 sequences at batch 512).
* The profile harness runs `max_model_len = 1 024` against the sweep's 4 096; KV depth per sequence
  is comparable (~596 tokens) but the two instruments are not bit-identical configurations.
