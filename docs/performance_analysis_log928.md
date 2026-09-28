# Performance Analysis — `slot` vs `main` on RTX 4090 (log928)

**First A/B of the `slot` branch against `main`.** Everything in
[`performance_analysis_log924.md`](./performance_analysis_log924.md) measured `main`-lineage code on a
*different* vast instance; this report measures both branches back-to-back on **one** box, so the
branch deltas here are the trustworthy part and any comparison to log924 is not (§8).

**Artifacts analysed**

| Artifact | Path |
| --- | --- |
| Concurrency sweep, `slot`, runs 1–2 | `log_vast/log928/benchmark_slot{,2}/` |
| Concurrency sweep, `main`, runs 1–2 | `log_vast/log928/benchmark_baseline{,2}/` |
| Pure-decode idle profile, `slot`, runs 1–2 | `log_vast/log928/profile_slot{,2}/` |
| Pure-decode idle profile, `main`, runs 1–2 | `log_vast/log928/profile_baseline{,2}/` |
| Run-to-run noise floor, `main` | `log_vast/log928/baseline_against_baseline2/mean_step_metrics.*.log` |
| Run-to-run noise floor, `slot` | `log_vast/log928/slot_against_slot2/mean_step_metrics.*.log` |
| Cross-branch step metrics, batch 1 | `log_vast/log928/mean_step_metrics.1.log` |
| GPU static inventory / host | `log_vast/log928/gpu.static.csv`, `host_info` |

Cross-branch step-metric tables for batches 8–512 were **not** in the log directory (only batch 1 was);
they were regenerated for this report from the same dumps with the same tool
(`test_profile.py::test_mean_step_metrics`, `STEP_METRICS_LINES=65:84,65:84`), writing into a scratch
copy so nothing in `log_vast/` was touched.

**Code under test** — `main` at `0603ceb` (also the merge-base) vs `slot` at `f879a01` (6 WIP commits,
+1 610 / −825 lines across 21 files; the substance is `scheduler.py` +684, `sampling.py`, `engine.py`,
`attention_metadata.py`). Switches on both sides: `compile_rope = false`, `pre_gather_cos_sin = true`,
`use_d_first_schedule = false`.

**Run order** (same instance, ~65 min, no reboot): `slot` sweep 05:18 → `slot2` 05:28 → `main` sweep
05:42 → `main2` 05:58 → profiles 06:10–06:18. `slot` ran *first*, so it did not benefit from a warmed
box.

---

## Executive summary

1. **The async/pipelined step works, and it is the whole story.** Peak throughput **6 015 → 7 232 tok/s
   (+20.2 %)**, and at batch 128 **4 638 → 6 791 tok/s (+46.4 %)** with TPOT **27.0 → 17.8 ms (−34 %)**.
   GPU *kernel* work per step is unchanged (**+0.8 % at batch 512**) — every gain came from deleting
   host stalls, not from cheaper kernels.
2. **GPU idle in a pure-decode step collapsed: 27.5 % → 1.5 % at batch 512**, 42.0 % → 7.1 % at 128,
   51.9 % → 28.9 % at 64. The host no longer blocks anywhere in the step: `dth` 2.91 → 0.04 ms,
   host-side `sample` 56.05 → 1.09 ms, `sched_ret_gpu` 0.83 → **0.00** ms.
3. **Iso-throughput latency improved 4.7×.** `main` needs batch 512 and 83.3 ms TPOT to reach
   6 015 tok/s; `slot` beats that at batch 128 with **17.8 ms TPOT** — 13 % more throughput at a
   quarter of the per-token latency.
4. **Batch ≤ 16 is unchanged to slightly worse** (batch 1: −1.9 % throughput, +1.9 % TPOT). `bld_meta`
   nearly doubled at batch 1 (0.163 → 0.314 ms) — the resident-table metadata build has a fixed cost
   that only pays for itself above ~batch 32 (§3.4).
5. **Batch 1024 now OOMs** where `main` delivered 5 861 tok/s. The failing allocation is
   `torch.sort` inside `apply_top_p` asking for **1.73 GiB** with 1.72 GiB free (§5.1). This is a
   capacity regression, not a perf one, but it caps the engine at 512 concurrency on a 24 GB card.
6. **Prefill latency is the price of the pipeline: TTFT +37 % to +78 % for batch ≥ 32** (121.6 →
   216.6 ms at batch 512). One-step lookahead means a request's first token is committed one
   iteration late, and at high concurrency that iteration is often another prefill chunk (§5.2). This
   is the textbook cost of async scheduling and it is a deliberate trade, but it is not free.
7. **Sampling is now 79 % of all device work and is the only thing left worth optimising.** At batch
   512, **40.6 of 51.5 ms/step**. A single `aten::sort` over the full 151 936-wide vocabulary is
   **15.4 ms/step — 30 % of the whole step** — and it runs *after* top-k already reduced the live
   candidates to **20**. §6 and §7.1: sorting inside the top-k window instead would cut the width
   ~150× and simultaneously fix the batch-1024 OOM.
8. **Reproducibility.** Run-to-run throughput agrees within **1.26 %** on `main` and **3.22 %** on
   `slot` (worst case batch 1; median 1.11 %); device-side GPU-busy time agrees within **0.02 %**.
   The 17–46 % gains are 6–40× the noise floor. `slot` is the noisier branch (§5.3).

---

## 1. System under test

### 1.1 Hardware — and how it differs from log924

| | log928 (this report) | log924 |
| --- | --- | --- |
| GPU | RTX 4090, 24 564 MiB | RTX 4090, 24 564 MiB |
| Driver / VBIOS | 550.127.08 / 95.02.3C.80.C8 | 580.142 / 95.02.18.00.51 |
| `enforced.power.limit` | **450 W** (max 600 W) | **250 W** (max 450 W) |
| `pcie.link.gen.current` | **1** (of max 4) | **4** (of max 4) |
| Host claim | 82.2 TFLOPS, 876.6 GB/s, Seoul | — |

Two differences matter:

* **No power cap this time.** `clocks_event_reasons.sw_power_cap` is Active for **0.1 %** of samples
  on `main` and **0.3 %** on `slot`; SM clocks hold 2 588 / 2 494 MHz mean, power averages
  250 / 229 W against a 450 W ceiling, temperature peaks at 61 °C. log924's finding that large-batch
  numbers were clock-limited **does not apply here** — these are clean, uncapped numbers.
* **The PCIe link negotiated gen 1.** Host↔device transfers are on a ~4 GB/s link. The `slot` branch's
  central win is deleting host↔device round-trips, so a degraded link flatters it. The removed costs
  are small-transfer *latency* (a 4 KB `dth`, an event sync) rather than bandwidth, so the effect is
  probably second-order — but the +20…46 % figures should be re-measured on a gen-4 box before being
  quoted as the branch's general speedup.

### 1.2 Model and workload

Qwen2.5-0.5B-Instruct (0.494 B params, bf16), 24 layers, `hidden=896`, 14 Q / 2 KV heads,
`head_dim=64`, **`vocab_size=151936`**. KV cache 4 096 blocks × 256 tokens = 12 288 MB.

Both instruments use the fixed-shape ShareGPT benchmarking config — `max_model_len=1024`,
`max_num_batched_tokens=8192`, 512-token prompts, `ignore_eos`, so all `batch_size` requests prefill
together and decode in lockstep. The sweep runs 11 batch sizes (1 → 1 024) with 10× batch requests;
the profile runs 7 batch sizes over 20 measured + 5 profiled steady-state decode steps.

### 1.3 One parity gap worth knowing

`main` ran with `rep_pen = 1.0`; `slot` ran with `repetition_penalty = 1.1` (the field was renamed and
the fixture's value differs). Neither branch branches on the value — `apply_penalties` launches the
same kernels for 1.0 as for 1.1 — so **this does not affect any timing in this report**, but the two
branches did not generate identical text, and a correctness A/B cannot be read off these runs.

---

## 2. Methodology

### 2.1 The two instruments

* **Sweep** (`test_benchmark_sweep_batch_size`) — end-to-end, prefill and decode interleaved, no
  profiler. This is the number a user feels. Its `TTFT` column is mean *prefill* latency
  (queueing excluded); its `TPOT`/`ITL` are per-token.
* **Profile** (`test_profile_decode_idle_fraction`) — pure decode, no admissions or preemptions. Reports
  `wall_clean_us/step` (un-profiled) and `gpu_busy_from_trace_us/step` (union of kernel/memcpy/memset
  spans from a Chrome trace, so overlap counts once). `gpu_idle_fraction = 1 − busy/wall`.

### 2.2 Three traps in these logs

1. **`step_0 + step_1` is not the step time — it double-counts `step_0`.** `step_1` is started on
   iteration *N*'s metrics object but stopped inside iteration *N+1*'s
   `sample_in_flight_step()` ([`engine.py:38`](../src/qwen/engine.py#L38),
   [`scheduler.py:706`](../src/qwen/scheduler.py#L706)), so it spans the tail of *N* plus the whole
   `step_0` of *N+1* — i.e. one full iteration. Median `step_1` tracks the profiler's independent
   `wall_clean` almost exactly:

   | batch | med `step_0` | med `step_1` | `step_0+step_1` | `wall_clean` |
   | --- | --- | --- | --- | --- |
   | 8 | 10.31 | 10.77 | 21.09 | 10.64 |
   | 32 | 10.69 | 11.17 | 21.86 | 11.06 |
   | 64 | 10.70 | 11.21 | 21.91 | 11.17 |
   | 128 | 14.69 | 15.29 | 29.98 | 15.30 |
   | 512 | 51.70 | 52.66 | 104.36 | 52.33 |

   **Use `step_1` as the step wall time.** As a side effect `step_1` inherits any spike from the
   *next* iteration, which is why it is the field the anomaly detector flags most often on `slot`.
2. **On `slot`, host-side `fwd` and `rope` are wait time, not work.** At batch 512, `fwd` = 46.8 ms
   CPU against `fwd_gpu` = 10.7 ms, and `rope` = 21.2 ms CPU against `rope_gpu` = 0.82 ms. The launch
   thread is blocking on a full CUDA launch queue — the signature of a *saturated GPU*, which is the
   goal. Reading these as regressions (`fwd` +377 %, `rope` +868 %) inverts the meaning.
3. **`*_gpu` fields are device-timeline *elapsed*, not kernel-busy.** They come from CUDA events
   bracketing a region, so they include device bubbles inside it. At batch 1, `fwd_gpu` = 8.5 ms while
   the trace says the whole step only keeps kernels busy 3.25 ms. Only
   `gpu_busy_from_trace` is occupancy.

### 2.3 Noise floor

| Quantity | `main` | `slot` |
| --- | --- | --- |
| Sweep throughput, run-to-run | ≤ 1.26 % (median 0.69 %) | ≤ 3.22 % (median 1.11 %) |
| `gpu_busy_from_trace`/step, run-to-run | ≤ 0.03 % | ≤ 0.02 % |
| Step-metric fields beyond 2σ | 0–8 of 18 per batch, all \|Δ\| < 3 % except batch 512 `step`/`sample` (−2.7 / −2.8 %) | 1–8 of 20 per batch, all \|Δ\| < 11 % |

Anomalous steps per 20-step window: **0/20 on `main` at every batch size**; **1–4/20 on `slot`**. The
branch is measurably less steady (§5.3).

---

## 3. Results

### 3.1 Concurrency sweep — mean of two runs each

| batch | `main` tok/s | `slot` tok/s | Δ tok/s | `main` TPOT ms | `slot` TPOT ms | Δ TPOT | `main` TTFT ms | `slot` TTFT ms | Δ TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 102 | 100 | **−1.9 %** | 9.83 | 10.01 | **+1.9 %** | 10.2 | 10.4 | +2.5 % |
| 2 | 187 | 186 | −0.2 % | 10.71 | 10.71 | +0.0 % | 11.6 | 11.6 | +0.1 % |
| 4 | 369 | 368 | −0.3 % | 10.78 | 10.79 | +0.1 % | 18.8 | 18.6 | −1.0 % |
| 8 | 717 | 722 | +0.7 % | 10.99 | 10.85 | −1.2 % | 33.0 | 32.5 | −1.5 % |
| 16 | 1 345 | 1 366 | +1.6 % | 11.49 | 11.21 | −2.5 % | 62.1 | 60.8 | −2.1 % |
| 32 | 2 275 | 2 668 | **+17.3 %** | 13.67 | 11.33 | **−17.1 %** | 58.6 | 80.7 | **+37.6 %** |
| 64 | 3 557 | 4 919 | **+38.3 %** | 17.53 | 12.17 | **−30.6 %** | 63.0 | 96.7 | **+53.5 %** |
| 128 | 4 638 | 6 791 | **+46.4 %** | 27.00 | 17.77 | **−34.2 %** | 73.2 | 118.1 | **+61.3 %** |
| 256 | 5 498 | 7 149 | **+30.0 %** | 45.64 | 34.30 | **−24.9 %** | 90.8 | 154.1 | **+69.6 %** |
| 512 | 6 015 | 7 232 | **+20.2 %** | 83.31 | 68.23 | **−18.1 %** | 121.6 | 216.6 | **+78.2 %** |
| 1024 | 5 861 | **OOM** | — | 169.68 | — | — | 199.6 | — | — |

Scaling efficiency (`tok/s/req` relative to batch 1) holds far longer: at batch 64 it is **0.76–0.78 on
`slot` vs 0.54–0.55 on `main`**, and the profitable-doubling marker sits at batch 64 for both but with
gain/cost **1.77 vs 1.22**.

### 3.2 Iso-throughput: the headline

| | `main` | `slot` |
| --- | --- | --- |
| Concurrency needed for ~6 000 tok/s | 512 | 128 |
| Throughput delivered | 6 015 | 6 791 (+13 %) |
| TPOT | 83.31 ms | **17.77 ms (4.7× lower)** |
| ITL p50 / p90 | 70.3 / 125.4 ms | 15.1 / 15.2 ms |
| Prefill latency | 121.6 ms | 118.1 ms |

At equal delivered throughput the `slot` branch is better on **every** axis including TTFT. The TTFT
regression in §3.1 only appears when you compare at equal *concurrency*, where `slot` is doing far
more work per second.

### 3.3 GPU idle fraction and GPU busy — pure decode

| batch | `main` idle (r1/r2) | `slot` idle (r1/r2) | `main` busy µs/step | `slot` busy µs/step | Δ busy |
| --- | --- | --- | --- | --- | --- |
| 1 | 68.2 / 68.0 % | 76.0 / 66.7 % | 3 203 | 3 255 | +1.6 % |
| 8 | 62.4 / 62.1 % | 59.8 / 59.4 % | 4 219 | 4 277 | +1.4 % |
| 32 | 59.1 / 59.9 % | 50.5 / 50.7 % | 5 414 | 5 470 | +1.0 % |
| 64 | 51.6 / 52.1 % | **29.2 / 28.5 %** | 7 856 | 7 910 | +0.7 % |
| 128 | 42.3 / 41.7 % | **7.1 / 7.0 %** | 14 083 | 14 229 | +1.0 % |
| 256 | 32.5 / 32.5 % | 12.2 / 13.0 % | 26 826 | 27 068 | +0.9 % |
| 512 | 28.4 / 26.5 % | **1.5 / 1.5 %** | 51 132 | 51 526 | +0.8 % |

Read the last two columns first: **the kernels are the same kernels**, within 1 %. The branch did not
make the GPU faster; it stopped leaving it idle. Un-profiled wall time per step follows:
batch 512 **70.5 → 52.3 ms (−25.8 %)**, batch 128 **24.3 → 15.3 ms (−37.0 %)**, batch 64
**16.3 → 11.1 ms (−31.9 %)**.

At batch 512 the residual idle is 0.8 ms of 52.3 — the host has essentially nothing left to hide.
Batch 256 is the exception (12.6 % idle, worse than batch 128's 7.1 %): see §5.3.

### 3.4 Host-side per-step breakdown, pure decode (ms, run 1 vs run 1)

| field | b=1 main → slot | b=64 main → slot | b=512 main → slot | what it is |
| --- | --- | --- | --- | --- |
| `sched` | 0.092 → **0.020** (−78 %) | 0.239 → **0.077** (−68 %) | 1.274 → **0.494** (−61 %) | schedule() total |
| `sched_ret` | 0.077 → **0.006** (−92 %) | 0.176 → **0.015** (−91 %) | 0.847 → **0.082** (−90 %) | building the SchedulerOutput |
| `sched_ret_gpu` | 0.063 → **0.000** | 0.162 → **0.000** | 0.831 → **0.000** | device work inside `sched_ret` — **gone entirely** |
| `bld_meta` | 0.163 → **0.314** (+92 %) | 0.219 → 0.344 (+57 %) | 0.647 → **0.506** (−22 %) | attention metadata |
| `sample` (host) | 0.785 → 0.683 (−13 %) | 5.754 → **0.688** (−88 %) | 56.054 → **1.089** (−98 %) | host time in the sampler |
| `sample_gpu` | 0.510 → 0.434 | 5.540 → 3.896 | 56.907 → 40.565 | device-elapsed in the sampler |
| `dth` | 0.023 → 0.035 | 0.110 → **0.034** | 2.913 → **0.036** (−99 %) | device→host token read |
| `ci` | 0.006 → 0.012 | 0.031 → 0.043 | 0.270 → 0.269 | commit/bookkeeping |

Three things stand out:

* **`sample` host time went flat.** On `main` it tracks `sample_gpu` almost 1:1 (56.05 vs 56.91 ms at
  batch 512) — the host was *waiting inside the sampler*. On `slot` it is ~0.7–1.1 ms at every batch
  size: purely launch cost.
* **`sched_ret_gpu` is exactly zero at every batch size.** Scheduling no longer touches the device.
* **`bld_meta` crosses over.** It is ~2× more expensive at batch 1–64 and ~22 % cheaper at batch 512:
  the resident slot-indexed tables have a fixed per-step cost that amortises only with width. That
  fixed cost, plus the extra `dth`/`ci` bookkeeping, is exactly the −1.9 % at batch 1.

Summing the non-forward segments at batch 512 gives **61.2 → 2.5 ms**, though on `main` most of that
61.2 ms is the host *waiting* inside `sample` rather than working. The clean statement is the one the
trace gives independently: **`main` left 19.4 ms/step of step wall time the GPU could not overlap
(70.5 wall − 51.1 busy); `slot` leaves 0.8 ms** (52.3 − 51.5).

### 3.5 Device-side op breakdown at batch 512 (`slot`, ms/step, from `key_averages`)

Total GPU busy **51.5 ms/step**. Sampler total (`sample_gpu`) **40.6 ms = 79 %**; model forward
**10.7 ms = 21 %**; logits 0.89 ms.

| op | ms/step | share of step | belongs to |
| --- | --- | --- | --- |
| **`aten::sort`** | **15.43** | **30.0 %** | top-p |
| `flash_attn::_flash_attn_varlen_forward` | 4.89 | 9.5 % | attention |
| `aten::mm` | 3.62 | 7.0 % | model GEMMs |
| `aten::sub` | 3.08 | 6.0 % | penalties |
| `aten::copy_` | 2.67 | 5.2 % | mixed |
| `aten::mul` | 2.66 | 5.2 % | mixed |
| `aten::div` | 2.50 | 4.9 % | penalties / temperature |
| `aten::_softmax` | 2.45 | 4.8 % | top-p + final softmax (2 calls/step) |
| `aten::where` | 2.24 | 4.4 % | penalties / top-k / greedy select |
| `aten::masked_fill_` | 2.18 | 4.2 % | penalties / top-p |
| Memcpy DtoD | 1.92 | 3.7 % | `rep.repeat(1, vocab)` |
| `aten::topk` | 1.87 | 3.6 % | top-k |
| `aten::scatter_` | 0.92 | 1.8 % | top-p scatter-back |
| `aten::cumsum` | 0.83 | 1.6 % | top-p |
| `aten::fill_`, `argmax`, `gt`, `sum` | 2.89 | 5.6 % | sampler tail |
| `aten::addmm` | 0.47 | 0.9 % | model |

Model + attention is **~9 ms**. Everything else is the sampler grinding `[512, 151936]` fp32 tensors —
311 MB per materialised temporary, and there are a dozen of them.

---

## 4. What the branch actually changed

Three mechanisms, in order of payoff.

**(a) One-step-lookahead pipeline** — [`engine.py`](../src/qwen/engine.py). `step()` is now
`step_0` (schedule + forward + sample **launch**, no sync) followed by `step_1`
(`sample_in_flight_step()`, which drains the *previous* step). `in_flight_steps` is a deque; the
sampled token is written into the token table **on device**
(`add_sampled_tokens_on_device`), and `update_projected_state_in_advance()` advances the scheduler's
view so step *N+1*'s metadata can be built without knowing step *N*'s token values. The only sync
left is `events.synchronize("dth")` on the previous step, by which time the copy has long landed.
This is what turned 19.4 ms/step of exposed host time into 0.8 ms.

**(b) Sync-free sampling setup** — [`sampling.py`](../src/qwen/sampling.py). `bin_counts_and_mask`'s
per-step Python list building and H2D scatter is gone; `prompt_mask` / `output_counts` /
`output_mask` are now resident device tensors gathered by slot index. `apply_top_k` takes `max_k`
from a **pinned host mirror** of the top-k table instead of `top_k.max().item()`, so choosing the
`topk` width no longer costs a device→host sync. `MAX_EFFECTIVE_TOP_K = 1024` caps the `topk` width.

**(c) Slot-indexed resident tables** — [`scheduler.py`](../src/qwen/scheduler.py). `TokenIdTable`,
`SamplingParamTable` and the request buffers are indexed by a stable slot per request, so per-step
work is `index_select` on device rather than Python iteration. `sched_ret_gpu → 0` is this change.

---

## 5. Regressions and risks

### 5.1 Batch 1024 OOMs inside `apply_top_p` — **blocking for 1024 concurrency**

```
File ".../qwen/sampling.py", line 152, in apply_top_p
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.73 GiB.
GPU 0 has a total capacity of 23.64 GiB of which 1.72 GiB is free.
```

`1024 × 151936 × 4 B = 622 MB` per fp32 temporary, and `sort` needs values **and** int64 indices plus
workspace — 1.73 GiB in one allocation. Peak `memory.used` is **24 200 MiB of 24 564** on `slot`
against 23 972 on `main`: the pipeline holds one extra step's tensors alive, so `slot` runs ~228 MiB
closer to the edge and tips over. `main` completed batch 1024 at 5 861 tok/s (below its own 512 peak,
so nothing valuable is lost in throughput terms — but the engine now *fails* instead of degrading).
§7.1 fixes the cause rather than the symptom.

### 5.2 TTFT +37 % … +78 % for batch ≥ 32

Prefill latency at batch 512 goes **121.6 → 216.6 ms**. Mechanism, consistent with the data: a
request's first token is produced by step *N* but only committed during step *N+1*'s drain, so TTFT
gains one whole following iteration — and at high concurrency that iteration is frequently another
8 192-token prefill chunk. The size of the penalty tracks step time (batch 64: +34 ms ≈ 3 steps'
decode or ~⅓ of a prefill step; batch 512: +95 ms ≈ one prefill step), which is what that mechanism
predicts. Decode ITL is *better* everywhere — at batch 512, p50 **70.3 → 52.4 ms** and p90
**125.4 → 119.5 ms** — so the cost is confined to the first token.

This is the standard, accepted cost of async scheduling. Flagging it because the sweep table reads
like a regression and it is worth a deliberate decision, not a surprise: if first-token latency is a
product requirement, the drain of a step that produced a *first* token could be prioritised, or
prefill chunks kept out of the iteration immediately after an admission.

### 5.3 `slot` is less steady, and batch 256 is a visible outlier

* Anomalous steps per 20-step window: **0/20 on `main` at every batch size**; on `slot` 1–4/20.
* Batch 256 idle fraction **12.6 %**, worse than batch 128's **7.1 %** — non-monotonic.
* The batch-256 dump shows the mechanism: two of twenty steps take ~90 ms against a 27–28 ms norm,
  and at line 79 the spike is **device-side** — `fwd_gpu` 64.1 ms against a 7.1 ms local median (9.1×)
  — not a host stall.
* `profiler overhead` goes **negative** for `slot` at batch 256 and 512 (−16.7 %, −15.2 %): the
  *profiled* run was faster than the "clean" one, which only happens when the clean window caught
  these stalls. Treat `slot`'s batch-256 idle fraction as an upper bound.

A device-side stall of 9× in a step whose kernels are unchanged, on a branch whose peak memory sits
364 MiB from the ceiling, points at allocator pressure: at batch 256 each full-vocab fp32 temporary is
156 MB, the pipeline keeps two steps' worth reachable, and a cache-miss `cudaMalloc` has to
synchronize and can force `cudaFree`. Same root cause as §5.1. Worth confirming with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a `torch.cuda.memory_stats()`
`num_alloc_retries` check before chasing anything else.

### 5.4 Batch 1 regressed ~2 %

Throughput 102 → 100 tok/s, TPOT 9.83 → 10.01 ms. `bld_meta` +0.150 ms and `dth` +0.013 ms and
`ci` +0.006 ms outweigh `sched` −0.072 ms. Small, real, and the expected shape of a change that trades
fixed per-step setup for per-item scaling. Note batch 1 is also `slot`'s noisiest point (3.22 %
run-to-run, and `profile_slot` run 1 shows a 76 % idle fraction against run 2's 66.7 %), so the true
figure is somewhere in −1 % … −3 %.

### 5.5 `step_0`/`step_1` instrumentation double-counts

See §2.2 trap 1. `step_0 + step_1` reads as 104 ms/step at batch 512 when the step is 52 ms. This
misleads the anomaly detector (`step_1` is its most-flagged field, inheriting the *next* step's
spikes) and will mislead anyone reading a dump. Cheap fix: stop `step_1` where it starts — or rename
the pair to make the overlap explicit and add a derived `step` = `step_1`.

---

## 6. What is left: sampling is 79 % of device work

With the host stalls gone, the engine at batch 512 spends **40.6 of 51.5 ms/step in the sampler** and
**9 ms doing the actual model**. The sampler's cost is not arithmetic, it is memory traffic over
`[bsz, 151936]` fp32 tensors, and most of it is avoidable:

1. **`apply_top_p` sorts the full vocabulary — 15.4 ms/step, 30 % of the step.** It runs *after*
   `apply_top_k` has already set everything outside the top **20** (`config.top_k = 20`) to `-inf`.
   Sorting 151 936 columns to rank 20 finite values is ~7 600× more width than the problem needs.
   Worse, `torch.topk` **already returns its values sorted descending** — the sort is redundant, not
   merely oversized. `constants.py` even documents the insight ("sort overhead is almost flat when
   top_k ≤ 1024") but applies the cap only to the `topk` width.
2. **`apply_penalties` materialises `[bsz, vocab]` temporaries to apply a per-row scalar.**
   `rep = repetition_penalty[:, None].repeat(1, vocab)` is a 311 MB DtoD copy (the 1.92 ms Memcpy DtoD
   row) purely so the next line can index it with a mask. Then `rep[~(prompt|output)] = 1.0`
   (`masked_fill_`, 2.18 ms) and `where(logits>0, logits/rep, logits*rep)` (`gt` + `div` + `mul` +
   `where`, ~8 ms). The whole thing is one fused elementwise pass over `logits` with a broadcast
   `[bsz,1]` scalar and two `[bsz,vocab]` bool masks.
3. **Two full-vocab softmaxes per step.** `apply_top_p` softmaxes the sorted logits, then `sample`
   softmaxes the full vocab again (`aten::_softmax`, 10 calls / 5 steps = 2/step, 2.45 ms).
4. **The dead-row guard and greedy path each add a full-vocab pass** (`sum`, `isfinite`,
   `masked_fill`, `argmax` — ~1.4 ms/step combined). Correct and deliberately branch-free (the comment
   explains why), but they would be ~150× cheaper inside a top-k window too.

---

## 7. Recommendations, in order

### 7.1 Do top-p inside the top-k window — removes ~15 ms/step *and* fixes the batch-1024 OOM

`apply_top_k` already computes `top_vals, top_idx = torch.topk(logits, max_k)`, **descending**. Run the
whole tail of the sampler on that `[bsz, max_k]` block (`max_k ≤ 1024`, 20 in this config):

* `softmax` → `cumsum` → `(cum - p) > top_p` on `[bsz, max_k]` — no `sort`, no `scatter_` back to
  vocab order, no `[bsz,vocab]` `masked_fill`.
* `multinomial` over the `[bsz, max_k]` probabilities, then map the result back through `top_idx`
  with a `gather`.
* Keep the current full-vocab path only for rows where top-k is genuinely disabled (`max_k <= 0`),
  which the code already detects host-side without a sync.

Expected: `sort` (15.4), `scatter_` (0.92), `cumsum` (0.83), one `_softmax` (~1.2) and most of the
top-p `masked_fill`/`where` traffic collapse to ~0.1 ms — **roughly 18 of 51.5 ms/step at batch 512**,
i.e. TPOT ~52 → ~34 ms and peak throughput plausibly past 10 000 tok/s. The 1.73 GiB sort allocation
disappears with it, unblocking batch 1024 (§5.1) and relieving the allocator pressure behind §5.3.

Sequencing note: this changes only the sampler's *shape*, not its distribution — top-k then top-k-window
top-p is mathematically identical to top-k then full-vocab top-p, since the excluded entries are
`-inf` either way. Verify with `test_sampling.py` plus a fixed-seed logits comparison before/after.

### 7.2 Fuse the penalty pass — removes ~8 ms/step

Drop the `.repeat(1, vocab)`. Compute `rep_row = where(prompt_mask | output_mask, rep[:,None], 1.0)`
implicitly inside a single expression over `logits`, or better, apply penalties *after* 7.1's top-k
narrowing so the pass is `[bsz, max_k]` wide. If `do_penalities` is on but the parameters are
identities (`rep == 1.0 and freq_pen == 0 and pres_pen == 0` — the `main` default, and detectable
host-side from the resident tables without a sync), skip the pass entirely.

### 7.3 Confirm and fix the allocator pressure

Run `slot` at batches 256/512/1024 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and log
`torch.cuda.memory_stats()['num_alloc_retries']` per step. If retries correlate with the 9× `fwd_gpu`
spikes, §5.3's non-monotonic batch-256 idle fraction and the batch-1024 OOM are the same bug and 7.1
mostly dissolves both. Low effort, high information.

### 7.4 Fix the `step_1` overlap before the next A/B

§5.5. Two lines, and it stops every future dump from over-reporting step time 2×.

### 7.5 Then, and only then, look at batch ≤ 16

Batch 1–16 is unchanged-to-1.9 %-worse, and batch 1 — the only small size the profile covers — is
still **68 % GPU-idle**: the host needs ~10 ms to launch a step whose kernels occupy 3.2 ms. That is a
CUDA-graph / reduced-dispatch problem, not a scheduling one, and it is untouched by this branch.
`bld_meta`'s doubled fixed cost (§3.4) is the one piece of it this branch made worse, and the cheapest
to look at.

### 7.6 Re-measure on a PCIe gen-4 instance

§1.1. The branch's headline numbers were taken on a gen-1 link, which flatters a change whose whole
point is fewer host↔device round-trips. Expect the gains to hold directionally and shrink somewhat.

---

## 8. Caveats

1. **Different instance from log924** — 450 W vs 250 W cap, driver 550 vs 580, PCIe gen 1 vs gen 4. No
   number here is comparable to log924. Within log928 both branches ran on the same box within 25
   minutes of each other, with `slot` first (no warm-box advantage).
2. **`repetition_penalty` 1.1 (`slot`) vs `rep_pen` 1.0 (`main`)** — same kernels, same timings,
   different generated text. §1.3.
3. **Cross-branch step-metric tables for batches 8–512 were regenerated locally**, not taken from the
   log directory (which held only batch 1). Same tool, same line range, same dumps; the scratch copy
   means `log_vast/` is unmodified.
4. **`slot`'s batch-256 and batch-1 profile numbers are contaminated** by the stalls of §5.3 — the
   negative "profiler overhead" is the tell. Batch 128 and 512 are clean in both runs and carry the
   argument.
5. **Sampler share is read from `key_averages`, where `aten::` rows and their kernel rows are both
   listed** — percentages there sum past 100 %. The §3.5 table uses `aten::` rows only, and its total
   (≈51 ms) cross-checks against the independent trace `gpu_busy` (51.5 ms) and against the CUDA-event
   fields (`fwd_gpu` + `logits_gpu` + `sample_gpu` = 52.2 ms).
6. **The `torch.cuda.set_sync_debug_mode("warn")` output is not in the log directory** — no `warning`
   file was present and neither `pytest.log` contains a synchronisation warning, so the claim that
   `slot`'s decode path is sync-free rests on `sched_ret_gpu → 0`, host-side `sample` → ~1 ms, and the
   1.5 % idle fraction rather than on the debug-mode evidence. Capturing that output would make the
   case airtight.
7. **20 decode steps per profile point** (lines 65–84), AR(1)-deflated `n_eff` typically 5–20. Fine for
   the 30–100 % effects reported; not fine for anything under ~5 %.
