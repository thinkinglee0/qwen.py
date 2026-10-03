# Performance analysis — fused top-k + top-p vs `async_scheduling` on RTX 4090 (log1001)

> 中文版：[`performance_analysis_log1001.zh.md`](./performance_analysis_log1001.zh.md)

**This is the follow-up that cashes in [`performance_analysis_log928.md`](./performance_analysis_log928.md) §7.1.**
That report ended with one recommendation above all others: the sampler was spending **15.4 ms/step
sorting all 151 936 vocabulary columns** to rank the **20** candidates that top-k had already left
finite, and removing it should be worth ~18 ms/step at batch 512 *and* fix the batch-1024 OOM. This
report measures the branch that did it.

**Data sources**

| Content | Path |
| --- | --- |
| Concurrency sweep, `fused_top_kp`, runs 1–2 | `log_vast/log1001/benchmark_fused_top_kp{,2}/` |
| Concurrency sweep, `async_scheduling`, runs 1–2 | `log_vast/log1001/benchmark_baseline{,2}/` |
| Pure-decode idle-fraction profile, `fused_top_kp`, runs 1–2 | `log_vast/log1001/profile_fused_top_kp{,2}/` |
| Pure-decode idle-fraction profile, `async_scheduling`, runs 1–2 | `log_vast/log1001/profile_baseline{,2}/` |
| Cross-branch step metrics, batch 1–512 | `log_vast/log1001/baseline_against_fused_top_kp/mean_step_metrics.*.log` |
| Run-to-run noise floor, `async_scheduling` | `log_vast/log1001/baseline_against_baseline2/mean_step_metrics.*.log` |
| Run-to-run noise floor, `fused_top_kp` | `log_vast/log1001/fused_top_kp_against_fused_top_kp2/mean_step_metrics.*.log` |
| GPU static inventory / host | `log_vast/log1001/gpu.static.csv`, `host_info` |

**Code under test** — baseline `async_scheduling` at `bedc00f`; candidate `fused_top_kp` at `49bb10d`.
The candidate is **exactly one commit ahead of the baseline** (`git rev-list --count
async_scheduling..fused_top_kp` = 1), and that commit touches three files: `src/qwen/sampling.py`
(+51/−19), `src/qwen/config.py` (one line), `tests/test_sampling.py` (+227). This is as clean an A/B as
this repo gets — no scheduler change, no engine change, no kernel change outside the sampler.

Engine switches identical on both sides: `compile_rope = false`, `pre_gather_cos_sin = true`,
`use_d_first_schedule = false`, `max_num_batched_tokens = 8192`.

**Run order** (one vast instance, ~45 minutes, no restart): `fused_top_kp` profile 23:40 → its run 2
23:41 → `fused_top_kp` sweep 23:44–23:53 → its run 2 23:53–00:02 → `async_scheduling` profile 00:04 →
its run 2 00:06 → `async_scheduling` sweep 00:08–00:16 → its run 2 00:17–00:25. **The candidate ran
first**, on the colder machine, so it did not collect a warm-up advantage.

---

## Executive summary

1. **The predicted win landed, and then some. Peak throughput 7 245 → 10 976 tok/s (+51.5 %)**, batch
   256 **7 166 → 10 704 (+49.4 %)**, batch 128 **6 816 → 8 707 (+27.7 %)**. TPOT at batch 512
   **68.1 → 44.4 ms (−34.8 %)**.
2. **The whole gain is one op disappearing.** `aten::sort` was **15.43 ms/step** of a 52.1 ms device
   step at batch 512 — 29.6 % of all device time. It is now **absent from the trace entirely**. With
   the traffic it dragged along (`masked_fill_`, `scatter_`, `cumsum`, one full-vocab `_softmax`, the
   DtoD copies) the step's device time falls **52.1 → 27.4 ms (−47.3 %)**.
3. **The batch-1024 OOM is fixed.** The baseline still dies in `apply_top_p`'s `torch.sort` asking for
   **1.73 GiB** with 1.72 GiB free (`benchmark_baseline/pytest.log:174`). `fused_top_kp` completes
   batch 1024 at **9 630 tok/s** — runnable, though past the throughput peak (§5.4).
4. **Nothing below batch 128 moved, and that is the correct result.** At batch ≤ 64 the engine is
   host-bound (GPU idle 50–67 %), so removing device work buys **+0.1 % … +3.4 %** throughput. The
   saving is real there too — device busy at batch 64 is **7.91 → 5.40 ms/step (−31.7 %)** — it just
   has nowhere to go.
5. **The GPU-bound knee moved from batch 64 to batch 128.** The sweep's last profitable doubling
   shifts 64 → 128, and scaling efficiency at batch 128 goes **0.52 → 0.67**. The engine's default
   `max_num_seqs = 128` now sits *at* the knee instead of one doubling past it.
6. **Prefill latency improved too, which pays back part of log928's TTFT regression.** Batch 512
   **216.8 → 172.8 ms (−20.3 %)**, batch 256 **−13.6 %**, batch 128 **−7.2 %**. Against log928's `main`
   (121.6 ms at batch 512) the async-scheduling TTFT penalty is now **46 % repaid** for free.
7. **Sampling is still the biggest line item: 16.35 of 27.3 ms/step (60 %) at batch 512.** Every
   remaining millisecond is still an `O(bsz × 151 936)` pass — temperature, penalties, the `-inf`
   scatter-back, the fp32 softmax, the dead-row guard, `multinomial`, `argmax`. ~23 full-vocab passes
   where the problem is 20 columns wide. §7.1–7.2 lay out how to get this to ~2.5 ms/step.
8. **Reproducibility is excellent.** Run-to-run throughput agrees to **≤ 1.14 %** on the baseline and
   **≤ 1.99 %** on the candidate (medians 0.45 % / 0.51 %); device-side `sample_gpu` at batch 512
   agrees to **0.00 %** across candidate runs. The baseline also reproduces log928's
   `async_scheduling` numbers on a *different* instance to within **0.2 %** (peak 7 245 vs 7 232 tok/s,
   TPOT 68.11 vs 68.23 ms, device busy 51 515 vs 51 526 µs/step). The 28–52 % gains are ~25× the noise.

---

## 1. System under test

### 1.1 Hardware

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24 564 MiB, driver 550.127.08 |
| Clocks / power | 3 135 MHz SM, 10 501 MHz mem, 450 W enforced (600 W max) |
| PCIe | **gen 4 × 16** (current = max) |
| CPU | AMD Ryzen 5 7500F, 6 cores / 12 threads, 63.9 GB RAM |
| Instance | vast 53604495, host 132677 |
| Roofline assumed by the sweep tool | 1 008 GB/s, 165 TFLOP/s |

Unlike log928's instance this one is PCIe gen 4 at full width, so the host-link caveat from that
report does not apply here. It does mean **cross-log absolute comparisons remain unsafe** (§8) — but
see summary point 8: the baseline landed within 0.2 % of log928's `async_scheduling`, which is a
stronger consistency check than this report needs.

### 1.2 Model and workload

Qwen2.5-0.5B-Instruct, bf16, `vocab_size = 151 936`, 24 layers, GQA 14/2, `head_dim = 64`.
Sampling config from `generation_config.json`: **`temperature = 0.7`, `top_k = 20`, `top_p = 0.8`,
`repetition_penalty = 1.1`** — identical on both branches (verified in both `pytest.log`s), so the
comparison is apples-to-apples. This matters: §5.3 explains why the `config.py` default change in the
diff is inert here.

* **Sweep**: 10 × batch requests, 512 input tokens / 128 output tokens each, `max_model_len = 1024`,
  single prefill chunk per request (`prefill_chunk.mean = 1.0` throughout). Batch 1 → 1024.
* **Profile**: pure decode, 20 measured steps + 5 traced steps per batch, `num_blocks = 4096`,
  `block_size = 256`. KV drift between runs 4.34 %, identical on both branches.
* Zero preemptions, zero cache exhaustions, zero rescheduled requests in **all 42 benchmark runs**.
  Step counts match exactly per batch across branches (e.g. 1 324 steps at batch 512 on both), so the
  two branches really did do the same work.

### 1.3 Anomaly counts

`test_mean_step_metrics` flagged **0/20** anomalous steps at batch 1/8/32/512 and **1/20** at batch
64/128/256 — the same pattern on both branches. Nothing was excluded.

---

## 2. Methodology

### 2.1 The two instruments

* **Sweep** (`test_benchmark.py`) — end-to-end, un-profiled: throughput, TPOT, prefill latency,
  queueing, ITL percentiles. This is the number that matters to a user.
* **Profile** (`test_profile.py::test_profile_decode_idle_fraction`) — steady-state pure decode with
  `torch.profiler` on 5 steps, plus a clean un-profiled wall measurement over 20 steps. Yields
  `gpu_busy_from_trace`, `gpu_idle_fraction`, and the per-op `key_averages` table. Profiler overhead
  is +69 % on wall, which is why idle fraction is computed against `wall_clean`, not trace wall.

`test_mean_step_metrics` then diffs two runs' step-metrics dumps field by field with a Welch z-test
over 20 decode steps, labelling each field `noise` or `SHIFT` at 2σ.

### 2.2 How to read the `*_gpu` fields — the one trap in these logs

`fwd_gpu`, `logits_gpu`, `sample_gpu` are **CUDA-event spans on the stream, not kernel sums**. They
tile the step, so they add up to the *wall*, not to the busy time:

| | batch 64 | batch 512 |
| --- | --- | --- |
| `fwd_gpu` + `logits_gpu` + `sample_gpu`, baseline | 6.57 + 0.36 + 3.90 = **10.82** | 10.66 + 0.89 + 40.56 = **52.11** |
| `wall_clean` / `gpu_busy`, baseline | 10.92 / **7.91** | 52.23 / **51.52** |

At batch 512 the device is saturated, so spans ≈ kernel time. At batch 64 the device is idle 27.6 % of
the step, and **the bubbles are inside the spans**. This is why
`baseline_against_fused_top_kp/mean_step_metrics.64.log` reports `fwd_gpu` **+39.1 %** and `rope_gpu`
**+42.9 %** on the candidate while total device busy *fell 31.7 %*: the candidate's step is more
starved, so more of the bubble lands inside the forward's event span. **No kernel got slower.** Cross-check:
`aten::mm` self-CUDA at batch 512 is 3.608 vs 3.614 ms/step, and `flash_attn` 4.888 vs 4.886 — the
model is bit-for-bit the same work.

The same effect explains the host-side `rope` field collapsing **21.15 → 3.39 ms** at batch 512 while
`rope_gpu` stays at 0.81 ms. `rope` is not doing CPU work; it is where backpressure lands when the
host runs a step ahead of a saturated device. Shorter device step → shorter stall. Treat host-side
`fwd`/`rope` at high batch as *stall accounting*, not cost.

### 2.3 Noise floor

Run-to-run, same branch, same instance:

| | baseline (r1 vs r2) | candidate (r1 vs r2) |
| --- | --- | --- |
| Throughput, worst batch | 1.14 % (batch 1) | 1.99 % (batch 32) |
| Throughput, median over batches | 0.45 % | 0.51 % |
| Throughput at batch 512 | 0.01 % | 0.04 % |
| `gpu_busy` at batch 512 | 51 515 vs 51 504 µs (0.02 %) | 27 305 vs 27 307 µs (0.01 %) |
| `sample_gpu` at batch 512 | −0.01 % | −0.00 % |
| Fields beyond 2σ, batch 512 | 1/20 (`logits_gpu`) | 1/20 (`ci`, +4.9 % of 0.019 ms) |

At batch 256 the baseline's two runs agree on **0/20** fields. The device-side instrument is
essentially exact; the end-to-end instrument is good to ~1 %, degrading at batch ≤ 32 where a single
scheduling hiccup is a measurable fraction of a 15-second run.

---

## 3. Results

### 3.1 Concurrency sweep — mean of two runs each

| batch | base tok/s | fused tok/s | Δ | base TPOT ms | fused TPOT ms | Δ | base prefill ms | fused prefill ms | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 102.4 | 102.0 | −0.4 % | 9.75 | 9.79 | +0.4 % | 10.1 | 10.2 | +0.7 % |
| 2 | 187.6 | 190.5 | +1.5 % | 10.64 | 10.48 | −1.5 % | 11.5 | 11.3 | −2.3 % |
| 4 | 370.2 | 377.9 | +2.1 % | 10.71 | 10.49 | −2.0 % | 18.5 | 18.1 | −2.3 % |
| 8 | 723.6 | 741.6 | +2.5 % | 10.83 | 10.56 | −2.5 % | 32.3 | 32.3 | −0.1 % |
| 16 | 1 384 | 1 432 | +3.4 % | 11.06 | 10.67 | −3.5 % | 60.6 | 60.2 | −0.6 % |
| 32 | 2 717 | 2 720 | +0.1 % | 11.12 | 11.11 | −0.1 % | 80.5 | 79.7 | −0.9 % |
| 64 | 5 057 | 5 102 | +0.9 % | 11.82 | 11.74 | −0.7 % | 96.4 | 93.1 | −3.3 % |
| 128 | 6 816 | **8 707** | **+27.7 %** | 17.70 | **13.67** | **−22.8 %** | 118.1 | 109.6 | −7.2 % |
| 256 | 7 166 | **10 704** | **+49.4 %** | 34.21 | **22.52** | **−34.2 %** | 154.0 | 133.0 | −13.6 % |
| 512 | 7 245 | **10 976** | **+51.5 %** | 68.11 | **44.44** | **−34.8 %** | 216.8 | 172.8 | **−20.3 %** |
| 1024 | **OOM** | 9 630 | — | — | 101.57 | — | — | 271.5 | — |

Secondary effects worth noting:

* **Scaling efficiency** (tok/s/req relative to batch 1) at batch 128: **0.52 → 0.67**; at batch 256
  **0.27 → 0.41**; at batch 512 **0.14 → 0.21**.
* **Last profitable doubling** (gain/cost > 1) moves **64 → 128**. At batch 128 gain/cost is
  **0.90 → 1.47**.
* **ITL p99** at batch 512: **120.7 → 97.4 ms**; at batch 128 **70.5 → 66.0 ms**.
* **MFU at batch 512**: 4.3 % → **6.6 %**. Still roofline-irrelevant — this model at this batch is
  memory- and sampler-bound, not FLOP-bound.

### 3.2 Iso-throughput

|  | `async_scheduling` | `fused_top_kp` |
| --- | --- | --- |
| Concurrency needed for ~7 200 tok/s | 512 | ~100 (between 64 and 128) |
| Throughput at batch 128 | 6 816 | **8 707 (+20 % over the baseline's *peak*)** |
| TPOT there | 68.11 ms (at batch 512) | **13.67 ms (5.0× lower)** |
| ITL p99 there | 120.8 ms | **66.0 ms** |
| Prefill latency there | 216.8 ms | **109.6 ms** |

The baseline needs 512-way concurrency and a 68 ms TPOT to reach its peak of 7 245 tok/s. The
candidate beats that peak by 20 % at batch 128, with **one fifth the per-token latency and half the
prefill latency**. Unlike log928's async-scheduling trade-off, this one has no latency cost to
apologise for — it is better on every axis at every batch ≥ 128.

### 3.3 GPU idle fraction and device busy — pure decode

| batch | base idle (r1/r2) | fused idle (r1/r2) | base busy ms/step | fused busy ms/step | Δ busy | base `wall_clean` | fused `wall_clean` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 66.8 / 66.4 % | 67.5 / 67.9 % | 3.25 | 3.15 | −3.2 % | 9.79 | 9.67 |
| 8 | 59.1 / 59.0 % | 63.2 / 61.8 % | 4.27 | 3.90 | −8.7 % | 10.46 | 10.61 |
| 32 | 50.8 / 49.6 % | 59.4 / 60.1 % | 5.46 | 4.36 | −20.2 % | 11.10 | 10.73 |
| 64 | 27.6 / 28.0 % | 50.8 / 49.7 % | 7.91 | 5.40 | −31.7 % | 10.92 | 10.96 |
| 128 | 6.6 / 6.7 % | 24.8 / 24.8 % | 14.24 | 8.35 | **−41.4 %** | 15.25 | **11.10** |
| 256 | 3.1 / 3.1 % | 5.6 / 5.6 % | 27.06 | 15.15 | **−44.0 %** | 27.93 | **16.04** |
| 512 | 1.4 / 1.4 % | 2.8 / 2.7 % | 51.52 | 27.31 | **−47.0 %** | 52.23 | **28.09** |

Read this table as the mirror image of log928's. There, device work was constant to within 1 % and the
branch removed host stalls. Here, host behaviour is unchanged and the branch removed **up to 47 % of
the device work**. Idle fraction therefore *rises* at every batch — at batch 64 from 27.6 % to 50.8 %,
at batch 128 from 6.6 % to 24.8 %. That is not a regression; it is the host overhead that was always
there becoming visible again once the device stopped being the long pole. Batch 128 has flipped from
device-bound to host-bound: 8.35 ms of device work inside an 11.10 ms step.

### 3.4 Host-side per-step breakdown, pure decode (ms)

From `baseline_against_fused_top_kp/mean_step_metrics.{128,512}.log`, 20 decode steps each.

| field | base @128 | fused @128 | base @512 | fused @512 | what it is |
| --- | --- | --- | --- | --- | --- |
| `step_0` | 14.327 | **10.613** | 49.212 | **26.306** | host wall of the step |
| `step_1` | 0.234 | 0.230 | 0.454 | 0.427 | in-flight sample drain |
| `sched` | 0.131 | 0.132 | 0.513 | 0.502 | scheduler total |
| `bld_meta` | 0.354 | 0.355 | 0.507 | 0.499 | attention-metadata build |
| `fwd` | 13.005 | **9.353** | 46.619 | **23.884** | host span over forward (**stall-dominated**) |
| `fwd_gpu` | 5.381 | 7.095 | 10.660 | 10.709 | forward event span (see §2.2) |
| `rope` | 2.481 | 2.088 | 21.150 | **3.394** | host span over rope (**pure backpressure**) |
| `rope_gpu` | 0.783 | 1.188 | 0.806 | 0.814 | rope event span |
| `logits_gpu` | 0.354 | 0.387 | 0.889 | 0.886 | lm_head event span |
| `sample` | 0.649 | **0.586** | 1.083 | **0.922** | host span over sampler |
| `sample_gpu` | **9.421** | **3.527** | **40.558** | **16.353** | sampler event span |
| `dth` / `dth_wait` | 0.034 / 0.005 | 0.034 / 0.005 | 0.036 / 0.005 | 0.035 / 0.005 | device→host copy of tokens |

The single row that explains the report is `sample_gpu`: **40.56 → 16.35 ms at batch 512 (−59.7 %)**
and **9.42 → 3.53 ms at batch 128 (−62.6 %)**, with z-scores of −5 461 and −2 660. Everything else in
the forward is untouched (`fwd_gpu` 10.660 → 10.709, +0.46 %), and every host-side field is either
unchanged or a stall-accounting artefact (§2.2).

Per-row sampler device cost, which is the cleanest way to see the asymptotics:

| batch | base µs/row | fused µs/row | ratio |
| --- | --- | --- | --- |
| 1 | 423.5 | 310.3 | 1.36× |
| 8 | 94.1 | 48.0 | 1.96× |
| 32 | 55.0 | 20.3 | 2.71× |
| 64 | 60.9 | 21.5 | 2.83× |
| 128 | 73.6 | 27.6 | 2.67× |
| 256 | 79.0 | 32.5 | 2.43× |
| 512 | 79.2 | 31.9 | **2.48×** |

The baseline's per-row cost *grows* from 55 µs at batch 32 to 79 µs at batch 512 — that is the
`O(vocab log vocab)` segmented radix sort getting less efficient per row as the batch widens. The
candidate's flattens at ~32 µs/row. Both are still far above what 20 candidates should cost (§6).

### 3.5 Device-side op breakdown, batch 512 (self CUDA, ms/step, from `key_averages` / 5 traced steps)

| op | baseline | fused | Δ |
| --- | --- | --- | --- |
| `decode_step` (CUDA total) | **52.109** | **27.448** | **−24.662** |
| `aten::sort` | **15.430** | **—** | **−15.430** |
| `aten::copy_` | 2.665 | 1.057 | −1.608 |
| `aten::masked_fill_` | 2.179 | 0.672 | −1.507 |
| `Memcpy DtoD` | 1.920 | 0.627 | −1.293 |
| `aten::_softmax` | 2.450 | 1.249 | −1.201 |
| `aten::sub` | 3.078 | 2.082 | −0.996 |
| `aten::scatter_` | 0.924 | 0.046 | −0.877 |
| `aten::cumsum` | 0.827 | 0.006 | −0.821 |
| `aten::where` | 2.240 | 1.531 | −0.709 |
| `Context Sync` | 9.246 | 4.384 | −4.862 |
| `aten::topk` | 1.868 | 1.867 | −0.001 |
| `aten::multinomial` (CUDA total) | 2.738 | 2.738 | −0.001 |
| `aten::div` | 2.504 | 2.504 | ±0.000 |
| `aten::mm` | 3.614 | 3.608 | −0.006 |
| `flash_attn::_flash_attn_varlen_forward` | 4.886 | 4.888 | +0.002 |

The nine op rows above `Context Sync` sum to **−24.4 ms**, which is the whole `decode_step` delta.
`aten::sort` alone is **63 %** of it, and the two `cub::DeviceSegmentedRadixSortKernel` rows backing it
(49.6 + 24.3 ms over 5 steps = 14.8 ms/step) vanish with it. `Context Sync` halves because it is the
host waiting on the stream — it tracks step length, it is not work.

The same shape holds at smaller batch: `sort` is 7.63 → 0 ms/step at batch 256, 3.75 → 0 at batch 128.

---

## 4. What the commit actually changed

Before (`sample`, two passes):

```python
logits = apply_top_k(logits, top_k, max_k)   # topk(max_k) -> kth threshold -> full-vocab mask
logits = apply_top_p(logits, top_p)          # torch.sort over ALL 151 936 columns, cumsum, scatter back
```

After (one pass, inside the top-k window):

```python
top_vals, indices = torch.topk(logits, max_k, dim=-1, sorted=True)   # [bsz, max_k], descending
top_vals = top_vals.masked_fill(arange(max_k) >= top_k[:, None], -inf) # per-row k, inside the slab
probs    = top_vals.softmax(-1)                                      # [bsz, max_k] — renormalised per row
remove   = (probs.cumsum(-1) - probs) > top_p[:, None]               # top-p, [bsz, max_k]
out      = torch.full_like(logits, -inf).scatter_(1, indices, top_vals.masked_fill(remove, -inf))
```

Three things make this correct rather than merely faster, and the new tests pin each one:

1. **`torch.topk` already returns descending values** — the baseline's `sort` was re-deriving an order
   it had been handed one line earlier. That is the 15.4 ms.
2. **The per-row k mask must be applied *before* the softmax** (`test_fused_renormalises_probs_within_each_row_k`).
   Probabilities have to be renormalised over that row's own k, not over the batch-wide `max_k` slab,
   or fused top-p comes out looser than sequential top-k→top-p.
3. **`max_k` is a candidate budget, not just `topk`'s width** — everything outside the top-`max_k`
   columns is hard `-inf`'d, so a narrow row cannot truncate a wide one
   (`test_fused_narrow_row_does_not_cap_wide_row`).

Supporting change: `SamplingParamTable.set_slot` now clamps the effective k into `[1, top_k_cutoff]`
with `top_k_cutoff = min(MAX_EFFECTIVE_TOP_K, vocab_size)` = **1 024**, so "top-k disabled" is
represented as `k = 1024` rather than `0`. `apply_penalties` also lost its
`repetition_penalty[:, None].repeat(1, vocab)` materialisation in favour of `torch.where` — worth
~0.7 ms/step of the `where`/DtoD deltas in §3.5, and exactly what log928 §7.2 asked for, partially.

`apply_top_k` and `apply_top_p` still exist but are **dead in the hot path** — only tests call them now.

---

## 5. Regressions and risks

### 5.1 No throughput regression anywhere, but batch ≤ 16 is within noise

Batch 1 is −0.4 % throughput / +0.4 % TPOT, which is inside the 1.14 % run-to-run floor. Batch 2–16
show +1.5 % … +3.4 %, also near the floor but consistently positive across both run pairs and
consistent with the measured device saving (−3 % to −9 % busy there). Nothing to fix.

### 5.2 "Top-k disabled" now means top-1024 — a real distribution change

With `vocab_size = 151 936 > MAX_EFFECTIVE_TOP_K = 1 024`, a request asking for `top_k = 0`,
`top_k < 0`, or `top_k ≥ 1024` is now clamped to **k = 1 024**, and the fused path hard-`-inf`s
everything outside that slab. The old path read the same request as *disabled* and let the whole
vocabulary through to top-p. `test_fused_truncates_at_cutoff_where_sequential_keeps_the_tail` documents
the divergence deliberately, and its own comment quantifies the cost: negligible on sharp logits,
**~5 % of the probability mass on plain N(0,1)**.

This does not affect any number in this report (`top_k = 20` throughout), but it is a semantic change
and it is not announced anywhere outside that test. If the engine ever needs honest unrestricted
sampling, it needs a separate full-vocab path (or `MAX_EFFECTIVE_TOP_K` raised with eyes open, since
the cutoff is also what keeps `topk` cheap).

### 5.3 The `config.py` default change is inert here — but not everywhere

The diff flips `ModelConfig.top_k` from `0` to `20`. Under `from_pretrained` this is unobservable:
`generation_config.json` supplies `top_k = 20` and `raw.update(raw2)` overwrites the dataclass default
(verified in both branches' logs). It *is* observable for any code path that constructs `ModelConfig()`
directly — synthetic configs, unit tests, embedded uses — which silently switches from "no top-k" to
top-20. Harmless in intent, invisible in effect, worth a line in the commit message it did not get.

### 5.4 Batch 1024 is runnable but past the peak

| batch | fused tok/s | TPOT ms | gain/cost |
| --- | --- | --- | --- |
| 512 | **10 976** | 44.44 | 0.52 |
| 1024 | 9 630 | 101.57 | 0.38 |

Doubling to 1024 *loses* 12 % throughput and more than doubles TPOT. The OOM fix is still worth having
— it removes a hard failure mode and the 1.73 GiB allocation spike that was pressuring the allocator —
but **1024 is not a useful operating point**, and the sweep tool's "saturated aggregate throughput ~
9 632 tok/s" line is misleading because it simply reports the last row. The real peak is
**10 976 tok/s at batch 512**.

### 5.5 Prefill latency still grows with concurrency

−20 % at batch 512 is a genuine improvement, but 172.8 ms is still 1.4× log928's `main` (121.6 ms) and
271.5 ms at batch 1024. The async-scheduling lookahead cost identified in log928 §5.2 is reduced, not
removed; it is bounded by one device step, and the device step is now 47 % shorter.

### 5.6 The scatter-back still allocates a full-vocab tensor every step

`torch.full_like(logits, -inf)` plus the fp32 `logits.float().softmax()` means two `[bsz, vocab]`
tensors per step — 155 MB (bf16) + 311 MB (fp32) at batch 512, and double that at 1024. Much better
than the baseline's 1.73 GiB sort workspace, but §7.1 removes both.

---

## 6. What is left: sampling is still 60 % of device work

At batch 512 the engine now spends **16.35 of 27.3 ms/step in the sampler** and 11.6 ms on the model.
The sampler's problem is unchanged in kind, only in degree: it is **memory traffic over
`[bsz, 151 936]` tensors**, and almost all of it is still there. One fp32 full-vocab pass at batch 512
is 311 MB read + 311 MB written ≈ **0.7 ms** at this card's achievable bandwidth, so 16.35 ms is
**~23 full-vocab passes** for a problem that is 20 columns wide.

Where they are, reading `sample()` and `apply_penalties()` against the batch-512 trace:

| source line | op rows (ms/step) | width |
| --- | --- | --- |
| `apply_penalties`: `prompt_mask \| output_mask` | `bitwise_or` 0.281 | full vocab, 2 bool masks |
| `apply_penalties`: `where(mask, rep[:,None], 1.0)` | part of `where` 1.531 | full vocab |
| `apply_penalties`: `where(logits>0, logits/rep, logits*rep)` | `gt` 0.257 + part of `div`/`mul`/`where` | full vocab, 4 passes |
| `sample`: `logits / t[:, None]` | part of `div` 2.504 | full vocab |
| `apply_fused_top_k_and_p`: `topk` | `topk` 1.867 | reads full vocab (**unavoidable**) |
| `apply_fused_top_k_and_p`: window softmax/cumsum/mask | `cumsum` 0.006, part of `_softmax` | `[bsz, 20]` (**already cheap**) |
| `apply_fused_top_k_and_p`: `full_like(-inf)` + `scatter_` | `fill_` 0.739 + `scatter_` 0.046 + `Memcpy DtoD` 0.627 | full vocab |
| `sample`: `logits.float().softmax(-1)` | `copy_` ~0.5 + `_softmax` ~1.2 | full vocab, 2 passes |
| `sample`: `probs.sum(-1)` dead-row guard | `sum` 0.695 | full vocab |
| `sample`: `probs.masked_fill(dead, 1.0)` | `masked_fill_` 0.672 | full vocab |
| `sample`: `multinomial(probs, 1)` | 2.738 CUDA total (`div`, `exponential_` 0.307, `max`/`min`/`searchsorted`) | full vocab |
| `sample`: `logits.argmax(-1)` greedy path | `argmax` 0.714 | full vocab |

Only **one** of these genuinely needs the full vocabulary: the `topk` that reads the logits. Everything
after it is operating on a tensor whose 151 916 non-candidate columns are all `-inf` or all zero.

---

## 7. Recommendations, in order

### 7.1 Sample *inside* the top-k window — removes ~9 ms/step at batch 512

log928 §7.1 proposed this in two halves; this commit shipped the first half (fuse top-p into the
window) and left the second (keep the tail of the sampler in the window too). Finish it:

```python
top_vals, indices = torch.topk(logits, max_k, dim=-1, sorted=True)  # the only full-vocab read
top_vals = top_vals / t[:, None]                                    # temperature: rank-invariant, safe after topk
top_vals = top_vals.masked_fill(beyond_k | beyond_p, -inf)          # as today, [bsz, max_k]
probs    = top_vals.float().softmax(-1)                             # [bsz, 20]
probs    = probs.masked_fill(~isfinite(probs.sum(-1, keepdim=True)) | (sum <= 0), 1.0)
local    = torch.multinomial(probs, 1)                              # [bsz, 1] index into the window
sampled  = indices.gather(1, local).squeeze(1)                      # map back to vocab ids
greedy_tok = indices[:, 0]                                          # topk is sorted -> argmax is free
return torch.where(greedy, greedy_tok, sampled)
```

This deletes the `full_like` + `scatter_` + DtoD (1.41), the full-vocab fp32 cast and softmax (~1.7),
the dead-row `sum`/`masked_fill` (1.37), `multinomial`'s full-vocab work (2.74), `argmax` (0.71) and the
temperature `div` (~0.8) — **~8.7 ms/step directly**, plus the `copy_`/`mul`/`sub` traffic that rides
along with them. Two notes:

* **Temperature after `topk` is safe**: dividing by a positive scalar is rank-preserving, so the top-k
  set is identical. Greedy rows (`t ≤ EPS`) take `indices[:, 0]` and never see the division.
* **`argmax` becomes free**: `topk(..., sorted=True)` already put the max in column 0.

### 7.2 Make penalties sparse — removes ~3.5 ms/step at batch 512

Penalties **must** run before `topk` (they change the ranking), so they cannot move into the window.
But they are a *sparse* update: only tokens the row has actually seen are affected, which is at most
`prompt_len + output_len ≤ 1 024` of 151 936 columns — 0.7 %. Instead of six full-vocab passes
(`bitwise_or`, `where`, `gt`, `div`, `mul`, `where`), gather the logits at the seen positions, apply the
penalty there, and scatter back: `O(bsz × n_seen)` ≈ 0.3 M elements at batch 512 instead of 78 M.
`SamplingTensors` already carries `prompt_mask`/`output_mask`; what it needs is the index list those
masks were built from, which the scheduler already has.

Also keep the existing cheap win: when `rep == 1.0 and freq_pen == 0 and pres_pen == 0` the whole pass
is an identity, and that is detectable host-side from the resident tables with no sync.

Together 7.1 + 7.2 should bring the sampler to **~4 ms/step at batch 512** (the `topk` read's 1.87 ms
plus window arithmetic), i.e. a device step of **~15 ms** against today's 27.3 — another ~1.8× on the
device side. Throughput will not follow 2× one-for-one, because at that point the host becomes the limiter
again (§7.3).

### 7.3 Then the host, because it is next

The candidate is already host-bound at batch ≤ 128 (24.8 % idle at 128, 50.8 % at 64) and 7.1+7.2 will
push that boundary to ~256. Un-profiled wall at batch 64 is 10.96 ms against 5.40 ms of device work:
**5.5 ms/step of host time that nothing hides**. The step-metrics fields that are real host work sum to
far less than that (`sched` 0.076 + `bld_meta` 0.339 + `sample` 0.598 + `dth` 0.034 ≈ 1.0 ms), so
~4.5 ms/step is unattributed — and the trace says what it is: **4 414 `aten::` calls per step at batch
64**, totalling **8.18 ms/step of self CPU** under the profiler, which at the measured +69 % profiler
overhead is ~4.8 ms un-profiled. That is almost exactly the gap. It is Python/dispatch cost, not
algorithmic work, and the call count barely grows with batch (6 158 at batch 512), so it is a fixed per-step
tax: CUDA graphs or a leaner eager path, not micro-optimisation of individual ops.

### 7.4 Re-run the sweep with `max_num_seqs = 128` as the headline config

The knee moved to 128 and that is the engine's default. The sweep's `gain/cost` marker and the
"saturated throughput" line should be read at 512 now (§5.4), and the 1024 row kept only as an OOM
regression guard.

### 7.5 Document the cutoff semantics

§5.2. One sentence in `constants.py` next to `MAX_EFFECTIVE_TOP_K` and one in the sampler docstring:
"top-k disabled" is implemented as top-1024, which is a truncation, not a no-op, whenever
`vocab_size > 1024`.

### 7.6 Delete or mark the dead paths

`apply_top_k` and `apply_top_p` are now test-only. Leaving them as reference implementations for the
differential tests is defensible, but say so in a comment, and remove the unused `indices` binding in
`apply_top_k`.

---

## 8. Caveats

1. **Cross-log comparisons are still unsafe.** log1001 is a different vast instance than log928 and
   log924 (PCIe gen 4 × 16 here, different CPU). What is trustworthy is the **within-log1001
   baseline-to-candidate delta**, measured back-to-back on one machine inside 45 minutes. The fact that
   this baseline reproduced log928's `async_scheduling` to 0.2 % is reassuring, not licensing.
2. **The candidate ran first**, on the colder machine. If there is a warm-up bias, it favours the
   baseline, i.e. the measured gains are a floor.
3. **The profile is pure decode.** It deliberately excludes prefill, chunking and admission, which is
   why §3.3's device savings (−47 %) exceed §3.1's TPOT improvement (−35 %) at batch 512: the
   benchmark's steps are a mix of decode and prefill chunks.
4. **`*_gpu` fields are event spans, not kernel sums** (§2.2). Any reading of
   `mean_step_metrics.{32,64,128}.log` that treats `fwd_gpu`/`rope_gpu` increases as regressions is
   wrong; cross-check against `gpu_busy_from_trace` and `key_averages`.
5. **Profiler overhead is +69 % on wall**, so idle fractions come from the separate un-profiled
   20-step measurement, and the `key_averages` ms/step figures are trace-time (5 steps) which inflate
   host-side columns but not device-side ones.
6. **Two runs per configuration** is enough to bound noise at ~1 % but not to characterise tails. The
   ITL p99 figures in §3.1 come from single runs of ~10 000 inter-token intervals each and should be
   treated as indicative.
7. **Sampling quality was not measured here.** The commit's 227 lines of new tests establish semantic
   equivalence for `top_k < cutoff` and document the deliberate divergence above it (§5.2); no
   end-to-end generation-quality comparison was run, and none is needed for `top_k = 20`.
