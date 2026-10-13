# Performance analysis — one engine, two hosts, the same GPU (log1009)

> 中文版：[`performance_analysis_log1009.zh.md`](./performance_analysis_log1009.zh.md)

**This round measures the measurement.** [log1003 §2.4](./performance_analysis_log1003.md) claimed that
qwen.py's device work is portable while its throughput is set by the host CPU, and inferred that from
two instances rented a day apart. log1009 tests it directly: the same commit, the same workload, two
vast instances with **the same GPU** and very different hosts, run back to back.

The short version: **the two GPUs are not different. One of them is attached to a 2016 Xeon.**

**Data sources**

| Content | Path |
| --- | --- |
| Concurrency sweep + pure-decode profile, host 1 | `log_vast/log1009/host1/{benchmark_baseline,profile_baseline,profile_baseline2}/` |
| Concurrency sweep ×2 + profile ×2, host 2 | `log_vast/log1009/host2/{benchmark_baseline,benchmark_baseline2,profile_baseline,profile_baseline2}/` |
| vLLM 0.30.0 arm A, host 2 (batch 1–2 only) | `log_vast/log1009/host2/benchmark_vllm/` |
| Run-to-run step-metric diffs | `log_vast/log1009/host{1,2}/baseline_against_baseline2/` |
| Host inventory | `log_vast/log1009/host{1,2}/{host_info,gpu.static.csv}` |

**Code under test** — one commit, both hosts. Verified from the dumps: identical sampling config
(`top_k=20, top_p=0.8`), identical switches (`compile_rope=false`, `pre_gather_cos_sin=true`), and the
same step-metrics schema including the `smp_prep`/`smp_pen`/`smp_pick`/`smp_post` split and `upd_proj`
added after log1003.

---

## Executive summary

1. **The device work is identical.** At batch 512, pure decode: `gpu_busy_from_trace` **27 053 vs
   27 032 µs/step (−0.1 %)**, and the sampler's event spans agree to three decimal places —
   `smp_gpu` 16.348 vs 16.349, `smp_pen_gpu` 6.848 vs 6.846, `smp_pick_gpu` 8.858 vs 8.864. Across all
   seven batch sizes the device agrees within **±3 %**.
2. **The throughput differs by up to 2.3×.** Batch 1 **66.2 vs 29.6 tok/s**, batch 64 **3 645 vs
   1 647**, converging to **1.01×** at batch 1024. Peak **11 117 @ batch 512** against **9 580 @ batch
   1024** — host 2 needs twice the concurrency and still does not catch up.
3. **The cause is the host's single-thread speed.** Ryzen 9 5900X (Zen 3, 2020) against Xeon E5-2680 v4
   (Broadwell-EP, 2016). At batch 512 **thirteen mutually unrelated host-side fields all land between
   2.03× and 2.35×** — a uniform penalty, which is a slower interpreter rather than a bottleneck
   anywhere specific. Note what is *not* established: the interpreter benchmark from
   [§1b of the runbook](../env/vastai/vast-evn-build.md) was not run on either host, so this report
   attributes the gap to the CPU but cannot decompose it into clock versus IPC (§4.1).
4. **It is not PCIe, despite appearances.** Host 2's link negotiated **gen 1** (`pcie.link.gen.current
   = 1`, max 3) against host 1's gen 4 × 16. The decode step does not care: per-step `Memcpy HtoD` is
   **4.6 vs 4.7 µs**, DtoH 1.2 vs 0.9 µs, DtoD 0.621 vs 0.621 ms. The link matters only for the
   one-time 0.99 GB weight load.
5. **It is not power or thermals either.** Host 2's enforced limit is **500 W**, *higher* than host 1's
   450 W. Host 1 spent 5.3 % of its busy samples at `sw_power_cap`, median 389 W / 2 715 MHz, 73 °C max.
6. **One field is hit harder than the uniform 2.1×: `bld_meta`, at 3.0–3.5×.** It is the most
   purely-Python phase in the step — five list comprehensions over the batch plus a `chain.from_iterable`
   flatten — so it has the least C++ tensor work to dilute the interpreter penalty. That makes it the
   next `.item()`-shaped target (§5).
7. **The new instrumentation pays off immediately: the sampler's 16.35 ms/step at batch 512 splits into
   `smp_pen` 6.85 (42 %) and `smp_pick` 8.86 (54 %)**, with `smp_prep` 0.59 and `smp_post` 0.04. That is
   the first measured split of what [log1001 §7.1–7.2](./performance_analysis_log1001.md) proposed to fix,
   and it prices the two halves separately for the first time (§6).
8. **log1003's "ratios transfer, absolutes do not" claim holds on a third host.** qwen.py against vLLM
   arm A at batch 1: **1.64×** on log1003's EPYC, **1.53×** here on the Xeon, while the absolute
   throughput moved 43 → 29.6 tok/s.
9. **The `.tolist()` fix landed exactly as predicted.** At batch 512, `aten::item` **515 → 3**,
   `_local_scalar_dense` 515 → 3, `select` 517 → 5, `as_strided` 1 041 → 529; total aten calls per step
   **6 158 → 4 112 (−33 %)**, and now flat with batch.

---

## 1. The two hosts

| | **host 1** (54975648) | **host 2** (55040273) |
| --- | --- | --- |
| GPU | RTX 4090, driver 595.84 | RTX 4090, driver 580.173.02 |
| Enforced power limit | 450 W (max 600) | **500 W** (max 600) |
| Panel bandwidth / TFLOPS / DLPerf | 879.2 GB/s / 81.8 / 97.2 | 879.1 GB/s / 82.6 / 97.5 |
| **PCIe** | **gen 4 × 16** (current = max) | **gen 1 × 16 current**, gen 3 max |
| **CPU** | **Ryzen 9 5900X** — Zen 3, 2020, 3.7 / 4.8 GHz | **Xeon E5-2680 v4** — Broadwell-EP, 2016, 2.4 / 3.3 GHz |
| CPU allocated | 24 / 24 | 14 / 28 |
| RAM | 62.2 GB, B550, DDR4 dual-channel | 32.1 GB, X99, DDR4 quad-channel but higher latency |
| Max CUDA | 13.2 | 13.0 |

By every GPU-facing number these are the same machine: same card, same memory bandwidth, same DLPerf
to within 0.3 %. Everything that differs is on the host side of the PCIe bus.

---

## 2. Methodology

### 2.1 What is held constant

One commit on both hosts, verified from the dumps rather than assumed: `top_k=20, top_p=0.8`,
`compile_rope=false`, `pre_gather_cos_sin=true`, `max_model_len=1024`,
`max_num_batched_tokens=8192`, `num_blocks=4096 × block_size=256`. Workload is
`test_benchmark_sweep_batch_size`: 10 × batch requests (minimum 64), fixed 512-in / 128-out,
`ignore_eos`. **Zero preemptions and zero errors across all 33 sweep points on both hosts.**

### 2.2 How to read the `*_gpu` fields

As established in [log1001 §2.2](./performance_analysis_log1001.md), `fwd_gpu` / `rope_gpu` / `smp_*_gpu`
are CUDA-event spans on the stream, not kernel sums: when the host is the bottleneck they contain the
bubble. Host 2 is 31 % idle at batch 512 and 86–91 % idle below batch 64, so **its event spans are
inflated wherever it is starved** — `rope_gpu` reads 4.18× and `fwd_gpu` 2.05× at batch 512 for exactly
this reason, while the authoritative `gpu_busy_from_trace` differs by 0.1 %.

The spans that are tight around real kernels tell the truth, and they are the sampler's: at batch 512
host 1 is 3.1 % idle and the sampler is 60 % of the step, so `smp_gpu` and its children are
bubble-free on both sides. Those are the numbers quoted in §3.1.

### 2.3 Noise floor

| | host 1 | host 2 |
| --- | --- | --- |
| Sweep repeats | **1** (no `benchmark_baseline2`) | 2 |
| Throughput run-to-run | — | worst **1.59 %**, median 0.71 % |
| Profile repeats | 2 | 2 |
| Fields beyond 2σ at batch 512 | 9/31 | 8/27 |
| Preemptions / errors | 0 / 0 | 0 / 0 |

**Host 1's throughput numbers have no run-to-run floor** — only one sweep was taken. Given host 2's
1.59 % worst case on identical code, the 2.0–2.3× ratios in §3.2 are ~100× any plausible noise, so the
conclusion is safe; but per [§6e of the runbook](../env/vastai/vast-evn-build.md) it is a gap, and the
cheapest fix is one more sweep next time.

---

## 3. Results

### 3.1 The device side is identical

Pure decode, `gpu_busy_from_trace`, µs/step:

| batch | host 1 | host 2 | Δ |
| --- | --- | --- | --- |
| 1 | 3 237.4 | 3 168.2 | −2.1 % |
| 8 | 4 020.6 | 3 937.0 | −2.1 % |
| 32 | 4 261.6 | 4 392.0 | +3.1 % |
| 64 | 5 309.6 | 5 415.4 | +2.0 % |
| 128 | 8 228.2 | 8 367.2 | +1.7 % |
| 256 | 14 972.6 | 14 918.9 | −0.4 % |
| 512 | **27 053.4** | **27 032.0** | **−0.1 %** |

And at batch 512, where neither host's spans contain much bubble on the sampler:

| device span, ms/step | host 1 | host 2 |
| --- | --- | --- |
| `smp_gpu` | 16.348 | **16.349** |
| `smp_prep_gpu` | 0.592 | 0.589 |
| `smp_pen_gpu` | 6.848 | **6.846** |
| `smp_pick_gpu` | 8.858 | **8.864** |
| `smp_post_gpu` | 0.041 | **0.041** |
| `logits_gpu` | 0.865 | 0.956 |

Agreement to three decimal places on a 16 ms quantity. There is no GPU difference to explain.

### 3.2 The throughput is not

| batch | host 1 tok/s | host 2 tok/s | h1 / h2 | h1 TPOT | h2 TPOT |
| --- | --- | --- | --- | --- | --- |
| 1 | 66.2 | 29.6 | **2.24×** | 14.99 ms | 33.81 ms |
| 2 | 128.7 | 55.6 | 2.31× | 15.40 | 35.95 |
| 8 | 506.0 | 219.9 | 2.30× | 15.57 | 36.36 |
| 32 | 1 808.1 | 842.5 | 2.15× | 17.06 | 37.73 |
| 64 | 3 644.5 | 1 647.4 | 2.21× | 16.79 | 38.41 |
| 128 | 6 385.4 | 3 129.4 | 2.04× | 19.06 | 40.24 |
| 256 | 10 557.3 | 5 764.6 | 1.83× | 22.85 | 43.37 |
| 512 | **11 117.2** | 9 195.3 | 1.21× | 43.84 | 53.75 |
| 1024 | 9 722.5 | **9 580.1** | **1.01×** | 100.59 | 102.48 |

| | host 1 | host 2 |
| --- | --- | --- |
| Peak throughput | **11 117 @ batch 512** | 9 580 @ batch 1024 |
| Last profitable doubling | batch 256 | batch 512 |
| Scaling efficiency @ 256 | 0.62 | 0.76 † |
| MBU @ batch 1 | 6.5 % | 2.9 % |

† Host 2's *higher* scaling efficiency is an artefact, not a virtue: efficiency is measured against its
own batch-1 throughput, and that baseline is already crippled. A host-bound engine scales beautifully
right up to the point where the device finally becomes the limit.

Monotone convergence from 2.3× to 1.01× is the textbook host-bound signature. The corroborating view is
idle fraction and un-profiled wall:

| batch | h1 idle | h2 idle | h1 `wall_clean` | h2 `wall_clean` |
| --- | --- | --- | --- | --- |
| 1 | 80.5 % | 90.6 % | 16.58 ms | 33.61 ms |
| 64 | 66.6 % | 85.6 % | 15.90 | 37.50 |
| 128 | 49.5 % | 78.3 % | 16.29 | 38.48 |
| 256 | 23.8 % | 61.2 % | 19.65 | 38.41 |
| 512 | **3.1 %** | **31.2 %** | 27.92 | 39.28 |

**Host 2's wall is pinned at 33–39 ms from batch 1 to 512** — a fixed per-step tax that the batch size
barely moves. Host 1's sits at 15–20 ms until the device takes over at batch 512.

### 3.3 The host side, field by field

Batch 512, ms/step, host 1 → host 2:

| field | h1 | h2 | ratio |
| --- | --- | --- | --- |
| `bld_meta` | 0.615 | 1.925 | **3.13×** |
| `smp_prep` | 0.250 | 0.588 | 2.35× |
| `smp_pen` | 0.113 | 0.254 | 2.26× |
| `smp_post` | 0.163 | 0.360 | 2.21× |
| `smp` | 0.971 | 2.139 | 2.20× |
| `logits` | 0.089 | 0.191 | 2.16× |
| `smp_run` | 0.537 | 1.149 | 2.14× |
| `smp_pick` | 0.400 | 0.843 | 2.11× |
| `upd_proj` | 0.097 | 0.204 | 2.10× |
| `dth` | 0.042 | 0.088 | 2.07× |
| `ci` | 0.021 | 0.044 | 2.05× |
| `dth_wait` | 0.006 | 0.012 | 2.03× |
| `step_1` | 0.527 | 0.983 | 1.87× |
| `sched` | 0.599 | 1.038 | 1.73× |

Thirteen mutually unrelated fields between 1.73× and 2.35×, clustered on ~2.1. **A uniform penalty
across everything host-side is a slower interpreter, not a bottleneck anywhere in particular.** One
field stands outside the cluster; §5.

---

## 4. Why — and what it is not

### 4.1 CPU single-thread — the attribution is solid, the decomposition is not measured

**What the data supports.** The penalty is **uniform across thirteen mutually unrelated host-side
fields** (§3.3) while the device is identical to ±0.1 %. That combination rules out every alternative
checked — PCIe (§4.2), power and thermals (§4.3), kernel or config differences (§2.1) — and leaves the
host's ability to execute the Python decode loop. CPython running that loop is close to a pure
single-thread integer/pointer-chasing workload: [log1001 §7.3](./performance_analysis_log1001.md)
measured ~4 400 `aten::` calls per step, 71–80 % of which launch no kernel at all.

**What the data does not support, and an earlier draft of this section claimed anyway.** That draft
decomposed the 2.1× as "clock 1.45× × IPC ~1.5×". The IPC figure was not measured and not cited — it
was reverse-engineered from the answer (2.2 ÷ 1.45), then presented as if the product independently
predicted the measurement. It is circular and it is removed. What is actually known about the two CPUs
here:

| | Ryzen 9 5900X | Xeon E5-2680 v4 | status |
| --- | --- | --- | --- |
| Microarchitecture | Zen 3 (2020) | Broadwell-EP (2016) | vendor fact |
| Nominal boost clock | 4.8 GHz | 3.3 GHz | vendor fact — **1.45×** |
| **Actual operating clock during the runs** | — | — | **not recorded** |
| **IPC on this workload** | — | — | **not measured** |
| **Interpreter loop time** | — | — | **not measured — §1b's test was not run** |
| Measured host-field penalty | — | — | **2.03–2.35×** |

Two reasons the nominal clock ratio cannot carry the argument on its own: neither host's sustained clock
was sampled (host 2's Xeon runs in a 14-of-28-core allocation on a shared server and may never reach
3.3 GHz), and a four-year, two-generation microarchitecture gap plausibly contributes more than the
clocks do. The honest statement is that **a ~2.1× single-thread gap between these two parts is
unsurprising, not that it was predicted.**

**The fix is one command, and it is already in the runbook.** The 3 M-iteration pure-Python loop added
to [§1b](../env/vastai/vast-evn-build.md) after log1003 measures exactly this quantity in one number
per host — log1003's reference is 276 ms on an EPYC 7402 held at ~2.24 GHz. **It was not run on either
host here**, nor were `uptime` or `scaling MHz` (§8.7). Had it been, this section would have a
measurement instead of a plausibility argument. That is the whole reason the gate exists.

Memory latency is a further untested contributor: host 1 is DDR4-3200 dual-channel on a desktop B550,
host 2 an X99 server platform whose quad-channel bandwidth does nothing for a latency-bound
interpreter. Also unmeasured.

### 4.2 Not PCIe — the obvious suspect, measured and dismissed

Host 2's link is at **gen 1** while the board supports gen 3, which is an eightfold bandwidth
difference against host 1's gen 4. It is also, for this workload, irrelevant:

| per decode step, batch 512 | host 1 | host 2 |
| --- | --- | --- |
| `Memcpy HtoD` | 0.0046 ms | 0.0047 ms |
| `Memcpy DtoH` | 0.0012 ms | 0.0009 ms |
| `Memcpy DtoD` | 0.6210 ms | 0.6212 ms |

**4.7 µs of host-to-device traffic per step.** The resident-buffer design from M8 is why: the staging
tensors are a few KB (slot index, want, needs-sample, cache slots, block table) and the token read-back
is `B` int32s. The link shows up once, in the 0.99 GB weight load at startup — ~0.3 s at gen 1 against
~0.04 s at gen 4, invisible in steady state.

This is worth recording because log928 flagged a gen-1 link as a caveat on *that* round's numbers. On a
decode workload with resident buffers, it is not one.

### 4.3 Not power, not thermals

Host 2's enforced limit is **500 W against host 1's 450 W** — more headroom, not less. Host 1's trace
shows `sw_power_cap` Active for 5.3 % of busy samples, median 389 W and 2 715 MHz, 73 °C maximum, no
thermal slowdown. Host 2 has no GPU trace during its own sweep (§8), but the ±0.1 % device-time
agreement is the stronger argument: a differently-throttled GPU could not produce identical kernel
times.

---

## 5. `bld_meta` — the one field worse than the interpreter penalty

| | batch 1 | batch 64 | batch 512 |
| --- | --- | --- | --- |
| `bld_meta` h2/h1 | 3.00× | **3.52×** | 3.13× |
| every other host field | ~2.0× | ~2.2× | ~2.1× |

`build_attn_metadata` → `RequestBuffer.set()` is the most purely-Python phase of the step:

```python
self.slot_idx_np[:bsz]     = [r.slot for r in reqs]
self.want_np[:bsz]         = [s.want for s in s_infos]
self.needs_sample_np[:bsz] = [r.projected_is_decoding or
                              (r.num_scheduled_tokens + s.want == r._num_projected_prefill_tokens)
                              for r, s in zip(reqs, s_infos)]
cache_slot_flat            = list(chain.from_iterable(s.cache_slots for s in s_infos))
```

Five comprehensions over the batch — one of them with a compound predicate per element — plus a flatten
that is O(total tokens) rather than O(batch), then numpy assignment from Python lists. Every other
host-side field has some C++ tensor call to dilute the interpreter penalty; this one barely does, so it
absorbs the 2.1× and then multiplies it.

**This is the same shape of finding as the `.item()` loop** that §9 of this report confirms fixed, and it
is now the largest remaining batch-scaling host cost. On host 1 at batch 512 it is 0.615 ms/step inside a
27.9 ms step, so the ceiling is modest — but on a host like host 2, or at batch 128–256 where the engine
is host-bound on *both* machines, it is the next thing worth batching.

---

## 6. What the new sampler split says

The `smp_prep` / `smp_pen` / `smp_pick` / `smp_post` events added after log1003 give the first measured
decomposition of the sampler. Host 1, batch 512, pure decode, where the GPU is 3.1 % idle so the spans
are kernel time:

| | ms/step | share of sampler | share of the step's device time |
| --- | --- | --- | --- |
| `smp_prep` — `build_sampling_tensors` | 0.592 | 3.6 % | 2.2 % |
| **`smp_pen` — `apply_penalties`** | **6.848** | **41.9 %** | **25.3 %** |
| **`smp_pick` — `sample()`** | **8.858** | **54.2 %** | **32.7 %** |
| `smp_post` — `add_sampled_tokens_on_device` | 0.041 | 0.3 % | 0.2 % |
| **`smp_gpu` total** | **16.348** | 100 % | **60.4 %** |
| `gpu_busy_from_trace` | 27.053 | | 100 % |

The nesting checks out: `prep + pen + pick + post` = 16.338 against `smp_gpu` 16.348, and `pen + pick`
= 15.705 against `smp_run_gpu` 15.710.

**This prices the two halves of log1001 §7.1–7.2 separately for the first time:**

* **§7.2, sparse penalties, now has a number: 6.85 ms/step.** `apply_penalties` must run before `topk`
  because it changes the ranking, so it cannot move into the candidate window — but it is a *sparse*
  update, touching at most `prompt_len + output_len ≤ 1 024` of 151 936 columns, 0.7 %. Gather, apply,
  scatter back instead of six full-vocabulary passes.
* **§7.1, sampling inside the top-k window, targets most of 8.86 ms.** Of that, `topk` itself is
  ~1.86 ms and irreducible; the rest is the full-vocabulary tail — temperature, the `-inf` scatter-back,
  the fp32 softmax, the dead-row guard, `multinomial`, `argmax`.

Together ~13 of 16.35 ms, i.e. the sampler to ~3.5–4 ms and the device step from 27.05 to roughly 14 ms.
The ordering question log1003 §6.4 raised — which half first — is now answerable by size alone: they are
within 30 % of each other, so do the one that is simpler to make correct, and §7.1's window change is
already half-built.

Caveat on the smaller batches: below batch 128 host 1 is 50–80 % idle, so those `smp_*_gpu` spans contain
bubble and the pen/pick split swings between 8 % and 53 % without meaning much. Only batch 256 and 512
are trustworthy here.

---

## 7. Two things this round validates

### 7.1 Ratios transfer across hosts; absolutes do not

[log1003 §2.4](./performance_analysis_log1003.md) argued that a ratio of two host-bound loops is
independent of interpreter speed, and used it to salvage the small-batch data. host 2 ran vLLM arm A
(no compile, no graphs) at batch 1–2 before being stopped — 3 minutes per run at that speed — which is
enough to test the claim on a third CPU:

| host | CPU | qwen.py | vLLM arm A | ratio |
| --- | --- | --- | --- | --- |
| log1003 | EPYC 7402 @ ~2.24 GHz | 43.0 | 70.8 | **1.64×** |
| log1009 host 2 | Xeon E5-2680 v4 | 29.6 | 45.1 | **1.53×** |

At batch 2: 1.64× against 1.54×. **The absolutes moved by 0.69× and the ratio held within 7 %.** The
engine-design gap to vLLM's eager path is ~1.5–1.6× at small batch on any host.

### 7.2 The `.tolist()` fix did what it was predicted to do

The previous session predicted that replacing `want_stage_snapshot[i].item()` with one `tolist()` would
take batch-512 call counts from 515/517/515 to "3/5/3":

| aten calls/step @ batch 512 | log1003 (before) | host 1 | host 2 |
| --- | --- | --- | --- |
| `aten::item` | 515 | **3** | **3** |
| `aten::_local_scalar_dense` | 515 | **3** | **3** |
| `aten::select` | 517 | **5** | **5** |
| `aten::as_strided` | 1 041 | 529 | 532 |
| **total** | **6 158** | **4 112** | **4 118** |

−33 % of all per-step aten calls, and the count is now flat with batch at ~4 100 — the fixed per-step
floor log1003 §2.4 identified. `upd_proj`, the span added to measure it, reads 0.097 ms/step at batch
512. As predicted it bought no throughput at batch 512 (device-bound there) and nothing at batch 1
(15 calls); its value is at batch 128–256.

---

## 8. Caveats

1. **Host 1 has one sweep, not two.** No run-to-run floor for its throughput column. Host 2's 1.59 %
   worst case on identical code is the best available proxy.
2. **Host 2 has no GPU clock/power trace during its own sweep** — only during the vLLM run, where the
   GPU never exceeded 50 % utilisation so the trace has no busy samples. Throttling is ruled out by the
   device-time agreement rather than directly.
3. **Host 2's vLLM arm covers batch 1 and 2 only.** Enough for §7.1's ratio test, not for a sweep.
4. **The `*_gpu` spans are event spans, not kernel sums** (§2.2). Host 2's `fwd_gpu` 2.05× and
   `rope_gpu` 4.18× at batch 512 are bubble, not slower kernels; `gpu_busy_from_trace` is the field to
   trust.
5. **Prefill latency crosses over at batch ≥ 256** — host 2 reports *lower* prefill than host 1
   (117.8 vs 125.6 ms at 256, 141.1 vs 164.8 at 512). Plausibly a second-order effect of host 1
   admitting requests faster and therefore batching more prefill chunks per step, but this report does
   not have the data to establish it. Noted, not explained.
6. **No host-side measurements were taken.** The interpreter loop, `uptime` and `scaling MHz` that
   [§1b](../env/vastai/vast-evn-build.md) and [§6e](../env/vastai/vast-evn-build.md) ask for are absent
   from both hosts, so §4.1's attribution rests on the uniformity of the penalty rather than on a direct
   number, and neither host's actual operating clock or neighbour load is known. One command per host
   next time.
7. **Cross-log comparison with log1001/log1003 mixes code and host.** log1009 runs a newer commit
   (the sampler split, `upd_proj`, the `.tolist()` fix). The device side is unchanged — batch-512
   `gpu_busy` is 27 053 µs here against log1001's 27 305 — so the sampler work from §6 is still
   outstanding, but TPOT comparisons across those logs are not clean.

---

## 9. What follows

1. **Finish the sampler, with §6's numbers to order the work** — 6.85 ms of penalties and 8.86 ms of
   pick, ~13 ms recoverable of a 27 ms device step. This remains larger than the entire measured gap to
   vLLM ([log1003 §3.2](./performance_analysis_log1003.md)).
2. **Batch `RequestBuffer.set()`'s list comprehensions** (§5). Same shape as the `.item()` fix, now the
   largest batch-scaling host cost, and worth most at batch 128–256.
3. **Rent host-1-class machines for engine comparisons.** A Zen 3 desktop part puts the knee at batch
   256 and 3.1 % idle at batch 512; a Broadwell server part pins the engine in its host-bound regime
   across the whole sweep, where the measurement says more about the CPU than the engine. The
   interpreter gate added to [§1b of the runbook](../env/vastai/vast-evn-build.md) after log1003 is what
   separates them before the rental is spent.
4. **Run the host-side measurements that §1b and §6e already ask for** — the 3 M-iteration interpreter
   loop, `uptime` after the run, `scaling MHz`, and `os.sched_getaffinity`. Four commands, one minute,
   and they are the difference between §4.1 being a measurement and §4.1 being an argument. log1003's
   276 ms on an EPYC 7402 is the only reference point the project has so far; two more would make it a
   scale.
5. **Add the second sweep on every host** (§8.1), and keep the GPU trace running for the qwen.py sweep
   and not only the vLLM arm (§8.2).
