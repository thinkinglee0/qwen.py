# Performance analysis — the sampler optimisation, same-host A/B (log1012)

**This round measures the first batch of [the sampler plan](./sampler_optimization_plan.md)**, with
both arms on one instance, back to back: `main` as the baseline and the `smp_opt` branch as the
candidate. It also answers a question from reading the sweep in isolation: **why is this engine about
12 % slower at low batch than log1009 host 1?**

The short version: **the sampler change worked as planned on the penalties and better than planned on
the pick, and some of that "better" is a dtype change the plan said not to make. The low-batch
slowdown is not a code regression: both branches show it equally, because this host's CPU is slower.**

**Data sources**

| Content | Path |
| --- | --- |
| Concurrency sweep, `main` | `log_vast/log1012/benchmark_baseline/` |
| Concurrency sweep, `smp_opt` | `log_vast/log1012/benchmark_smp_opt/` (plus `concurrency_sweep.txt`) |
| Pure-decode profile ×2, `main` | `log_vast/log1012/profile_baseline{,2}/` |
| Pure-decode profile ×2, `smp_opt` | `log_vast/log1012/profile_smp_opt{,2}/` |
| Run-to-run diffs | `log_vast/log1012/{baseline_against_baseline2,smp_opt_against_smp_opt2}/` |
| **`main` vs `smp_opt` diff** (generated for this report) | `log_vast/log1012/baseline_against_smp_opt/` |
| Host inventory | `log_vast/log1012/{host_check.log,gpu.static.csv}` |
| Cross-host reference | `log_vast/log1009/host1/benchmark_baseline/` |

**Code under test**

| arm | ref | difference |
| --- | --- | --- |
| baseline | `main` @ `2c169ce` | — |
| smp_opt | `smp_opt` @ `b60fd3f` = `main` + one commit | `sampling.py`, `scheduler.py`, `model.py` (+168 / −55) |

Both arms have the same config, checked from the dumps: `temperature=0.7, top_k=20, top_p=0.8,
repetition_penalty=1.1, frequency_penalty=presence_penalty=0`, `compile_rope=false`,
`pre_gather_cos_sin=true`, `max_num_batched_tokens=8192`, `num_blocks=4096 × 256`. The workload is
`test_benchmark_sweep_batch_size` (10 × batch requests, minimum 64; 512 in / 128 out; `ignore_eos`).
**Neither arm had a preemption or an error at any of the 22 sweep points.**

---

## Executive summary

1. **Peak throughput rose 24 %: 11 199 → 13 913 tok/s at batch 512**, and **+29 % at batch 1024**
   (9 806 → 12 660). Batch-512 TPOT went from 44.2 to 35.4 ms. At batch ≤ 128 the two arms are within
   **±0.4 %**, which is expected, because that regime is host-bound (§3.1).
2. **Device time at batch 512 fell by 11.65 ms/step**: `gpu_busy_from_trace` went from 26 844 to
   15 192 µs and `smp_gpu` from 16.33 to 4.67 ms (−71 %). `smp_pen_gpu` went from **6.85 to 0.09 ms**,
   which is the plan's modelled −6.85 ms for A2 + B1–B3, matched almost exactly (§5.1).
3. **`smp_pick_gpu` fell 8.85 → 3.94 ms, about 4× what the plan priced for the items that landed.**
   About **2.9 ms of the 4.9 ms comes from the sampling tail running in bf16 instead of fp32**. The
   evidence is that the same kernels run the same number of times and each costs half as much.
   [The plan's "Do NOT change these"](./sampler_optimization_plan.md#do-not-change-these) lists the
   fp32 cast first. This is a numerical change that has not been checked, and it needs a decision
   before merge (§6).
4. **The engine is now host-bound at batch 512 as well.** Idle time is **3.6 % → 23.9 %**. Wall time
   fell only 7.9 ms (27.84 → 19.97 ms) of the 11.65 ms the device saved. The ~19–20 ms step is the
   host floor. On this host, the plan's modelled end state (12.9 ms device step) cannot appear in wall
   time, and the rest of the plan (C2–C5, D1/D2) pays off only at batch 1024 (§7).
5. **Host-side cost of the change is net zero.** `smp_prep` gained a constant **+0.07 ms** (three
   host-side `.any().item()` switch reads plus the temperature flags). `smp_run` lost **−0.07 ms**
   because it launches fewer ops. No device sync was added: `dth_wait` did not move.
6. **Why the low-batch numbers look worse than log1009: the host CPU.** At batch 1, both `main` and
   `smp_opt` run 58.4 tok/s, against 66.2 on log1009 host 1 (−12 %). This host has a Ryzen 9 3900X
   (Zen 2); log1009 host 1 had a 5900X (Zen 3). The unchanged host-side fields are all slower by the
   same factor, **1.10–1.15×**. Where the step is GPU-bound, the gap disappears: `main` at batch 512
   does 11 199 here against 11 117 there, and its `gpu_busy` is −0.8 % (§4).
7. **One field does not fit the CPU explanation: `bld_meta`, 2.05× (0.41 → 0.83 ms at batch 1).** It
   is identical in both arms, so `smp_opt` is not responsible. It costs ~0.37 ms/step beyond the CPU
   factor, and it is either the scheduler commits since log1009 or the CPU sensitivity that log1009
   already saw in this field. A same-host A/B would settle it (§4.3).

---

## 1. The host

| | log1012 | log1009 host 1 (reference) |
| --- | --- | --- |
| GPU | RTX 4090, driver 595.71.05, 450 W enforced | RTX 4090, driver 595.84, 450 W |
| PCIe | gen 4 × 16 under load (gen 1 at idle in `host_check.log`, which is power-saving) | gen 4 × 16 |
| **CPU** | **Ryzen 9 3900X**, Zen 2 (2019), 24/24 threads | **Ryzen 9 5900X**, Zen 3 (2020), 24/24 |
| Measured clock | 4.57 GHz single-core, 3.95 GHz all-core | not measured |
| **Interpreter loop** (runbook §1b) | **197–210 ms / 3 M iters** | not measured |
| cgroup | quota 23.04 CPUs; 7 throttled periods (0.59 s) before the runs | — |

`scaling_max_freq` reads 3.8 GHz, but the measured clock is 4.57 GHz, so the cap is not enforced.
The cgroup throttling is negligible for a single-threaded engine loop.

**Both arms had the same GPU conditions** during their sweeps. Busy samples (utilisation ≥ 50 %):

| | baseline | smp_opt |
| --- | --- | --- |
| median SM clock | 2 805 MHz | 2 805 MHz |
| median / max power | 386 / 450 W | 423 / 453 W |
| max temperature | 66 °C | 66 °C |
| `sw_power_cap` active | 2.3 % | 4.9 % |

`smp_opt` draws more power because it keeps the GPU busier per second. Neither arm throttled enough
to matter.

---

## 2. Method

### 2.1 Run order and noise

The two arms ran sequentially on one instance: `smp_opt` from 15:41 to 15:58, then `main` from 16:14
to 16:32. Each arm has one sweep and two pure-decode profiles.

| | baseline | smp_opt |
| --- | --- | --- |
| extra batch-512 sweep run at the start of the pytest session | 11 041 vs 11 193 tok/s (**1.4 %**) | 13 658 vs 13 903 (**1.8 %**) |
| profile `wall_clean` run-to-run, all batches | ≤ 1.7 % | ≤ 3.2 % |
| anomalous steps (batch 512) | 1/20 | 0/20 |

So the noise floor is about 2 %. The +24 % and +29 % are an order of magnitude above it. The ±0.4 % at
low batch is inside it.

### 2.2 Reading the `*_gpu` fields

As established in [log1001 §2.2](./performance_analysis_log1001.md), `fwd_gpu`, `rope_gpu`,
`logits_gpu` and `smp_*_gpu` are CUDA-event spans on the stream, not kernel sums. When the host is
the bottleneck, they include idle bubbles. That is why `smp_opt` *appears* to make the model slower
at batch 512: `fwd_gpu` +39 %, `rope_gpu` +255 %, `logits_gpu` +7.5 %. The model code is untouched.
Now that the step is host-bound, those spans have absorbed bubbles. The figure to trust for device
time is `gpu_busy_from_trace`. The sampler spans are still reliable, because the sampler is a dense
run of kernels with no host work in between.

---

## 3. Results

### 3.1 Throughput (same host)

| batch | baseline tok/s | smp_opt tok/s | Δ | TPOT base → smp (ms) |
| --- | --- | --- | --- | --- |
| 1 | 58.4 | 58.4 | −0.0 % | 17.12 → 17.12 |
| 8 | 438.4 | 437.2 | −0.3 % | 18.21 → 18.27 |
| 64 | 3 173.6 | 3 168.4 | −0.2 % | 19.78 → 19.82 |
| 128 | 5 811.7 | 5 835.1 | +0.4 % | 21.48 → 21.41 |
| 256 | 9 636.0 | 9 898.9 | **+2.7 %** | 25.69 → 25.05 |
| 512 | 11 199.1 | **13 912.9** | **+24.2 %** | 44.16 → **35.35** |
| 1024 | 9 805.6 | **12 659.8** | **+29.1 %** | 100.73 → **77.73** |

The knee does not move: the last profitable doubling ends at batch 256 in both arms. Peak is at batch
512 in both. Batch 1024 is still past the knee, 9 % below peak, and its time is dominated by prefill
(84 of 102 s).

Prefill steps benefit too. Mixed steps also run the sampler, so batch-512 prefill time fell from
31.4 to 27.4 s, and batch 1024 from 101.3 to 84.4 s.

### 3.2 Device side (pure decode, `gpu_busy_from_trace`, µs/step)

| batch | baseline | smp_opt | Δ | idle base → smp |
| --- | --- | --- | --- | --- |
| 1 | 3 167 | 3 130 | −1.2 % | 81.3 % → 81.4 % |
| 8 | 3 939 | 3 861 | −2.0 % | 78.3 % → 78.4 % |
| 32 | 4 191 | 4 133 | −1.4 % | 77.3 % → 77.4 % |
| 64 | 5 175 | 4 325 | −16.4 % | 72.3 % → 76.9 % |
| 128 | 8 114 | 5 412 | −33.3 % | 57.4 % → 71.4 % |
| 256 | 14 805 | 8 549 | −42.3 % | 22.7 % → 55.7 % |
| 512 | **26 844** | **15 192** | **−43.4 %** | **3.6 % → 23.9 %** |

The device saving grows with batch, as a `[B, V]` sampler cost should. Wall time follows only where
the GPU was the bottleneck: batch 512, and partly 256.

### 3.3 Sampler spans (batch 512, profile, ms/step)

| span | baseline | smp_opt | Δ |
| --- | --- | --- | --- |
| `smp_gpu` | 16.329 | 4.671 | **−11.657** |
| `smp_prep_gpu` | 0.588 | 0.595 | +0.007 |
| `smp_pen_gpu` | 6.848 | 0.090 | **−6.758** |
| `smp_pick_gpu` | 8.845 | 3.938 | **−4.906** |
| `smp_post_gpu` | 0.040 | 0.039 | — |
| host `smp_prep` | 0.299 | 0.370 | +0.071 |
| host `smp_run` | 0.648 | 0.576 | −0.072 |
| host `smp` | 1.156 | 1.156 | 0.000 |

---

## 4. Why the low-batch numbers are worse than log1009

### 4.1 It is not the branch

At batch ≤ 128, `main` and `smp_opt` are within ±0.4 % of each other (§3.1). Both are 12–13 % below
log1009 host 1:

| batch | log1009 host 1 | log1012 `main` | log1012 `smp_opt` |
| --- | --- | --- | --- |
| 1 | 66.2 | 58.4 (−11.8 %) | 58.4 (−11.8 %) |
| 64 | 3 644.5 | 3 173.6 (−12.9 %) | 3 168.4 (−13.1 %) |
| 256 | 10 557.3 | 9 636.0 (−8.7 %) | 9 898.9 (−6.2 %) |
| 512 | 11 117.2 | **11 199.1 (+0.7 %)** | 13 912.9 (+25.1 %) |

The batch-512 row is the control. Both hosts run `main`'s sampler, and the step there is GPU-bound,
at 3.1 % and 3.6 % idle. In that case, the hosts agree within 1 %, and so does `gpu_busy`: 27 053
against 26 844 µs. **The GPU is the same; only the host differs.**

### 4.2 It is the host CPU

In the batch-1 sweep, the step is 81 % idle on the GPU, so its length is the host's time. Decode-step
medians in ms:

| field | log1009 h1 | log1012 `main` | ratio |
| --- | --- | --- | --- |
| `step_0` | 14.264 | 16.379 | 1.148 |
| `fwd` (kernel launches for 24 layers) | 12.749 | 14.305 | **1.122** |
| `rope` | 3.089 | 3.553 | **1.150** |
| `smp` | 0.962 | 1.093 | **1.136** |
| `smp_post` | 0.154 | 0.169 | **1.097** |
| `bld_meta` | 0.406 | 0.831 | **2.047** |
| `sched` | 0.024 | 0.016 | 0.67 (code: `2c169ce`) |

Every field whose code did not change is slower by the same 1.10–1.15×. A uniform factor like that
points to a slower interpreter, not to a bottleneck in one place. It is the same pattern log1009 found
at 2.1× on a Broadwell Xeon, here at a smaller scale. A Zen 2 to Zen 3 step in single-thread speed fits
this size.

**The interpreter loop predicts the gap.** log1009 host 1 has no loop number, so the 5900X cannot be
placed directly. But this round and log1011 both measured it, and the ratio carries across:

| | interpreter loop | `fwd` at batch 1 |
| --- | --- | --- |
| log1011 (EPYC 7452) | 263–271 ms | 19.145 ms |
| log1012 (Ryzen 3900X) | 197–210 ms | 14.339 ms |
| ratio | **1.31×** | **1.34×** |

That makes the 3 M-iteration loop a usable predictor of this engine's low-batch throughput. It is the
second data point after log1003's 276 ms on an EPYC 7402.

### 4.3 The one outlier: `bld_meta`

`bld_meta` is 2.05× against the 1.12× CPU factor. That is ~0.37 ms/step that the CPU does not
explain, about 2.3 % of the batch-1 step. It is identical in both arms (0.831 / 0.828 ms at batch 1;
1.053 / 1.053 ms at batch 512), so `smp_opt` is not the cause. Two explanations remain, and this data
cannot separate them:

* **Code.** Commits after log1009 touched the path it times: `687f265` (request slots sized by
  pipeline depth), `43e66f5`, and `2c169ce` (per-request columns built in `sched_ret`).
* **CPU sensitivity.** log1009 already saw this field hit harder than any other on a slower CPU:
  3.0–3.5× against a uniform 2.1×. In those runs, `bld_meta / fwd` went from 0.032 to 0.049 with
  *identical code*. Here it is 0.058.

Settling it costs one profile: run log1009's commit on this host.

---

## 5. What the sampler change did, item by item

### 5.1 Against the plan

Predicted and measured figures are ms/step at batch 512. The profile measures spans, not items, so
each landed item's measured effect appears in its span's total row.

| span | plan item | landed? | predicted | measured |
| --- | --- | --- | --- | --- |
| `smp_pen` | A2 per-term penalty gating (freq/pres skipped when 0) | ✓ | −3.13 | |
| `smp_pen` | B1 / B2 / B3 sparse, in-place repetition via `[B, L]` gather/scatter | ✓ (B3 subsumes B1, B2) | −0.42 / −1.10 / −2.2 | |
| **`smp_pen`** | **total** | | **−6.85** | **−6.76** |
| `smp_pick` | A3 skip greedy `argmax` when no row is greedy | ✓ | −0.76 | |
| `smp_pick` | C1 temperature in place (`div_`), skipped when every `t == 1` | ✓ | −0.30 | |
| `smp_pick` | A4 `all_greedy` early return | ✓ (cold path here, `t = 0.7`) | — | |
| **`smp_pick`** | **total** | | **−1.06** | **−4.91** |
| `smp_pick` | C2–C5 sampling tail inside the `[B, max_k]` window | ✗ | −6.2 | — |
| `smp_prep` | D1 / D2 skip `output_counts` and the masks | ✗ | −0.54 | — (`smp_prep_gpu` unchanged at 0.59) |

`smp_pen` came in at the predicted figure. The plan's model for it had already been validated to
0.1 %. `smp_pick` came in at 4.91 ms against roughly 1.06 ms predicted for A3 + C1. §6 accounts for
the difference.

### 5.2 Where the `smp_pick` saving came from

These are aten totals from the batch-512 profile, divided by the 5 profiled steps. Each op below is
called **the same number of times in both arms**, on the same shapes. The only change is the dtype the
kernels report:

| op | calls | baseline (fp32) | smp_opt (bf16) | Δ ms/step |
| --- | --- | --- | --- | --- |
| `topk` (`radixFindKthValues<float>` → `<BFloat16>`) | 5 / 5 | 1.856 | 0.719 | −1.137 |
| `_softmax` (`cunn_SoftMaxForward<…, float, …>` → `<…, BFloat16, …>`) | 10 / 10 | 1.252 | 0.429 | −0.823 |
| `sum` | 10 / 10 | 0.693 | 0.352 | −0.341 |
| `masked_fill_` | 15 / 15 | 0.675 | 0.380 | −0.295 |
| `exponential_` (inside `multinomial`) | 5 / 5 | 0.303 | 0.149 | −0.154 |
| `max` | 15 / 15 | 0.329 | 0.176 | −0.153 |
| **subtotal** | | | | **−2.90** |

Each one roughly halves, which is what halving the bytes per element does to a bandwidth-bound
kernel. The other ~2.0 ms comes from the items that landed: `argmax` 0.714 → 0.222 (A3, and bf16
for the one that remains), and the temperature/`where` traffic (C1).

---

## 6. The fp32 cast is gone

In `main`, the sampling tail ran in fp32. That was an accident of the repetition penalty: the `rep`
tensor is fp32, so `torch.where(…, logits / rep, logits * rep)` promoted the logits. On top of that,
`sample()` cast explicitly: `logits.float().softmax(dim=-1)`. `smp_opt` removes both:

* `apply_fre_penalty_sparse` writes the penalised values back with `.to(logits.dtype)`, so the logits
  stay bf16.
* `sample()` drops `.float()` before the full-vocab softmax.

So `topk`, the top-p `cumsum` against `top_p`, the full-vocab softmax and the `multinomial` input
are all bf16 now. bf16 carries about 3 significant digits. That affects two things:

* **Ranking near the top-k and top-p boundaries.** The plan's reason for keeping fp32: the repetition
  penalty is ranking-sensitive.
* **The probability mass passed to `multinomial`.**

The plan said this explicitly ([Do NOT change these](./sampler_optimization_plan.md#do-not-change-these)):
make the fp32 cast deliberate, but do not try to stay in bf16. The plan's verification table also
asks for a **fixed-seed, token-for-token comparison against the old path**. The branch's "all cases
passed" refers to `tests/test_sampling.py`, which is a different check.

**Restoring fp32 is cheap where it matters.** At batch 512 the step is now host-bound, with ~4.8 ms
of device idle (19.97 wall − 15.19 busy). Adding the ~2.9 ms back would fit inside that idle time and
should leave wall time roughly unchanged. Only batch 1024, which is still GPU-heavy, would pay for it.
That is an inference from the idle margin and has not been measured; it needs one profile with the
cast restored. Once the sampling tail moves into the `[B, max_k]` window (C2–C5), the fp32 cost falls
to the single `topk` read the plan priced at 0.17 ms.

---

## 7. The bottleneck has moved to the host

| batch 512, profile | baseline | smp_opt |
| --- | --- | --- |
| `wall_clean` | 27.84 ms | 19.97 ms |
| `gpu_busy_from_trace` | 26.84 ms | 15.19 ms |
| idle | 3.6 % | 23.9 % |

The host floor on this machine is about **18.6–19.9 ms per decode step** at batch 256–512. Most of it
is `fwd`, ~14–15.6 ms of kernel launches for 24 layers, and that cost is the same with or without the
sampler. As a result:

* **The remaining sampler items (C2–C5, D1/D2) will not show up in wall time at batch ≤ 512 on this
  host.** The plan priced them at ~6.7 ms against the fp32 tail. Against what is left now
  (`smp_pick_gpu` 3.94 + `smp_prep_gpu` 0.59 ms), the realistic saving is closer to 2.5 ms. They still help at batch 1024, where the device
  step (`fwd_gpu` 17.2 + `smp_gpu` 9.2 ms) is above the host floor.
* **The lever for every batch ≤ 512 is now host time per step**, in this order:
  1. `fwd` launch overhead: decode CUDA graphs, or fewer, fused launches.
  2. `bld_meta` (§4.3).
  3. `sched`, which scales with batch: 0.76 ms at 512.

One more consequence: the plan's modelled end state, a 12.9 ms device step, is a device number. On a
3900X it cannot become a 12.9 ms wall step until the host floor drops below it.

---

## 8. Caveats

1. **One sweep per arm.** The extra batch-512 run at the start of each pytest session gives a 1.4–1.8 %
   repeat for that point only. The other points rely on the ±0.4 % agreement between arms at low batch
   as an implicit noise check.
2. **The arms ran sequentially, not interleaved**, with `smp_opt` first and about 16 minutes apart.
   The identical GPU clock, temperature and low-batch throughput argue against drift, but the order was
   not randomised.
3. **The log1009 comparison mixes host and code.** §4's CPU attribution rests on the unchanged fields
   and on the batch-512 control. `bld_meta` is the exception and is left open (§4.3).
4. **log1009 host 1 has no interpreter-loop number**, so the 5900X cannot be placed on the scale in
   §4.2. Only the log1011 and log1012 pair confirms the loop as a predictor.
5. **The event spans `fwd_gpu`, `rope_gpu` and `logits_gpu` mislead at batch ≥ 256 in `smp_opt`**
   (§2.2). They grew because they now contain bubbles, not because the kernels got slower.
6. **§5.2 is an op-level attribution, not a line-level one.** The aten totals cover the whole step.
   At batch 512 the six listed ops are dominated by the sampler's `[B, V]` calls, but the model's own
   small calls are included in the totals.

---

## 9. What follows

1. **Decide on fp32 before merging `smp_opt`.** The recommendation is to restore one explicit
   `.float()` at the sampler entry. Then:
   * re-profile batch 512 and 1024 to price it (§6 expects ~0 ms of wall time at 512);
   * run the plan's fixed-seed token-for-token check against `main`.
2. **Land D1/D2.** They follow from B3, which has landed. The masks are no longer read when
   `freq_pen = pres_pen = 0`, but `bin_count_and_mask` still builds them: `smp_prep_gpu` is 0.59 ms
   at 512.
3. **Defer C2–C5 until the host floor drops.** On this class of host, they move only batch 1024.
4. **Turn to host time per step**, starting with decode CUDA graphs for `fwd` (§7). This is now the
   limit on every batch ≤ 512.
5. **Profile log1009's commit on a log1012-class host** to resolve `bld_meta` (§4.3). Keep running
   the interpreter loop on every rental, so cross-host comparisons have a number behind them.
