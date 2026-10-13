# Performance analysis — the windowed sampler: bf16 `topk`, fp32 window, Gumbel-max (log1013, window run)

**This round measures items C2–C5 and E of [the sampler plan](./sampler_optimization_plan.md).** It
runs `topk` on the bf16 logits, casts only the `[B, max_k]` window to fp32, applies temperature,
top-k, top-p and sampling inside the window, and samples with Gumbel-max instead of `multinomial`.
The baseline is `main`, which is the fp32-at-entry sampler measured in
[log1013](./performance_analysis_log1013.md).

The short version: **the plan's predictions held.**

* `smp_pick_gpu` fell from **8.53 to 0.80 ms**; the plan predicted ~0.8.
* The device step fell from **20.6 to 12.4–12.6 ms**.
* Throughput rose **+25.4 % at batch 512** (12 754 → 15 999 tok/s, the project's best so far) and
  **+22.6 % at batch 1024**.

**Batch 512 is now balanced between host and device, and prefill is 69 % of the run.** The sampler is
no longer what limits throughput.

**Data sources** (`log_vast/log1013/sampler-bf16-logits-fp32-topk-window/`)

| Content | Path |
| --- | --- |
| Concurrency sweep ×2, `main` | `benchmark_baseline/`, `benchmark_baseline2/` |
| Concurrency sweep ×2, candidate | `benchmark_fp32_topk_window/`, `benchmark_fp32_topk_window2/` |
| Pure-decode profile ×2 per arm | `profile_{baseline,baseline2,fp32_topk_window,fp32_topk_window2}/` |
| Step diffs | `baseline_against_fp32_topk_window/`, plus both run-to-run diffs |
| Host inventory | `host_check.log`, `gpu.static.csv` |

**Code under test**

| arm | ref | sampler |
| --- | --- | --- |
| baseline | `main` @ `42edf53` | fp32 cast at entry, full-vocab tail, `multinomial` |
| fp32_topk_window | `perf/sampler-bf16-logits-fp32-topk-window` @ `b4ebaf7` | bf16 `topk`, fp32 window, Gumbel-max |

**`test_info` swaps the two hashes.** It lists `main` as `b4ebaf7` and the candidate as `42edf53`. Git
says otherwise: `b4ebaf7` exists only on the candidate branch, and `main` is at `42edf53`. The dumps
agree with git: the `fp32_topk_window` runs have `smp_pick_gpu` 0.80 ms, which is the window code.
`be525e1` (the branch head) changes only tests, so its `src/` is identical to `b4ebaf7`.

The config is the same as log1013 in all four runs: `temperature=0.7, top_k=20, top_p=0.8,
repetition_penalty=1.1, frequency_penalty=presence_penalty=0`. **No preemptions and no errors at any
of the 44 sweep points.**

---

## Executive summary

1. **Throughput** (mean of two sweeps per arm):

   | batch | Δ |
   | --- | --- |
   | ≤ 64 | within ±3 %, inside this host's noise |
   | 128 | +3.7 % |
   | 256 | +5.2 % |
   | 512 | **+25.4 %** |
   | 1024 | **+22.6 %** |

   Batch-512 TPOT fell 38.6 → 30.6 ms.
2. **Every device prediction landed** (batch 512):
   * `smp_pick_gpu` 8.53 → **0.80** (plan: ~0.8);
   * the entry cast went 0.51 → **0.005** (plan: 0);
   * `gpu_busy_from_trace` 20.60 → **12.36–12.60** (plan for C + E: ~12.1).
3. **bf16 `topk` costs 0.76 ms, against 1.87 in fp32.** It takes 2 radix passes per step instead of 4.
   This is the single biggest part of what remains.
4. **What remains of the sampler is 1.57 ms**, made of:
   * `topk` 0.76;
   * `smp_prep` 0.60, which is D1/D2, still open: 0.45 ms/step of `fill_` is zero-filling
     `output_counts` and the masks;
   * penalties 0.09;
   * the whole window, which is a few µs per op.
5. **The full 8.1 ms device saving reached wall time at batch 512.** The decode step went
   20.36 → 12.27 ms. Host and device are now level there: host `fwd` is 9.6–10.2 ms and `fwd_gpu` is
   10.6 ms, at 3.7 % idle in the clean profile. Batch 128–256 is host-bound (37–61 % idle), so it
   gains only from fewer host-side launches.
6. **Prefill is now the largest cost at high batch.** At batch 512, decode time fell 20.0 → 12.5 s and
   prefill time 30.1 → 27.3 s, so prefill is 69 % of the run. At batch 1024 it is 82 s against
   13.7 s. A full 8 k-token prefill step takes ~61 ms and is compute-bound; the sampler's share is
   0.42 ms.
7. **This host was busy with other tenants.**
   * Load average 15.4; on average 48 % busy across all 32 CPUs; the interpreter loop ranged
     152–197 ms.
   * As a result, the host-side fields of the two arms are **not comparable**. Unchanged fields such
     as `sched` and `bld_meta` differ by ~20 % between them, because `main` was profiled 45 minutes
     later.
   * Device spans and sweep throughput at batch ≥ 256 are clean, with run-to-run differences of
     ≤ 0.1 % and ≤ 0.6 %. They carry this report.

---

## 1. The host

| | this run | log1013 host 2 (same CPU model) |
| --- | --- | --- |
| CPU | Ryzen 9 7950X3D, 32 threads | Ryzen 9 7950X3D, 32 threads |
| single-core clock | 4.83–4.97 GHz | 4.73–5.09 GHz |
| **interpreter loop** | **152–197 ms** | 148–155 ms |
| **load average** | **15.37 / 15.43 / 15.32** | 0.89 / 7.50 / 12.44 |
| CPU busy at host check | 5–96 %, mean **48 %** | one CPU at 1.5 % |
| all-core scheduled | 61–67 % | — |
| GPU | RTX 4090, driver 550.120, 450 W, gen 4 × 16 | RTX 4090, driver 580.173.02, 450 W |

The GPU was healthy throughout. At utilisation ≥ 95 % the SM clock was 2 685 MHz in all four sweeps,
power was 390 W (`main`) and 415 W (candidate), `sw_power_cap` was active in ≤ 0.5 % of samples, and
the temperature peaked at 62 °C.

The CPU was not healthy, for a reason that is not ours: other tenants kept it about half busy, and the
spread of the interpreter loop (152 to 197 ms within one check) shows the contention moving over time.
`main` therefore reads 8 % slower at batch 1 than log1013 host 2, which ran the same code: 90.7 against
99.0 tok/s. On the device side the two hosts agree: `main` has `gpu_busy` 20.60 here against 20.38
there, and `smp_gpu` 9.78 against 9.77.

---

## 2. Method and noise

Run order:

* candidate profiles at 14:08 and 14:13;
* candidate sweeps at 14:16 and 14:24;
* `main` sweeps at 14:35 and 14:46;
* `main` profiles at 14:54 and 14:56.

Noise:

| | baseline | candidate |
| --- | --- | --- |
| sweep run-to-run, batch ≥ 256 | ≤ 0.1 % | ≤ 0.6 % |
| sweep run-to-run, batch 1–128 | ≤ 4.4 % | ≤ 6.3 % |
| profile `gpu_busy`, batch 512 | 20.60 / 20.59 ms | 12.36 / 12.60 ms |
| profile `wall_clean`, batch 512 | 21.50 / 21.37 ms | **14.66 / 13.08 ms** |
| anomalous steps, batch 512 | 0/20 | 6/20 (first run), all `rope_gpu`/`smp_*` bubbles |

What this means for the comparison:

* **Low-batch throughput deltas inside ±6 % are noise.** That covers batch ≤ 64.
* **The candidate's first batch-512 profile hit a contention spike.** It has 15.7 % idle against 3.7 %
  in the second run, and its wall time is 12 % higher. Both runs agree on device time. §4 uses the
  second run for wall time.
* **Host-side fields cannot be compared across arms.** Unchanged code differs by 17–23 % between them
  (`sched`, `bld_meta`, `rope`), and the CV inside `main`'s batch-512 profile is 10–16 %. Host-side
  claims below are directional only.

---

## 3. Results

### 3.1 Throughput (tok/s, two sweeps per arm)

| batch | baseline | baseline2 | candidate | candidate2 | Δ (means) | TPOT base → cand (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 89.2 | 92.2 | 91.7 | 90.7 | +0.5 % | 11.02 → 10.96 |
| 8 | 656.4 | 686.3 | 706.0 | 664.0 | +2.0 % | 11.82 → 11.59 |
| 64 | 4 579.1 | 4 504.4 | 4 580.8 | 4 500.9 | −0.0 % | 13.66 → 13.67 |
| 128 | 7 815.0 | 7 884.2 | 8 039.7 | 8 233.7 | +3.7 % | 15.72 → 15.16 |
| 256 | 12 206.1 | 12 211.5 | 12 804.8 | 12 874.7 | +5.2 % | 20.08 → 19.09 |
| 512 | 12 761.4 | 12 746.1 | **15 953.1** | **16 043.9** | **+25.4 %** | 38.64 → **30.58** |
| 1024 | 10 957.7 | 10 942.2 | **13 421.4** | **13 437.4** | **+22.6 %** | 90.00 → **73.30** |

Peak throughput is at batch 512 in both arms. At 16 000 tok/s the candidate is the highest the project
has measured. Before this, the best was log1012's bf16 `smp_opt` at 13 913.

### 3.2 Device spans (batch 512, profile, ms/step)

| span | baseline | candidate | Δ |
| --- | --- | --- | --- |
| `smp_gpu` | 9.784 | 1.571 | **−8.213** |
| `smp_run_gpu` | 9.136 | 0.905 | −8.231 |
| `smp_pen_gpu` | 0.094 | 0.103 | — |
| `smp_pick_gpu` | 8.532 | **0.797** | **−7.736** |
| `smp_run_gpu − pen − pick` (entry cast) | 0.510 | **0.005** | −0.505 |
| `smp_prep_gpu` | 0.596 | 0.613 | — |
| `fwd_gpu` | 10.653 | 10.919 | noise: the event span absorbs bubbles |
| `gpu_busy_from_trace` | 20 605 / 20 592 µs | **12 356 / 12 596 µs** | −8.1 ms |

The sweep medians agree. At batch 512 `smp_gpu` is 9.78 → 1.53 ms; at batch 1024 it is
19.29 → 2.87 ms.

### 3.3 What is left inside `smp_pick` (batch 512, per step)

| op | baseline (fp32 tail) | candidate | note |
| --- | --- | --- | --- |
| `topk` | 1.868 | **0.763** | bf16: 2 `radixFindKthValues` passes + `gatherTopK`, against 4 passes in fp32 |
| `_softmax` | 1.255 (full vocab) | 0.002 (window) | |
| `div` + `div_` | 1.664 | 0.007 | temperature now on the window; `multinomial`'s `div` gone |
| `sum` + `masked_fill_` | 1.367 | 0.004 | dead-row guard no longer needed |
| `argmax` | 0.385 | 0.005 | Gumbel-max argmax over 20 columns |
| `exponential_` / `uniform_` | 0.306 | 0.002 | `multinomial`'s noise → `rand_like` on the window |
| `max` | 0.329 | 0 | |

Everything after `topk` costs a few µs per op, so the window is effectively free on the device. The
`fill_` that remains in the step (0.45 ms) is `bin_count_and_mask` zero-filling `output_counts` and the
masks in `smp_prep`, which is D1/D2.

### 3.4 Against the plan

| item | plan | measured |
| --- | --- | --- |
| C2–C5 + E: `smp_pick` | 8.52 → ~0.8 | 8.53 → **0.80** |
| E: entry cast | 0.51 → 0 | 0.51 → **0.005** |
| E: `topk` in bf16 | −1.14 | **−1.10** (1.868 → 0.763) |
| device step after C + E | ~12.1 ms | **12.36–12.60 ms** |
| `smp_prep` (D1/D2 not yet done) | 0.60 | 0.60–0.61 |

---

## 4. Wall time: the whole saving arrived, and batch 512 is now at the host floor

Decode-step medians from the sweeps, ms:

| batch | baseline | candidate | Δ |
| --- | --- | --- | --- |
| 128 | 11.75 | 11.15–11.58 | −0.2 to −0.6 |
| 256 | 11.79 | 11.21–11.43 | −0.4 to −0.6 |
| 512 | **20.36** | **12.18–12.35** | **−8.1** |
| 1024 | 37.49 | 21.10 | −16.4 |

* **Batch 512: the full device saving reached wall time.** In the candidate the step is now level on
  both sides. Host `fwd` is 9.6–10.2 ms and `fwd_gpu` is 10.6 ms. The clean profile shows 3.7 % idle
  at 13.08 ms of wall time, which is about the host floor on this host. That floor is 11–12 ms at
  batch ≤ 128 and is itself raised by the neighbour load.
* **Batch 128–256 are host-bound in both arms** (37–61 % idle in the candidate). Their +4–5 % comes
  from the host side: the window path launches fewer full-vocab ops, and the host time inside `smp`
  drops by roughly 0.1–0.3 ms. The exact figure is lost in this host's cross-arm drift (§2).
* **Batch 1024 is still GPU-bound** and takes the whole saving: 19.3 → 2.9 ms of `smp_gpu` against a
  16.4 ms shorter step.

---

## 5. The new bottleneck is prefill

| | batch 512 base | batch 512 cand | batch 1024 base | batch 1024 cand |
| --- | --- | --- | --- | --- |
| decode time | 20.0 s | **12.5 s** | 23.1 s | **13.7 s** |
| prefill time | 30.1 s | 27.3 s | 94.7 s | 82.1 s |
| prefill share | 60 % | **69 %** | 80 % | **86 %** |
| full 8 k-token prefill step | 62.2 ms | 61.3 ms | 62.3 ms | 61.3 ms |
| of which `smp_gpu` | 1.52 ms | 0.42 ms | 1.51 ms | 0.42 ms |

Prefill time also fell. Mixed steps run the sampler over every decode row they carry, so they too
benefit. But a full prefill step is 61 ms of compute-bound `fwd`, and the sampler is now 0.7 % of it.
From here, throughput at batch ≥ 512 is set by prefill:

* attention and GEMM efficiency on 8 k-token chunks;
* chunk size and how prefill is interleaved with decode.

For batch ≤ 256 it is set by host time per step.

---

## 6. Correctness

* **Distribution.** Gumbel-max uses the RNG differently from `multinomial`, so a fixed seed no longer
  reproduces the old tokens, and token-for-token comparison does not apply (plan, §Verification).
  `tests/test_sampling.py` now has:
  * **`test_sample_matches_softmax_distribution[fp32, bf16]`**: draws 20 000 samples per group for two
    parameter groups in one batch, (T, k, p) = (0.7, 8, 0.8) and (1.3, 20, 0.95). It checks that no
    token outside the top-k/top-p set is ever drawn, and runs a χ² test against an fp64 reference of
    softmax(logits / T) restricted to that set, at a 10⁻⁶ tail. The test has power: a sampler that
    ignores temperature scores χ² = 490 against a critical value of 37.5.
  * **`test_sample_greedy_rows_take_window_argmax_in_mixed_batch`**: checks that greedy rows in a
    mixed bf16 batch return `indices[:, 0]`, and that this equals `logits.argmax()`.
* **Cold paths.** `be525e1` turns on top-p and all three penalties in the end-to-end
  `tmp_target_driver` fixture, so the freq/pres branches now run in a full engine test.
* **Dead code.** `sample()` no longer calls `apply_fused_top_k_and_p`; only its tests still use it.
  Either delete both, or keep it explicitly as a reference oracle.

---

## 7. Caveats

1. **Neighbour load** (§1, §2). Host-side fields are not comparable across arms, and low-batch
   throughput deltas inside ±6 % are noise. The conclusions rest on device spans and on batch ≥ 256
   throughput, which repeat to within 0.6 %.
2. **The arms were not interleaved.** The candidate ran first throughout. The contention moved over
   time, which is why host-side fields drifted by ~20 %.
3. **The candidate's first batch-512 profile hit a contention spike** (6/20 anomalous steps, 15.7 %
   idle). Its device time agrees with the second run.
4. **The op table in §3.3 uses aten totals for the whole step.** At batch 512 the baseline's listed ops
   are dominated by the sampler's `[B, V]` calls. In the candidate those ops are so small that model
   calls may make up most of what is left.

---

## 8. What follows

1. **Merge the candidate** once the new distribution tests are committed, and fix `test_info`'s
   hashes.
2. **D1/D2** (−0.55 ms): stop zero-filling `output_counts` and the masks when freq/pres are off. This
   needs the switches computed before `bin_count_and_mask`. At batch 512 it will show only once host
   time drops, because host and device are level.
3. **Host time per step.** Decode CUDA graphs for `fwd` would lower the floor at every batch ≤ 512 and
   expose the device-side work again.
4. **Prefill.** It is 69–86 % of the run at batch ≥ 512, at ~61 ms per 8 k-token chunk. Profile one
   full prefill step and compare it with vLLM's prefill on the same chunk.
5. **Rent a quiet host for host-side work.** This run's 152–197 ms interpreter loop and 48 % neighbour
   load make host-side fields unusable across arms. Add the load average and per-CPU busy figures from
   `check_instance.sh` to the rental gate, next to the power limit.
