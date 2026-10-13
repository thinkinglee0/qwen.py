# Performance analysis — the sampler change with fp32 restored, on two hosts (log1013)

> 中文版：[`performance_analysis_log1013.zh.md`](./performance_analysis_log1013.zh.md)

**This round measures what [log1012](./performance_analysis_log1012.md) recommended**: the
penalty-skip and sparse-penalty work with the fp32 cast put back, done once at the sampler entry.
`main` is the baseline and `perf/sampler-skip-and-sparse-penalty` is the candidate. The same A/B ran
on two instances the same day:

* **host 1**: a slow, power-capped machine;
* **host 2**: the fastest host the project has rented so far.

The short version:

* **The device saving is the same on both hosts: 6.6 ms per step at batch 512.** How much of it turns
  into throughput depends on the host:
  * host 1: **+5.8 %** at batch 512, because it is host-bound;
  * host 2: **+16.6 %**, because it is GPU-bound.
* **fp32 costs more than log1012 said.** Restoring it costs **5.1 ms** of device time per step at
  batch 512, not 2.9 ms. On a fast host that is about a quarter of the step.
* **Two of the plan's `smp_pick` predictions were too high** (§3.4).

**Data sources** (`log_vast/log1013/`)

| Content | host 1 | host 2 |
| --- | --- | --- |
| Concurrency sweep, `main` / candidate | `host1/benchmark_{baseline,sparse_pen}/` | `host2/benchmark_{baseline,sparse_pen}/` |
| Pure-decode profile ×2 per arm | `host1/profile_{baseline,baseline2,sparse_pen,sparse_pen2}/` | `host2/profile_…` (same names) |
| `main` vs candidate step diff | `host1/baseline_against_sparse_pen/` | `host2/baseline_against_sparse_pen/` |
| Run-to-run diffs | `host1/{baseline_against_baseline2,sparse_pen_against_sparse_pen2}/` | `host2/…` (same names) |
| Host inventory | `host1/{host_check.log,gpu.static.csv}` | `host2/{host_check.log,gpu.static.csv}` |

The bf16 reference is log1012's `smp_opt` arm: `log_vast/log1012/{benchmark,profile}_smp_opt/`.

**Code under test**

| arm | ref | difference |
| --- | --- | --- |
| baseline | `main` @ `2c169ce` | — |
| sparse_pen | `perf/sampler-skip-and-sparse-penalty` @ `1bd9f72` = `main` + one commit | `sampling.py`, `scheduler.py`, `model.py`, `tests/test_sampling.py` |

The commit was amended between the two hosts' runs, from `7b9b350` to `1bd9f72`. Only
`tests/test_sampling.py` changed, and `src/` is byte-identical, so both hosts ran the same engine
code.

Compared with log1012's `smp_opt` arm, the candidate differs in one way that matters.
`QwenForCausalLM.sampler()` now calls `logits.float()` at entry, so the penalties, temperature,
`topk`, softmax and `multinomial` all run in fp32 whatever switches are on.

The config is identical in all four runs and the same as log1012: `temperature=0.7, top_k=20,
top_p=0.8, repetition_penalty=1.1, frequency_penalty=presence_penalty=0`, `compile_rope=false`,
`pre_gather_cos_sin=true`. **No preemptions and no errors at any of the 44 sweep points.**

---

## Executive summary

1. **The same code saved the same device time on both hosts, but the throughput gains differ by
   about 3×:**

   | batch | host 1 | host 2 |
   | --- | --- | --- |
   | 256 | +1.1 % | **+16.0 %** |
   | 512 | +5.8 % | **+16.6 %** |
   | 1024 | +12.4 % | **+14.2 %** |

   At batch ≤ 128 both hosts are within ±1 %. Peak throughput rose 10 048 → 10 630 tok/s on host 1
   and 11 137 → 12 990 on host 2.
2. **Device work is the same on every host measured so far.** At batch 512, `main`'s `smp_gpu` is
   16.33 / 16.36 / 16.35 ms on log1012, host 1 and host 2, and the candidate's is 9.771 / 9.772 ms on
   host 1 and host 2. The gain depends on whether the host keeps up with the device (§5):
   * **Host 2's host floor is ~10–11 ms**, so batch 256–512 is GPU-bound at 3–7 % idle, and the full
     6.6 ms saving reaches wall time (27.70 → 21.09 ms).
   * **Host 1's host floor is ~25 ms**, so it recovers only 1.9 ms of the 6.6 ms.
3. **The penalty saving holds in fp32.** `smp_pen_gpu` fell **6.85 → 0.09 ms** at batch 512 on both
   hosts. The entry cast (0.51 ms) now sits outside both sub-spans. Counting it, the net penalty
   saving is **6.25 ms**.
4. **Running in fp32 rather than bf16 costs 5.1 ms per step at batch 512** (§4). log1012 §6 counted
   six ops and estimated ~2.9 ms, and it said restoring fp32 would cost "≈ 0 ms of wall time".
   **Both claims are wrong.** On host 2, which is GPU-bound, the 5.1 ms is about a quarter of the
   decode step. That makes the sampling-tail rewrite (C2–C5) the next lever on fast hosts.
5. **In fp32, `smp_pick_gpu` fell only 8.87 → 8.52 ms (−0.34)**, all of it from A3, the dropped
   greedy `argmax`. The plan priced A3 at −0.76 because it counted both argmaxes in the step, but
   one is inside `multinomial`. It priced C1 (temperature in place) at −0.30, but `div_` moves the
   same bytes as `div`, so the measured saving is about zero (§3.4).
6. **The two hosts differ in two ways.**
   * **CPU.** Host 2 has a **Ryzen 9 7950X3D** (Zen 4); its batch-1 `fwd` takes 8.35 ms against
     19.07 on host 1's EPYC 7K62. Batch-1 throughput is 99 tok/s against 44.
   * **GPU power limit.** Host 1 is capped at **260 W**, which makes full 8 k-token prefill steps
     take 69 ms against 61 on host 2's 450 W card. Decode is bandwidth-bound and unaffected.
7. **The interpreter loop stops predicting across CPU generations.** Across three Zen 2 hosts,
   `fwd`/loop is constant at 0.068–0.072. On the Zen 4 X3D it is **0.055**: the engine runs about 22 %
   faster than the loop predicts (§6).

---

## 1. The hosts

| | host 1 | host 2 | log1012 (reference) |
| --- | --- | --- | --- |
| CPU | **AMD EPYC 7K62**, Zen 2 | **AMD Ryzen 9 7950X3D**, Zen 4, 3D V-Cache | Ryzen 9 3900X, Zen 2 |
| threads visible / cgroup quota | 96 / **23.04** | 32 / 30.72 | 24 / 23.04 |
| single-core clock | 3.25–3.27 GHz | **4.73–5.09 GHz** | 4.57 GHz |
| **interpreter loop** (3 M iters) | 271–289 ms | **148–155 ms** | 197–210 ms |
| throttled periods before runs | 82 / 1 007 | 5 | 7 |
| GPU | RTX 4090, driver 580.142 | RTX 4090, driver 580.173.02 | RTX 4090, driver 595.71.05 |
| **enforced power limit** | **260 W** | 450 W | 450 W |
| PCIe | gen 4 × 16 | gen 4 × 16 | gen 4 × 16 |

**Host 1's cgroup.** The 23-CPU quota is spread across 96 visible CPUs, so in the all-core test each
CPU was scheduled only about 25 % of the time. The single-core test was scheduled 100 %, and the
engine loop is single-threaded, so the main path should be unaffected.

**Host 2's load before the run.** Its `uptime` showed a load average of 0.89 / 7.50 / 12.44, so
another tenant had been busy shortly before the run. By the time of the host check the machine was
quiet: one CPU at 1.5 % busy.

**GPU during the sweeps** (samples at utilisation ≥ 95 %):

| | host 1 base | host 1 cand | host 2 base | host 2 cand |
| --- | --- | --- | --- | --- |
| median SM clock | 2 205 MHz | 2 085 MHz | 2 745 MHz | 2 745 MHz |
| median power | 259.5 W | 258.4 W | 389 W | 408 W |
| `sw_power_cap` active | 96.6 % | 95.9 % | 9.7 % | 13.1 % |
| max temperature | 70 °C | 70 °C | 72 °C | 72 °C |

**What the 260 W cap changes.** Decode device time is the same on every host (§3.3), because decode
is bandwidth-bound and the lower SM clock does not slow it. Prefill is compute-bound, and full 8 k
-token prefill steps show it:

| | host 1 (260 W) | host 2 (450 W) | log1012 (450 W) |
| --- | --- | --- | --- |
| full-prefill `fwd_gpu`, `main`, batch 512 | 69.2 ms | 61.1 ms | 59.6 ms |

On host 1 the candidate also clocks about 5 % lower than `main`, because it does more work per second
within the same 260 W. If anything, that biases host 1's comparison slightly against the candidate.

---

## 2. Method and noise

Each host ran its arms back to back:

* **host 1**: candidate profiles 01:39 and 01:45, then `main` profiles 01:48 and 01:50, then the
  `main` sweep 01:53–02:08, then the candidate sweep 02:09–02:24;
* **host 2**: candidate profiles 04:01 and 04:02, then the candidate sweep 04:05–04:13, then `main`
  profiles 04:14 and 04:16, then the `main` sweep 04:18–04:27.

| | host 1 base | host 1 cand | host 2 base | host 2 cand |
| --- | --- | --- | --- | --- |
| extra batch-512 sweep run at the start of the session | 1.4 % | 2.0 % | **0.9 %** | **1.2 %** |
| profile run-to-run, batch 1 `step_0` | +0.5 % | +1.3 % | −0.5 % | −0.2 % |
| profile run-to-run, batch 512 `step_0` | −0.1 % | −0.4 % | 0.0 % | +0.1 % |
| profile run-to-run, batch 512 `smp_gpu` | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| anomalous steps at batch 512 | 1/20 | 1/20 | 0/20 | 0/20 |

Host 2 is the quietest instance the project has rented. Throughput noise is about 1 % there and
about 2 % on host 1. The sampler's device spans repeat to within 0.03 % on both hosts.

At batch 1, host 1's profile diff shows the candidate +1.5 % slower. Almost all of that is in `fwd`
and `rope`, whose code did not change, and host 1's sweep shows the opposite (+0.3 %). Host 2 shows
−0.02 %. So host 1's 1.5 % is run-to-run drift, not a change caused by the code.

As in earlier reports, the `fwd_gpu`, `rope_gpu` and `logits_gpu` event spans swell when the host is
the bottleneck. On host 1 at batch 512 the candidate's `fwd_gpu` reads +45 %. On host 2, which is
GPU-bound, `fwd_gpu` is unchanged (10.36 vs 10.33 ms). `gpu_busy_from_trace` is the field to trust.

---

## 3. Results

### 3.1 Throughput (tok/s)

| batch | host 1 base | host 1 cand | Δ | host 2 base | host 2 cand | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 43.8 | 44.0 | +0.3 % | 99.0 | 99.1 | +0.0 % |
| 8 | 328.9 | 326.8 | −0.6 % | 735.9 | 732.3 | −0.5 % |
| 64 | 2 394.2 | 2 405.9 | +0.5 % | 5 027.2 | 5 012.9 | −0.3 % |
| 128 | 4 453.7 | 4 467.6 | +0.3 % | 8 639.2 | 8 631.6 | −0.1 % |
| 256 | 7 522.0 | 7 607.5 | +1.1 % | 10 832.7 | **12 567.9** | **+16.0 %** |
| 512 | 10 048.5 | **10 629.8** | **+5.8 %** | 11 136.7 | **12 990.4** | **+16.6 %** |
| 1024 | 8 918.1 | **10 022.4** | **+12.4 %** | 9 756.6 | **11 143.5** | **+14.2 %** |

TPOT at batch 512:

| | `main` | candidate |
| --- | --- | --- |
| host 1 | 49.07 ms | 46.33 ms |
| host 2 | 44.46 ms | 37.95 ms |

Peak throughput is at batch 512 for every arm on both hosts. On host 2, the candidate also moves the
last profitable doubling from batch 128 to 256.

### 3.2 Device time and idle (pure decode, profile)

| batch | host 1 `gpu_busy` base → cand (µs) | idle | host 2 `gpu_busy` base → cand (µs) | idle |
| --- | --- | --- | --- | --- |
| 1 | 3 152 → 3 141 | 86 % → 86 % | 3 200 → 3 202 | 68 % → 68 % |
| 64 | 5 405 → 4 680 | 78 % → 81 % | 5 262 → 4 530 | 52 % → 59 % |
| 128 | 8 372 → 6 712 | 67 % → 73 % | 8 208 → 6 544 | 28 % → 41 % |
| 256 | 14 809 → 11 499 | 42 % → 55 % | 14 923 → 11 605 | **5.5 % → 6.8 %** |
| 512 | **26 866 → 20 307** | **6.1 % → 23.9 %** | **26 999 → 20 385** | **2.5 % → 3.4 %** |

Wall time per decode step (`wall_clean`, ms):

| batch | host 1 | host 2 |
| --- | --- | --- |
| 256 | 25.68 → 25.45 | **15.79 → 12.45** |
| 512 | 28.61 → 26.68 | **27.70 → 21.09** |

### 3.3 Sampler spans (batch 512, profile, ms/step)

| span | host 1 base | host 1 cand | host 2 base | host 2 cand |
| --- | --- | --- | --- | --- |
| `smp_gpu` | 16.360 | 9.771 | 16.346 | 9.772 |
| `smp_run_gpu` | 15.719 | 9.127 | 15.705 | 9.129 |
| `smp_pen_gpu` | 6.851 | 0.093 | 6.849 | 0.094 |
| `smp_pick_gpu` | 8.865 | 8.523 | 8.852 | 8.525 |
| `smp_run_gpu − pen − pick` (the entry cast) | 0.003 | **0.511** | 0.004 | **0.510** |
| `smp_prep_gpu` | 0.592 | 0.597 | 0.590 | 0.595 |
| host `smp_prep` | 0.393 | 0.487 | 0.182 | 0.232 |
| host `smp_run` | 0.849 | 0.774 | 0.591 | 0.517 |

The two hosts agree on every device span to within 0.02 ms.

* **Where the cast is timed.** On `main`, the fp32 promotion happened inside `smp_pen`, as a side
  effect of `where(…, logits / rep, …)`. On the candidate the cast is explicit and sits before the
  `smp_pen` span. That is why `smp_pen` looks 0.51 ms better than the penalty work really got.
* **Host-side cost is net zero.** `smp_prep` gains a constant +0.05 to +0.09 ms from the three
  host-side `.any().item()` switch reads and the temperature flags. `smp_run` loses about 0.075 ms
  because it launches fewer ops. Neither adds a device sync: `dth_wait` did not move.

### 3.4 Ops (batch 512, host 1 profile; host 2 matches within 1 %)

The table gives aten totals divided by the 5 profiled steps. At this batch the sampler dominates every
op listed:

| op | baseline | candidate | what changed |
| --- | --- | --- | --- |
| `copy_` | 1.023 | 1.527 | **+0.50**: the explicit `logits.float()` |
| `where` | 1.532 | 0.030 | −1.50: dense repetition `where` gone (B3) |
| `mul` | 2.614 | 0.682 | −1.93: repetition and freq/pres multiplies gone |
| `sub` | 2.076 | 0.054 | −2.02: freq/pres subtractions skipped (A2) |
| `bitwise_or`, `gt` | 0.541 | 0.005 | −0.54: `prompt_mask | output_mask`, `logits > 0` |
| `argmax` | 0.715 (2 calls) | 0.384 (1 call) | **−0.33**: greedy argmax skipped (A3) |
| `div` + `div_` | 2.504 | 1.664 | −0.84: the dense `logits / rep` is gone; the temperature divide is unchanged |
| `topk`, `_softmax`, `sum`, `masked_fill_`, `max`, `exponential_` | 5.109 | 5.116 | unchanged: same fp32 work |

**Two plan predictions for `smp_pick` were too high:**

* **A3, skip the greedy argmax: predicted −0.76, measured −0.33.** `main` runs two `argmax` per
  step, but only one belongs to the greedy path. The other is inside `torch.multinomial`: for one
  sample it computes `argmax(probs / exponential_noise)`. That explains the one `exponential_`, one
  `div` and one `argmax` still left in the candidate. The plan counted both argmaxes.
* **C1, temperature in place: predicted −0.30, measured ≈ 0.** An fp32 `[512, 151 936]` divide
  reads 311 MB and writes 311 MB whether or not it is in place. `div_` saves an allocation, not
  memory traffic. The temperature divide costs 0.66 ms either way.

---

## 4. What fp32 costs compared with bf16

Device work carries across hosts in this setup (§3.3), so log1012's bf16 `smp_opt` arm can be
compared directly with this fp32 candidate:

| batch 512, ms/step | bf16 (log1012 `smp_opt`) | fp32 (log1013 candidate) | fp32 cost |
| --- | --- | --- | --- |
| `smp_run_gpu` | 4.033 | 9.127 / 9.129 | **+5.09** |
| `smp_pen_gpu` | 0.090 | 0.093 / 0.094 | — |
| `smp_pick_gpu` | 3.938 | 8.523 / 8.525 | +4.59 |
| entry cast (outside the sub-spans) | 0.005 | 0.511 / 0.510 | +0.51 |
| `gpu_busy_from_trace` | 15 192 µs | 20 307 / 20 385 µs | +5.1 to +5.2 |

Breaking the fp32 cost down by op (host 1, per step):

| op | bf16 | fp32 | Δ |
| --- | --- | --- | --- |
| `topk` | 0.719 | 1.859 | +1.14 |
| `copy_` (the entry cast, among other copies) | 0.686 | 1.527 | +0.84 |
| `_softmax` | 0.429 | 1.256 | +0.83 |
| `div` (inside `multinomial`) | 0.497 | 1.001 | +0.50 |
| `sum` | 0.352 | 0.694 | +0.34 |
| `masked_fill_` | 0.380 | 0.675 | +0.30 |
| `div_` (temperature) | 0.379 | 0.663 | +0.28 |
| `fill_` (`full_like(-inf)`) | 0.571 | 0.735 | +0.16 |
| `argmax` (inside `multinomial`) | 0.222 | 0.384 | +0.16 |
| `max`, `exponential_` | 0.325 | 0.632 | +0.31 |
| **total** | | | **+4.86** |

**Correction to log1012 §6.** That section listed six ops and estimated ~2.9 ms. It left out:

* the cast itself;
* `multinomial`'s internal `div` and `argmax`;
* the temperature `div_`;
* the `full_like` fill.

It then concluded that restoring fp32 would cost "≈ 0 ms of wall time" at batch 512 on the 3900X,
because that host had 4.8 ms of device idle. The measured cost is 5.1 ms, so on that host the fp32
step would have been marginally GPU-bound.

**On host 2 the cost is fully visible.** At batch 512 host 2 is GPU-bound at 3.4 % idle, and its host
floor is far below 15 ms. In bf16 its step would be about 15.2 ms of device time plus a small
residual, roughly 15.9 ms of wall time, against the 21.09 ms measured in fp32. So on this host fp32
costs about **a quarter of the batch-512 decode step**. This is a projection from device time that
carries across hosts. bf16 was not run on host 2.

None of this argues for going back to bf16. The plan's numerical argument still stands. Most of the
fp32 cost is in the `[B, V]` sampling tail: `topk`, softmax, `multinomial`, the temperature divide
and the `-inf` fill. That is exactly the work C2–C5 removes by moving into the `[B, max_k]` window,
which keeps fp32 and drops most of its cost.

---

## 5. Why the same saving pays so differently

A decode step's wall time is roughly `max(host time, device time)`:

| batch 512 | host 1 | host 2 |
| --- | --- | --- |
| host floor (wall time at batch 128–256 while still host-bound) | **~25–26 ms** | **~10–11 ms** |
| device step, `main` → candidate | 26.9 → 20.3 ms | 27.0 → 20.4 ms |
| bound, `main` → candidate | device → **host** | device → device |
| wall saved / device saved | 1.9 / 6.6 ms | **6.6 / 6.6 ms** |

* **Host 1 flips from device-bound to host-bound** partway through the saving, so most of it turns
  into idle time (23.9 %).
* **Host 2 stays device-bound**, so all of the saving reaches wall time, and batch 256 joins it.
  On host 2 `main` was already GPU-bound at batch 256 (5.5 % idle), and the candidate cut that step
  from 15.79 to 12.45 ms.

**Batch 1024 is decode-identical across hosts.** The candidate's sweep decode step is 37.67 ms on
both. The 1 121 tok/s gap between hosts at that batch is entirely prefill: 104.2 s against 93.4 s,
which is the 260 W cap.

What this means for priorities:

* **Fast hosts (host 2 class):** the GPU is the bottleneck from batch 256 up. The remaining sampler
  work pays in full: C2–C5 against the 8.5 ms fp32 tail, and D1/D2 against the 0.6 ms in
  `smp_prep_gpu`.
* **Slow hosts (host 1 class):** the host floor caps everything at batch ≤ 512. Host time per step is
  the lever there, starting with decode CUDA graphs for `fwd`.

---

## 6. Cross-host comparison, and the interpreter loop

| host | arch | loop (ms) | batch-1 `fwd` (ms, `main`) | `fwd` / loop | batch-1 tok/s |
| --- | --- | --- | --- | --- | --- |
| log1011 EPYC 7452 | Zen 2 | 263–271 | 19.15 | 0.072 | 44 |
| log1013 host 1, EPYC 7K62 | Zen 2 | 271–289 | 19.07 | 0.068 | 44 |
| log1012 Ryzen 3900X | Zen 2 | 197–210 | 14.31 | 0.070 | 58 |
| **log1013 host 2, Ryzen 7950X3D** | **Zen 4** | **148–155** | **8.35** | **0.055** | **99** |

Within Zen 2, the 3 M-iteration loop predicts the host cost of this engine well: `fwd`/loop stays at
0.068–0.072 across a 1.4× range of speeds. **It does not carry to Zen 4.** The loop says the 7950X3D
is 1.35× faster than the 3900X, but `fwd` is 1.71× faster.

`fwd` at batch 1 is dominated by PyTorch's dispatcher and launch path, which is a larger working set
than a tight bytecode loop. Zen 4's higher IPC and the X3D's 128 MB L3 plausibly help that more than
they help the loop. This report cannot separate the two.

The loop is still a useful gate for ruling out slow hosts. It is not a substitute for measuring
`fwd` itself. The fastest proxy is batch-1 TPOT from a single sweep point: 10.1 ms on host 2 against
22.8 on host 1.

---

## 7. Caveats

1. **One sweep per arm per host.** The extra batch-512 run in each pytest session repeats only that
   point: 0.9–2.0 %.
2. **The arms were not interleaved.**
   * On host 1, the candidate profiled first but `main` swept first.
   * On host 2, the candidate ran first throughout.

   Run-to-run drift is about 1 % on host 1 and under 0.5 % on host 2.
3. **On host 1 both arms ran at the 260 W limit, and the candidate at a ~5 % lower SM clock.** Decode
   device time is unaffected, but the prefill share of host 1's sweep is slightly biased against the
   candidate.
4. **Host 1's cgroup quota (23 of 96 CPUs) and a neighbour at 99.5 % on one CPU** were not measured
   during the runs. Host 2 had recent tenant load before the run (load average 12.4 over 15 min), but
   was quiet at the host check.
5. **§4's bf16 comparison and host-2 projection cross hosts.** They rely on device time carrying
   across hosts, which three hosts now support to within 0.1 %. bf16 itself was never run on host 1
   or host 2.
6. **The op attribution in §3.4 and §4 uses aten totals for the whole step.** At batch 512 the
   sampler's `[B, V]` calls dominate the listed ops, but small model-side calls are included.

---

## 8. What follows

1. **Merge the candidate.**
   * The fp32 cast is explicit and does not depend on which penalties are on, which is the property
     the plan asked for.
   * Batch ≤ 128 is unchanged on both hosts, and batch ≥ 256 is faster on both.
   * Before merging, run the plan's fixed-seed token-for-token check against `main`. Both arms
     compute in fp32 and draw from the RNG in the same order, so they should match.
2. **Do C2–C5 next.** On a host-2-class machine they now carry the biggest sampler gain: the
   8.5 ms fp32 tail at batch 512, on a step that is GPU-bound. Update the plan with this report's
   corrections:
   * A3 is worth −0.33, not −0.76;
   * C1 is worth ≈ 0;
   * `multinomial`'s internal `div`/`argmax`/`exponential_` (~1.7 ms in fp32) is what C5's Gumbel-max
     replaces.
3. **Land D1/D2.** `smp_prep_gpu` is still 0.60 ms at batch 512 on both hosts. On host 2 that is
   wall time.
4. **Decode CUDA graphs for `fwd`** remain the lever for host-1-class machines and for every host at
   batch ≤ 128.
5. **Correct log1012 §6.** The fp32 cost is 5.1 ms, not 2.9 ms, and it does not fit within the
   3900X's idle time. Point to §4 here.
6. **Rental gates.**
   * Check `enforced.power.limit`: 260 W costs about 13 % on prefill against 450 W.
   * Treat the interpreter loop as a floor check, not a predictor, across CPU generations (§6).
   * Prefer host-2-class machines for sampler and model work. Only on those does a device-side change
     show up fully in throughput.
