# Performance analysis — qwen.py vs vLLM 0.30.0 on RTX 4090 (log1003)

> 中文版：[`performance_analysis_log1003.zh.md`](./performance_analysis_log1003.zh.md)

**The first external yardstick.** Every report before this one
([log910](./performance_analysis_log910.md) … [log1001](./performance_analysis_log1001.md)) measured
qwen.py against itself — branch versus branch, one commit at a time. This one measures it against
vLLM on the same host, in the same window, on the workload
`test_benchmark_sweep_batch_size` already defines.

**Data sources**

| Content | Path |
| --- | --- |
| Concurrency sweep, qwen.py, runs 1–2 | `log_vast/log1003/benchmark_fused_top_kp{,2}/` |
| Pure-decode idle-fraction profile, qwen.py, runs 1–2 | `log_vast/log1003/profile_fused_top_kp{,2}/` |
| vLLM arm A (no compile, no graphs) | `log_vast/log1003/benchmark_vllm/vllm_eager.{jsonl,log}` |
| vLLM arm B (inductor + CUDA graphs) | `log_vast/log1003/benchmark_vllm/vllm_cudagraph.{jsonl,log}` |
| GPU clock / power / throttle traces, 200 ms | `log_vast/log1003/**/gpu.4090.*.csv` |
| Host inventory, CPU, affinity, interpreter speed | `log_vast/log1003/host_info`, `host_cpu_info`, `gpu.static.csv` |
| Dependency snapshots | `log_vast/log1003/constraints.txt` (main venv), `vllm_pip_freeze.txt` (vLLM venv) |
| The previous (same-branch, faster host) baseline | `log_vast/log1001/` |

**Code under test** — qwen.py at the commit that follows `49bb10d` (fused top-k+top-p, plus the
`freq_pen`→`frequency_penalty` rename and the `o_tok_throughput`/`io_tok_throughput` additions to
`analyze_metrics`). Provenance is checkable from the dumps themselves: log1001's `pytest.log` records
`freq_pen=0.0` while log1003's records `frequency_penalty=0.0`, and log1003's `benchmark_metrics`
carries the two new throughput keys. vLLM at **0.30.0** (torch 2.13.0+cu130, CUDA 13.0, Python
3.12.3), installed in an isolated venv per
[`vast-evn-build.md` §6a](../env/vastai/vast-evn-build.md).

**The harness** — [`benchmark/tool/vllm_sweep.py`](../benchmark/tool/vllm_sweep.py), which mirrors
`test_benchmark_sweep_batch_size` field by field: offline closed loop (every request enqueued before
the timer starts), fixed 512-in / 128-out, `ignore_eos`, `max_num_seqs` as the sweep variable, and the
sampling parameters copied item for item from `generation_config.json`. §2.2 explains why each of
those is load-bearing.

**Run order** (one instance, ~50 minutes, no restart, one engine at a time): vLLM arm A 03:51–04:30 →
arm B 04:33–05:00 → qwen.py sweep run 1 05:02–05:14 → run 2 05:17–05:29 → qwen.py profile run 1
05:33 → run 2 05:38. **vLLM ran first**, on the colder machine.

---

## Executive summary

1. **At batch 512, where both engines are device-bound, qwen.py delivers 10 970 tok/s against
   vLLM's 13 505 (no compile, no graphs) and 15 848 (compiled + graphed) — 81 % and 69 %.**
   Peak to peak it is **10 970 vs 18 032 tok/s = 61 %**.
2. **The gap factorises cleanly at batch 512: 1.23 × 1.17 = 1.44.** The 1.23 is engine design, measured
   like-for-like (both eager, both `FLASH_ATTN`, identical KV capacity, identical sampling params). The
   1.17 is what `torch.compile` + CUDA graphs buy vLLM — a capability qwen.py does not have at all.
   **`enforce_eager` does not disable vLLM's sampler kernels** (FlashInfer + five Triton kernels, in
   both arms), so the whole of its sampler advantage sits inside the 1.23, not the 1.17 (§2.1).
3. **That second factor is enormous at low concurrency: 7.28× at batch 1**, decaying monotonically
   through 3.58× at batch 64, 1.72× at 256, 1.17× at 512, and 0.98× at 1024. This is the first external
   price tag on the per-step host tax [log1001 §7.3](./performance_analysis_log1001.md) identified as
   ~4 400 `aten::` calls per step.
4. **The engine-design factor is remarkably flat at 1.59–1.65× for every batch ≤ 64**, then closes to
   1.23× at 512. Both engines are host-bound there and paying the *same* interpreter, so this ratio is
   a measurement of relative host-loop efficiency — and unlike the absolute numbers, **a ratio of two
   host-bound loops transfers across hosts** (§2.4).
5. **This host's Python is ~2.3× slower than log1001's, and the data proves it is only the host.**
   Same code, same branch: `gpu_busy_from_trace` agrees with log1001 **within ±2 % at every batch**,
   while `wall_clean` is 2.26–2.38× higher at batch ≤ 128 and converges to 1.01× at 512. Cause:
   `CPU max MHz 2800` × `scaling 80 %` ≈ **2.24 GHz**, held down by neighbour load (`load average
   12.11` measured 12 minutes *after* our last run ended, with no users logged in).
6. **So read the two halves of the sweep differently.** Batch 512/1024 is a measurement and transfers.
   Batch ≤ 256 absolute throughput does not transfer; the *ratios* in point 4 do.
7. **The device-side picture from log1001 is confirmed, not disturbed.** The sampler is still
   **61 % of qwen.py's device work at batch 512** (16.32 of 26.93 ms/step) — reproducing log1001's
   60 % on a different host. [log1001 §7.1–7.2](./performance_analysis_log1001.md) remains the single
   largest identified win, and it is worth ~1.8× on device time.
8. **The power cap biases *against* vLLM, so the measured gap is a floor.** The 400 W limit was Active
   for **64.7–68.6 %** of vLLM's busy samples versus **42.3–42.6 %** of qwen.py's (median 394–397 W
   vs 356–359 W; median SM clock 2700 vs 2775 MHz). On a 450 W host vLLM gains more than qwen.py.
9. **Reproducibility.** qwen.py run-to-run: worst 2.87 % (batch 16), < 0.5 % at batch ≥ 256, identical
   step counts, zero preemptions across 22 runs. vLLM: 3 runs per point, median reported. Device-side
   `sample_gpu` agrees to **0.03 %** between qwen.py runs.

---

## 1. System under test

### 1.1 Hardware — and why it is the main thing to know about this report

| | log1003 (this report) | log1001 (previous) |
| --- | --- | --- |
| GPU | RTX 4090, driver **580.142** | RTX 4090, driver 550.127.08 |
| Enforced power limit | **400 W** (max 600) | 450 W (max 600) |
| HBM, measured | **919 GB/s** | 876.8 GB/s (vendor panel) |
| CPU | **AMD EPYC 7402**, 24C/48T, `max 2800 MHz`, `scaling 80 %` → **≈ 2.24 GHz** | Ryzen 5 7500F, 6C/12T, 3.7/5.0 GHz |
| L3 | **128 MiB in 8 instances** (8 CCX × 3 cores × 16 MiB) | 32 MiB, 1 CCD |
| CPU allocation | CFS quota `cfs_quota_us 1152000` = **11.52 cores**; affinity **all 48**; 1 NUMA node | 12/12 |
| Neighbour load | **`load average 12.11 / 12.81 / 12.96`**, taken 12 min after our runs ended, 0 users | not recorded |
| 3 M-iteration Python loop | **276 / 275 / 285 ms** | not recorded |
| `aten` launch overhead, median | 7.56 µs (passes the §1b < 10 µs gate) | not recorded |
| PCIe | gen 4 × 16 | gen 4 × 16 |

The GPU is healthy: the vendor panel's 387.5 GB/s claim was simply wrong, and the measured 919 GB/s is
91 % of the card's theoretical 1 008. No thermal throttling anywhere (≤ 60 °C,
`hw_thermal_slowdown` never Active).

The **host** is the problem, and §2.4 is devoted to it. Note what the §1b gate did *not* catch: launch
overhead passed comfortably at 7.56 µs, because that microbenchmark times one op in a tight loop,
while the engine pays Python dispatch on ~4 400 `aten::` calls per step. The pure-interpreter loop is
the measurement that would have caught it, and it has now been added to the runbook.

### 1.2 Workload — identical by construction

Qwen2.5-0.5B-Instruct, bf16, `vocab_size = 151 936`. From `test_benchmark_sweep_batch_size`:

* `10 × batch` requests (minimum 64), **fixed 512 input / 128 output tokens**, `max_model_len = 1024`.
* `ignore_eos` — every request produces exactly 128 tokens. Asserted on both sides: qwen.py's
  `o_tok_num` and the harness's `assert o_tok == req_num * OUT_LEN`.
* Random uniform token ids from `SEED = 1234`. Content carries no data dependence (output length is
  pinned, ids are uniform), so prompts were regenerated from the seed rather than transported; arm B's
  header records `prompts_sha256_16 = ffc9f6a0ecf792c5` as proof of the set.
* **Offline closed loop**: all requests enqueued *before* `t0`, then drained. No HTTP, no arrival rate,
  no tokenizer in the loop. `engine.benchmark()` and `LLM.generate()` are timed at the same boundary.

### 1.3 What was aligned, and verified from the logs

| Knob | qwen.py | vLLM | Verified |
| --- | --- | --- | --- |
| KV capacity | `num_blocks 4096 × block_size 256` = **1 048 576 tokens** (12.0 GiB) | `block_size 16`, `num_gpu_blocks_override 65536` | vLLM log: `GPU KV cache size: 1,048,576 tokens`. The override **reduced** vLLM from its own 110 679 blocks (1.77 M tokens), so it received no extra room |
| Attention kernel | flash-attn 2.8.3 | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` | both arms, 11/11 points — **kernel family is not a confound** |
| Sampling | temp 0.7, top_k 20, top_p 0.8, rep_pen 1.1, freq/pres 0 | same, item for item | §2.2 |
| Prefill batching | `max_num_batched_tokens 8192`, `long_prefill_token_threshold 8192` | `max_num_batched_tokens 8192`, chunked prefill on | `prefill_chunk.mean = 1.0` on the qwen.py side |
| Prefix caching | not implemented | `enable_prefix_caching=False` | vLLM non-default args |
| Preemption | 0 across 22 runs | 0; KV headroom logged as `1024.00x` concurrency | both |
| `OMP_NUM_THREADS` | 8 | 8 (**inherited** from `~/.bashrc`) | recorded; see §5.4 |
| CPU binding | none | none | same on both arms |

---

## 2. Methodology

### 2.1 Three arms, and what `enforce_eager` actually switches

| Arm | Configuration | Attributes |
| --- | --- | --- |
| qwen.py | the engine under development | — |
| **A** — `enforce_eager=True` | vLLM log: `'mode': <CompilationMode.NONE: 0>`, zero `Capturing CUDA graph` lines | **engine design**: scheduler, KV management, sampler, kernel selection |
| **B** — default | `'mode': <CompilationMode.VLLM_COMPILE: 3>`, backend `inductor`, `Dynamo bytecode transform time: 3.89 s`, CUDA graphs captured | A→B = **`torch.compile` fusion *and* CUDA graphs, together** |

**The A→B delta is not "CUDA graphs" alone.** `enforce_eager=True` turns off inductor compilation *and*
graph capture, so the two effects are inseparable in this data. Earlier framing in this project called
this "the CUDA-graph arm"; the logs say it is compile + graphs. Splitting them needs a third arm
(compile on, `cudagraph_mode=NONE`) — §6.4.

**And `enforce_eager` does not touch vLLM's sampler.** Both arms log, once per batch point:

```
[topk_topp_sampler.py:78] Using FlashInfer for top-p & top-k sampling.
[jit_monitor.py:141] Triton kernel JIT compilation during inference: _topk_topp_kernel
                                                                     _topp_sb_stats_kernel
                                                                     _topp_sb_step_kernel
                                                                     _topp_sb_mask_kernel
                                                                     _gumbel_sample_kernel
```

So vLLM samples through custom kernels in arm A as well as arm B, and **none of its sampler advantage
is inside the 1.17×** — all of it sits in the 1.23× that §3.2 labels engine design. §4 reads that
contributor accordingly.

The JIT lines land after `init engine … took 94.65 s` and inside the first `generate()`, which is the
harness's warm-up call, so the compilation spike is outside the timed runs. Arm B's slow first runs at
batches 128/256/512 (§2.3) are more likely inductor and capture warm-up than Triton JIT.

### 2.2 Throughput definition, and the three ways to get this comparison wrong

**Throughput is output tokens per second**, `o_tok_num / elapsed`, on both sides. This is the metric
every report in this series has used since log910, and it is also what vLLM calls *Output token
throughput*. At this workload's fixed 512-in / 128-out shape, the "total token throughput" that
vLLM's own harnesses print is **exactly 5.0× larger** for every single request, so it is a constant
multiple carrying no information — and quoting it against qwen.py's number would manufacture a 5×
gap. `analyze_metrics` now emits `o_tok_throughput` and `io_tok_throughput` explicitly so the
distinction cannot be lost again.

The other two traps, both avoided here:

* **Sampling parameters are the subject under test, not an incidental knob.** After log1001 the
  sampler is 61 % of qwen.py's device work at batch 512. Running vLLM at `temperature=0` would reduce
  its sampler to a single full-vocab `argmax` (0.71 ms/step in qwen.py's own trace) against qwen.py's
  16.32 ms — a ~2.1× phantom gap with no code difference in it. Worse, vLLM takes a cheaper code path
  when **no** request needs penalties/top-k/top-p, so *partial* alignment is worse than none: it looks
  aligned while the two engines do different work.
* **`detokenize=False`.** vLLM detokenizes incrementally by default; qwen.py never detokenizes inside
  the loop (`save_output=False` for this test).

### 2.3 Noise floor

| | value |
| --- | --- |
| qwen.py run-to-run throughput, worst | **2.87 %** (batch 16) |
| qwen.py run-to-run, batch ≥ 256 | < 0.5 % (batch 512: 0.42 %, batch 1024: 0.05 %) |
| qwen.py step counts, run 1 vs run 2 | identical at every batch (e.g. 1 324 at batch 512) |
| qwen.py device `sample_gpu` at batch 512, run-to-run | **0.03 %** |
| qwen.py host fields at batch 512, run-to-run | all within 4.6 % |
| vLLM | 3 runs per point, median reported, all three in `elapsed_all` |
| vLLM arm B spread at batch 512 | **11.6 %** (38.997 / 41.352 / 43.517 s) — this point carries ≈ ±6 % |
| `aten` launch overhead spread | 6.5 %, vs the runbook's ±3 % reference |

Arm B shows a slow first run at batches 128, 256 and 512 (12.487 vs 9.857/9.687; 20.878 vs
18.173/18.072) — graph capture and inductor warm-up leaking into the first timed iteration despite the
warm-up call. The median absorbs it, but batch 512's ±6 % is the loosest point in the report. The 1.44×
gap there is ~7× that uncertainty.

### 2.4 The host confound, and exactly how far it reaches

Same code, same branch, same GPU model, different host:

| batch | `gpu_busy` log1001 → log1003 | `wall_clean` log1001 → log1003 | throughput ratio |
| --- | --- | --- | --- |
| 1 | 3 146 → 3 184 µs (**+1.2 %**) | 9 674 → 22 733 µs (**2.35×**) | 0.42 |
| 8 | 3 903 → 3 960 (+1.5 %) | 10 613 → 23 875 (2.25×) | 0.44 |
| 32 | 4 360 → 4 421 (+1.4 %) | 10 727 → 25 524 (2.38×) | 0.45 |
| 64 | 5 398 → 5 445 (+0.9 %) | 10 963 → 24 752 (2.26×) | 0.46 |
| 128 | 8 346 → 8 396 (+0.6 %) | 11 100 → 25 399 (2.29×) | 0.50 |
| 256 | 15 149 → 14 857 (−1.9 %) | 16 040 → 26 013 (1.62×) | 0.72 |
| 512 | 27 305 → 26 930 (−1.4 %) | 28 092 → 28 354 (**1.01×**) | **1.00** |

The device does identical work to within ±2 % at every batch; the host is 2.3× slower until the device
becomes the long pole at 512. A field-by-field split of the host side confirms it is uniform, not
localised: `sched` 1.72×, `sched_run` 1.97×, `bld_meta` 2.10×, `sample` 2.35×, `dth` 1.97×,
`dth_wait` 1.90×, `ci` 1.97×. **Every unrelated pure-CPU field scales by the same ~2× — that is a
slower interpreter, not a bottleneck.** The 3 M-iteration Python loop (276/275/285 ms, only 3.6 %
spread) says the same thing and says it is clock-limited rather than contention-jittered.

Three consequences, and they are different for different parts of the report:

1. **Device-side conclusions transfer.** ±2 % agreement across hosts is as good as this project's
   instrument gets.
2. **Absolute throughput at batch ≤ 256 does not transfer.** Both engines are depressed by the same
   slow interpreter.
3. **Ratios between two host-bound loops *do* transfer.** If qwen.py's loop issues `N_q` Python
   operations per step and vLLM's issues `N_v`, both at interpreter rate `R`, the step times are
   `N_q/R` and `N_v/R` and the ratio `N_q/N_v` is independent of `R`. At batch 1 qwen.py is 86 % GPU-idle
   and vLLM arm A is ~77 % idle, so both qualify. This is why §3.2's *engine factor* column is usable
   even where the absolute columns are not.

> **One thing this host could not measure at all**: qwen.py's small-batch absolute numbers on a fast
> host, against vLLM measured in the same window. That needs a different box, not more time on this
> one. §6.1.

---

## 3. Results

### 3.1 The three-way sweep — one host, one window, one engine at a time

qwen.py is the mean of two runs; vLLM is the median of three per point. All figures are **output
tokens/s**. (`concurrency_sweep.txt` prints run 1 alone, so its batch-512 peak reads 10 993 where the
mean of two runs is 10 970; the 81 % / 61 % fractions are the same either way.)

| batch | qwen.py | vLLM A (eager) | **A/qwen** | vLLM B (compiled+graphed) | **B/qwen** | **B/A** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 43 | 71 | **1.64×** | 516 | 11.9× | **7.28×** |
| 2 | 82 | 134 | 1.64× | 900 | 11.0× | 6.70× |
| 4 | 162 | 267 | 1.65× | 1 781 | 11.0× | 6.66× |
| 8 | 324 | 535 | 1.65× | 3 318 | 10.3× | 6.20× |
| 16 | 632 | 1 014 | 1.60× | 5 796 | 9.2× | 5.72× |
| 32 | 1 218 | 1 958 | 1.61× | 9 193 | 7.5× | 4.70× |
| 64 | 2 375 | 3 773 | 1.59× | 13 518 | 5.7× | 3.58× |
| 128 | 4 392 | 6 502 | 1.48× | 16 622 | 3.8× | 2.56× |
| 256 | 7 672 | 10 470 | 1.36× | **18 032** | 2.4× | 1.72× |
| **512** | **10 970** | **13 505** | **1.23×** | 15 848 | **1.44×** | **1.17×** |
| 1024 | 9 622 | 13 038 | 1.36× | 12 841 | 1.33× | 0.98× |

| | qwen.py | vLLM A | vLLM B |
| --- | --- | --- | --- |
| Peak throughput | **10 970 @ batch 512** | 13 505 @ 512 | **18 032 @ batch 256** |
| qwen.py as a fraction of peak | — | 81 % | **61 %** |

Three shapes worth reading off the table:

* **qwen.py and vLLM A peak at the same place (512) and then both fall off at 1024** — the workload's
  KV capacity is exactly 1 024 × 1 024 tokens, so batch 1024 is the point where nothing is spare.
* **vLLM B peaks two doublings earlier, at 256**, and *loses* 12 % by 512 and 29 % by 1024. Compiled +
  graphed execution reaches the device roof sooner, so the useful operating range moves down.
* **B/A crosses 1.0 at batch 1024** (0.98×): once the step is long enough, compilation and graphs buy
  nothing and the capture overhead is marginally negative.

### 3.2 The gap factorises

At batch 512, where both engines are device-bound and the host confound has vanished (§2.4):

```
   vLLM B / qwen.py  =  1.44×
                     =  1.23×        ×  1.17×
                        engine design   compile + CUDA graphs
                        (A/qwen)        (B/A)
```

The 1.23× is the honest like-for-like engine comparison: both eager, both `FLASH_ATTN`, identical KV
capacity, identical sampling parameters, same host, same window. **qwen.py's engine is within 23 % of
vLLM's at the batch size where the GPU is the bottleneck** — which is the regime the engine was built
for.

The 1.17× is a capability gap, not a design gap: qwen.py has no `torch.compile` path and no CUDA graph
capture, so there is nothing in it to compare.

Across the sweep the two factors behave completely differently:

| | batch 1 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| engine factor (A/qwen) | 1.64× | 1.59× | 1.48× | 1.36× | **1.23×** | 1.36× |
| compile+graphs factor (B/A) | **7.28×** | 3.58× | 2.56× | 1.72× | 1.17× | 0.98× |

The engine factor is **flat at 1.59–1.65× for every batch ≤ 64** and improves monotonically from there.
Per §2.4's third consequence, that flat 1.6× is a ratio of two host-bound loops on a shared
interpreter, so it transfers: **qwen.py's host-side step costs roughly 1.6× what vLLM's eager
host-side step costs**, on any host. In log1001 §7.3's terms, that is the price of ~4 400 `aten::`
calls per step against whatever vLLM's eager path issues.

The compile+graphs factor is where the 12× headline at batch 1 comes from, and it is almost entirely
the host path: it decays to nothing exactly as the device becomes the bottleneck.

### 3.3 qwen.py's own device side — log1001 reproduced on a different host

Pure decode, from `profile_fused_top_kp{,2}`:

| batch | `gpu_busy` ms/step | GPU idle | `sample_gpu` ms/step | sampler share | `wall_clean` ms/step |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.18 | **86.0 %** | 1.07 † | 34 % † | 22.73 |
| 8 | 3.96 | 83.4 % | 1.03 † | 26 % † | 23.87 |
| 32 | 4.42 | 82.7 % | 1.16 † | 26 % † | 25.52 |
| 64 | 5.45 | 78.0 % | 1.39 | 26 % | 24.75 |
| 128 | 8.40 | 66.9 % | 3.50 | 42 % | 25.40 |
| 256 | 14.86 | 42.9 % | 8.29 | 56 % | 26.01 |
| 512 | **26.93** | 5.0 % | **16.32** | **61 %** | 28.35 |

† `sample_gpu` is a CUDA-event span, not a kernel sum, so at high idle it contains host bubbles
([log1001 §2.2](./performance_analysis_log1001.md)). Below batch 64 these cells are not kernel time —
compare against log1001's values at the same batch (0.31 / 0.38 / 0.65 ms), which were measured at
50–67 % idle rather than 83–86 %. From batch 64 up the two hosts agree: 1.37→1.39, 3.53→3.50,
8.31→8.29, 16.35→16.32 ms.

**The sampler is 61 % of device work at batch 512, reproducing log1001's 60 % on different silicon
and a different driver.** So [log1001 §7.1–7.2](./performance_analysis_log1001.md) is unaffected by
anything in this report and remains the largest identified win: moving the sampler's tail inside the
top-k window plus making penalties sparse should take 16.32 ms/step down to ~4 ms, i.e. a device step
of ~15 ms against today's 26.9 — about **1.8× on the device side**.

That matters for the gap: 1.8× on device time is larger than the entire 1.44× gap to vLLM B at
batch 512. The sampler work is not a secondary cleanup; it is the single biggest lever qwen.py has.

### 3.4 Clocks and power — the bias runs against vLLM

From the 200 ms `nvidia-smi` traces, over samples with `utilization.gpu > 50 %`:

| run | samples | `sw_power_cap` Active | median SM | median power | max temp |
| --- | --- | --- | --- | --- | --- |
| vLLM arm A | 5 065 | **68.6 %** | 2 700 MHz | 397.3 W | 60 °C |
| vLLM arm B | 5 792 | **64.7 %** | 2 700 MHz | 394.6 W | 59 °C |
| qwen.py run 1 | 3 008 | 42.6 % | 2 775 MHz | 359.0 W | 58 °C |
| qwen.py run 2 | 3 112 | 42.3 % | 2 775 MHz | 355.8 W | 59 °C |

This host's 400 W cap (log1001's was 450 W) binds on both engines, but **half again as often on
vLLM**, which runs ~40 W hotter and 75 MHz slower. The reason is visible in §3.3: qwen.py leaves the
GPU idle 5–86 % of the time and simply cannot draw as much power. The direction matters: **on a 450 W
host vLLM would gain more than qwen.py, so every gap figure in this report is a floor, not a ceiling.**
No thermal throttling on any run.

---

## 4. Where the gap comes from

Collecting what the data supports, at batch 512 unless stated:

**1. Compile + CUDA graphs — the whole of the low-batch gap, 1.17× of the high-batch gap.** B/A decays
7.28× → 1.17× → 0.98× as the device takes over. Nothing in qwen.py corresponds to it.

**2. qwen.py's host loop costs ~1.6× vLLM's eager host loop.** Flat across batch ≤ 64, and a ratio
that transfers across hosts (§2.4). This is the same quantity log1001 §7.3 measured from the inside as
~4 400 `aten::` calls and ~4.8 ms/step of unattributed dispatch.

**3. The device side is closer than the headline suggests, but not equal.** Arm A is device-bound only
at the top of the sweep, so the cleanest device-side read is batch 512's 1.23×. Two contributors are
identified:

* **The sampler, 16.32 of 26.93 ms/step (61 %).** With identical parameters, vLLM runs this through
  FlashInfer plus five JIT-compiled Triton kernels, **in both arms** (§2.1), against qwen.py's eager
  `aten::` ops. Its cost there is not measured (§5.5), so "most of the 1.23×" is not a claim this data
  supports — but the sampler is the largest single candidate, and the kernel difference is now
  evidenced rather than inferred.

  **The diagnosis is not "qwen.py should use Triton", and acting on that reading would be a mistake.**
  At batch 512 the logits are `[512, 151 936]` — **155.6 MB in bf16, 311.2 MB once cast to fp32**. At
  the measured 919 GB/s one full-vocabulary elementwise pass therefore costs **0.68 ms** read+write in
  fp32 and **0.34 ms** in bf16 or read-only. That unit is worth checking rather than trusting, and the
  two ops whose access pattern is unambiguous confirm it: the dead-row `sum` measures **0.346 ms/call**
  and `argmax` **0.357 ms/call** against 0.339 predicted for a read-only fp32 pass. On that scale the
  sampler's 16.32 ms is **roughly 24–48 full-vocabulary pass-equivalents**, the range being how much of
  it runs in bf16 rather than fp32.

  How much of that is attributable per op, and how much is not: `sampler()` runs **once per step**, on
  the `[B, vocab]` logits of the last position — not per layer — so `key_averages` aggregates its ops
  together with the model's 24 layers under the same names. Only the ops whose every call belongs to
  the sampler can be read off directly:

  | op | ms/step | calls/step | sampler's calls |
  | --- | --- | --- | --- |
  | `aten::multinomial` (CUDA total, incl. `exponential_` 0.31) | 2.735 | 1 | 1 |
  | `aten::topk` | 1.857 | 1 | 1 |
  | `aten::_softmax` (the window one + the full-vocab one) | 1.254 | 2 | 2 |
  | `aten::masked_fill_` | 0.672 | 3 | 3 |
  | `aten::scatter_` | 0.046 | 3 | 3 |
  | **attributable with certainty** | **6.56** | | **40 % of 16.32** |

  The other **9.76 ms** sits in op types the model also uses — `div` (3 calls/step, 2 the sampler's),
  `where` (9 / 4), `argmax` (2 / 1), `sum` (2 / 1), `fill_` (11 / 2), `copy_` (**191 / 1**) — and
  `key_averages` cannot split them. Per-call averaging would not rescue it either, because the calls
  differ in size by orders of magnitude: of `copy_`'s 191 calls the sampler makes the single
  311 MB `.float()` cast and the model makes 190 small ones. **An earlier draft of this section quoted
  those whole-step totals as if they were the sampler's; they are not.** The `sample_prep` /
  `sample_run` / `sample_post` split added after these measurements is the first step toward closing
  that gap, though even it resolves phases rather than ops.

  Three caveats on the pass-equivalent estimate, none of which weaken the conclusion: 919 GB/s came
  from a pure 1 GiB device-to-device copy, so it is a best case and each pass really costs a little
  more; `_softmax` (~1.8 passes) and `multinomial` (~4) are multi-pass rather than single; and `topk`'s
  1.86 ms is not an elementwise pass at all but a radix select running at **11× its 0.17 ms minimum
  read**, and it is **irreducible** — it must see all 151 936 columns.

  The argument does not actually need the pass count, which is why the loose range is fine. It needs
  only the operand size: **the live candidate set is `[512, 20]` = 41 kB, 7 600× smaller than the
  311 MB fp32 tensor these ops traverse.** A Triton rewrite that still touches `[512, 151 936]` at
  every step costs the same, because the price is bandwidth × traffic and a change of kernel language
  moves neither — unless it also fuses the steps, and the fusion that pays is precisely the one that
  moves them into the window. That fusion is plain PyTorch. log1001 §7.1–7.2's 16.32 → ~4 ms estimate
  assumes no Triton at all; the residue is then dominated by `topk`'s irreducible 1.86 ms. **That** is
  where a fused select-and-sample kernel would start to earn its keep — algorithm first, kernel
  language second.

  One idea worth stealing at the PyTorch level: `_gumbel_sample_kernel` says vLLM does not use
  `multinomial` at all. Gumbel-max — add `-log(-log(u))` to the logits and take the argmax — is one
  pass plus a reduction, against `torch.multinomial`'s normalise-then-search, which costs qwen.py
  **2.74 ms/step** here and would be near-free inside a 20-column window.
* **The forward pass is further from the bandwidth roof than vLLM's.** At batch 1 qwen.py's device
  work is **3.18 ms/step** (measured, `gpu_busy_from_trace`) while the weights alone are 0.99 GB —
  **1.08 ms at the measured 919 GB/s**. That puts qwen.py at ~34 % of the weight-bandwidth roof.
  For vLLM arm B the same quantity is not measured but can be bounded: 15.877 s / 64 requests =
  248 ms per request, which is one 512-token prefill plus 128 decode steps, so the decode step is
  **under 1.94 ms** even if prefill were free — i.e. **above 56 % of the roof**. Treat the vLLM half
  as a bound, not a measurement (§5.5). Part of the difference is inductor fusion, which arm A does not
  have, so this contributor also sits inside factor 1.

**4. What is *not* a contributor, and was checked:** attention kernel family (both `FLASH_ATTN`), KV
capacity (both 1 048 576 tokens, with vLLM's own larger allocation overridden *down* to parity),
prefix caching (off), preemption (zero on both), EOS handling (exact 128-token outputs asserted on
both sides), and GPU thermals.

The sampler implementation is deliberately **not** on that list. Sampling *parameters* were aligned
(§2.2); the kernels behind them are each engine's own choice and are part of what is being compared,
not a confound to be removed.

---

## 5. Threats to validity

### 5.1 Absolute small-batch numbers are host-specific

§2.4. Batch ≤ 256 throughput on this host is roughly 0.42–0.72× what the same qwen.py code produces on
a faster interpreter. The ratio columns in §3.2 are the transferable part; the absolute columns at
batch ≤ 256 are not. Any future splice of log1003's vLLM numbers onto log1001's qwen.py numbers would
be exactly the error [log1001 §8](./performance_analysis_log1001.md) warns about.

### 5.2 Arm A→B conflates compilation with graph capture

§2.1. `enforce_eager=True` disables both, so this report cannot say how much of the 7.28× at batch 1
is inductor fusion and how much is launch elimination. The decomposition in §3.2 is therefore
*engine design* × *(compile + graphs)*, and the second factor is a bundle.

### 5.3 vLLM arm B's batch-512 point is the loosest number here

11.6 % spread across three runs (§2.3), with the slow run first — warm-up leaking past the warm-up
call. Treat 15 848 tok/s as ±6 %. The 1.44× gap survives comfortably; a hypothetical 1.1× claim at
that batch would not.

### 5.4 Parity items that were matched but are not ideal

* **`block_size` 256 vs 16.** vLLM has no 256, so total KV capacity was matched instead. Block-table
  traversal and fragmentation behaviour differ.
* **`OMP_NUM_THREADS=8` on both**, inherited from `~/.bashrc` rather than chosen. It satisfies §6e's
  "both or neither" rule, but by accident rather than by decision; the runbook now says so explicitly.
* **vLLM V1 runs its EngineCore in a separate process**, so it uses an extra core and `/dev/shm`. That
  is vLLM as shipped, and qwen.py has no equivalent, but it is an asymmetry.
* **No CPU binding on either side.** With 48 CPUs visible, one NUMA node and **8 separate L3
  instances**, the single hot Python thread was free to migrate across CCXs all run. This hurts both
  engines, and it hurts the eager paths most.

### 5.5 Not measured

* **TTFT / prefill latency for vLLM.** The harness records only elapsed time and output token counts,
  so this report has no latency comparison — only throughput. qwen.py's own prefill latency is in
  `concurrency_sweep.txt` (23.8 ms at batch 1 rising to 137.5 ms at 512) with nothing to compare it to.
* **vLLM's internal step breakdown.** There is no vLLM equivalent of `step_metrics` here, so the
  device-side attribution in §4.3 is inferred from totals and the roofline, not measured per op. The
  engine log names vLLM's sampler kernels (§2.1) but not what they cost, so the sampler's share of
  *its* step is unknown — which is exactly why §4.3 stops at "largest single candidate".
* **A per-op breakdown of qwen.py's *own* sampler.** `sampler()` is called once per step, so
  `key_averages` lumps its ops in with the 24 model layers under the same op names; only 6.56 of the
  16.32 ms is attributable with certainty (§4.3). Closing this needs NVTX ranges around the sampler's
  phases, or the `sample_prep`/`sample_run`/`sample_post` events added after this report's runs.
* **Sampling output quality.** Parameters were matched; distributions were not compared. Fine for a
  throughput comparison, and not claimed beyond that.

---

## 6. Recommendations, in order

### 6.1 Do not re-rent to chase the small-batch numbers — fix the sampler first

The transferable results say qwen.py is at 81 % of vLLM arm A and 69 % of arm B at batch 512, and that
the identified sampler work (log1001 §7.1–7.2) is worth ~1.8× on device time there — **more than the
entire gap**. Spending the next rental on a faster host to re-measure what is already bounded buys
less than spending the next session on `sample()`.

**Do the algorithm, not the kernel language.** Seeing FlashInfer and Triton in vLLM's log invites the
conclusion that qwen.py needs Triton kernels; §4.3 is the arithmetic for why that is backwards. The
~4× available here comes from operating on 20 columns instead of 151 936, which is plain PyTorch;
a Triton kernel that still makes ~24 full-vocabulary passes costs the same ~16 ms. Revisit kernel
language once the sampler is ~4 ms/step and `topk`'s irreducible 1.86 ms dominates it.

### 6.2 Then add a compiled / graphed path, because that is the other half

Arm A→B is 7.28× at batch 1 and 1.17× at 512 on an engine that is otherwise the same. qwen.py's
decode step is static in shape (fixed `max_num_seqs` slots, resident parameter tables, no dynamic
control flow in the hot path), which is the easy case for graph capture. Sequence it *after* 6.1:
capturing a step that spends 61 % of its device time in an avoidable sampler bakes that in.

### 6.3 Re-measure on a host that passes the new §1b gate

When 6.1 and 6.2 land, the comparison needs a host where the 3 M-iteration Python loop is near the
reference rather than 276 ms, so that the batch ≤ 256 absolute numbers become measurements instead of
ratios. Prefer a high-clock consumer part (Ryzen 7000/9000) over a shared EPYC, and bind to one CCX
([§3 of the runbook](../env/vastai/vast-evn-build.md)).

### 6.4 Add a third vLLM arm to split compilation from graph capture

Arm A′: compilation on, `cudagraph_mode=NONE`. A→A′ is inductor fusion; A′→B is launch elimination.
That tells you which of 6.2's two halves to build first, and it is one more sweep on an already-built
venv.

### 6.5 Record TTFT on the vLLM side

`vllm_sweep.py` currently keeps only elapsed and token counts. vLLM's `RequestOutput` carries per
request timing; adding TTFT and ITL percentiles to the JSONL rows would make the next report a latency
comparison as well as a throughput one, at no extra runtime.

---

## 7. Caveats

1. **One host, one window, one engine at a time** — satisfied, and that is what makes the
   within-log1003 comparisons valid. Do not join these numbers to log1001's or log928's.
2. **vLLM ran first**, on the colder machine. Any warm-up bias favours vLLM, so the gap figures are
   conservative in that direction too (as is the power-cap bias, §3.4).
3. **qwen.py: two runs; vLLM: three per point.** §6e asks for ≥ 3. qwen.py's two agree to 2.87 % worst
   case, so the shortfall is not material, but it is a shortfall.
4. **The code under test is one commit past `49bb10d`** and was uncommitted while the runs happened;
   provenance was re-established from the dumps (`frequency_penalty` in the config line,
   `o_tok_throughput` in `benchmark_metrics`) and the commit now exists. Future runs should commit
   first.
5. **`vllm_pip_freeze.txt` was captured twice.** The first attempt silently recorded the *system*
   Python (155 lines of jupyter/autobahn, no `vllm`, no `torch`) because `$VLLM_VENV` was unset in that
   shell, so `"$VLLM_VENV/bin/pip"` resolved to `/bin/pip`. The file in the log directory is the
   corrected capture: 198 lines, `vllm==0.30.0`, `torch==2.13.0`. The version facts that matter are
   also in every JSONL header (vllm 0.30.0, torch 2.13.0+cu130, CUDA 13.0, Python 3.12.3), which is
   what the headers are for.
6. **`log_vast/` is gitignored** (repo convention since log910), so these dumps live only on the
   analysis machine. `git add` of a log directory silently does nothing.
7. **vLLM 0.30.0 pulled a cu13 torch and therefore requires driver ≥ 580.** The same
   `pip install vllm==0.30.0` would fail on log1001's host (driver 550 / CUDA 12.4). This is recorded
   because it makes the vLLM side of this report un-reproducible on an older box.
