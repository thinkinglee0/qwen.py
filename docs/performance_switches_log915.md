# Optimisation Switches — Isolated Measurement on RTX 4090 (log915)

> 中文版：[`performance_switches_log915.zh.md`](./performance_switches_log915.zh.md)

Supersedes [`performance_switches_log914.md`](./performance_switches_log914.md), which had to infer
per-switch effects from a single combined configuration. This round varies **one switch at a time**.

| Switch | GPU kernel time | Step wall time | Verdict |
| --- | --- | --- | --- |
| `pre_gather_cos_sin` | **−0.3 … −4.2 %** (clear at all 5 batch sizes) | **−1.6 … −10.9 %** (clear at 3, same sign at all 5) | **Turn it on.** The only switch that cuts host *and* device work. |
| `compile_rope` | −1.0 … −12.3 % (clear at all 5) | **+1.2 … +4.7 %** (clear at 3, same sign at all 5) | **Leave off.** Third measurement, third time it trades GPU time for more wall time. |
| `stage_sampling_params` | ±0.1 % | −0.1 … −1.8 % (same sign at all 5, never clear) | **Still below resolution.** Consistently negative but never outside noise. |

**Artifacts**

| Configuration | `compile_rope` | `pre_gather_cos_sin` | `stage_sampling_params` | Path |
| --- | :-: | :-: | :-: | --- |
| baseline | false | false | false | `log_vast/log915/profile_baseline{,2}/` |
| compile_rope | **true** | false | false | `…/profile_compile_rope{,2}/` |
| pre_gather | false | **true** | false | `…/profile_pre_gather_cos_sin{,2}/` |
| stage_sampling | false | false | **true** | `…/profile_stage_sampling{,2}/` |
| sweep (baseline only) | false | false | false | `…/benchmark_baseline{,2}/` |

Every configuration ran **twice**, five batch sizes (1, 8, 32, 128, 512), **20 clean steps** per point
— four times the window of log914. The sweep ran twice over 11 batch sizes.

Two things changed in the engine since log914: `pre_gather_cos_sin` is now actually read by
[`attention.py:134`](../src/qwen/attention.py#L134) (in log914 it was declared and ignored), and
`make_sampling_tensor_strategy: int` was renamed to `stage_sampling_params: bool`.

**Still missing:** the sweep was run for the baseline only, so there is **no throughput number for
any switch**. Everything in §3 is per-step, from the profile harness.

---

## 1. Two measurement artefacts found while reducing this data

Both were found by cross-checking the profile harness against the sweep, which ran on the same host
within the same hour. They matter more than any switch below.

### 1.1 `sync_debug_mode="warn"` is free — hypothesis tested and rejected

[`test_profile.py:169`](../tests/test_profile.py#L169) turns on `torch.cuda.set_sync_debug_mode("warn")`
for the whole clean measurement window. The obvious worry is that it taxes every CUDA call and
inflates the very numbers the window exists to produce.

It does not. Comparing the last 20 **warmup** steps (sync debug off) against the 20 **measure** steps
(sync debug on) — same engine, same run, nothing else different:

| batch | 1 | 8 | 32 | 128 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rope` | −0.7 % | −0.6 % | +0.1 % | −1.6 % | −0.9 % |
| `fwd` | −1.2 % | −0.3 % | +0.2 % | −1.3 % | −1.3 % |
| `step` | −0.9 % | −0.2 % | +0.7 % | −0.1 % | +0.3 % |

Within noise, both signs. The switch can stay where it is.

### 1.2 The profiler leaves ~30 % of host overhead behind in the process

The real artefact. Host-side `rope`, by batch point **in run order**, for all eight profile sessions:

| session | bs 1 | bs 8 | bs 32 | bs 128 | bs 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | **6.27** | 8.12 | 8.21 | 8.27 | 8.27 |
| baseline 2 | **6.80** | 8.19 | 8.52 | 8.20 | 8.25 |
| compile_rope | **7.90** | 8.94 | 9.10 | 9.23 | 9.30 |
| compile_rope 2 | **8.12** | 8.95 | 8.97 | 9.12 | 9.27 |
| pre_gather | **4.83** | 6.26 | 6.52 | 6.40 | 6.32 |
| pre_gather 2 | **4.93** | 6.41 | 6.32 | 6.35 | 6.41 |
| stage_sampling | **6.42** | 8.09 | 8.07 | 8.16 | 8.49 |
| stage_sampling 2 | **6.42** | 8.22 | 8.18 | 8.09 | 8.18 |

In **every** session the first batch point is 20–30 % cheaper than every later one, which then sit
flat. `rope`'s host cost is batch-independent (log914 §3.2), so this is not a batch effect — and the
sweep harness proves it directly. Eleven engine constructions in one session, never profiled:

```
bs    1    2    4    8   16   32   64  128  256  512 1024
rope 6.11 6.10 6.12 6.15 6.17 6.24 6.24 6.24 6.26 6.29 6.36
```

Flat. So it is not engine re-creation, not allocator state, not batch size. The one thing that
happens between the profile harness's first and second points is that **`torch.profiler` has been
entered and exited once**, and something it leaves in the process taxes every subsequent ATen op on
the host. (Kineto registering global observer callbacks on first use is the obvious suspect, but the
experiment below identifies the *scope* of the state, not which piece of it.)

**Consequences.**

* Profile-harness *absolute* host-side numbers are inflated ~30 % at every point except the first,
  and with them `wall_clean_us` — so **the `gpu_idle_fraction` this harness reports is overstated**
  everywhere but batch 1. Device-side kernel time is unaffected (it reproduces to 0.1 %).
* Profile numbers must **not** be compared against sweep numbers. At batch 512 the same engine reads
  `fwd` 28.57 ms in the profile harness and 22.48 ms in the sweep.
* The switch verdicts in §3 **survive**, because the contamination is common-mode: each configuration
  ran in its own process with the same batch order, so bs 8–512 are equally taxed in all four. The
  *relative* deltas hold; the absolute millisecond savings are inflated by roughly the same 30 %.
* The cleanest points in the whole dataset are the **batch-1** columns, which are uncontaminated.

**Fix:** one batch size per pytest process, or run the clean window before the profiled one has ever
executed in that process. Until then, read the profile harness for kernel time and for A/B ratios —
not for absolute step times.

### 1.3 Confirmed by process isolation (log916)

The prediction is sharp: give each batch size its own pytest process and *every* point becomes a
first point, so the inflation should vanish entirely. Run as

```bash
for bz in 1 8 32 128 512; do SWEEP_PROFILE_BATCH_SIZES=$bz pytest -x -s \
  tests/test_profile.py::test_profile_decode_idle_fraction \
  --compile-rope=False --stage-sampling-params=false --pre-gather-cos-sin=false; done
```

it does (`log_vast/log916/profile_baseline{,2}`), on both the field that exposed the problem and on
the step time as a whole:

| | bs 1 | bs 8 | bs 32 | bs 128 | bs 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rope`, log915 — one process, five batches | 6.27 | **8.12** | **8.21** | **8.27** | **8.27** |
| `rope`, log916 — process per batch | 6.10 | 6.15 | 6.35 | 6.52 | 6.22 |
| `rope`, log916 — process per batch, run 2 | 6.12 | 6.27 | 6.01 | 5.93 | 6.20 |
| `rope`, sweep harness (never profiles) | 6.11 | 6.15 | 6.24 | 6.24 | 6.29 |
| **step**, log915 — one process | 23.19 | **31.18** | **34.76** | **48.74** | **104.82** |
| **step**, log916 — process per batch | 23.21 | 24.82 | 28.95 | 44.09 | 99.73 |
| **step**, log916 — run 2 | 22.72 | 24.96 | 27.09 | 40.87 | 99.92 |
| **step**, sweep harness decode step | 22.69 | 24.55 | 27.73 | 42.47 | 100.17 |

Two things fall out:

* **The inflation is entirely gone** — up to +28 % at batch 8 in log915, nothing in log916.
* **The two harnesses now agree.** They had no reason to disagree in the first place; at batch 512 the
  isolated profile harness reports `wall_clean_us` = 100.218 ms against the sweep's 100.17 ms decode
  step — **0.05 % apart**, having been 5 % apart before.

The reported idle fraction moves with it: at batch 512, **50.2 % isolated against 52.3 % contaminated**
(§1.2 predicted ~49.8 %). Every published `gpu_idle_fraction` from a multi-batch session should be
read down by roughly two points.

> **Caveat for the loop above:** `log_file` in `pyproject.toml` opens in truncate mode, so each
> iteration overwrites `log/pytest.log` and only the last batch size's session survives. The idle
> fractions for batches 1–128 were lost. Set `log_file_mode = "a"` (already there, commented out) or
> give each iteration its own `--log-dir`.

---

## 2. Method

Per point: 20 pure-decode steps from the clean (unprofiled) window, `rows[-25:-5]` of each dump,
×2 runs. A delta is marked **clear** only when its magnitude exceeds the run-to-run spread of both
configurations at that batch size.

Kernel time is the union of `kernel`/`gpu_memcpy`/`gpu_memset` spans from the chrome trace, which
reproduces run-to-run to ~0.1 % and is the instrument the verdicts rest on.

---

## 3. Results — each switch against the same baseline

### 3.1 GPU kernel time

| batch | compile_rope | pre_gather | stage_sampling |
| ---: | ---: | ---: | ---: |
| 1 | **−12.3 %** clear | **−4.2 %** clear | −0.1 % |
| 8 | **−10.7 %** clear | **−2.4 %** clear | +0.0 % |
| 32 | **−7.4 %** clear | **−2.0 %** clear | +0.0 % |
| 128 | **−3.3 %** clear | **−0.7 %** clear | −0.0 % |
| 512 | **−1.0 %** clear | **−0.3 %** clear | −0.0 % |

Both rope switches genuinely reduce device work. In absolute terms each saves a near-constant
~0.4 ms (compile) and ~0.15 ms (pre-gather) per step; the percentage shrinks with batch only because
sampling inflates the denominator.

### 3.2 Step wall time

| batch | compile_rope | pre_gather | stage_sampling |
| ---: | ---: | ---: | ---: |
| 1 | +4.7 % | **−10.9 %** | −1.8 % |
| 8 | **+3.2 %** clear | **−7.0 %** clear | −0.2 % |
| 32 | +2.6 % | −2.8 % | −0.9 % |
| 128 | **+3.1 %** clear | **−3.8 %** clear | −1.1 % |
| 512 | **+1.2 %** clear | **−1.6 %** clear | −0.1 % |

### 3.3 Host-side `rope` and `fwd`

| batch | `rope` compile | `rope` pre_gather | `fwd` compile | `fwd` pre_gather |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +22.6 % | **−25.3 %** | +5.1 % | −11.5 % |
| 8 | +9.7 % | **−22.3 %** | +3.4 % | −8.0 % |
| 32 | +7.9 % | **−23.3 %** | +3.3 % | −5.5 % |
| 128 | +11.4 % | **−22.6 %** | +4.3 % | −7.0 % |
| 512 | +12.4 % | **−23.0 %** | +4.8 % | −7.4 % |

All clear except `fwd`/compile at batch 1 and 32. `sample` moves nowhere for any switch, as expected.

Absolute, batch 512 (ms, mean of two runs — inflated ~30 % per §1.2, ratios are the usable part):

| | baseline | compile_rope | pre_gather | stage_sampling |
| --- | ---: | ---: | ---: | ---: |
| step | 104.57 | 105.84 | **102.91** | 104.42 |
| `fwd` | 28.57 | 29.95 | **26.46** | 28.72 |
| `rope` | 8.26 | 9.28 | **6.37** | 8.34 |
| `bld_meta` | 1.16 | 1.18 | 1.26 | 1.19 |
| GPU kernel | 50.31 | 49.79 | 50.17 | 50.30 |

---

## 4. Per-switch analysis

### 4.1 `pre_gather_cos_sin` — the first switch that is simply worth having

It removes 23 of the 24 per-layer `cos_cached[position_ids]` gathers, doing one in
[`build_attn_metadata`](../src/qwen/attention.py#L134) instead, and routes each layer through
`rope.forward2`. Both sides of the ledger improve:

* **Host** — `rope` falls 22–25 % at every batch size (8.26 → 6.37 ms at batch 512), and `fwd` with
  it. The one gather that remains shows up as +0.10 ms on `bld_meta`, which is where it should be.
* **Device** — 23 fewer gather kernels, −0.15 ms of kernel time.

Net −1.6 % to −10.9 % on the step. This is what a real optimisation looks like: less work on both
sides, no trade.

Read the **batch-1 column** for the honest magnitude — it is the uncontaminated point (§1.2):
`rope` 6.53 → 4.88 ms, step −10.9 %.

**`config.py` already defaults it to `True`.** These runs turned it *off* explicitly, and so did
log914's, because `tests/conftest.py` passes the CLI option through and the run scripts set false.
Nothing needs changing in the engine — only in how the sweeps are invoked.

### 4.2 `compile_rope` — measured three times, negative three times

| | log910 | log914 | log915 |
| --- | --- | --- | --- |
| GPU kernel time | −13 % | −12.7 % | **−12.3 %** (batch 1) |
| Step wall time | slower | +2.7 … +14.0 % | **+1.2 … +4.7 %** |
| `do_sample` | false | true | true |
| Isolated? | yes | no (bundled) | **yes** |

The device-side win is real and reproducible: `torch.compile` fuses the five-op pointwise chain and
takes ~0.4 ms of kernel time out of every step, at every batch size. The host cost of reaching it is
larger: guards and the compiled wrapper run **48 times per step** (24 layers × q and k), and `rope`'s
host time rises 8–23 %.

In an engine that is idle 50–85 % of every step, that trade can only lose. **Re-test it after the
decode path is CUDA-graphed**, not before — the sign should flip once launch cost is amortised.

Note it also stacks badly with §4.1: `compile_rope` raises `rope` by ~1.0 ms while `pre_gather`
lowers it by ~1.9 ms, and they touch the same 48 calls.

### 4.3 `stage_sampling_params` — consistent in direction, invisible in magnitude

Step wall time is lower in all five batch sizes (−0.1 to −1.8 %) and kernel time is identical to
four decimal places, which is exactly what replacing six pageable H2D copies with two pinned async
ones should look like. But **not one of those deltas clears the run-to-run spread**, even with a
20-step window.

That is a statement about the experiment, not the switch: the tensors are `[5, batch]` and `[batch]`
floats — 10 KB at batch 512 — so the saving is bounded at a few hundred microseconds against a
~100 ms step. log910 §5.2 identified the copies that actually cost something (the `flat_idx` lists
`bin_counts_and_mask` rebuilds in Python every step, ~1 MB of int64 at batch 512); this switch does
not touch them.

**To settle it:** a sweep with only this switch flipped — thousands of steps instead of forty. Worth
doing only after `bin_counts_and_mask` is fixed.

---

## 5. The new baseline sweep

Two runs, 11 batch sizes, `pre_gather_cos_sin=false`, `use_d_first_schedule=false`.

| batch | tok/s | Δ run 2 | step ms | `fwd_gpu` | `sample_gpu` | sample % step | sample % run | TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 43.6 | +0.30 % | 22.69 | 20.50 | 1.10 | 4.8 % | 4.8 % | 22.9 |
| 8 | 322.1 | +0.26 % | 24.55 | 21.75 | 1.68 | 6.8 % | 6.8 % | 24.8 |
| 32 | 1 120.6 | −0.40 % | 27.73 | 22.54 | 3.92 | 14.2 % | 13.8 % | 28.2 |
| 128 | 2 834.5 | +0.96 % | 42.47 | 22.72 | 17.90 | 42.1 % | 38.9 % | 44.6 |
| 256 | 3 801.3 | +0.36 % | 62.41 | 22.60 | 37.18 | 59.6 % | 52.7 % | 66.5 |
| 512 | 4 681.7 | +0.50 % | 100.17 | 22.47 | 73.24 | 73.1 % | 60.4 % | 107.5 |
| 1024 | 5 094.6 | +0.47 % | 171.31 | 23.47 | 139.79 | 81.6 % | 58.6 % | 195.3 |

**Reproducibility improved sharply**: every point agrees within **1.21 %** (log914 reached 5.5 %).
Both structural findings reproduce unchanged — `fwd_gpu` is flat at 20.5–23.5 ms across a 1024×
range of batch size, and sampling is 60.4 % of the whole run at batch 512.

**Against log914, three things moved at once**, so the sweeps are not directly comparable:

| | log914 | log915 |
| --- | --- | --- |
| `pre_gather_cos_sin` | declared false, **ignored** — pre-gathered path ran | false, **honoured** — per-layer path ran |
| `use_d_first_schedule` | true | false |
| Host | one vast.ai instance | another |
| tok/s @ 1024 | 5 300 | 5 095 (−3.9 %) |
| `rope` @ 512 | 4.31 ms | 6.29 ms (+46 %) |

The `rope` regression is the flag change, not a code regression: log914's "baseline" was in fact
running with pre-gather on. **log915's `pre_gather` configuration is the one comparable to log914's
baseline**, and the log915 baseline is a genuinely slower setting that nothing should be run in.

---

## 6. Recommendations

| Priority | Action |
| :-: | --- |
| 1 | **Stop passing `--pre-gather-cos-sin=false`.** It is a free 2–11 % per step and `config.py` already defaults to `True`; only the sweep invocations turn it off. Re-run the baseline sweep with it on — the current baseline numbers understate the engine. |
| 2 | **Fix the profiler contamination** (§1.2): one batch size per process. Every `gpu_idle_fraction` published so far is overstated for all but the first batch point. |
| 3 | **Keep `compile_rope` off** (§4.2) and re-test only after CUDA graphs. |
| 4 | **Leave `stage_sampling_params` at its default** (§4.3) until `bin_counts_and_mask` is fixed and a sweep can resolve it. |
| 5 | **Run the sweep, not just the profiler, for whichever switch is next.** Three rounds of switch measurement have now produced no throughput number for any switch. |

None of this changes the ranking in the [baseline report](./performance_analysis_log914.md#4-bottlenecks-ranked-by-what-they-cost):
sampling is ~60 % of the run and the model forward costs ~22 ms regardless of batch. The best switch
here is worth ~2 ms.
