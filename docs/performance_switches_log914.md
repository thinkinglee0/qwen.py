# Optimisation Switches — Effectiveness on RTX 4090 (log914)

> **Superseded by [`performance_switches_log915.md`](./performance_switches_log915.md)**, which
> varies one switch at a time. Kept for the log914 measurements and for §4.1's account of the
> `pre_gather_cos_sin` flag being inert at the time.

Do the three tuning switches earn their keep?

| Switch | Config field | Verdict |
| --- | --- | --- |
| `--compile-rope` | `compile_rope` | **Net negative.** Saves a flat ~0.45 ms of GPU kernel time per step and costs 0.7–4.3 ms of host time. Leave it off. |
| `--pre-gather-cos-sin` | `pre_gather_cos_sin` | **Unmeasured.** The flag was not read by the code these runs used, so both settings executed the same path; it was wired up on 2026-09-15, after the runs. Needs measuring from scratch. |
| `--make-sampling-tensor-strategy` | `make_sampling_tensor_strategy` | **Below resolution.** Its mechanism caps the possible win at ~0.4 ms/step, and this harness cannot resolve better than ±2–3 %. Not measurable here — not the same as "no effect". |

**Artifacts analysed**

| Configuration | Path | `compile_rope` | `pre_gather_cos_sin` | `…tensor_strategy` |
| --- | --- | :-: | :-: | :-: |
| **A — all off** | `log_vast/log914/profile_baseline/`, `…_baseline2/` | false | false | 0 |
| **B — all on** | `…/profile_pre_gather_cos_sin_plus_staging_sampling_plus_compile_rope/`, `…2/` | true | true | 1 |

Each configuration was run **twice**, over five batch sizes (1, 8, 32, 128, 512), 74 steps per run.

**What this data cannot answer.** Two gaps in log914, both worth closing before the next round:

* `log_vast/log914/profile_pre_gather_cos_sin_plus_staging_sampling/` and its `…2` twin are
  **empty directories** — the one configuration that would have separated `compile_rope` from the
  other two produced no data.
* The `benchmark_*` folders in log914 are **baseline only**, so there is no throughput or TPOT
  number for configuration B. Every figure below is per-step, from the profile harness.

Attribution survives anyway, but by reading the code rather than the sweep: §4.1 shows
`pre_gather_cos_sin` is inert by construction, and §4.3 bounds the staging switch at ~0.4 ms. What
is left — all of the measured movement — is `compile_rope`.

---

## 1. What each switch actually does

### 1.1 `compile_rope`

[`get_apply_rotary`](../src/qwen/rope.py#L16) returns either the eager `apply_rotary` or
`torch.compile(apply_rotary, dynamic=True, fullgraph=True)`, cached in a module global so the
compile happens once. The function is a five-op pointwise chain over `[T, H, D]`, called **twice per
layer** (query and key) — **48 calls per step**.

The docstring already notes the numerical consequence: the compiled path keeps intermediates in
fp32 and rounds once, the eager path rounds after every op, and the two drift ~1 bf16 ulp per layer.
That is why HF-parity tests must run eager.

### 1.2 `pre_gather_cos_sin`

Intended to hoist the `cos_cached[position_ids]` / `sin_cached[position_ids]` gather out of the
per-layer loop and do it once per step, routing attention through
[`rope.forward2(q, k, cos, sin)`](../src/qwen/rope.py#L64) instead of
[`rope.forward(q, k, position_ids)`](../src/qwen/rope.py#L72).

Whether it was a choice at all depends on when you look — see §4.1.

### 1.3 `make_sampling_tensor_strategy`

Strategy 0 ([`from_sampling_list_0`](../src/qwen/sampling.py#L86)) builds the six per-step sampling
parameter tensors with six separate `torch.tensor([...], device=cuda)` calls — six **pageable** H2D
copies, each of which is synchronous and drains the stream.

Strategy 1 ([`from_sampling_list_1`](../src/qwen/sampling.py#L102)) packs five of them into one
pinned `[5, batch]` staging tensor and issues a single `non_blocking=True` copy, plus one more for
the int64 `top_k`: **six pageable copies → two pinned async copies**.

---

## 2. Method

### 2.1 Which steps are compared

The profile harness lays out each run as `WARMUP_STEPS` (64) → a clean unprofiled window (5 steps)
→ the same window again under `torch.profiler` (5 steps). All 74 steps are dumped to
`step_metrics.*.json`, so the **clean window is `rows[-10:-5]`** — the last five steps before the
profiled ones. Everything in §3 is the mean of those five steps, which are pure-decode by
construction (`n_p == 0` verified).

Using the clean window matters: the profiled window carries 10–70 % profiler overhead on the host
(§2.5 of the [log914 baseline report](./performance_analysis_log914.md)), which would swamp effects
of this size.

### 2.2 The two instruments have very different precision

Four runs give a direct noise floor per configuration. Run-to-run spread on the **same** config:

| Instrument | Source | Spread, batch 1 | Spread, batch 512 |
| --- | --- | ---: | ---: |
| Step wall time | `step` field, 5-step mean | **13.6 %** | 2.2 % |
| GPU kernel time | union of trace kernel spans | **0.1 %** | 0.0 % |

Five steps is not enough to time a step to better than ~10 % at low batch. The chrome trace's
kernel-span union, by contrast, reproduces to the third digit. **So the kernel-time column below
carries the verdicts, and the wall-time column is only trusted where it is consistent across all
five batch sizes.**

A delta is marked **clear** only when its magnitude exceeds the run-to-run spread of both
configurations at that batch size.

---

## 3. Results — configuration B vs A

Each cell is the mean of two runs; `Δ` is B relative to A.

| batch | GPU kernel time A → B | Δ | | step wall A → B | Δ | | `rope` A → B | Δ |
| ---: | --- | ---: | :-: | --- | ---: | :-: | --- | ---: |
| 1 | 3.283 → 2.866 ms | **−12.7 %** | clear | 20.11 → 20.84 ms | +3.6 % | noise | 4.32 → 5.58 ms | **+29.1 %** clear |
| 8 | 4.333 → 3.842 ms | **−11.3 %** | clear | 26.89 → 27.90 ms | +3.7 % | noise | 5.60 → 6.44 ms | +14.9 % noise |
| 32 | 5.534 → 5.120 ms | **−7.5 %** | clear | 31.04 → 35.39 ms | **+14.0 %** clear | | 5.77 → 7.83 ms | **+35.7 %** clear |
| 128 | 14.153 → 13.748 ms | **−2.9 %** | clear | 40.63 → 42.45 ms | +4.5 % | noise | 5.26 → 6.13 ms | **+16.6 %** clear |
| 512 | 50.787 → 50.273 ms | **−1.0 %** | clear | 93.73 → 96.22 ms | +2.7 % | noise | 5.31 → 6.32 ms | **+19.1 %** clear |

Three directions are consistent across **every** batch size and **both** repeats:

| Field | Direction | Deltas | Clear at |
| --- | --- | --- | --- |
| GPU kernel time | **down** | −12.7, −11.3, −7.5, −2.9, −1.0 % | all five |
| GPU idle fraction | **up** | +3.0, +2.7, +4.0, +3.7, +4.1 points | all five |
| `rope` / `rope_gpu` | **up** | +15 … +36 % | four of five |
| `fwd` | **up** | +5.5, +4.3, +15.3, +6.3, +8.1 % | two of five |
| `step` | **up** | +3.6, +3.7, +14.0, +4.5, +2.7 % | one of five |
| `sample` / `sample_gpu` | — | −10 … +9 %, no pattern | none |

**The saving is a constant, not a fraction.** In absolute terms the kernel-time reduction is
417, 491, 415, 405 and 514 µs — **~450 µs at every batch size**, which is exactly what a rope
optimisation should look like, since rope's cost is batch-independent (log914 baseline §4.2). The
percentage shrinks from 12.7 % to 1.0 % only because the denominator grows with batch.

`sched` also moves consistently down (−0.06 to −0.09 ms, flat across batch). It is real but an order
of magnitude below the costs above, and unexplained; not pursued here.

---

## 4. Verdicts

### 4.1 `pre_gather_cos_sin` — inert when these runs were made, live since

The code that produced the log914 runs did not read the flag. At `HEAD` (`330e3db`),
[`attention.py:134`](../src/qwen/attention.py#L134) reads:

```python
cos_sin = rope.gather_cos_sin(position_ids) if rope is not None else None
```

and [`engine.forward`](../src/qwen/engine.py#L51) always passes `rope=self.model.model.rope`, so
`cos_sin` was never `None`, [`attention.py:278`](../src/qwen/attention.py#L278) always took
`forward2`, and the per-layer gather in `rope.forward` was unreachable. `grep -rn "pre_gather" src/`
returned exactly one hit — the declaration in `config.py`. The flag was set (by
`tests/conftest.py:251`, which is how the run directories got their names) and ignored.

**Wired on 2026-09-15 17:14**, a day after the runs (2026-09-14 19:10), the line now reads:

```python
cos_sin = rope.gather_cos_sin(position_ids) if rope is not None and config.pre_gather_cos_sin else None
```

with an intact fallback — `meta.position_ids` is always populated
([`attention.py:147`](../src/qwen/attention.py#L147)), so with the flag off each layer gathers its
own cos/sin through `rope.forward`. It is a real switch now. **It has never been measured as one.**

**The measurements agree with that history.** Had the flag been live during these runs,
configuration A (`pre_gather_cos_sin=false`) would have carried 24 per-layer gathers inside its
`rope` timing while B carried one in `bld_meta`. The data shows the opposite: A's `rope` is *lower*
than B's at every batch size (4.32 vs 5.58 ms at batch 1) and `bld_meta` is unchanged (0.31 vs
0.30 ms). Two independent lines of evidence, same conclusion.

**Consequence for this report:** configuration B differed from A in `compile_rope` and
`make_sampling_tensor_strategy` only. Combined with §4.3, every delta in §3 is `compile_rope` — the
attribution the empty directories would otherwise have cost.

**Action:** measure it. Now that the flag reaches the code, the eager per-layer path is reachable
again and the switch belongs in the next matrix as its own factor. Expect a *small* effect in both
directions: pre-gathering removes 23 of 24 gather kernels (GPU work down, like `compile_rope`) but
moves one gather into `bld_meta` on the host — and in a launch-bound engine the host side is what
decides (§5).

### 4.2 `compile_rope` — buys 0.45 ms of GPU, spends 0.7–4.3 ms of host

The trade is unambiguous and it is on the wrong side:

| batch | kernel time saved | wall time added | net |
| ---: | ---: | ---: | ---: |
| 1 | −0.42 ms | +0.73 ms | **+0.31 ms** |
| 8 | −0.49 ms | +1.01 ms | **+0.52 ms** |
| 32 | −0.41 ms | +4.35 ms | **+3.94 ms** |
| 128 | −0.41 ms | +1.82 ms | **+1.41 ms** |
| 512 | −0.51 ms | +2.49 ms | **+1.98 ms** |

`torch.compile` does what it promises on the device — it fuses the five-op pointwise chain, and the
measured kernel time falls by a repeatable ~450 µs. But the **host** cost of reaching that kernel is
higher than an eager call: guard evaluation and the compiled wrapper run **48 times per step**, and
`rope`'s host time rises 15–36 % (+1.0 ms at batch 512).

Note what happens to `rope_gpu`: it rises 17–39 %, *in the same direction as the host time*, while
total kernel time falls. That is the window-versus-busy trap
([log914 §7.2](./performance_analysis_log914.md#7-three-traps-in-this-instrumentation)) in a single
measurement — `rope_gpu` is the span between two CUDA events, so a host that launches more slowly
stretches the window even as the work inside it shrinks. Reading `rope_gpu` alone would say the
switch made the GPU work *harder*; reading the trace says the opposite. Both are correct about what
they measure.

**Do not re-test this switch on a wall clock until the engine is CUDA-graphed.** Once launch cost is
amortised, the same 450 µs becomes a real 450 µs.

### 4.3 `make_sampling_tensor_strategy` — below this harness's resolution

No field moves consistently: `sample` goes −8.2, +0.9, +8.3, +2.6, +1.1 % across the sweep, with
run-to-run spread of 2–15 % — a pattern indistinguishable from noise.

That is the expected outcome, and it is a statement about the *experiment*, not about the switch.
The mechanism replaces six small pageable H2D copies with two pinned async ones; the tensors are
`[5, batch]` and `[batch]` floats — 10 KB at batch 512. The saving is bounded by the host-block cost
of four pageable copies, i.e. a few hundred microseconds at most, against a 93 ms step. **A 0.4 %
effect cannot be seen through a ±2–3 % noise floor**, let alone the ±13.6 % at batch 1.

log910 §5.2 reached the same place from the other direction: the copies that actually hurt are the
`flat_idx` lists rebuilt in Python by `bin_counts_and_mask` every step (~1 MB of int64 at batch
512), and this switch does not touch them.

**To settle it**, run the benchmark sweep — thousands of steps instead of five — with only this
switch flipped. Fix `bin_counts_and_mask` first; measuring a 0.4 ms saving while a 10 ms one sits
next to it is not worth the machine time.

---

## 5. Why "less GPU work, more wall time" is the expected result

It follows directly from the baseline finding. The engine is idle **45–85 % of every decode step**
(log914 §3.4). When the GPU is idle most of the time, GPU work is not on the critical path — **the
host is**. Any change that trades host time for device time is therefore negative by construction,
whatever it does to kernel counts.

This is the second time the same experiment has produced the same answer: log910 §4.5 measured
`compile_rope` reducing GPU-busy time by 13 % and making the step *slower*, under `do_sample=false`.
log914 reproduces it at −12.7 % under `do_sample=true`, with sampling now dominating the step. The
conclusion is robust to the sampling configuration because it never depended on it.

**The corollary is the useful part:** micro-optimising kernels is not merely low-value here, it is
*negative-value*, and will stay that way until the launch overhead is gone. Fix the step structure
(CUDA graphs) and the sampling algorithm first; re-evaluate every kernel-level switch afterwards,
because their sign may flip.

---

## 6. How to measure a switch properly next time

1. **One switch per run.** The empty `…pre_gather_cos_sin_plus_staging_sampling/` directories cost
   this report its per-switch attribution; it was recovered from source, which will not always be
   possible.
2. **Run the benchmark sweep, not only the profiler.** Five steps resolve ±13 % at batch 1. The
   sweep gives thousands of steps, a throughput number, and works with
   `test_mean_step_metrics`'s two-run diff, which already deflates for autocorrelation.
3. **Judge kernel-level switches on trace kernel time and step wall time together.** Either one
   alone gives the wrong answer here: kernel time says `compile_rope` wins by 12.7 %, wall time says
   it loses.
4. **Assert the switch is live.** A test that sets a config field and asserts the code path actually
   changed would have caught §4.1 at the source — before it named two directories after a flag the
   binary was ignoring.

---

## 7. Recommendations

| Priority | Action |
| :-: | --- |
| 1 | **Measure `pre_gather_cos_sin`** (§4.1). It became a real switch on 2026-09-15 and has no data at all; it must be its own factor in the next matrix. |
| 2 | **Keep `compile_rope` off** (§4.2), and re-evaluate only after the decode step is CUDA-graphed. |
| 3 | **Leave `make_sampling_tensor_strategy` at 0** for now (§4.3) — not because it is harmful, but because its effect is unmeasurable until `bin_counts_and_mask` is fixed. |
| 4 | **Re-run the switch matrix as benchmark sweeps, one switch at a time**, after the two structural fixes in the [baseline report §8](./performance_analysis_log914.md#8-recommendations-in-order). |

None of these three switches is where the performance is. The baseline report's ranking stands
unchanged: sampling is 61.5 % of the run, and the model forward costs 18.5 ms regardless of what
goes into it. These switches move ~0.45 ms.
