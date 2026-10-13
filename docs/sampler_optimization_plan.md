# Sampler optimisation plan

**Original goal: 16.35 → ~2.2 ms/step at batch 512.**

**Status (log1013 window run): everything except D1/D2 and B4 has landed.**

* A, B (in `1bd9f72`, on `main`): the sampler went **16.35 → 9.77 ms**.
* C2–C5 + E (in `perf/sampler-bf16-logits-fp32-topk-window` @ `b4ebaf7`): the sampler went
  **9.77 → 1.57 ms**, and the device step **20.6 → 12.4–12.6 ms**. Batch-512 throughput rose +25.4 %
  ([report](./performance_analysis_log1013_window.md)).

What remains is bf16 `topk` (0.76 ms, irreducible without a custom kernel), the mask build in
`smp_prep` (0.60 ms, D1/D2) and the penalties (0.09 ms). **The sampler is no longer the bottleneck.**

* At batch 512 the decode step is level with the host floor.
* At batch ≥ 512 prefill is 69–86 % of the run.
* D1/D2 will show in wall time only once host time drops.

Sources:

* [log1009 host 1](./performance_analysis_log1009.md): the original measurements, batch 512, pure
  decode.
* [log1012](./performance_analysis_log1012.md): bf16 variant of batch 1 + 3.
* [log1013](./performance_analysis_log1013.md): fp32 variant, two hosts. Device spans repeat to within
  0.02 ms across three hosts, so the numbers below carry across hosts.
* [log1013 window run](./performance_analysis_log1013_window.md): C2–C5 + E on a 7950X3D.

Targets were **modelled** as traffic ÷ the measured 919 GB/s. The model was validated to **0.1 %**
on `apply_penalties`, and it **over-predicted two of the four `smp_pick` items it was first used for**
(A3 and C1, §Corrections). The revised C + E prediction, built on measured per-op costs, landed within
5 % (§The list).

vLLM citations are from the local checkout at
`/Users/jaydeli/ai_master_workspace/inference_experiment/vllm`, commit `0f112d1800`
(= `v0.31.1rc0-99`). That is **newer than the 0.30.0 we benchmarked**. The claims marked ✓0.30 were
re-checked against the `v0.30.0` tag and hold there too (line numbers shift by ~1). The rest were read
at the newer commit only.

---

## Where the time is (measured, batch 512, ms/step)

| span | log1009 (start) | log1013 (A + B, fp32) | window run (now) | what |
| --- | --- | --- | --- | --- |
| `smp_prep` | 0.592 | 0.595 | 0.613 | `build_sampling_tensors` → `bin_count_and_mask` |
| `smp_pen` | 6.848 | 0.093 | 0.103 | `apply_penalties` |
| entry cast | (inside `smp_pen`) | 0.511 | 0.005 | `logits.float()`, now only on the window |
| `smp_pick` | 8.858 | 8.524 | **0.797** | `sample()` |
| `smp_post` | 0.041 | 0.041 | 0.047 | token write-back |
| **total** | **16.348** | **9.771** | **1.571** | |
| device step (`gpu_busy`) | 27.05 | 20.31 | **12.36–12.60** | |

In the A + B version the fp32 promotion was an explicit cast at the sampler entry, outside every
sub-span, which is why it has its own row. Before that, it was a side effect of
`where(…, logits / rep, …)` inside `smp_pen`.

In the window run, `smp_pick` is the bf16 `topk` (0.763 ms: 2 radix passes plus `gatherTopK`) plus a
few µs per window op. The rest of the step's sampler time is D1/D2's zero-fill in `smp_prep`, a
`fill_` of 0.45 ms/step.

For the record, here is what `smp_pick` spent before C + E, by op (log1013, fp32, per step):

| op | ms | what it is |
| --- | --- | --- |
| `topk` (full vocab, fp32) | 1.859 | 4 × `radixFindKthValues` passes |
| `multinomial` internals: `div` + `argmax` + `exponential_` | 1.688 | `argmax(probs / Exp(1) noise)` over `[B, V]` |
| `_softmax` (full vocab) | 1.256 | |
| `fill_` (`full_like(-inf)`) + `scatter_` back | ~0.8 | rebuilding a `[B, V]` tensor from the window |
| `sum` + `masked_fill_` (dead-row guard) | 1.369 | |
| `div_` (temperature) | 0.663 | |
| `max` and small ops | ~0.9 | |

Everything except `topk` exists only because the tail goes back to `[B, V]` after `topk`.

---

## The list

| # | optimisation | plan target | measured | status |
| --- | --- | --- | --- | --- |
| **A2** | per-term penalty gating | −3.13 | } | ✅ landed |
| **B1** | don't materialise `rep` as `[B,V]` | −0.42 | } **`smp_pen` −6.76** | ✅ subsumed by B3 |
| **B2** | apply repetition in place | −1.10 | } (−6.25 net of the cast) | ✅ subsumed by B3 |
| **B3** | sparse repetition via `[B, L]` gather/scatter | −2.2 | } | ✅ landed |
| **A3** | skip `argmax` when no row is greedy | −0.76 | **−0.33** | ✅ landed, over-predicted |
| **C1** | temperature in place | −0.30 | **≈ 0** | ✅ landed, no gain (§Corrections) |
| **A1** | whole-block penalty gate | 6.85 (needs `rep=1.0`) | — | ✅ landed (`rep_switch_on`), cold path |
| **A4** | `all_greedy` early return | whole sampler | — | ✅ landed (`greedy_host.all()`), cold path |
| **C2–C5** | keep the tail in the `[B, max_k]` window | −6.6 (revised; fp32 `topk` stays at 1.86) | } **`smp_pick` −7.74** | ✅ landed (`b4ebaf7`) |
| **E** | cast to fp32 **after** `topk`, not at entry | −1.65 (−0.51 cast, −1.14 `topk`) | } (−8.24 with the cast) | ✅ landed (`b4ebaf7`) |
| **D1/D2** | stop building the dense counts and masks | −0.54 | `smp_prep_gpu` unchanged | ⏳ open |
| **B4** | sparse freq/pres over the output history | cold path only | — | optional |

C2–C5 + E predicted −8.2 ms and measured −8.24 ms (`smp_pick` 8.53 → 0.80, and the cast
0.51 → 0.005).

Remaining:

1. **D1/D2**, about **−0.55 ms**. It needs the switches before `bin_count_and_mask` runs (§D). At
   batch 512 the step is now level with the host floor (log1013 window run, §4), so D1/D2 will reach
   wall time only once host time drops, for example with decode CUDA graphs. Until then it pays only at
   batch 1024, which is still GPU-bound.
2. **B4**: optional, cold path.

---

## Corrections from measurement

* **A3 was priced at both argmaxes; only one was the greedy path.** `main` runs two `argmax` per step
  (0.715 ms in total). The second is inside `torch.multinomial`, which for one sample computes
  `argmax(probs / exponential_noise)`. A3 removes only the greedy one: **−0.33**, not −0.76. The
  `multinomial` internals (`div` + `argmax` + `exponential_`, 1.69 ms in fp32) are what C5 replaces.
* **C1 cannot save traffic.** An fp32 `[512, 151 936]` divide reads 311 MB and writes 311 MB whether
  or not it is in place. `div_` saves an allocation, not bytes, and measures 0.663 ms against the same
  for `div`. In-place pays only when the op touches a subset (B3's `scatter_`) or when an address must
  stay fixed (CUDA graphs, KV cache). The real fix for the temperature pass is C2: divide the
  `[B, max_k]` window, not the vocab.
* **fp32 `topk` costs 2.6× bf16, not 2×.** The old "Do NOT change" note priced the fp32/bf16
  difference at the single `topk` read, 0.17 ms by bytes. Measured, it is **1.859 against 0.719 ms**.
  Radix select runs one pass per key-digit group: 4 `radixFindKthValues` passes per step on 32-bit
  keys, 2 on 16-bit keys. That makes the dtype of `topk`'s input the largest single lever left after
  C, and it is why §E exists.
* **The fp32 cost of the current tail is 5.1 ms, not 2.9.** log1012 §6 counted six ops. log1013 §4
  measured the whole difference between the bf16 and fp32 variants: 5.09 ms of `smp_run_gpu`.

---

## A — host-side "nothing to do" gates (landed)

All are computed in `SamplingParamTable.gather()` from `fields_stage`, the **pinned host mirror**, so
none costs a device sync. log1013 confirms `dth_wait` did not move. The host cost is a constant
+0.05–0.09 ms in `smp_prep`, offset by −0.075 ms in `smp_run` from launching fewer ops.

vLLM tracks the same conditions as sets of request ids, added only when the param is non-identity:
`vllm/v1/worker/gpu_input_batch.py:415-425`, exposed as `no_penalties` at `:1126`. A later cleanup
could follow vLLM and maintain counters at admission in `set_slot`, so the per-step check becomes an
integer comparison instead of three `.any().item()` calls.

* **A2, per-term gating.** `rep_switch_on`, `freq_switch_on` and `pres_switch_on` gate their lines
  independently. This is **finer than vLLM**, whose `no_penalties` is all-or-nothing across the
  three. Qwen's `generation_config.json` has `freq_pen = pres_pen = 0`, so the freq/pres lines are
  skipped on every step.
* **A3, greedy `argmax`.** Runs only when some row is greedy, and is taken before top-k/p. vLLM does
  the same: `if sampling_metadata.all_random: greedy_sampled = None` ✓0.30.
* **A1, whole-block gate.** Fires only when every row has `repetition_penalty == 1.0`. Qwen's default
  of 1.1 keeps it cold in the benchmark.
* **A4, all-greedy.** `greedy_host.all()` returns `argmax` before scaling, top-k/p and sampling ✓0.30.
  Cold in the benchmark.

---

## B — repetition penalty (landed as B3)

vLLM's kernel is at `csrc/libtorch_stable/sampler.cu:16`: penalty in a register, `if (is_repeated)`
skipping unseen columns, written in place. qwen.py now goes further. `apply_rep_penalty_sparse()`
gathers the logits at `seen_toks` `[B, L]`, rescales them and `scatter_`s them back. The `[B, V]`
masks are never read on this path. Measured: **`smp_pen_gpu` 6.85 → 0.09 ms.**

The two rules this depends on, now in the code:

* **Padding.** `seen_toks` pads with the row's first prompt token, so a padded slot writes a real
  token's correctly penalised value again rather than scattering into an unrelated column.
* **Duplicates.** A token seen twice is penalised once, because `scatter_` writing the *same* value is
  idempotent. Never use it to accumulate: repeated-index writes of *different* values are
  order-undefined.

vLLM's own note, at `vllm/v1/sample/ops/penalties.py:26`:
`"The penalties implementation is currently quite inefficient and will be reworked anyhow."`

### B4 — sparse frequency/presence (optional)
Both penalties touch only output tokens, so they too could work over the `[B, L]` history instead of
`[B, V]`: `scatter_add_` of `-freq` per output occurrence, `scatter_` of `-pres` per distinct output
token. That would let D1/D2 drop the dense buffers even when these penalties are on. It is a cold path
today, so it is low priority, but it is the only way the masks can go away entirely.

---

## C — the sampling tail (`smp_pick`: 8.52 ms in fp32 → 0.80 ms, landed in `b4ebaf7`)

`topk` must read all 151 936 columns. Everything after it can live in `[B, max_k]` = **41 kB** at
`max_k = 20`, against 311 MB for the full-vocab fp32 tensor. That is a **7 600×** smaller operand,
which is why this is an algorithmic win and not a kernel-language one.

`MAX_EFFECTIVE_TOP_K` bounds the window at 1 024 columns, 2 MB in fp32, which is still negligible.

| # | change | removes (measured, fp32) |
| --- | --- | --- |
| **C2** | temperature *after* `topk`, on the window. Dividing by a positive scalar preserves rank. | `div_` 0.66 |
| **C3** | softmax, top-p and the dead-row guard on the window; never rebuild `[B, V]` | `full_like` + `scatter_` ~0.8, softmax 1.26, `sum` + `masked_fill_` 1.37, `max` and small ops ~0.9 |
| **C4** | greedy for free: `topk(sorted=True)` already put the max in column 0, so it is `indices[:, 0]` | the remaining greedy `argmax` (cold path) |
| **C5** | Gumbel-max on the window instead of `multinomial` on `[B, V]`: add `-log(-log(u))` and take the argmax | `multinomial` internals 1.69 |

Map the window index back with `indices.gather(1, local)`.

**After C + E**, predicted: `topk` in bf16 (0.72 ms, measured in log1012) plus a dozen window-sized
kernels, launch-bound at a few µs each, so ~0.05–0.1 ms, for **`smp_pick` ≈ 0.8 ms**.

**Measured** (log1013 window run): `smp_pick_gpu` **0.797 ms**, of which bf16 `topk` is 0.763. The
window ops cost 2–7 µs each: `_softmax` 2, `div` 7, `argmax` 5 and `uniform_` 2 µs per step.

vLLM's equivalent is a fused select-and-sample over the candidates: FlashInfer when available,
otherwise the Triton kernels our own logs named, `_topk_topp_kernel` and `_gumbel_sample_kernel`. C5
is directly evidenced by that `_gumbel_sample_kernel` line.

**Sampling semantics to preserve.** Gumbel-max over the window draws from the same distribution as
`multinomial` over the window. It will not reproduce `multinomial`'s exact tokens for a given seed,
because it consumes the RNG differently. The fixed-seed token-for-token check (§Verification)
therefore applies to everything *except* C5. Verify C5 statistically instead (§Verification).

---

## D — `bin_count_and_mask` (`smp_prep`, 0.595 ms)

* **D1.** Skip `output_counts` (`[B, V] int32`, 311 MB of zeros plus a `scatter_add_`) when
  `freq_switch_on` is false.
* **D2.** Skip the `[B, 2V+16] bool` masks (156 MB) when both `freq_switch_on` and `pres_switch_on`
  are false. Since B3, repetition reads `seen_toks`, not the masks.
* **Ordering problem.** `build_sampling_tensors()` calls `bin_count_and_mask()` *before*
  `SamplingTensors.from_table()`, and it is `from_table()` → `gather()` that computes the switches.
  Compute the switches first, and pass them in, or return `None` for the skipped tensors.
* **D3: change nothing else here.** vLLM builds two `[B, V+1] int64` tensors, **1.24 GB** at
  batch 512. qwen.py's `int32` + `bool` in one packed `2V+16` buffer is already the better layout.

---

## E — where the fp32 cast belongs (landed in `b4ebaf7`; replaces the old "Do NOT change the fp32 cast")

The old rule was "cast to fp32 deliberately, do not try to stay in bf16". `1bd9f72` follows it, with
one explicit `logits.float()` at the sampler entry. It was the right fix for the bf16 regression in
log1012, but it puts the cast in the wrong place once C lands:

* **bf16 → fp32 is lossless.** `lm_head` produces bf16. Casting adds no information, so `topk` over
  `x.float()` selects exactly the same set, in the same order, as `topk` over `x`.
* **What needs fp32 is the probability arithmetic**: softmax, the top-p cumulative sum compared
  against `top_p`, and the Gumbel/`multinomial` draw. Every one of these moves into the `[B, max_k]`
  window under C.
* **The full-vocab cast is pure cost.** It is 0.51 ms for the copy, and it doubles `topk`'s radix
  passes (+1.14 ms).

**The new rule:**

1. **Penalties work on the bf16 logits, using fp32 arithmetic.** `apply_rep_penalty_sparse` gathers,
   computes in fp32 (`rep` is fp32), and writes back with `vals.to(logits.dtype)`. `scatter_` requires
   the source dtype to match. The cold freq/pres lines must not promote the whole `[B, V]` tensor:
   either write them in place in bf16 (`sub_`), or do B4.
2. **`topk` runs on bf16 `[B, V]`.**
3. **Cast the `[B, max_k]` window to fp32**, then apply temperature, top-k mask, softmax, top-p,
   dead-row guard and sampling, all in fp32.

This keeps the plan's numerical requirement: no probability is ever computed in bf16. The only new
rounding is the penalised logits being written back in bf16, at a step of 0.125 for logits of 16–32.
`rep = 1.1` moves a logit of 20 by 1.8, about 30× that step. A ranking can flip only for a penalised
token that lands within half a step of another token's logit, and those two are near-ties in the
model's own bf16 output anyway.

**E only makes sense together with C**, which is how it landed. With a full-vocab tail, softmax and
`multinomial` run on `[B, V]` and must see fp32. Do not reintroduce a full-vocab probability step on
the bf16 logits.

Measured: the entry-cast residual went 0.51 → 0.005 ms, and `topk` went 1.868 → 0.763 ms (−1.10,
against −1.14 predicted).

---

## Still do NOT change these

* **No device sync in the sampler.** The old dead-row guard was branch-free because `if dead.any()`
  would cost a device sync every step. The window path no longer needs a guard at all: column 0 of
  the window is always finite, because top-k keeps at least one column and top-p always keeps the
  first. A row of all `-inf` would make `argmax` return column 0 rather than crash
  (`test_sample_all_neg_inf_row_does_not_crash`). Keep it that way: never add a host-side check on a
  device tensor here.
* **`MAX_EFFECTIVE_TOP_K` clamping.** Keep it. The truncation semantics deserve a line in the docs
  (log1003 §5.2): with `vocab_size > 1024`, "top-k disabled" is implemented as top-1024, which is a
  truncation, not a no-op.
* **The `greedy_tokens` ordering.** Greedy rows must be taken from the scaled-but-unfiltered logits.
  Under C4 that is `indices[:, 0]` of the bf16 `topk`, which is the same token.

---

## Verification

| must move | must NOT move |
| --- | --- |
| ✅ C2–C5 + E → `smp_pick_gpu` (8.53 → 0.80) and the entry-cast residual (0.51 → 0.005) | `tests/test_sampling.py` all green |
| D1/D2 → `smp_prep_gpu` (0.60 → ~0.05) | `smp_pen_gpu` (stays ~0.09) |
| ✅ `gpu_busy_from_trace` 20.6 → 12.4–12.6 ms at batch 512; D1/D2 should take ~0.55 more | `fwd_gpu`, `logits_gpu` (the model is untouched) |
| | zero preemptions, zero errors |

**Output checks, by stage:**

* **Before C5:** fixed seed, token for token against the current fp32 path. C2–C4 and E change the
  order of operations, not the distribution or the RNG use. The only allowed differences are
  penalised near-ties (§E), which should be zero or a handful per thousands of tokens.
* **C5:** it changes RNG consumption, so token-for-token equality is not expected. Check the
  distribution instead. This is now `test_sample_matches_softmax_distribution[fp32, bf16]`:
  * two (T, k, p) groups in one batch, 20 000 draws each;
  * no token outside the top-k/top-p set may be drawn;
  * a χ² test against an fp64 reference at a 10⁻⁶ tail.

  The test has power: a sampler that ignores temperature scores χ² = 490 against a critical value of
  37.5. `test_sample_greedy_rows_take_window_argmax_in_mixed_batch` covers C4.

**Disciplines:**

* **Every gate creates a cold path.** `freq_pen != 0`, `pres_pen != 0`, `rep == 1.0`, all-greedy
  batches and mixed greedy/sampled batches all go somewhere the benchmark never exercises. Each needs
  a test, or the cold branch rots unnoticed.
* **Profile on a GPU-bound host.** On a slow host, batch 512 is already host-bound after batch 1 + 3
  (log1013 host 1: 23.9 % idle). Device savings there become idle time and the throughput sweep
  cannot see them. Read `gpu_busy_from_trace` and the `smp_*_gpu` spans, and run the sweep on a
  host-2-class machine.

---

## Expected end state

| span | log1009 | log1013 (A + B) | window run (A + B + C + E) | after D |
| --- | --- | --- | --- | --- |
| `smp_prep` | 0.592 | 0.595 | 0.613 | ~0.05 |
| `smp_pen` | 6.848 | 0.093 | 0.103 | ~0.09 |
| entry cast | — | 0.511 | 0.005 | 0 |
| `smp_pick` | 8.858 | 8.524 | **0.797** | 0.8 |
| `smp_post` | 0.041 | 0.041 | 0.047 | 0.04 |
| **sampler** | **16.35** | **9.77** | **1.57** | **~1.0** |
| **device step** | **27.05** | **20.31** | **12.36–12.60** | **~11.9** |

The original ~2.2 ms target assumed an fp32 tail after C. With E, the measured sampler is 1.57 ms, and
about 1.0 ms after D.

**What the wall clock showed.** On a 7950X3D, the batch-512 decode step went 20.36 → 12.27 ms, and
throughput rose +25.4 %. The step is now level with the host floor: host `fwd` is 9.6–10.2 ms and
`fwd_gpu` is 10.6 ms. As this plan predicted, the sampler is no longer the bottleneck. The next levers
are:

* **host time per step**, through decode CUDA graphs for `fwd`;
* **prefill**, which is 69–86 % of the run at batch ≥ 512.
