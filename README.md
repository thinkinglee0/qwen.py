# qwen.py — A Qwen2.5 Inference Engine from Scratch

A from-scratch implementation of the Qwen2.5 forward pass in **pure PyTorch**, built to
understand LLM inference at the mechanism level. The model code (attention, RoPE, RMSNorm, SwiGLU, weight loading) depends only on `torch` + `safetensors`; `transformers` is a **dev-only** dependency, used solely for the tokenizer and as the reference model during numerical validation.

Development target: **Qwen2.5-0.5B** (fp32, CPU). Performance target: **Qwen2.5-7B on an NVIDIA RTX 4090**.

> *Special contributor: **Claude** maintaining `docs` and the profiling/visualizing tools (`tests/test_profile.py`, `benchmark/tool/bench_viz.py`)*

---

## Status

| Milestone | Scope                                                           | State                                                                                                                                                                                                                                                                    |
| --------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **M1**    | Forward pass — embed → 24 × decoder block → final norm → logits | ✅ done, logits validated against the `transformers` reference layer-by-layer                                                                                                                                                                                             |
| **M2**    | KV cache (incremental decode, `past_len` plumbing)              | ✅ done, logits validated against the non-kv-cache mode by pytest                                                                                                                                                                                                         |
| **M3**    | Generation loop, Streaming HTTP service                         | ✅ done, introduce FastAPI, async                                                                                                                                                                                                                                         |
| **M4**    | sampling, restructure the project layout                        | ✅ done, add repetition/frequency/presence penalties, temperature, top_k/top_p, multinomial; isolate source code from unit tests; extract attention/mlp/decode_layer/norm from model.py, and bind weights to the `nn.Module` tree through the `load_state_dict` function. |
| **M5**    | static batching                                                 | ✅done. pack a list of **variable-length** id sequences into a 1-dim id list, opt for SDPA attention in my local macbook for quick functional verifications; add request to `StaticScheduler`, and then scheduler in a fixed batch.                                       |
| **M6**    | continuous batching                                             | ✅done. continuous batching with paged kv cache; two interchangable scheduling strategies (preemptive_schedule and D_first_preemptive_schedule); watermark block reservation; expential backoff; KVCache.verify_invariant periodically checks; metrics (schedule metrics, step metrics, and request metrics).|
| **M7**    | performance profiling     | ✅ done for the 0.5B baseline. Three rounds on an RTX 4090 — concurrency sweeps (batch 1 → 1 024) and a decode-step profiler reporting GPU idle fraction from the chrome trace, every configuration run **twice** for a measured noise floor. Reports: [log910](./docs/performance_analysis_log910.md), [log914](./docs/performance_analysis_log914.md), [log915](./docs/performance_analysis_log915.md) ([中文](./docs/performance_analysis_log915.zh.md)) and the switch matrix [log915](./docs/performance_switches_log915.md) ([中文](./docs/performance_switches_log915.zh.md)). |
| **M8**    | async scheduling (one decode step of lookahead)            | ✅ done. `step_0` launches a step with no sync; `step_1` drains the previous one. Request state splits into projected/actual so the next batch is planned before the current lands; token ids and penalty masks stay resident on device. **+20.2 % peak throughput, +46.4 % at batch 128, GPU idle in a pure-decode step 27.5 % → 1.5 % with kernel time unchanged (+0.8 %)**. Report: [log928](./docs/performance_analysis_log928.md). |
| later     | Fused kernels in Trition | planed  |

Correctness is the gate for every milestone: a milestone is "done" only when its activations match the reference within tolerance (see [Validation](#validation)).

---

## What's implemented

**Continuous batching & scheduling**
- **Continuous batching** — scheduling runs per forward step instead of per fixed batch: a request is admitted as soon as there's room and evicted the moment it finishes, instead of waiting on the slowest member of a static batch.
- **Two interchangeable scheduling strategies** — `preemptive_schedule` lets Prefill (Ps) and Decode (Ds) requests interleave within a step; `D_first_preemptive_schedule` always schedules Ds before Ps to protect in-flight decode latency, which changes its `_pick_victim` logic accordingly. See `scheduler.py`.
- **Chunked prefill** — long prompts are split across steps (`long_prefill_token_threshold`) so they can't stall decode-phase requests and blow up ITL P99.
- **Exponential backoff on preemption** — a preempted request is barred from re-admission for `backoff_base ** preempt_count` steps (capped at `backoff_cap`), avoiding thrash under sustained cache pressure.
- **Watermark block reservation** — a reserved slice of the KV pool (`respect_watermark`) so already-running requests can't be starved by new admissions.
- **Recompute on preemption** — a preempted request drops its KV cache and returns to the waiting queue, keeping its already-generated output tokens; those tokens are replayed as input when it's rescheduled.
- **Truncation tracking** — requests that hit `max_new_tokens` before EOS are counted separately (`num_truncated`) from normal completions.
- **Async scheduling — one step of lookahead** — `step_0` schedules and launches forward + sample without a single device sync; `step_1` drains the *previous* step (event sync, commit, release). The host never waits on the step it just issued, which is what takes GPU idle in a pure-decode step from 27.5 % to 1.5 % at batch 512. `in_flight_steps` is a deque and `ModelRequest.num_in_flight` is a count, so nothing assumes a pipeline depth of one.
- **Projected vs actual request state** — with a step in flight, "what this request has computed" and "what the scheduler has committed it to" are different numbers. `num_scheduled_tokens` / `projected_is_decoding` / `projected_finished` drive scheduling and KV-block accounting; `num_computed_tokens` / `output_ids` / `finished` advance only when the step lands. Preemption splits the same way, and `_settle_and_check` re-admits a waiting request only once no in-flight batch can still mutate its actual state.
- **Resident per-step buffers** — `RequestBuffer` holds one pinned host tensor and one device tensor per per-step quantity (slot index, `want`, `needs_sample`, cache slots, block table), allocated once at capacity and refilled through numpy views, so a step costs one async H2D per row instead of fresh tensor construction. `SchedulerOutput.s_infos` is a list positionally aligned with `reqs`, replacing a dict keyed by `request_id`.

**Paged KV cache**
- **Block-based, ref-counted paged cache** (`cache.py`) — fixed-size blocks (`block_size=256`, required to be a multiple of 256 by flash-attn's paged-KV kernel) allocated from a `BlockPool`; `verify_invariant` runs periodically to catch block-table/ref-count drift (cache leaks) early.
- **Explicit teardown** — `KVCacheData`, `KVCache`, `Scheduler`, and `LLMEngine` each expose `teardown()` to drop KV-cache/model references deterministically instead of waiting on GC; `ServingDriver.stop()` tears its engine down automatically.

**Attention & model**
- **Flash and SDPA attention** — `flash_attn_varlen_func` with a paged `block_table` on CUDA; falls back to a reference SDPA-over-gathered-cache path (`sdpa_from_cache`) for functional verification on machines without flash-attn (e.g. a local MacBook).
- **RoPE** — `default`, `linear`, and `dynamic-NTK` scaling variants behind a common base class, selected from `config.rope_scaling`. Half-dim cos/sin cache; rotation convention is bit-equivalent to HF's `rotate_half`.
- **RMSNorm** — variance computed in fp32 then cast back, identical to the reference.
- **SwiGLU MLP** — `silu(gate_proj(x)) * up_proj(x) → down_proj`.
- **Causal masking** — additive `-inf` mask built once per forward; generalizes to `k_len > q_len` for the KV-cache case.
- **Tied embeddings** — `lm_head` falls back to `embed_tokens.weight` when `tie_word_embeddings=True` (0.5B); a separate `lm_head.weight` is used when present (7B).
- **Weight loading** — `safetensors` → flat dict, dtype cast, config parsed from `config.json`/`generation_config.json` (via `orjson`) into a typed dataclass.
- **Static batching** — packs a list of variable-length id sequences into a 1-D id list (`pack_sequences`, `scatter_to_kv_cache`) and schedules them as a fixed batch via `StaticScheduler`; kept alongside continuous batching for comparison/benchmarking.

**Sampling**
- **Sampling** — parses `generation_config.json`; applies repetition/frequency/presence penalties right after `forward`, then samples (temperature, top_k, top_p, multinomial) when `do_sample` is on.
- **Device-resident penalty masks** — `TokenIdTable` keeps the full token-id history on device, one row per request slot, and derives `output_counts` plus the prompt/output presence masks there (`bin_count_and_mask`): both masks share one `[B, 2V+16]` buffer, so it is a single memset and a single scatter. The sampler needs nothing from the host, which is the precondition for launching a step without a sync.
- **Sync-free sampler setup** — `apply_top_k` reads `max_k` from a pinned host mirror of the top-k column instead of `top_k.max().item()`; `MAX_EFFECTIVE_TOP_K = 1024` rounds a larger top-k up to a no-op rather than paying a wider `topk()` for something statistically indistinguishable from none. `SamplingParams.validate()` rejects out-of-range parameters at submission.
- **Known cost** — `apply_top_p` still sorts the full 151 936-wide vocabulary, which is now **30 % of a batch-512 decode step** and the cause of the batch-1024 OOM. See [§2.3](#23-async-scheduling--the-async_scheduling-ab-log928) and roadmap item 1.

**Serving & observability**
- **Streaming HTTP service** — FastAPI `/generate_stream` and `/health`; `async_generate` interleaves `_decode_step` into the running event loop so one worker serves multiple concurrent streams; `@asynccontextmanager`/`@pytest_asyncio.fixture`/`@pytest.fixture` ensure model weights load only once across sync, async, and endpoint tests.
- **Three-tier metrics** — logged as JSON per run: **request metrics** (TTFT/TPOT/ITL per request), **scheduler metrics** (cumulative preemption/cache-exhaustion/truncation/reschedule counters), and **per-step metrics** (batch size, prefill/decode token counts, KV-block utilization, step latency).
- **Benchmarking** — a functional benchmark plus a ShareGPT-based benchmark for performance profiling (see [Performance profiling](#performance-profiling)).

---

## Performance profiling

**Dataset**: `ShareGPT_V3_unfiltered_cleaned_split.json`

**Random seed for shuffle**: 0

### 1. Static Batching

#### 1.1 Platform: CPU Intel Core i7

**Max number of output tokens**: 512

**Attention**: SDPA(Scaled Dot-Product Attention)

**Conclusion**: 

1. **Compute-bound in Prefill phase**: `Prefill_mean ∝ batch_size`, and it degrades along `batch_size` increasing, so it's compute-bound.

2. **Sweet point** lies in batch size 16 - 32.
   
   Given `decode throughput = 1000*batch_size/ITL_mean`,
   
   `decode throughputs`: [8.13, 17.24, 31.24, 52.76, 70.08, 84.89, 90.57]
   
   `decode throughput ratio`: [**1.12**, 0.81, 0.69, **0.33**, **0.21**, 0.07], 
   
   `ITL ratio`: [-0.06, 0.1, 0.18, **0.51**, **0.65**, 0.87]

3. **Anomaly analisis**: decode throughput ratio is 1.12 from batch_isze 1 to 2, greater than 1. It was caused by CPU existing from Turbo mode due to my operations (1. `caffeinate -i -m`; 2. run benchmark; 3. press power button of my MacBook).

Full per-batch-size latency table: [`docs/static_batching_cpu_benchmark.md`](./docs/static_batching_cpu_benchmark.md).

#### 1.2 Platform: NVIDIA RTX 4090

stay tuned

### 2 Continuous Batching

#### 2.1 Platform: NVIDIA RTX 4090

**Model**: Qwen2.5-0.5B-Instruct, bf16 · **Workload**: 512 random ids in, 128 tokens out, EOS ignored, closed-loop

Three rounds, every configuration swept **twice with no code change in between** — so the noise floor
is measured, not assumed. The latest round reproduces to **1.21 % on all 11 batch sizes**, which is
the bar every claim below has to clear.

| | log910 | log914 | **log915** |
| --- | --- | --- | --- |
| `do_sample` | false (penalties + `argmax`) | true | true |
| `pre_gather_cos_sin` | — | declared false, **ignored by the code** | false, **honoured** |
| Peak throughput | 7 028 tok/s @ 1024 | 5 300 tok/s @ 1024 | 5 095 tok/s @ 1024 |
| TPOT @ batch 1 | 22.8 ms | 19.0 ms | 22.9 ms |
| Last profitable doubling | batch 256 | batch 256 | **batch 128** |
| Run-to-run spread | — | ≤ 5.5 % | **≤ 1.21 %** |
| Report | [log910](./docs/performance_analysis_log910.md) | [log914](./docs/performance_analysis_log914.md) | [log915](./docs/performance_analysis_log915.md) · [中文](./docs/performance_analysis_log915.zh.md) |

The log915 sweep is also plotted —
[throughput / latency trade-off, six panels](./docs/attachments/concurrency_sweep.log915.baseline.run1.png)
([run 2](./docs/attachments/concurrency_sweep.log915.baseline.run2.png)) — where the marginal-effect
panel shows the last profitable doubling directly.

The three baselines are **not** directly comparable: `do_sample` was off in log910, and
`pre_gather_cos_sin` was a declared-but-unread flag until log915 (so log914's "baseline" was silently
running with pre-gather on). Each report states its own configuration; the structural findings below
hold across all three.

**Conclusions**:

1. **The GPU is not the limit — it is idle 50–85 % of every decode step.** 3.3 ms of kernels inside a
   19.1 ms step at batch 1. At peak throughput the model's weight traffic is ~5.8 GB/s against the
   card's 1 008 GB/s: **4.3 % MBU at batch 1, 2.6 % MFU at batch 1 024.**

2. **The model forward costs ~22 ms regardless of what goes into it** — 20.5 ms at batch 1 and
   23.5 ms at batch 1 024, +14 % for 1 024× the tokens; a prefill step carrying 7 511 tokens spends
   77 ms in the same window. That is ~280 eager kernel launches per step with ~50 µs of gap between
   them, not model compute. It sets the **latency floor** and is why batch 1 reaches only 44 tok/s.
   `rope` alone is 28 % of it.

3. **Sampling is the throughput ceiling, and it scales linearly with batch** — **60.4 % of total
   wall-clock time** at batch 512, 82 % of a decode step at batch 1 024, a **127×** rise for a 1 024×
   batch. Two full-vocabulary passes per step are responsible: `apply_top_p` sorts all 151 936 logits
   per sequence (a 78 M-element sort at batch 512), and `apply_penalties` materialises
   `[batch, vocab]` fp32 tensors (311 MB at batch 512, 622 MB at batch 1 024). With `top_k = 20`,
   top-p only needs the 20 survivors ranked.

4. **Chunked prefill pays the sampling bill twice.** A prefill step also samples every running decode
   row, so at batch 1 024 prefill steps hold 54 % of all steps and **63.7 s of their 149 s is
   sampling**, against 79.5 s of genuine prefill compute.

5. **The kernels themselves are healthy.** The LM-head projection hits 146 TFLOP/s (**88 % of the
   4090's BF16 peak**) and prefill reaches 70 TFLOP/s (42 % MFU) on 8 192-token steps. The deficit is
   orchestration, not arithmetic. Zero preemptions and zero cache exhaustions across 11 batch sizes
   × 2 runs, in every round.

> Conclusions 1 and 2 describe the **pre-M8** engine, and only at large batch. Async scheduling (§2.3)
> removed that idle without touching a kernel — a pure-decode step at batch 512 is now 1.5 % idle, not
> 27.5 % — but **batch 1 is still 68 % idle**, still launch-bound, and still needs CUDA graphs
> (roadmap item 2). Conclusion 3 survived intact and is now the whole story: see §2.3 conclusion 4.

#### 2.2 Optimisation switches

Measured one at a time in log915 — [report](./docs/performance_switches_log915.md) ·
[中文](./docs/performance_switches_log915.zh.md):

| Switch | GPU kernel time | Step wall time | Verdict |
| --- | --- | --- | --- |
| `pre_gather_cos_sin` | −0.3 … −4.2 % | **−1.6 … −10.9 %** | **Keep it on** (and it already defaults to `True`) — the only switch that cuts host *and* device work |
| `compile_rope` | −1.0 … −12.3 % | **+1.2 … +4.7 %** | Off. Measured three times, negative three times: it trades GPU time for more wall time |
| `stage_sampling_params` | ±0.1 % | −0.1 … −1.8 % | Off. Consistent in direction, never outside noise — its mechanism caps the win at ~0.4 ms/step |

**In a launch-bound engine, cutting GPU work is not merely low-value — it is negative-value.**
`compile_rope` genuinely removes ~0.4 ms of kernel time per step and still makes the step slower,
because the host pays more to reach the fused kernel (48 guard evaluations per step). That sign
should flip once the decode path is CUDA-graphed; until then, kernel-level tuning is premature.

**Reading the metrics** — five traps, documented in
[log914 §7](./docs/performance_analysis_log914.md#7-three-traps-in-this-instrumentation) and
[log915 switches §1.2](./docs/performance_switches_log915.md), and
[log928 §7](./docs/performance_analysis_log928.md):

* CPU-side timings are meaningless in mixed prefill/decode steps — the host is blocking on the GPU backlog there.
* A `*_gpu` field is a CUDA-event **window**, not busy time; gaps inside it are counted.
* `rope_gpu` at batch 1 024 is a known-bad metric (per-layer event merge).
* **`torch.profiler` leaves ~30 % of host overhead behind in the process**, so every batch point after
  the first in a profiling session is inflated — and with it the reported GPU idle fraction. Confirmed
  by isolation: give each batch size its own pytest process and the two harnesses agree to **0.05 %**
  at batch 512 (100.218 vs 100.17 ms), having been 5 % apart; the reported idle fraction there drops
  52.3 % → 50.2 %. The sweep harness is unaffected; where the two disagree, trust the sweep.
* **On the async path, `step_0 + step_1` double-counts `step_0`.** `step_1` is started on iteration *N*
  but stopped inside iteration *N+1*'s drain, so it already spans a full step: it reads as 104 ms at
  batch 512 where the step is 52 ms. **Use `step_1` as the step wall time** — its median matches the
  profiler's independent `wall_clean` at every batch size. For the same reason host-side `fwd` and
  `rope` are now launch-queue back-pressure, not work (46.8 ms CPU against 10.7 ms `fwd_gpu` at batch
  512); reading them as regressions inverts the meaning.

#### 2.3 Async scheduling — the `async_scheduling` A/B (log928)

**Workload**: as above · `max_model_len=1024`, `compile_rope=false`, **`pre_gather_cos_sin=true`**
(so this round is not pessimistic the way the published baselines are)

Both branches on **one** box back to back, two runs each, `async_scheduling` first —
[report](./docs/performance_analysis_log928.md). The `main` column here *is* the pre-M8 baseline on this
box; [log924](./docs/performance_analysis_log924.md) is the last standalone measurement of that engine,
and log928 cites its noise floor. These numbers are **not** comparable to log910/914/915
above: different vast instance (450 W enforced cap, driver 550, and the PCIe link negotiated **gen 1**)
and a different config. Only the branch delta is meaningful, and the gen-1 link flatters a change whose
point is fewer host↔device round-trips.

| batch | `main` tok/s | branch tok/s | Δ | `main` TPOT | branch TPOT | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 102 | 100 | −1.9 % | 9.83 ms | 10.01 ms | +1.9 % |
| 16 | 1 345 | 1 366 | +1.6 % | 11.49 ms | 11.21 ms | −2.5 % |
| 64 | 3 557 | 4 919 | **+38.3 %** | 17.53 ms | 12.17 ms | **−30.6 %** |
| 128 | 4 638 | 6 791 | **+46.4 %** | 27.00 ms | 17.77 ms | **−34.2 %** |
| 512 | 6 015 | 7 232 | **+20.2 %** | 83.31 ms | 68.23 ms | **−18.1 %** |
| 1 024 | 5 861 | **OOM** | — | 169.68 ms | — | — |

Run-to-run throughput spread: ≤ **1.26 %** on `main`, ≤ **3.22 %** on the branch (worst case batch 1).
Every gain above is at least 6× that noise floor.

**Conclusions**:

1. **The kernels never changed.** GPU busy time per step agrees within **1.6 % at every batch size**
   (+0.8 % at batch 512). The whole gain is host stalls that stopped happening: GPU idle in a
   pure-decode step falls **27.5 % → 1.5 %** at batch 512 and **42.0 % → 7.1 %** at batch 128. Where
   §2.1 said "the GPU is idle 50–85 % of every decode step", the answer was not a faster kernel.

2. **19.4 ms of a 70.5 ms step was host work the GPU could not overlap.** At batch 512, per step:
   `sched` 1.274 → 0.494 ms, `sched_ret_gpu` (device work inside the scheduler) 0.831 → **0.000** ms,
   host time in the sampler 56.054 → **1.089** ms, `dth` (the token read-back) 2.913 → **0.036** ms.
   Residual unoverlapped host time is now 0.8 ms inside a 52.3 ms step.

3. **Same throughput at a quarter of the latency.** `main` needs batch 512 and 83.3 ms TPOT to reach
   6 015 tok/s; the branch beats that at batch 128 with **17.8 ms TPOT** — and at that operating point
   it is better on every axis, prefill latency included. Scaling efficiency at batch 64 holds
   **0.76–0.78** against `main`'s 0.54–0.55.

4. **Sampling is now 79 % of all device work** — 40.6 of 51.5 ms/step at batch 512, against ~9 ms for
   the model itself. A single `aten::sort` inside `apply_top_p` is **15.4 ms, 30 % of the whole step**,
   sorting all 151 936 vocabulary columns *after* top-k has cut the live candidates to 20 — and
   `torch.topk` already returns its values sorted. Roadmap item 1 is no longer merely the largest win
   available; it is the only one left of that size.

**Two costs, both measured**:

* **Prefill latency +37 … 78 % for batch ≥ 32** (121.6 → 216.6 ms at batch 512). A request's first
  token is produced by step *N* but committed during step *N+1*'s drain, so TTFT gains one whole
  following iteration — which at high concurrency is often another 8 192-token prefill chunk. Decode
  ITL improves at every percentile (p50 70.3 → 52.4 ms, p90 125.4 → 119.5 ms at batch 512), so the
  cost is confined to the first token. This is the standard cost of async scheduling, but it is not
  free.
* **Batch 1 024 OOMs** where `main` degraded to 5 861 tok/s: `apply_top_p`'s full-vocabulary
  `torch.sort` asks for **1.73 GiB in one allocation** with peak memory already at 24 200 of
  24 564 MiB, because the pipeline keeps one extra step's tensors reachable. The engine now caps at
  512 concurrency on a 24 GB card. Roadmap item 1 removes the allocation.

The branch is also **less steady**: 1–4 anomalous steps per 20-step window against `main`'s 0, and
batch 256's idle fraction (12.6 %) is worse than batch 128's (7.1 %). The spike is **device-side**
(`fwd_gpu` 64 ms against a 7 ms local median), not a host stall, which makes allocator pressure the
leading hypothesis — same root cause as the OOM. Roadmap item 3.

---

## Quickstart

build the development platform, see [vast-evn-build.md](./env/vastai/vast-evn-build.md) for details.

```bash
# verify
pytest

# start FastAPI service
uvicorn qwen.api:app --host 0.0.0.0 --port 8001

# test
curl -N -X POST "http://127.0.0.1:8001/generate_stream_plain"      -H "Content-Type: application/json"      -d '{"prompt": "The capital of France is", "max_new_tokens": 400}'
The capital of France is Paris. The French language belongs to the Romance languages and was spoken in France from 12th century onwards until 1804 when it was banned due to its influence on French culture.
Paris, the capital city of France, has been a UNESCO World Heritage Site since 1985. It was also listed as a City of History and Culture in 2013 by the Government of France.
Paris is home to many famous landmarks such as Notre Dame Cathedral, the Louvre Museum, the Eiffel Tower, Champs-Élysées, and the Arc de Triomphe.
It's important to note that there are many other cities with their own unique cultural attractions. Some examples include:
- Lyon: Known for its stunning medieval architecture
- Nice: Famous for its beautiful beaches and historic harbor
- Marseille: Home to the famous Port du Plein and its vibrant nightlife scene
- Toulouse: A city known for its wine industry and rich history
In conclusion, Paris is a major cultural hub and a UNESCO World Heritage site that offers visitors an opportunity to explore its rich history, architecture, and diverse cultural offerings. Its status as a UNESCO World Heritage Site underscores its importance as a global cultural and historical center. Visitors can enjoy breathtaking views of the Seine River, marvel at the iconic Notre Dame Cathedral, or take a stroll through the charming streets of the Latin Quarter. The city is also renowned for its cuisine, art, music, and fashion. Whether you're a fan of French culture, gastronomy, or simply looking for a relaxing destination, Paris is sure to offer something special. So if you ever find yourself in Paris, don't miss out! ���✨
Note: The information provided here is general and may not reflect current events or specific locations. Always check local authorities' latest updates before visiting any location. #ParisCulture #History #Cuisine #Relaxation #WorldHeritage #UNESCO #France ��[DONE]
```

`pytest` results all pass as expected. The letters from the `curl` response display like a typewriter.

## Validation

The engine is validated by **layer-by-layer activation alignment** against
`transformers.models.qwen2.modeling_qwen2`: run both models on the same input, capture intermediate tensors, and compare in execution order. The first mismatch localizes the bug; everything downstream is just propagation.

Capture uses PyTorch forward hooks for `nn.Module` outputs and module-level monkey-patching for inline functions like `apply_rotary_pos_emb` that aren't hookable. Comparison uses `torch.testing.assert_close` on tensors (which reports mismatch fraction and the largest abs/rel diff with its index) and `==` on integers. Two requirements make the comparison valid:

- Load the reference with `attn_implementation="eager"` — the fused SDPA/FlashAttention kernels differ in accumulation order and produce benign ~1e-3 diffs that masquerade as bugs.
- Match dtype on both sides (fp32 ↔ fp32 here) so tolerances stay tight.

The full set of pitfalls found this way — RoPE `inv_freq` exponent, `view`/`transpose` memory layout, GELU-vs-SiLU, hook signatures, batch-dim indexing, and more — is written up in [`qwen25_inference_alignment_notes.md`](./docs/qwen25_inference_alignment_notes.md) on **M1** milestone.

Other following verifications see `tests/` folder for details. The main ones `test/test_model.py` include:

**Against reference**: `test_prefill_matches_reference_on_math`, `test_prefill_matches_reference` and `test_decode_matches_reference`. **Troubleshoot** on ligits mismatch sees `HookManager` for details.

**Against self**: `test_kv_cache_correctness` (`prefill(L) == prefill(L-P) + decode(1)*P steps`). **Troubleshoot** on ligits mismatch sees `compare_cache_against_kv_after_rope` for details.

---

## Model architecture (Qwen2.5-0.5B)

|                  |                    |
| ---------------- | ------------------ |
| Layers           | 24                 |
| Hidden size      | 896                |
| Query / KV heads | 14 / 2 (GQA)       |
| Head dim         | 64                 |
| Activation       | SiLU (gated)       |
| Norm             | RMSNorm (pre-norm) |
| Position         | RoPE, θ = 1e6      |
| Embeddings       | tied               |

---

## Roadmap

Ordered by measured cost — see [log915 §7](./docs/performance_analysis_log915.md#7-recommendations-in-order)
and [log928 §6](./docs/performance_analysis_log928.md), which supersedes it where they differ.

1. **Rewrite sampling to work on the candidate set** — `topk(k)` already returns its values sorted, so
   run softmax → cumsum → threshold → `multinomial` over *k* and map back through the top-k indices,
   instead of sorting all 151 936 logits; penalties on the same narrowed block instead of a
   `[batch, vocab]` materialisation. Mathematically identical for any row where top-k is enabled.
   Now unambiguously the top item: post-M8 the sampler is **79 % of all device work** and its
   `aten::sort` alone is **30 % of a batch-512 step** (≈18 of 51.5 ms/step recoverable, TPOT
   52 → ~34 ms), and the same allocation is what OOMs batch 1 024. It still pays twice, because
   chunked-prefill steps sample every running decode row too.
2. **CUDA-graph the decode step** — now specifically a *small-batch* item. M8 took batch ≥ 128 to under
   8 % idle, but batch 1 is still **68 % idle**: the host needs ~10 ms to launch a step whose kernels
   occupy 3.2 ms. Shapes are static once the batch is fixed. This is the only change that improves
   single-stream latency, and the prerequisite for kernel-level tuning to have the sign one expects.
3. **Settle the allocator pressure behind the batch-256 stalls and the batch-1024 OOM** — run with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and log `num_alloc_retries` per step. If retries
   correlate with the 9× `fwd_gpu` spikes, §2.3's non-monotonic batch-256 idle fraction and the OOM
   are one bug, and item 1 mostly dissolves both. One env var, high information — do it before
   chasing anything else in that area.
4. **Reduce the pipeline's first-token cost** — §2.3 puts prefill latency up 37–78 % for batch ≥ 32,
   because a first token is committed one iteration late. Prioritising the drain of a step that
   produced a first token, or keeping prefill chunks out of the iteration immediately after an
   admission, should recover most of it without giving up the decode win.
5. **Make one-batch-per-process the harness default** — one pytest process per batch size is the
   verified fix for the ~30 % residual profiler overhead (log916), and log928 got it only from a shell
   loop. Every `gpu_idle_fraction` published from a multi-batch session is overstated by ~2 points for
   all but its first batch point.
6. **Promote `set_sync_debug_mode` from `warn` to `error`** — the decode path is sync-free as of M8 and
   `test_profile.py` already arms the warning across the measurement window. Making it fatal turns
   "no accidental sync" into a test rather than an observation, once the prefill path is clean too.
7. **Fused kernels in Triton**, once the orchestration overhead above no longer hides them.
8. **Scale to the 7B target** on the same harness.

## License

TODO — add a license.
