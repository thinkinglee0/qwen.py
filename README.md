# qwen.py — A Qwen2.5 Inference Engine from Scratch

A from-scratch implementation of the Qwen2.5 forward pass in **pure PyTorch**, built to
understand LLM inference at the mechanism level. The model code (attention, RoPE, RMSNorm, SwiGLU, weight loading) depends only on `torch` + `safetensors`; `transformers` is a **dev-only** dependency, used solely for the tokenizer and as the reference model during numerical validation.

Development target: **Qwen2.5-0.5B** (fp32, CPU). Performance target: **Qwen2.5-7B on an NVIDIA RTX 4090**.

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
| **M7**    | performance profiling     | 🔜 next. performance profiling of continuous batching on NVIDIA 4090    |
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

stay tuned

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

1. **Performance** — move to GPU, profile against the 7B / RTX 4090 target.
2. **Continuous batching** — evict finished requests and add waiting request in flight.

## License

TODO — add a license.
