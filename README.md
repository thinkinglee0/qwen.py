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
| **M4**    | sampling, restructure the project layout                        | ✅ done, add repetition/frequency/presence penalties, temperature, top_k/top_p, multinomial; isolate source code from unit tests; extract attention/mlp/decode_layer/norm from model.py, and bind weights to the `nn.Module` tree through the `load_state_dict` function. |
| **M5**    | static batching                                                 | ✅done. pack a list of **variable-length** id sequences into a 1-dim id list, opt for SDPA attention in my local macbook for quick functional verifications; add request to `StaticScheduler`, and then scheduler in a fixed batch.                                       |
| **M6**    | continuous batching                                             | 🔜 next                                                                                                                                                                                                                                                                  |
| later     | performance of static and continuous batchings on 7B / RTX 4090      | planed                                                                                                                                                                                                                                                                   |

Correctness is the gate for every milestone: a milestone is "done" only when its activations match the reference within tolerance (see [Validation](#validation)).

---

## What's implemented

- **GQA attention** — 14 query heads / 2 KV heads, `head_dim=64`; `repeat_kv` expands KV to query-head count *after* RoPE, matching HuggingFace's ordering.
- **RoPE** (`rope.py`) — `default`, `linear`, and `dynamic-NTK` scaling variants behind a common base class, selected from `config.rope_scaling`. Half-dim cos/sin cache; rotation convention is bit-equivalent to HF's `rotate_half`.
- **RMSNorm** — variance computed in fp32 then cast back, identical to the reference.
- **SwiGLU MLP** — `silu(gate_proj(x)) * up_proj(x) → down_proj`.
- **Causal masking** — additive `-inf` mask built once per forward; written so it generalizes to `k_len > q_len` (the KV-cache case in M2).
- **Tied embeddings** — `lm_head` falls back to `embed_tokens.weight` when
  `tie_word_embeddings=True` (0.5B); a separate `lm_head.weight` is used when present (7B).
- **Weight loading** — `safetensors` → flat dict, dtype cast, config parsed from `config.json` into a typed dataclass.
- **KV cache** — `prefill` and `decode` share the same `forward`; `init_kv_cache`while `cache`is None in `forward`function; return `cache` on the end of `forward` for the next iteration.
- **Streaming HTTP service** — `async_generate`throws `_decode_step`into the current `event loop`, and yields CPU after `_decode_step`returns; `/generate_stream`and `/health`endpoints implemented by FastAPI; `@asynccontextmanager`, `@pytest_asyncio.fixture` and `@pytest.fixture` ensure that the model **Weights** only loads **once** in testing scenarios of sync functions, async functions, and FastAPI endpoints.
- **Sampling** — parse `generation_conf.json`, apply repetition/frequency/presence penalties just after `forward`, then do sampling if `do_sample` swtich is on; sampling includes temperature, top_k, top_p, multinomial.
- **Restructure the project layout** — rename `qwen.py` to `model.py`, `main.py` to `api.py`, put sync/async generations into `engine.py`, place source code files in the `src/qwen` folder, and unit tests in `tests`.
- **Static batching** — pack a list of **variable-length** id sequence into an 1-dim id list by `pack_sequences`, `scatter_to_kv_cache` after the projection and rope of K and V; select flash_attn for cloud RTX 4090 VPS, falls back to SDPA attention in my locl macbook for quick functional verifications.
- **Benchmarking and statistic** — statisticize `TTFT`, `TPOT`, and `ITL` for each request, and triggered periodically after each step; add a regular benchmark for functionality verification and ShareGPT benchmark for performance profiling.

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

| Batch size /<br/>Request number | Prefill<br/>=TTFT - Queue_delay                                                                                                                            | TPOT                                                                                                                                                | ITL                                                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1/16                            | {<br/> "n": 512,<br/> "mean": 132.789,<br/> "std": 20.351,<br/> "p50": 123.835,<br/> "p90": 158.678,<br/> "p99": 200.014,<br/> "max": 238.765<br/> }       | {<br/> "n": 512,<br/> "mean": 123.055,<br/> "std": 6.967,<br/> "p50": 125.854,<br/> "p90": 127.2,<br/> "p99": 128.289,<br/> "max": 136.474<br/> }   | {<br/> "n": 53223,<br/> "mean": 123.057,<br/> "std": 10.703,<br/> "p50": 126.926,<br/> "p90": 130.222,<br/> "p99": 132.54,<br/> "max": 718.706<br/> }   |
| 2/32                            | {<br/> "n": 512,<br/> "mean": 154.021,<br/> "std": 32.566,<br/> "p50": 142.686,<br/> "p90": 198.823,<br/> "p99": 268.931,<br/> "max": 304.286<br/> }       | {<br/> "n": 512,<br/> "mean": 116.047,<br/> "std": 1.061,<br/> "p50": 115.814,<br/> "p90": 117.372,<br/> "p99": 120.021,<br/> "max": 121.719<br/> } | {<br/> "n": 50226,<br/> "mean": 116.033,<br/> "std": 5.726,<br/> "p50": 115.211,<br/> "p90": 119.312,<br/> "p99": 133.598,<br/> "max": 293.065<br/> }   |
| 4/64                            | {<br/> "n": 512,<br/> "mean": 248.721,<br/> "std": 72.967,<br/> "p50": 236.974,<br/> "p90": 351.284,<br/> "p99": 461.978,<br/> "max": 493.999<br/> }       | {<br/> "n": 512,<br/> "mean": 128.17,<br/> "std": 2.515,<br/> "p50": 128.205,<br/> "p90": 130.687,<br/> "p99": 133.46,<br/> "max": 138.427<br/> }   | {<br/> "n": 46170,<br/> "mean": 128.044,<br/> "std": 11.133,<br/> "p50": 127.213,<br/> "p90": 132.165,<br/> "p99": 149.898,<br/> "max": 1086.059<br/> } |
| 8/64                            | {<br/> "n": 512,<br/> "mean": 522.491,<br/> "std": 114.651,<br/> "p50": 504.209,<br/> "p90": 657.909,<br/> "p99": 809.124,<br/> "max": 809.124<br/> }      | {<br/> "n": 512,<br/> "mean": 151.551,<br/> "std": 3.149,<br/> "p50": 151.117,<br/> "p90": 156.75,<br/> "p99": 160.548,<br/> "max": 160.548<br/> }  | {<br/> "n": 41722,<br/> "mean": 151.636,<br/> "std": 9.636,<br/> "p50": 149.803,<br/> "p90": 159.913,<br/> "p99": 185.699,<br/> "max": 317.199<br/> }   |
| 16/64                           | {<br/> "n": 512,<br/> "mean": 993.516,<br/> "std": 148.212,<br/> "p50": 1009.939,<br/> "p90": 1172.503,<br/> "p99": 1313.25,<br/> "max": 1313.25<br/> }    | {<br/> "n": 512,<br/> "mean": 228.547,<br/> "std": 6.539,<br/> "p50": 227.93,<br/> "p90": 237.638,<br/> "p99": 243.826,<br/> "max": 243.826<br/> }  | {<br/> "n": 38854,<br/> "mean": 228.307,<br/> "std": 12.012,<br/> "p50": 226.788,<br/> "p90": 239.941,<br/> "p99": 276.275,<br/> "max": 351.846<br/> }  |
| 32/96                           | {<br/> "n": 512,<br/> "mean": 1871.921,<br/> "std": 191.012,<br/> "p50": 1918.657,<br/> "p90": 2165.994,<br/> "p99": 2214.558,<br/> "max": 2214.558<br/> } | {<br/> "n": 512,<br/> "mean": 376.967,<br/> "std": 7.716,<br/> "p50": 378.624,<br/> "p90": 384.873,<br/> "p99": 391.298,<br/> "max": 391.729<br/> } | {<br/> "n": 35717,<br/> "mean": 376.976,<br/> "std": 18.841,<br/> "p50": 376.089,<br/> "p90": 391.813,<br/> "p99": 447.585,<br/> "max": 595.678<br/> }  |
| 64/256                          | {<br/> "n": 512,<br/> "mean": 3860.661,<br/> "std": 375.43,<br/> "p50": 3868.442,<br/> "p90": 4454.093,<br/> "p99": 4454.093,<br/> "max": 4454.093<br/> }  | {<br/> "n": 512,<br/> "mean": 706.547,<br/> "std": 7.933,<br/> "p50": 707.658,<br/> "p90": 718.253,<br/> "p99": 718.253,<br/> "max": 718.253<br/> } | {<br/> "n": 33484,<br/> "mean": 706.657,<br/> "std": 32.015,<br/> "p50": 701.086,<br/> "p90": 739.056,<br/> "p99": 808.442,<br/> "max": 1032.269<br/> } |

#### 1.2 Platform: NVIDIA RTX 4090

stay tuned

### 2 Continuous Batching

stay tuned

---

## Quickstart

```bash
pip install torch safetensors transformers fastapi uvicorn pytest_asyncio pytest

# fetch the dev model (≈1 GB)
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ../qwen2.5-0.5b

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
