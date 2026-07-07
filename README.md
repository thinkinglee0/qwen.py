# qwen.py — A Qwen2.5 Inference Engine from Scratch

A from-scratch implementation of the Qwen2.5 forward pass in **pure PyTorch**, built to
understand LLM inference at the mechanism level. The model code (attention, RoPE, RMSNorm, SwiGLU, weight loading) depends only on `torch` + `safetensors`; `transformers` is a **dev-only** dependency, used solely for the tokenizer and as the reference model during numerical validation.

Development target: **Qwen2.5-0.5B** (fp32, CPU). Performance target: **Qwen2.5-7B on an NVIDIA A10**.

---

## Status

| Milestone | Scope                                                           | State                                                                                                                                                          |
| --------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **M1**    | Forward pass — embed → 24 × decoder block → final norm → logits | ✅ done, logits validated against the `transformers` reference layer-by-layer                                                                                   |
| **M2**    | KV cache (incremental decode, `past_len` plumbing)              | ✅ done, logits validated against the non-kv-cache mode by pytest                                                                                               |
| **M3**    | Generation loop, Streaming HTTP service                         | ✅ done, introducing FastAPI, async                                                                                                                             |
| **M4**    | sampling, restructure the project layout                        | ✅ done, repetition/frequency/presence penalties, temperature, top_k/top_p, multinomial; mv source code and test cases into src and tests folders respectively. |
| **M5**    | static batching                                                 | 🔜 next — `bsz`already stubbed with `1`                                                                                                                        |
| later     | continuous batching, performance work on 7B / A10               | planned                                                                                                                                                        |

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
- **Restructure the project layout** — rename `qwen.py` to `model.py`, `main.py` to `api.py`, put sync/async generations into `engine.py`, place source code files in `src/qwen` folder, and unit tests in `tests`.

---

## Repository layout

| File                                       | Responsibility                                                                                                                                                                                                                                                  |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/qwen/config.py`                       | `ModelConfig` dataclass, `from_pretrained` (parse `config.json` and `generation_config.json`), `load_qwen_weights` (safetensors + dtype + tied-embedding handling)                                                                                              |
| `src/qwen/rope.py`                         | `BaseRoPE` + `Default` / `Linear` / `DynamicNTK` variants, `init_rope` factory                                                                                                                                                                                  |
| `src/qwen/model.py`                        | `QwenModel` — the full forward pass                                                                                                                                                                                                                             |
| `src/qwen/engine.py`                       | generation loop and async generation loop                                                                                                                                                                                                                       |
| `src/qwen/api.py`                          | FastAPI endpoints                                                                                                                                                                                                                                               |
| `src/qwen/sampling.py`                     | repetition/frequency/presence penalties, temperature, top_k/top_p, multinomial;                                                                                                                                                                                 |
| `src/qwen/utils.py`                        | some common functions                                                                                                                                                                                                                                           |
| `tests/test_model.py`                      | unit tests by pytest for `QwenModel`, including FastAPI endpoints, and comparasons between `prefill(seq)` and `prefill(sub_seq) + decode(rest_seq)` step by step , between `QwenModel` and `modeling_qwen2.py`, between that with `KV cache` and non-`KV cache` |
| `tests/test_engine.py`                     | unit tests for generation loop, comparason between `generate` and `async_generate`                                                                                                                                                                              |
| `tests/test_api.py`                        | unit tests for FastAPI                                                                                                                                                                                                                                          |
| `tests/test_rope.py`                       | unit tests by pytest for `rope.py`                                                                                                                                                                                                                              |
| `docs/qwen25_inference_alignment_notes.md` | Engineering notes — the pitfalls hit while aligning against HuggingFace, and the methodology used to find them                                                                                                                                                  |

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

Capture uses PyTorch forward hooks for `nn.Module` outputs and module-level monkey-patching for inline functions like `apply_rotary_pos_emb` that aren't hookable. Comparison uses `torch.testing.assert_close` (which reports mismatch fraction and the largest abs/rel diff with its index). Two requirements make the comparison valid:

- Load the reference with `attn_implementation="eager"` — the fused SDPA/FlashAttention
  kernels differ in accumulation order and produce benign ~1e-3 diffs that masquerade as bugs.
- Match dtype on both sides (fp32 ↔ fp32 here) so tolerances stay tight.

The full set of pitfalls found this way — RoPE `inv_freq` exponent, `view`/`transpose` memory layout, GELU-vs-SiLU, hook signatures, batch-dim indexing, and more — is written up in [`qwen25_inference_alignment_notes.md`](./docs/qwen25_inference_alignment_notes.md).

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

1. **Batched decode** — padding / position handling for `bsz > 1`.
2. **Performance** — move to GPU, profile against the 7B / A10 target.

---

## License

TODO — add a license.
