# qwen.py — A Qwen2.5 Inference Engine from Scratch

A from-scratch implementation of the Qwen2.5 forward pass in **pure PyTorch**, built to
understand LLM inference at the mechanism level. The model code (attention, RoPE, RMSNorm, SwiGLU, weight loading) depends only on `torch` + `safetensors`; `transformers` is a **dev-only** dependency, used solely for the tokenizer and as the reference model during numerical validation.

Development target: **Qwen2.5-0.5B** (fp32, CPU). Performance target: **Qwen2.5-7B on an NVIDIA A10**.

---

## Status

| Milestone | Scope                                                           | State                                                                        |
| --------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **M1**    | Forward pass — embed → 24 × decoder block → final norm → logits | ✅ done, logits validated against the `transformers` reference layer-by-layer |
| **M2**    | KV cache (incremental decode, `past_len` plumbing)              | ✅ done, logits validated against the non-kv-cache mode by pytest             |
| **M3**    | Streaming HTTP service                                          | ✅ done, introducing FastAPI, async                                           |
| **M4**    | static batching                                                 | 🔜 next — `bsz`already stubbed with `1`                                      |
| later     | continuous batching, performance work on 7B / A10               | planned                                                                      |

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

---

## Repository layout

| File                                       | Responsibility                                                                                                                                                                                                                                                                                           |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                                | `ModelConfig` dataclass, `from_pretrained` (parse `config.json`), `load_qwen_weights` (safetensors + dtype + tied-embedding handling)                                                                                                                                                                    |
| `rope.py`                                  | `BaseRoPE` + `Default` / `Linear` / `DynamicNTK` variants, `init_rope` factory                                                                                                                                                                                                                           |
| `qwen.py`                                  | `QwenModel` — the full forward pass, generation loop and async generation loop                                                                                                                                                                                                                           |
| `main.py`                                  | FastAPI endpoints                                                                                                                                                                                                                                                                                        |
| `test_qwen.py`                             | unit tests by pytest for `QwenModel`, including FastAPI endpoints, and comparasons between `prefill(seq)` and `prefill(sub_seq) + decode(rest_seq)` step by step , between `QwenModel` and `modeling_qwen2.py`, between that with `KV cache` and non-`KV cache`, between `generate` and `async_generate` |
| `test_rope.py`                             | unit tests by pytest for `rope.py`                                                                                                                                                                                                                                                                       |
| `utils.py`                                 | some functions                                                                                                                                                                                                                                                                                           |
| `docs/qwen25_inference_alignment_notes.md` | Engineering notes — the pitfalls hit while aligning against HuggingFace, and the methodology used to find them                                                                                                                                                                                           |

---

## Quickstart

```bash
pip install torch safetensors transformers fastapi uvicorn pytest_asyncio pytest

# fetch the dev model (≈1 GB)
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ../qwen2.5-0.5b

# verify
pytest

# start FastAPI service
uvicorn main:app --host 0.0.0.0 --port 8001

# test
$ curl -N -X POST "http://127.0.0.1:8001/generate_stream_plain"      -H "Content-Type: application/json"      -d '{"prompt": "The capital of France is", "max_len": 200}'
The capital of France is Paris. It is the largest city in Europe and the third largest city in the world. It is located in the south of France, on the banks of the Seine River. It is situated on the Île de la Cité, which is a small island in the center of the city. The city is surrounded by the Seine River and the Mediterranean Sea. It is also surrounded by the Pyrenees mountains. The city is known for its beautiful architecture, its rich history, and its beautiful parks and gardens. Paris is a city of contrasts, with its modern and old parts, its rich and poor parts, and its diverse and multicultural population. It is a city of art, culture, and science, and it is a city of innovation and progress. Paris is a city of love, and it is a city of hope. It is a city of dreams, and it is a city of reality. It is a city of beauty, and it is a city of wonder. Paris
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

1. **Sampling** — greedy + temperature / top-p, decoupled from the model.
2. **Batched decode** — padding / position handling for `bsz > 1`.
3. **Performance** — move to GPU, profile against the 7B / A10 target.

---

## License

TODO — add a license.
