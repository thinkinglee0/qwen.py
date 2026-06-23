# qwen.py — A Qwen2.5 Inference Engine from Scratch

A from-scratch implementation of the Qwen2.5 forward pass in **pure PyTorch**, built to
understand LLM inference at the mechanism level. The model code (attention, RoPE, RMSNorm, SwiGLU, weight loading) depends only on `torch` + `safetensors`; `transformers` is a **dev-only** dependency, used solely for the tokenizer and as the reference model during numerical validation.

Development target: **Qwen2.5-0.5B** (fp32, CPU). Performance target: **Qwen2.5-7B on an NVIDIA A10**.

---

## Status

| Milestone | Scope                                                                    | State                                                                        |
| --------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| **M1**    | Forward pass — embed → 24 × decoder block → final norm → logits          | ✅ done, logits validated against the `transformers` reference layer-by-layer |
| **M2**    | KV cache (incremental decode, `past_len` plumbing)                       | 🔜 next — `past_len` already stubbed in `forward()`                          |
| later     | sampling / generation loop, batched decode, performance work on 7B / A10 | planned                                                                      |

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

---

## Repository layout

| File                                       | Responsibility                                                                                                                        |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                                | `ModelConfig` dataclass, `from_pretrained` (parse `config.json`), `load_qwen_weights` (safetensors + dtype + tied-embedding handling) |
| `rope.py`                                  | `BaseRoPE` + `Default` / `Linear` / `DynamicNTK` variants, `init_rope` factory                                                        |
| `qwen.py`                                  | `QwenModel` — the full forward pass                                                                                                   |
| `t.py`                                     | Validation harness: hooks + monkey-patch to capture intermediates and `compare` against the `transformers` reference                  |
| `docs/qwen25_inference_alignment_notes.md` | Engineering notes — the pitfalls hit while aligning against HuggingFace, and the methodology used to find them                        |

---

## Quickstart

```bash
pip install torch safetensors transformers

# fetch the dev model (≈1 GB)
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./qwen2.5-0.5b

# point the harness at the weights, then run the reference comparison
python t.py
```

`t.py` resolves the model directory via `model_dir` near the top of the file — set it to
wherever you placed the weights. A passing run prints aligned logits between `qwen.py` and the `transformers` reference, and the same next-token prediction for the prompt
`"The capital of France is"`.

Minimal inference:

```python
from transformers import AutoTokenizer
from qwen import QwenModel

model_dir = "./qwen2.5-0.5b"
tok = AutoTokenizer.from_pretrained(model_dir)
model = QwenModel(model_dir)

ids = tok("The capital of France is", return_tensors="pt").input_ids
logits = model(ids).logits                 # [1, seq, vocab]
next_id = logits[0, -1].argmax(dim=-1)     # greedy
print(tok.decode(next_id))                 # -> " Paris"
```

---

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

1. **M2 — KV cache.** Replace `past_len = 0` with the running cache length; cache K/V per layer; mask becomes `(q_len, past_len + q_len)`. Validate incremental decode against a full re-prefill.
2. **Generation loop & sampling** — greedy + temperature / top-p, decoupled from the model.
3. **Batched decode** — padding / position handling for `bsz > 1`.
4. **Performance** — move to GPU, profile against the 7B / A10 target.

---

## License

TODO — add a license.
