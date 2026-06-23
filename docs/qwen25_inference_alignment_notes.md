# Qwen2.5 Inference from Scratch: Activation Alignment Notes

Engineering notes from building a Qwen2.5 inference engine from scratch and validating it layer-by-layer against HuggingFace `transformers` (`modeling_qwen2.py`). Each section is a concrete pitfall hit during alignment, with the actual symptom, the diagnostic path that localized it, and the fix.

**Model under test:** Qwen2.5-0.5B (`hidden_size=896`, `num_attention_heads=14`,
`num_key_value_heads=2`, `head_dim=64`, `vocab_size=151936`, `rope_theta=1e6`). All shapes and numbers below come from a 5-token prompt (`bsz=1`, `q_len=5`).

**Environment:** macOS Intel, CPU-only, `torch_dtype=torch.float32`. See §8 for why float32 is the right choice for the alignment phase regardless of the eventual deployment dtype.

---

## 1. Strategy

Run your implementation and the reference model on identical input, capture intermediate activations at matching points, and compare them with `torch.allclose`. The first layer (or operation) that fails to match localizes the bug; everything downstream is just propagated error.

Two capture mechanisms are needed:

- **`register_forward_hook`** for anything that is an `nn.Module` (projections, norms,
  attention block, MLP, whole decoder layers).
- **Function-level monkey-patching** for things that are *not* modules — most importantly
  `apply_rotary_pos_emb`, which is a free function called inline inside
  `Qwen2Attention.forward()` and therefore cannot be hooked.

---

## 2. Hook Signatures: `forward_hook` vs `forward_pre_hook`

The two hook types have different signatures, and mixing them up fails in a way that does not name the real cause:

```python
# forward hook — fires AFTER forward(), receives (module, input, output)
def hook(m, i, o): ...

# pre-hook — fires BEFORE forward(), receives (module, input) only. No output arg.
def pre_hook(m, i): ...
```

**Symptom:** registering a 3-arg function as a pre-hook throws, mid-forward, deep inside
`torch/nn/modules/module.py`:

```
TypeError: save.<locals>.fn() missing 1 required positional argument: 'o'
```

The traceback points at `args_result = hook(self, args)` in `_call_impl`, not at your registration line, so it is easy to misread as a model bug rather than a signature mismatch.

**Fix:** match the signature to the hook type.

```python
cap = {}
def save_i(name):
    def pre_hook(module, input):
        val = input[0] if isinstance(input, tuple) else input
        cap[name] = val.detach().clone()    # clone instantly, avoiding modified by following flow
    return pre_hook
def save_o(name):
    def hook(module, input, output):
        cap[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    return hook

# example
ref.model.layers[0].self_attn.q_proj.register_forward_pre_hook(save_i("L0.attn_q_i"))
ref.lm_head.register_forward_hook(save_o("lm_head_o"))
```

**What can be hooked.** The module tree is the menu:

```
ref                          # AutoModelForCausalLM
└── model                    # Qwen2Model
    ├── embed_tokens          # Embedding
    ├── layers[i]             # Qwen2DecoderLayer  × num_layers
    │   ├── self_attn         # Qwen2Attention
    │   │   ├── q_proj        # Linear
    │   │   ├── k_proj        # Linear
    │   │   ├── v_proj        # Linear
    │   │   └── o_proj        # Linear
    │   ├── mlp               # Qwen2MLP
    │   │   ├── gate_proj     # Linear
    │   │   ├── up_proj       # Linear
    │   │   └── down_proj     # Linear
    │   ├── input_layernorm   # RMSNorm
    │   └── post_attention_layernorm  # RMSNorm
    └── norm                  # RMSNorm (final)
└── lm_head                   # Linear (unembedding)
```

`self_attn`'s output is a tuple `(attn_output, attn_weights, past_key_value)`; take `[0]` for the hidden-state output. RoPE is *not* in this tree — it is applied inline, hence §3.

**Practical tip:** capturing the input to `q_proj`/`k_proj`/`v_proj` is more cleanly done by
hooking the *output* of the upstream `input_layernorm` (its output is exactly their shared input) than by fighting with a pre-hook on `q_proj`.

---

## 3. Capturing RoPE I/O via Monkey-Patching

Because `apply_rotary_pos_emb` is a plain function, replace it at module scope to capture its inputs and outputs without editing the library:

```python
import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

original_rope = qwen2_modeling.apply_rotary_pos_emb

def patched_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cap.setdefault("q_before_rope", []).append(q.detach().clone())
    cap.setdefault("k_before_rope", []).append(k.detach().clone())
    q_out, k_out = original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
    cap.setdefault("q_after_rope", []).append(q_out.detach().clone())
    cap.setdefault("k_after_rope", []).append(k_out.detach().clone())
    return q_out, k_out

qwen2_modeling.apply_rotary_pos_emb = patched_rope   # must run before from_pretrained
ref = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)
```

**Pitfall 1 — patch must precede `from_pretrained`.** Module-internal call sites bind the
function reference at import/instantiation time. Patch after the model is built and the call sites still hold `original_rope`; the patch silently does nothing.

**Pitfall 2 — the patch fires once per decoder layer, and a scalar dict key keeps only the last.** If captures go to `cap["q_before_rope"] = ...`, each layer overwrites the previous one and the final value belongs to the *last* layer, not layer 0. This is the single most confusing symptom in the whole alignment, because the key is *named* `L0` but holds layer 27.

It first showed up as a large mismatch between a (correctly reconstructed) layer-0 q and the capture:

```python
q = cap["L0.attn_q_o"].view(1, 5, 14, 64).transpose(1, 2)   # layer-0 q, reshaped correctly
compare(cap["L0.q_before_rope"], q)        # differ, max_diff ~ 77.7   (k: ~128.7)
```

The reshape is fine (see §4); the two tensors are simply different layers. Confirm with a
three-way comparison: reconstruct the layer-0 q two different ways and check both against the capture:

```python
a_reshaped = query_states_before_reshape.view(1,5,14,64).transpose(1,2)   # from your impl
b_reshaped = cap["L0.attn_q_o"].view(1,5,14,64).transpose(1,2)            # from HF q_proj out
compare(a_reshaped, b_reshaped)            # close          -> reshape itself is fine
compare(a_reshaped, cap["L0.q_before_rope"])  # differ
compare(b_reshaped, cap["L0.q_before_rope"])  # differ
```

Both reshaped tensors disagreeing with `q_before_rope` is the tell: the captured value is not layer 0 at all. Accumulate into a list and index by layer:

```python
cap["q_before_rope"][0]   # layer 0
```

In HuggingFace, nothing modifies `q` between the `view/transpose` reshape and the
`apply_rotary_pos_emb` call, so layer-0 `q_before_rope` must equal the reshaped layer-0`q_proj` output. Once indexed correctly, it does.

---

## 4. Keep `bsz` Explicit — and: same shape is not same data, but check *why*

**Keep `bsz` in the engine.** The batch-less reshape only works at `bsz=1`:

```python
# bsz-less: fine at bsz=1, RAISES at bsz>1
query_states.view(q_len, num_heads, head_dim).transpose(-2, -3)
# correct: works for any bsz
query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
```

For `bsz>1`, `view(q_len, ...)` is a `numel` mismatch (`bsz*q_len*hidden` vs `q_len*hidden`) and raises a `RuntimeError` — it does **not** silently corrupt. So the reason to keep `bsz`
is to support batches at all, not to dodge silent data corruption.

---

## 5. Wrong Activation: GELU vs SiLU

Qwen2.5's MLP is a gated SiLU MLP:

```python
# modeling_qwen2.py — act_fn = nn.functional.silu  (config: "hidden_act": "silu")
down_proj(act_fn(gate_proj(x)) * up_proj(x))
```

Using GELU instead is a real semantic error that hides on easy inputs:

```python
mlp_act = F.gelu(gate) * up    # wrong
mlp_act = F.silu(gate) * up    # correct
```

- SiLU: `x * sigmoid(x)`
- GELU (exact): `x * 0.5 * (1 + erf(x / sqrt(2)))`

They nearly coincide near `x=0` and diverge as `|x|` grows. At `bsz=1` with a short prompt the MLP-output difference can be small enough that greedy decode still emits the right token, so it survives end-to-end smoke tests. Hook the MLP output and `allclose` to surface it.

---

## 6. Silent Empty Output After Adding the Batch Dimension

`output[-1]` changes meaning when `bsz` is introduced, and raises nothing:

```python
# before bsz: output is [seq, vocab]      -> output[-1] = last token logits [vocab]   OK
# after  bsz: output is [bsz, seq, vocab] -> output[-1] = last batch's logits [seq, vocab]
sampling(output[-1])          # argmax over [seq, vocab]; flat index; decodes to ""
```

Observed directly:

```python
print(output.shape)                  # torch.Size([1, 5, 151936])
print(sampling(output[-1]))          # -> (empty string), no error
```

With `bsz=1`, `output[-1] == output[0]`, so the shape is `[seq, vocab]` either way — no IndexError, no shape complaint. `argmax()` over the full 2-D tensor returns an offset into the flattened `seq*vocab` buffer, which detokenizes to empty/garbage.

**Fix — index batch and token explicitly:**

```python
next_id = output[0, -1].argmax()     # batch 0, last position
```

---

## 7. RoPE `inv_freq` Exponent Off by 2x

The highest-value bug of the session, because it passed every casual check.

**Symptom:** logits did **not** match HuggingFace, yet greedy decode produced the **same**
token. The "wrong numbers, right answer" pattern points at a uniform numerical distortion upstream of `argmax`, not a structural error.

**False lead — precision.** First suspicion was a dtype mismatch (cos/sin computed in float32 vs the model's compute dtype, with rounding cast). Ruled out immediately:

```python
print(ref.dtype)    # torch.float32
```

The whole graph is float32 on CPU, so there is no mixed-precision rounding to blame. Back to the RoPE math.

**The fast diagnostic.** Do **not** start by comparing post-RoPE `q_rot`/`k_rot` — compare
the `cos`/`sin` tables themselves. A table mismatch localizes the bug to
`precompute_freq_cis` in one step, skipping all reasoning about `rotate_half` and broadcast dims. The tables disagreed, and the cause was the frequency exponent:

```python
# WRONG: arange(0, head_dim, 2) is already [0, 2, 4, ...] = 2i; the extra *2 gives 4i/d
inv_freq = 1.0 / (base ** (2 * torch.arange(0, head_dim, 2).float() / head_dim))

# CORRECT: theta_i = base^(-2i/d)
inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

`torch.arange(0, head_dim, 2)` already encodes the `2i` stride; the formula
`theta_i = base^(-2i/d)` needs no further factor of 2. The comment said `-2i/d` while the code computed `-4i/d`.

**Why it hides.** A scaled exponent still yields *valid* rotation angles — the frequency
spectrum is uniformly compressed, not broken. RoPE neither errors nor NaNs, and on short sequences the top-1 token often survives the distortion. Only layer-wise `allclose` (or a direct cos/sin comparison) reliably exposes it.

---

## 8. float32 First on CPU (macOS Intel)

Do alignment in float32, not bf16, regardless of where the engine will eventually run:

- On Intel CPUs without `AVX512_BF16`, many bf16 ops fall back to "upcast to f32, compute, downcast" — a different rounding path from that of native bf16 on an A10. Therefore, even the *reference* model produces different intermediates on CPU-bf16 from these on GPU-bf16.
- Running both sides in float32 removes the mixed-precision variable entirely: cos/sin dtype, cast timing, and accumulation order stop mattering, so a failing `allclose` means a real logic bug, not a rounding artifact.
- bf16 on CPU is emulated (slow) and numerically unlike the deployment target. Validate logic in float32 now; switch to bf16 only when benchmarking on the actual A10.

**Model-specific aside (not a bug, but config-dependent):** reusing `embed_tokens.weight` for the final projection is correct for Qwen2.5-0.5B because it sets `tie_word_embeddings=True`. Larger Qwen2.5 variants (7B and up) set it to `False` and ship a separate `lm_head.weight`; check `ref.config.tie_word_embeddings` before assuming the LM head is tied.

---

## 9. Alignment Criteria

`torch.allclose` as a pass/fail gate. float32 tolerances (tighten before trusting BF16 on
GPU):

| Location                    | `atol` |
| --------------------------- | ------ |
| `embed_tokens`              | 1e-5   |
| `input_layernorm` (RMSNorm) | 1e-3   |
| `q` / `k` / `v` projections | 1e-3   |
| `self_attn` output          | 1e-3   |
| `mlp` output                | 1e-3   |
| Deep layers (accumulated)   | 2e-3   |

- **Find the first failing layer.** Downstream failures are propagated, not independent — do not chase them as separate bugs.
- **Read the magnitude of `max_diff`.** ~1e-3 is precision; ~10–100 means the two tensors are *different objects* — wrong layer captured, wrong activation function, or a genuinely different layout. (In this project the 10–100 diffs were a wrong-layer capture (§3) and the GELU/SiLU swap (§5), not a reshape bug.) The two regimes demand different searches.
- **Tensor alignment is necessary, not sufficient.** Every layer can pass `allclose` and decode can still diverge if a logit gap at a token boundary flips `argmax`. Conversely, decode can match while logits differ (see §7). The final gate is greedy-decode token-ID equality, not tensor distance.

---

## 10. Recommended Workflow

1. Monkey-patch `apply_rotary_pos_emb` **before** `from_pretrained`; accumulate per-layer captures into lists, index by layer.
2. `register_forward_hook` on every decoder block (and key sub-modules) to capture outputs.
3. Run the HuggingFace forward in float32 to populate `cap`.
4. Run your forward, saving intermediates at the same points.
5. Layer-wise `allclose`; find the first FAIL.
6. Within that layer, narrow operation by operation:
   `norm -> q/k/v_proj -> reshape -> RoPE (compare cos/sin first) -> attn -> o_proj -> mlp`.
7. Fix, re-run, repeat until all layers pass.
8. Greedy-decode both implementations; compare output token IDs as the final gate.
