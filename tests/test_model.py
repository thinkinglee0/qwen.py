import pytest
import torch
import logging

import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

import qwen
from qwen.cache import init_kv_cache2
from qwen.attention import build_prefill_metadata, build_decode_metadata, AttentionMetadata
from qwen.rope import DefaultRoPE
from qwen.config import ModelConfig

from constants import *


logger = logging.getLogger(__name__)


def sampling_batch(tok, last_logits):
    new_token_ids = last_logits.argmax(dim=-1)
    return [tok.decode(token_id.item()) for token_id in new_token_ids], new_token_ids

class HookManager:
    def __init__(self):
        self.hooks = {}

    def save_i(self, name):
        def pre_hook(module, input):
            val = input[0] if isinstance(input, tuple) else input
            self.hooks[name] = val.detach().clone()    # clone instantly, avoiding modified by following flow
        return pre_hook
    
    def save_o(self, name):
        def hook(module, input, output):
            self.hooks[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        return hook
    
    def patch_rope(self):
        # ref
        original_ref_rope = qwen2_modeling.apply_rotary_pos_emb
        def patched_ref_func(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            self.hooks.setdefault("ref_model.q_before_rope", []).append(q.detach().clone())
            self.hooks.setdefault("ref_model.k_before_rope", []).append(k.detach().clone())
            if position_ids is not None:
                self.hooks.setdefault("ref_model.position_ids_on_rope", []).append(position_ids.detach().clone())
            else:
                self.hooks.setdefault("ref_model.position_ids_on_rope", []).append(None)
            q_embed, k_embed = original_ref_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
            self.hooks.setdefault("ref_model.q_embed_returned", []).append(q_embed.detach().clone())
            self.hooks.setdefault("ref_model.k_embed_returned", []).append(k_embed.detach().clone())
            return q_embed, k_embed

        qwen2_modeling.apply_rotary_pos_emb = patched_ref_func

        # target, rope
        original_target_rope = DefaultRoPE.forward
        def patched_target_func(slf, q, k, position_ids):
            self.hooks.setdefault("target_model.q_before_rope", []).append(q.detach().clone())
            self.hooks.setdefault("target_model.k_before_rope", []).append(k.detach().clone())
            if position_ids is not None:
                self.hooks.setdefault("target_model.position_ids_on_rope", []).append(position_ids.detach().clone())
            else:
                self.hooks.setdefault("target_model.position_ids_on_rope", []).append(None)
            q_embed, k_embed = original_target_rope(slf, q, k, position_ids)
            self.hooks.setdefault("target_model.q_embed_returned", []).append(q_embed.detach().clone())
            self.hooks.setdefault("target_model.k_embed_returned", []).append(k_embed.detach().clone())
            return q_embed, k_embed

        DefaultRoPE.forward = patched_target_func
        
        # target, sdpa_one_seq
        original_target_sdpa = qwen.attention.sdpa_one_seq
        def patched_target_sdpa(q, k, v):
            self.hooks.setdefault("target_model.q_before_sdpa", []).append(q.detach().clone())
            self.hooks.setdefault("target_model.k_before_sdpa", []).append(k.detach().clone())
            self.hooks.setdefault("target_model.v_before_sdpa", []).append(v.detach().clone())
            out = original_target_sdpa(q, k, v)
            self.hooks.setdefault("target_model.out_after_sdpa", []).append(out.detach().clone())
            return out

        qwen.attention.sdpa_one_seq = patched_target_sdpa

# target_model == ref_model on rectangular tensor?
@pytest.mark.parametrize("B", [1, 2])
@torch.inference_mode()
def test_prefill_matches_reference_on_math(target_model, ref_model, B:int):
    torch.manual_seed(0)        # ramdom seed, to fix the executing process.
    S, P = 100, 5
    input_ids = torch.randint(0, target_model.config.vocab_size, (B, P))    # rectangular tensor

    handles = []
    try:
        hm = HookManager()
        handles.append(ref_model.model.embed_tokens.register_forward_hook(hm.save_o("ref_model.embed_tokens_o")))
        handles.append(target_model.model.embed_tokens.register_forward_hook(hm.save_o("target_model.embed_tokens_o")))

        handles.append(ref_model.model.layers[0].self_attn.q_proj.register_forward_hook(hm.save_o("ref_model.L0.q_proj")))
        handles.append(target_model.model.layers[0].self_attn.q_proj.register_forward_hook(hm.save_o("target_model.L0.q_proj")))

        handles.append(ref_model.model.layers[0].self_attn.k_proj.register_forward_hook(hm.save_o("ref_model.L0.k_proj")))
        handles.append(target_model.model.layers[0].self_attn.k_proj.register_forward_hook(hm.save_o("target_model.L0.k_proj")))

        handles.append(ref_model.model.layers[0].self_attn.v_proj.register_forward_hook(hm.save_o("ref_model.L0.v_proj")))
        handles.append(target_model.model.layers[0].self_attn.v_proj.register_forward_hook(hm.save_o("target_model.L0.v_proj")))

        hm.patch_rope()

        handles.append(ref_model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(hm.save_i("ref_model.L0.o_proj_in")))
        handles.append(target_model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(hm.save_i("target_model.L0.o_proj_in")))

        handles.append(ref_model.model.layers[0].self_attn.o_proj.register_forward_hook(hm.save_o("ref_model.L0.o_proj")))
        handles.append(target_model.model.layers[0].self_attn.o_proj.register_forward_hook(hm.save_o("target_model.L0.o_proj")))

        for index in range(P, S):
            # target model
            packed_ids, md = build_prefill_metadata(input_ids.unbind(), None, target_model.device, target_model.config.cache_len)
            hidden = target_model.forward(packed_ids, md)       # [total_tokens, H]

            # gather each seq's LAST token -> logits -> first generated token
            last_idx = md.cu_seqlens_q[1:] - 1          # [B]
            target_logits = target_model.compute_logits(hidden[last_idx])   # [B, vocab]

            # reference model
            ref_output = ref_model.forward(input_ids)
            ref_logits = ref_output.logits[:, -1, :]        # [B, vocab]

            try:
                torch.testing.assert_close(target_logits, ref_logits, rtol=0, atol=1e-3,
                                    msg=lambda s: f"logits mismatch, index={index}\n{s}")
            except AssertionError as e:
                torch.testing.assert_close(hm.hooks["target_model.embed_tokens_o"], hm.hooks["ref_model.embed_tokens_o"][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.L0.q_proj"], hm.hooks["ref_model.L0.q_proj"][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.L0.k_proj"], hm.hooks["ref_model.L0.k_proj"][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.L0.v_proj"], hm.hooks["ref_model.L0.v_proj"][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")

                # rope
                torch.testing.assert_close(hm.hooks["target_model.q_before_rope"][0], hm.hooks["ref_model.q_before_rope"][0][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.k_before_rope"][0], hm.hooks["ref_model.k_before_rope"][0][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.q_embed_returned"][0], hm.hooks["ref_model.q_embed_returned"][0][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.k_embed_returned"][0], hm.hooks["ref_model.k_embed_returned"][0][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                if hm.hooks["ref_model.position_ids_on_rope"][0] is not None:
                    torch.testing.assert_close(hm.hooks["target_model.position_ids_on_rope"][0], hm.hooks["ref_model.position_ids_on_rope"][0][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                # before sdpa, after scatter by cu_seqlens_q/cu_seqlens_k
                torch.testing.assert_close(hm.hooks["target_model.q_embed_returned"][0], hm.hooks["target_model.q_before_sdpa"][0][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.k_embed_returned"][0], hm.hooks["target_model.k_before_sdpa"][0][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")

                torch.testing.assert_close(hm.hooks["target_model.L0.o_proj_in"], hm.hooks["ref_model.L0.o_proj_in"][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")
                torch.testing.assert_close(hm.hooks["target_model.L0.o_proj"], hm.hooks["ref_model.L0.o_proj"][0], rtol=0, atol=1e-3, msg=lambda s: f"mismatch, index={index}\n{s}")

                logger.info(f"index: {index}, hooked values are equal, please continue adding more hooks")
                raise e

            # shape [B, seq_len]
            next_tokens = target_logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], -1)
    finally:
        for handle in handles:
            handle.remove()

@pytest.mark.parametrize("L", [
    1,          # structural degenerate
    5,          # calibrated noise-floor anchor
    64, 65,     # tile boundary (if BLOCK_M == 64)
    127, 128, 129,   # tile boundary (128) + one-past
    # 512,        # long enough to cross any short-seq flash fallback
], ids=lambda n: f"L{n}")
@torch.inference_mode()
def test_prefill_matches_reference(L, target_model, ref_model, tokenizer, solo_long_input_ids_list, solo_long_encoding):
    # padded batch [bsz, seq_len]
    assert solo_long_encoding.input_ids.shape[1] > L
    assert all(len(sub) > L for sub in solo_long_input_ids_list)

    slice_list = [sub[:L] for sub in solo_long_input_ids_list]
    slice_tensor = solo_long_encoding.input_ids[:, :L]
    slice_attention_mask = solo_long_encoding.attention_mask[:, :L]

    # target model
    packed_ids, md = build_prefill_metadata(slice_list, None, target_model.device, target_model.config.cache_len)
    hidden = target_model.forward(packed_ids, md)       # [total_tokens, H]

    # gather each seq's LAST token -> logits -> first generated token
    last_idx = md.cu_seqlens_q[1:] - 1          # [B]
    target_logits = target_model.compute_logits(hidden[last_idx])   # [B, vocab], last one

    # reference model
    ref_output = ref_model.forward(slice_tensor,
                                    attention_mask=slice_attention_mask,
                                    num_logits_to_keep=1,          # lm_head sees M = B, matching the engine
                                    )        # [bsz, seq_len, vocab]
    ref_logits = ref_output.logits[:, -1, :]        # [bsz, vocab], last one
    
    # (list, tensor)
    target_new_tokens, target_new_ids = sampling_batch(tokenizer, target_logits)
    ref_new_tokens, ref_new_ids = sampling_batch(tokenizer, ref_logits)
    logger.info(f"target: {target_new_ids.tolist()} - |{target_new_tokens}| <-> ref: {ref_new_ids.tolist()} - |{ref_new_tokens}|")

    torch.testing.assert_close(target_logits, ref_logits, rtol=0, atol=1e-3)

# only for B=1
def compare_cache_against_kv_after_rope(B: int, meta: AttentionMetadata, cfg: ModelConfig):
    if B != 1:
        logger.error("compare_cache_against_kv_after_rope is only implemented for B=1")
        return

    start, end = 0, 0
    logger.info(f"compare_cache_against_kv_after_rope, debug_k_list.len: {len(meta.debug_k_list)}")
    for i, (k, v) in enumerate(zip(meta.debug_k_list, meta.debug_v_list)):
        # k, v [T, H, D]
        logger.info(f"compare_cache_against_kv_after_rope, shape, k: {k.shape}, v: {v.shape}")
        layer_index = i % cfg.num_hidden_layers
        if layer_index == 0:
            seq_len = k.shape[0]
            start, end = end, end + seq_len

        # k_cache/v_cache [B, T, H, D], B=1
        k_cache, v_cache = meta.cache.data[layer_index]
        k_cache, v_cache = k_cache[0, start:end], v_cache[0, start:end]
        logger.info(f"compare_cache_against_kv_after_rope, i: {i}, layer_index: {layer_index}, start: {start}, end: {end}")
        logger.info(f"compare_cache_against_kv_after_rope, max, k: {k.abs().max().item()}, k_cache: {k_cache.abs().max().item()}, v: {v.abs().max().item()}, v_cache: {v_cache.abs().max().item()}")
        assert (k-k_cache).abs().max() < 1e-3
        assert (v-v_cache).abs().max() < 1e-3

# prefill(S) == prefill(P) + decode(range(P, S))?
@pytest.mark.parametrize("B", [1, 2])
@torch.inference_mode()
def test_kv_cache_correctness(target_model, B:int):
    torch.manual_seed(0)        # ramdom seed, to fix the executing process.
    L= 100
    ids = torch.randint(0, target_model.config.vocab_size, (B, L))

    # sample 1: prefill
    cache1 = init_kv_cache2(target_model.config, B, target_model.config.cache_len)
    packed_ids, meta_prefill = build_prefill_metadata(ids.unbind(), cache1, target_model.device, target_model.config.cache_len)
    hidden = target_model.forward(packed_ids, meta_prefill)       # [total_tokens, H]
    last_idx = meta_prefill.cu_seqlens_q[1:] - 1          # [B]
    logits_only_prefill = target_model.compute_logits(hidden[last_idx])   # [B, vocab]

    for P in [1, 2, L//2, L-1]:
        # sample 2: prefill + decode
        cache2 = init_kv_cache2(target_model.config, B, target_model.config.cache_len)
        packed_ids, meta_prefill = build_prefill_metadata(ids[:, :P].unbind(), cache2, target_model.device, target_model.config.cache_len)

        ###########################################################################
        #  !! This line is omitted due to my mistake !!
        #  It leads to a logits mismatch, which is troubleshot by comparing kv cache with kv list recorded after projection and rope.
        #  See the function `compare_cache_against_kv_after_rope` for details.
        ###########################################################################
        hidden = target_model.forward(packed_ids, meta_prefill)       # [total_tokens, H]

        debug_k_list, debug_v_list = [], []
        meta_prefill.debug_k_list, meta_prefill.debug_v_list = debug_k_list, debug_v_list   # turn on debug
        last_idx = meta_prefill.cu_seqlens_q[1:] - 1          # [B]
        logits_prefill_decode = target_model.compute_logits(hidden[last_idx])   # [B, vocab]

        past_lens = meta_prefill.cache_seqlens
        for t in range(P, L):                       # decode the rest 1-by-1
            meta_decode = build_decode_metadata(past_lens, cache2, target_model.config.cache_len)
            meta_decode.debug_k_list, meta_decode.debug_v_list = debug_k_list, debug_v_list   # turn on debug
            hidden = target_model.forward(torch.cat(ids[:, t:t+1].unbind()), meta_decode)  # input_ids = [B], one token per seq
            logits_prefill_decode = target_model.compute_logits(hidden)    # [B, vocab]  (decode: every row is a last token)
            
            past_lens = meta_decode.cache_seqlens

        # shape [B, V]
        try:
            # logits
            diff = (logits_only_prefill - logits_prefill_decode).abs()
            logger.info(f"max abs logit diff: {diff.max().item()}, B: {B}, P: {P}")
            assert diff.max().item() < 1e-3          # safe upper bound of FP32 floor
            assert (logits_only_prefill.argmax(-1) != logits_prefill_decode.argmax(-1)).sum() == 0

            # kv cache
            for layer in range(target_model.config.num_hidden_layers):
                k1, v1 = cache1.data[layer][0], cache1.data[layer][1]
                k2, v2 = cache2.data[layer][0], cache2.data[layer][1]
                torch.testing.assert_close(k1, k2, rtol=0, atol=1e-3)
                torch.testing.assert_close(v1, v2, rtol=0, atol=1e-3)
        except AssertionError as e:
            # compare kv cache with kv list recorded after projection and rope
            compare_cache_against_kv_after_rope(B, meta_decode, target_model.config)

            raise e     # re-raise


@pytest.mark.parametrize(
    ("encoding_fixture", "list_fixture"),
    [
        pytest.param("solo_encoding", "solo_input_ids_list", id="single"),
        pytest.param("batch_encoding", "batch_input_ids_list", id="batch"),
    ],
)
@torch.inference_mode()
def test_decode_matches_reference(target_model, ref_model, request, encoding_fixture, list_fixture):
    encoding = request.getfixturevalue(encoding_fixture)
    input_list = request.getfixturevalue(list_fixture)
    # target model
    B = len(input_list)
    cache = init_kv_cache2(target_model.config, B, target_model.config.cache_len)

    # target model's prefill
    packed_ids, md_prefill = build_prefill_metadata(input_list, cache, target_model.device, target_model.config.cache_len)
    hidden = target_model.forward(packed_ids, md_prefill)       # [total_tokens, H]
    last_idx = md_prefill.cu_seqlens_q[1:] - 1          # [B]
    target_logits = target_model.compute_logits(hidden[last_idx])   # [B, vocab], last one
    target_next_ids = target_logits.argmax(dim=-1)

    assert len(input_list) == len(target_next_ids)

    # target model's decode
    past_lens = md_prefill.cache_seqlens
    md_decode = build_decode_metadata(past_lens, cache, target_model.config.cache_len)   # is_prefill=False, fresh each step
    hidden = target_model.forward(target_next_ids, md_decode)  # input_ids = [B], one token per seq -> list[int]
    target_logits = target_model.compute_logits(hidden)    # [B, vocab]  (decode: every row is a last token)
    target_next_ids = target_logits.argmax(dim=-1)

    # reference's prefill
    ref_output = ref_model.forward(encoding.input_ids,
                                    attention_mask=encoding.attention_mask,
                                    num_logits_to_keep=1,          # lm_head sees M = B, matching the engine
                                    use_cache=True,
                                    )        # [bsz, seq_len, vocab]
    ref_logits = ref_output.logits[:, -1, :]        # [bsz, vocab], last one
    ref_next_ids = ref_logits.argmax(dim=-1)
    past_key_values = ref_output.past_key_values

    # reference's decode
    old_mask = encoding.attention_mask
    new_mask = torch.cat([old_mask, old_mask.new_ones((B, 1))], -1)
    ref_output = ref_model.forward(ref_next_ids.unsqueeze(-1),  # [B, 1]
                                    attention_mask=new_mask,
                                    num_logits_to_keep=1,          # lm_head sees M = B, matching the engine
                                    use_cache=True,
                                    past_key_values=past_key_values
                                    )        # [bsz, seq_len, vocab]
    ref_logits = ref_output.logits[:, -1, :]        # [bsz, vocab], last one

    # assert logits
    assert (target_logits - ref_logits).abs().max() < 1e-3


