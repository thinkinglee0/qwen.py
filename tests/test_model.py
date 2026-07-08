import pytest
import torch
import logging

from qwen.model import KVCache, init_kv_cache2, make_causal_mask, init_kv_cache
from qwen.utils import resolve_device, default_dtype
from constants import *


logger = logging.getLogger(__name__)


def sampling(tok, last_logit):
    new_token_id = last_logit.argmax()
    return tok.decode(new_token_id.item()), new_token_id

# map[platform:xx]
CUR_HOST_PLATFORM = "mac"
EXPECTED_DEVICE_DTYPE_MAP = {
    "mac": {"device": torch.device("cpu"), "dtype": torch.float32},
    "a10": {"device": torch.device("cuda"), "dtype": torch.bfloat16},
}

def get_expected_value(key: str):
    return EXPECTED_DEVICE_DTYPE_MAP[CUR_HOST_PLATFORM][key]


def test_device_dtype():
    device = resolve_device()
    dtype = default_dtype(device)
    assert device == get_expected_value("device")
    assert dtype == get_expected_value("dtype")

def test_causal_mask():
    m = float("-inf")   # masked
    z = 0.0             # zero

    mask = make_causal_mask(2, 2, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    expected = torch.tensor([[[
        [z, m],
        [z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = make_causal_mask(3, 3, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    expected = torch.tensor([[[
        [z, m, m],
        [z, z, m],
        [z, z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = make_causal_mask(1, 2, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    # expected = torch.zeros(1, 1, 1, 2)
    expected = torch.tensor([[[
        [z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = make_causal_mask(2, 3, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    expected = torch.tensor([[[
        [z, z, m],
        [z, z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

# prefill(S) == prefill(P) + decode(range(P, S))?
@pytest.mark.parametrize("use_cache", [True, False])
def test_forward_and_kv_cache_correctness(target_model, use_cache: bool):
    torch.manual_seed(0)        # ramdom seed, to fix the executing process.
    bsz = 1
    S, P = 100, 5
    ids = torch.randint(0, target_model.config.vocab_size, (bsz, S))

    # sample 1: prefill
    logits_only_prefill = target_model.forward(ids, cache=None).logits         # [bsz, V], no cache

    # sample 2: prefill + decode
    cache: KVCache | None = init_kv_cache2(target_model.config, bsz, target_model.device, target_model.dtype) if use_cache else None
    output = target_model.forward(ids[:, :P], cache=cache)       # prefill P tokens
    cache = output.past_key_values
    logits_prefill_decode = output.logits       # shape [bsz, V]
    for t in range(P, S):                       # decode the rest 1-by-1
        if use_cache:       # kv cache
            output = target_model.forward(ids[:, t:t+1], cache=list(cache))
            cache = output.past_key_values
        else:
            output = target_model.forward(ids[:, :t+1], cache=None)
        
        logits_prefill_decode = output.logits

    # shape [bsz, V]
    diff = (logits_only_prefill - logits_prefill_decode).abs()
    logger.info(f"max abs logit diff: {diff.max().item():e}")
    assert diff.max().item() < 1e-3          # safe upper bound of FP32 floor
    assert (logits_only_prefill.argmax(-1) != logits_prefill_decode.argmax(-1)).sum() == 0

# target_model == ref_model?
def test_forward_comparason_with_reference_on_math(target_model, ref_model):
    torch.manual_seed(0)        # ramdom seed, to fix the executing process.
    S, P = 100, 5
    input_ids = torch.randint(0, target_model.config.vocab_size, (1, P))

    for index in range(P, S):
        target_output = target_model.forward(input_ids)
        ref_output = ref_model.forward(input_ids)
        new_token_id = target_output.logits[0].argmax()

        torch.testing.assert_close(target_output.logits, ref_output.logits[:, -1, :], rtol=1e-3, atol=1e-3,
                                   msg=lambda s: f"logits mismatch, index={index}\n{s}")

        # shape [bsz, seq_len]
        input_ids = torch.cat([input_ids, torch.full((1,1), new_token_id)], -1)

# fixed inputs, (e.g.: The capital of France is)
def test_forward_comparason_with_reference(target_model, ref_model, tokenizer, inputs):
    input_ids = inputs.input_ids.detach().clone()
    for index in range(MAX_NEW_TOKEN_NUM):
        target_output = target_model.forward(input_ids)
        ref_output = ref_model.forward(input_ids)
        target_new_token, target_new_token_id = sampling(tokenizer, target_output.logits[0])
        ref_new_token, ref_new_token_id = sampling(tokenizer, ref_output.logits[0])
        logger.info(f"index: {index}, target: {target_new_token_id} - |{target_new_token}| <-> ref: {ref_new_token_id} - |{ref_new_token}|")

        torch.testing.assert_close(target_output.logits, ref_output.logits[:, -1, :], rtol=1e-3, atol=1e-3,
                                   msg=lambda s: f"logits mismatch, index={index}\n{s}")

        # shape [bsz, seq_len]
        input_ids = torch.cat([input_ids, torch.full((1,1), target_new_token_id)], -1)


