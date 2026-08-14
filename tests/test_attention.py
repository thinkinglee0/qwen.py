import torch

from qwen.utils import resolve_device, default_dtype
from qwen.attention import _bottom_right_causal_bias, build_attn_metadata
from qwen.scheduler import SchedulerOutput, ModelRequest, ScheduledInfo
from qwen.cache import KVCacheData, cdiv
from qwen.sampling import Sampling, TensorSampling


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
    device=get_expected_value("device")
    dtype=get_expected_value("dtype")

    m = torch.finfo(dtype).min   # masked
    z = 0.0             # zero

    mask = _bottom_right_causal_bias(2, 2, device=device, dtype=dtype)
    expected = torch.tensor([[[
        [z, m],
        [z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = _bottom_right_causal_bias(3, 3, device=device, dtype=dtype)
    expected = torch.tensor([[[
        [z, m, m],
        [z, z, m],
        [z, z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = _bottom_right_causal_bias(1, 2, device=device, dtype=dtype)
    # expected = torch.zeros(1, 1, 1, 2)
    expected = torch.tensor([[[
        [z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = _bottom_right_causal_bias(2, 3, device=device, dtype=dtype)
    expected = torch.tensor([[[
        [z, z, m],
        [z, z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

def test_build_attn_metadata(tmp_target_config):
    sampling = Sampling(temperature=1.0, top_k=3)
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 100))[0].tolist()    # rectangular tensor
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling, max_new_tokens=1000)
    req1.num_computed_tokens = 10
    assert req1.is_decoding == False

    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 200))[0].tolist()    # rectangular tensor
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling, max_new_tokens=1000)
    req2.num_computed_tokens = 20

    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 300))[0].tolist()    # rectangular tensor
    req3 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling, max_new_tokens=1000)
    req3.num_computed_tokens = 30

    # uuid
    assert req1.request_id != req2.request_id and req1.request_id != req3.request_id and req2.request_id != req3.request_id

    s_info_1 = ScheduledInfo(want=1, slots=[110])
    s_info_2 = ScheduledInfo(want=2, slots=[120, 121])
    s_info_3 = ScheduledInfo(want=3, slots=[130, 131, 132])

    scheduled: dict[str, ScheduledInfo] = {
        req1.request_id: s_info_1,
        req2.request_id: s_info_2,
        req3.request_id: s_info_3,
    }

    block_tables: list[list[int]] = [[100], [200, 400], [300]]

    sch_out = SchedulerOutput(reqs=[req1, req2, req3], scheduled=scheduled, block_tables=block_tables, config=tmp_target_config)
    assert sch_out.tensor_sampling.temperature is not None and sch_out.tensor_sampling.temperature.tolist() == [1.0]*3
    assert sch_out.tensor_sampling.top_k is not None and sch_out.tensor_sampling.top_k.tolist() == [3]*3

    packed_ids, md = build_attn_metadata(sch_out, cache_data=None, device=tmp_target_config.device)

    assert len(packed_ids) == 6
    assert packed_ids.tolist() == req1.input_ids[req1.num_computed_tokens:req1.num_computed_tokens+s_info_1.want] \
        +req2.input_ids[req2.num_computed_tokens:req2.num_computed_tokens+s_info_2.want] \
        +req3.input_ids[req3.num_computed_tokens:req3.num_computed_tokens+s_info_3.want]
    assert md.cache == None

    # query side
    assert md.cu_seqlens_q.tolist() == [0, 1, 3, 6]
    assert md.max_seqlen_q == 3

    # cache
    kv_lens = md.cache_seqlens.tolist()
    assert len(kv_lens) == 3
    assert kv_lens == [req1.num_computed_tokens+s_info_1.want, req2.num_computed_tokens+s_info_2.want, req3.num_computed_tokens+s_info_3.want]

    # position_id for rop
    assert md.position_ids.tolist() == [10, 20, 21, 30, 31, 32]
    assert md.slot_mapping.tolist() == [110, 120, 121, 130, 131, 132]

    # block tbale
    block_table_lst = md.block_table.tolist()
    assert md.block_table.size() == (3, 2)
    assert md.block_table.tolist() == [[100, 0], [200, 400], [300, 0]]


