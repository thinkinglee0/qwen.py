import torch

from qwen.utils import resolve_device, default_dtype
from qwen.attention import _bottom_right_causal_bias


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


