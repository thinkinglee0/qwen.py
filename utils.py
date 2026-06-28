import torch


def resolve_device(prefer: str | None = None) -> torch.device:
    if prefer:                       # override explictly for reproducing or debuging
        return torch.device(prefer)
    if torch.cuda.is_available():    # A10
        return torch.device("cuda")

    return torch.device("cpu")       # Intel Mac

def default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16        # A10
    return torch.float32             # CPU

def compare(target, ref, name="", rtol=1e-3, atol=1e-3):
    try:
        torch.testing.assert_close(
            target, ref,
            rtol=rtol, atol=atol,
            check_dtype=False, check_device=False,   # ignore dtype and device
        )
        if isinstance(target, torch.Tensor):
            print(f"[PASS] {name}, shape={tuple(target.shape)}")
        else:
            print(f"[PASS] {name}, type={type(target)}")
        return True
    except AssertionError as e:
        # assert_close error messages include shape mismatch / Mismatched elements
        print(f"[FAIL] {name}\n{e}")
        return False

