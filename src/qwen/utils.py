import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


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

def compare(target, ref, name="", rtol=0, atol=1e-3):
    try:
        torch.testing.assert_close(
            target, ref,
            rtol=rtol, atol=atol,
            check_dtype=False, check_device=False,   # ignore dtype and device
        )
        if isinstance(target, torch.Tensor):
            logger.info(f"[PASS] {name}, shape={tuple(target.shape)}")
        else:
            logger.info(f"[PASS] {name}, type={type(target)}")
        return True
    except AssertionError as e:
        # assert_close error messages include shape mismatch / Mismatched elements
        logger.error(f"[FAIL] {name}, {e}")
        return False


def pad_token_ids(seqs, vocab_size, device):
    # seqs: list[list[int]];空序列(还没生成)给空 list
    max_len = max((len(s) for s in seqs), default=1)
    max_len = max(max_len, 1)
    out = torch.full((len(seqs), max_len), vocab_size,
                     dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        if s:
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return out

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)