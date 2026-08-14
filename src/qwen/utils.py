import torch
from torch import nn
import logging
import orjson, random

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


def sample_sharegpt(path, tokenizer, num_requests=256, max_p_len=1024, max_model_len=2048, seed=0) -> list[list[int]]:
    with open(path) as f:
        raw = orjson.loads(f.read())
    raw = [d for d in raw if len(d["conversations"]) >= 2]
    random.Random(seed).shuffle(raw)

    reqs = []
    for d in raw:
        prompt, completion = d["conversations"][0]["value"], d["conversations"][1]["value"]
        p_input_ids = tokenizer(prompt).input_ids
        p_len = len(p_input_ids)
        o_len = len(tokenizer(completion).input_ids)
        # vLLM-compatible filter — must match exactly for a valid A/B
        if p_len < 4 or o_len < 4:          # degenerate turns skew the tail
            continue
        if p_len >= max_p_len or p_len + o_len > max_model_len:   # keep every request inside one cache slab
            continue
        reqs.append(p_input_ids)
        if len(reqs) == num_requests:
            break
    return reqs


# nested dict/list: round recursively before dumps, since default won't help
def round_floats(o, nd=2):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [round_floats(v, nd) for v in o]
    return o
