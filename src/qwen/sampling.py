import logging
import torch
from dataclasses import dataclass, InitVar

from qwen.config import ModelConfig
from qwen.constants import EPS


logger = logging.getLogger(__name__)

@dataclass
class SamplingParams:
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    rep_pen: float | None = None
    freq_pen: float | None = None
    pres_pen: float | None = None


# resident sampling params on device, one column per sequence, for async H2D copy
class SamplingParamTable:
    FLOAT32_FIELDS = ("temperature", "top_p", "rep_pen", "freq_pen", "pres_pen")

    def __init__(self, config: ModelConfig):
        '''Called once in the constructor of Scheduler'''

        n = config.max_num_seqs
        self._f32 = torch.empty(len(self.FLOAT32_FIELDS), n, dtype=torch.float32, device=config.device)
        self._stage = torch.empty(len(self.FLOAT32_FIELDS), n, pin_memory=True)    # host tensor for async H2D copy

        self._top_k = torch.empty(n, dtype=torch.int64, device=config.device)
        self._k_stage = torch.empty(n, dtype=torch.int64, pin_memory=True)    # host tensor for async H2D copy

    def set_slot(self, slot: int, user_sampling: SamplingParams | None, config: ModelConfig):
        """Called once in _alloc_resources_on_admission when a request is admitted, not once per step."""

        for i, name in enumerate(self.FLOAT32_FIELDS):
            v = getattr(user_sampling, name, None) if user_sampling is not None else None
            self._stage[i, slot] = v if v is not None else getattr(config, name)
        self._f32[:, slot].copy_(self._stage[:, slot], non_blocking=True)

        self._k_stage[slot] = user_sampling.top_k if user_sampling is not None and user_sampling.top_k is not None else config.top_k
        self._top_k[slot].copy_(self._k_stage[slot], non_blocking=True)

    def gather(self, slot_idx: torch.Tensor):
        """Per step: one index_select on device, zero Python iteration."""
        sel = self._f32.index_select(1, slot_idx)
        return sel.unbind(0), self._top_k.index_select(0, slot_idx)

@dataclass
class SamplingTensors:
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    rep_pen: torch.Tensor
    freq_pen: torch.Tensor
    pres_pen: torch.Tensor

    @classmethod
    def from_params_list(cls, samplings: list[SamplingParams | None], config: ModelConfig):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"from_params_list, len: {len(samplings)}")
        return cls(
            temperature=torch.tensor([sampling.temperature if sampling is not None and sampling.temperature is not None else config.temperature for sampling in samplings], device=config.device, dtype=torch.float32),
            top_k=torch.tensor([sampling.top_k if sampling is not None and sampling.top_k is not None else config.top_k for sampling in samplings], device=config.device, dtype=torch.int64),
            top_p=torch.tensor([sampling.top_p if sampling is not None and sampling.top_p is not None else config.top_p for sampling in samplings], device=config.device, dtype=torch.float32),
            rep_pen=torch.tensor([sampling.rep_pen if sampling is not None and sampling.rep_pen is not None else config.rep_pen for sampling in samplings], device=config.device, dtype=torch.float32),
            freq_pen=torch.tensor([sampling.freq_pen if sampling is not None and sampling.freq_pen is not None else config.freq_pen for sampling in samplings], device=config.device, dtype=torch.float32),
            pres_pen=torch.tensor([sampling.pres_pen if sampling is not None and sampling.pres_pen is not None else config.pres_pen for sampling in samplings], device=config.device, dtype=torch.float32)
        )

    @classmethod
    def from_table(cls, sampling_param_tab: SamplingParamTable, slot_idx: torch.Tensor):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"from_table, {slot_idx}")
        (temperature, top_p, rep_pen, freq_pen, pres_pen), top_k = sampling_param_tab.gather(slot_idx=slot_idx)

        return cls(temperature=temperature, top_k=top_k,
                top_p=top_p, rep_pen=rep_pen, freq_pen=freq_pen, pres_pen=pres_pen)

def bin_counts_and_mask(
    token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin-count token ids per sequence, no padding involved.

    token_ids: one variable-length id list per seq; empty lists are allowed.
    Returns (counts, mask), both [bsz, vocab_size], counts is int32.
    """
    bsz = len(token_ids)

    # Row-major linearization on the CPU: one H2D copy instead of bsz of them.
    # flat_idx max is bsz * vocab_size, but scatter_add_ requires an int64 index.
    flat_idx = [row * vocab_size + t for row, seq in enumerate(token_ids) for t in seq]

    counts = torch.zeros(bsz * vocab_size, dtype=torch.int32, device=device)
    if flat_idx:    # all-empty on the first decode step -> nothing to scatter
        idx = torch.tensor(flat_idx, dtype=torch.int64).to(device, non_blocking=True)
        counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
    counts = counts.view(bsz, vocab_size)    # contiguous, unlike the old narrow()

    return counts, counts > 0

def apply_penalties2(logits, prompt_tokens: list[list[int]], output_tokens: list[list[int]],
                     sampling_tensors: SamplingTensors, vocab_size: int):
    assert sampling_tensors.rep_pen is not None and sampling_tensors.freq_pen is not None and sampling_tensors.pres_pen is not None
    return apply_penalties(logits, prompt_tokens, output_tokens,
                           sampling_tensors.rep_pen, sampling_tensors.freq_pen, sampling_tensors.pres_pen, vocab_size)

def apply_penalties(logits, prompt_tokens: list[list[int]], output_tokens: list[list[int]],
                    rep_pen: torch.Tensor, freq_pen: torch.Tensor, pres_pen: torch.Tensor, vocab_size: int):
    # logits [bsz, vocab_size]
    bsz = logits.shape[0]

    # mask now means presence/existing, shape [bsz, vocab_size]
    _,            prompt_mask  = bin_counts_and_mask(prompt_tokens, vocab_size, logits.device)  # ditch the count of prompt
    out_counts,   output_mask  = bin_counts_and_mask(output_tokens, vocab_size, logits.device)

    # -> shape [bsz, vocab_size]
    rep = rep_pen[:, None].repeat(1, vocab_size)

    # repetition penalty based on prompt and output
    rep[~(prompt_mask | output_mask)] = 1.0     # unseen set to 1.0
    logits = torch.where(logits > 0, logits / rep, logits * rep)    # ensure new <= old while rep>1.0, no matter the signedness of ligits

    # frequency/presence based on output token
    logits = logits - freq_pen[:, None] * out_counts
    logits = logits - pres_pen[:, None] * output_mask
    return logits

def apply_top_k(logits: torch.Tensor, top_k: torch.Tensor):
    # logits [n, vocab]
    # top_k: [n] int; <=0 or >=vocab treated as no-op (disabled)
    n, vocab = logits.shape
    disabled = (top_k <= 0) | (top_k >= vocab)               # [n] bool
    if disabled.all():                                       # every row is no-op
        return logits

    # max_k driven by valid rows ONLY — a disabled row's huge top_k must not
    # blow max_k up to vocab and force a full-width topk on every row
    max_k = int(top_k[~disabled].max().item())
    max_k = min(max(max_k, 1), vocab)

    top_vals, _ = torch.topk(logits, max_k, dim=-1)          # [n, max_k] descending, top-max_k of all lines
    # k-th largest per row as threshold; clamp k into [1, max_k]
    idx = (top_k.clamp(min=1, max=max_k) - 1).unsqueeze(1)   # [n,1], each line's top-k index
    kth = top_vals.gather(1, idx)                            # [n,1], each line's top-k-th element
    # disabled rows: threshold = -inf so nothing gets cut
    kth = torch.where(disabled.unsqueeze(1),
                      torch.full_like(kth, float("-inf")), kth)
    return torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)    # <kth, set to -inf, otherwise keep

def apply_top_p(logits: torch.Tensor, top_p: torch.Tensor):
    # top_p: [n] float in (0,1]; 1.0 treated as no-op
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)                    # softmax must be on the sorted logits, ensuring the first token is the most likely
    cumprobs = probs.cumsum(dim=-1)
    # "cumulative prob before this token already exceeds p" -> remove; this always keeps the first token
    sorted_remove = (cumprobs - probs) > top_p[:, None]
    # scatter back to original vocab order
    remove = torch.zeros_like(sorted_remove)
    remove.scatter_(1, sorted_idx, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))

def sample2(logits: torch.Tensor, sampling_tensors: SamplingTensors):
    assert sampling_tensors.temperature is not None and sampling_tensors.top_k is not None and sampling_tensors.top_p is not None
    return sample(logits, sampling_tensors.temperature, sampling_tensors.top_k, sampling_tensors.top_p)

def sample(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor):
    # logits: [n, vocab] (penalties already applied); all three params are [n]
    greedy = temperature <= EPS
    t = torch.where(greedy, torch.ones_like(temperature), temperature)
    logits = logits / t[:, None]                             # temperature; greedy rows unscaled

    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    # softmax in fp32 for numerical stability
    probs = logits.float().softmax(dim=-1)                   # [n, vocab]

    # Guard against all-(-inf) rows: softmax of all -inf -> all NaN, and even a
    # near-underflow row can sum to 0, both of which make multinomial throw /
    # return garbage. Detect dead rows by their probability mass, not by scanning
    # logits for -inf.
    row_sum = probs.sum(dim=-1)                              # [n]
    dead = ~torch.isfinite(row_sum) | (row_sum <= 0)         # [n] bool
    if dead.any():
        # fall back to a uniform row so multinomial stays well-defined;
        # these rows will be overwritten by argmax below iff greedy, otherwise
        # uniform sampling is the least-bad recovery
        probs = torch.where(dead.unsqueeze(1),
                            torch.full_like(probs, 1.0 / probs.shape[-1]),
                            probs)

    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)    # select by probability
    argmax = logits.argmax(dim=-1)                           # greedy rows take argmax
    return torch.where(greedy, argmax, sampled)              # [n]


