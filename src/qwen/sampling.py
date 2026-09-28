import logging
import torch
from dataclasses import dataclass, InitVar

from qwen.config import ModelConfig
from qwen.constants import EPS, MAX_EFFECTIVE_TOP_K


logger = logging.getLogger(__name__)

@dataclass
class SamplingParams:
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    freq_pen: float | None = None
    pres_pen: float | None = None

    def validate(self, config: ModelConfig):
        if self.temperature is not None:
            if not (self.temperature >= 0):
                raise ValueError("temperature must be >= 0")

        if self.top_k is not None:
            if not (0 <= self.top_k <= config.vocab_size):
                raise ValueError(f"top_k must be in [0, config.vocab_size]")

        if self.top_p is not None:
            if not (0. < self.top_p <= 1.):
                raise ValueError(f"top_p must be in [0., 1.]")

        if self.repetition_penalty is not None:
            if not (0. < self.repetition_penalty):
                raise ValueError("repetition_penalty must be > 0.")

        if self.freq_pen is not None:
            if not (-2. <= self.freq_pen <= 2.):
                raise ValueError(f"freq_pen must be in [-2., 2.]")

        if self.pres_pen is not None:
            if not (-2. <= self.pres_pen <= 2.):
                raise ValueError(f"pres_pen must be in [-2., 2.]")


# resident sampling params on device, one column per sequence, for async H2D copy
class SamplingParamTable:
    FLOAT32_FIELDS = ("temperature", "top_p", "repetition_penalty", "freq_pen", "pres_pen")

    def __init__(self, config: ModelConfig):
        '''Called once in the constructor of Scheduler'''

        bsz = config.max_num_seqs
        self.fields_device = torch.empty(len(self.FLOAT32_FIELDS), bsz, dtype=torch.float32, device=config.device)
        self.fields_host = torch.empty(len(self.FLOAT32_FIELDS), bsz, dtype=torch.float32, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(len(self.FLOAT32_FIELDS), bsz, dtype=torch.float32)

        self.top_k_device = torch.empty(bsz, dtype=torch.int64, device=config.device)
        self.top_k_host = torch.empty(bsz, dtype=torch.int64, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(bsz, dtype=torch.int64)

    def set_slot(self, slot: int, user_sampling: SamplingParams | None, config: ModelConfig):
        """Called once in _alloc_resources_on_admission when a request is admitted, not once per step."""

        for i, name in enumerate(self.FLOAT32_FIELDS):
            v = getattr(user_sampling, name, None) if user_sampling is not None else None
            self.fields_host[i, slot] = v if v is not None else getattr(config, name)
        self.fields_device[:, slot].copy_(self.fields_host[:, slot], non_blocking=True)   # async H2D copy

        self.top_k_host[slot] = user_sampling.top_k if user_sampling is not None and user_sampling.top_k is not None else config.top_k
        self.top_k_device[slot].copy_(self.top_k_host[slot], non_blocking=True)     # async H2D copy

    def gather(self, slot_idx: torch.Tensor, slot_idx_host: torch.Tensor):
        """Per step: one index_select on device, zero Python iteration."""
        sel = self.fields_device.index_select(1, slot_idx)

        top_k = self.top_k_host.index_select(0, slot_idx_host)
        max_k = int(top_k.max().item())

        # (temperature, top_p, repetition_penalty, freq_pen, pres_pen), top_k, max_k
        return sel.unbind(0), self.top_k_device.index_select(0, slot_idx), max_k

@dataclass
class SamplingTensors:
    temperature: torch.Tensor
    top_k: torch.Tensor
    max_k: int
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    freq_pen: torch.Tensor
    pres_pen: torch.Tensor
    prompt_mask: torch.Tensor
    output_counts: torch.Tensor
    output_mask: torch.Tensor

    @classmethod
    def from_table(cls, sampling_param_tab: SamplingParamTable, slot_idx: torch.Tensor, slot_idx_host: torch.Tensor,
                   prompt_mask: torch.Tensor, output_counts: torch.Tensor, output_mask: torch.Tensor):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"from_table, {slot_idx}")
        (temperature, top_p, repetition_penalty, freq_pen, pres_pen), top_k, max_k = sampling_param_tab.gather(slot_idx=slot_idx, slot_idx_host=slot_idx_host)

        return cls(
            temperature=temperature, top_k=top_k, max_k=max_k,
            top_p=top_p, repetition_penalty=repetition_penalty, freq_pen=freq_pen, pres_pen=pres_pen,
            prompt_mask=prompt_mask,
            output_counts=output_counts, output_mask=output_mask,
        )

def apply_penalties(logits, sampling_tensors: SamplingTensors, vocab_size: int):
    # logits [bsz, vocab_size]
    bsz = logits.shape[0]

    # -> shape [bsz, vocab_size]
    rep = sampling_tensors.repetition_penalty[:, None].repeat(1, vocab_size)

    # repetition penalty based on prompt and output
    rep[~(sampling_tensors.prompt_mask | sampling_tensors.output_mask)] = 1.0     # unseen set to 1.0
    logits = torch.where(logits > 0, logits / rep, logits * rep)    # ensure new <= old while rep>1.0, no matter the signedness of ligits

    # frequency/presence based on output token
    logits = logits - sampling_tensors.freq_pen[:, None] * sampling_tensors.output_counts
    logits = logits - sampling_tensors.pres_pen[:, None] * sampling_tensors.output_mask
    return logits

def apply_top_k(logits: torch.Tensor, top_k: torch.Tensor, max_k: int):
    # logits [n, vocab]
    # top_k: [n] int; <=0 or >=cutoff treated as no-op (disabled)
    # max_k: host-side max of top_k over the batch, so topk()'s width costs no D2H sync
    n, vocab = logits.shape

    # top-k at or above the cutoff is statistically indistinguishable from no top-k
    # at all, so round it up to a no-op rather than paying a wider topk() for it.
    cutoff = min(MAX_EFFECTIVE_TOP_K, vocab)
    max_k = min(max_k, cutoff)
    if max_k <= 0:                                           # whole batch disabled (top_k=0 is the default)
        return logits                                        # plain Python int -> branch is free, no sync

    disabled = (top_k <= 0) | (top_k >= cutoff)              # [n] bool

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
    return sample(logits, sampling_tensors.temperature, sampling_tensors.top_k, max_k=sampling_tensors.max_k, top_p=sampling_tensors.top_p)

def sample(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, max_k:int, top_p: torch.Tensor):
    # logits: [n, vocab] (penalties already applied); all three params are [n]
    greedy = temperature <= EPS
    t = torch.where(greedy, torch.ones_like(temperature), temperature)
    logits = logits / t[:, None]                             # temperature; greedy rows unscaled

    logits = apply_top_k(logits, top_k, max_k=max_k)
    logits = apply_top_p(logits, top_p)

    # softmax in fp32 for numerical stability
    probs = logits.float().softmax(dim=-1)                   # [n, vocab]

    # Guard against all-(-inf) rows: softmax of all -inf -> all NaN, and even a
    # near-underflow row can sum to 0, both of which make multinomial throw /
    # return garbage. Detect dead rows by their probability mass, not by scanning
    # logits for -inf.
    #
    # Applied unconditionally: an `if dead.any()` branch would stall the launch
    # queue on a device->host sync every single step, just to skip one elementwise
    # kernel in the rare case. Flat weights make the row uniform -- multinomial
    # normalizes internally, so there is no need to divide by vocab. Greedy rows
    # get overwritten by argmax below either way.
    row_sum = probs.sum(dim=-1, keepdim=True)                # [n, 1]
    dead = ~torch.isfinite(row_sum) | (row_sum <= 0)         # [n, 1] bool, broadcasts over vocab
    probs = probs.masked_fill(dead, 1.0)

    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)    # select by probability
    argmax = logits.argmax(dim=-1)                           # greedy rows take argmax
    return torch.where(greedy, argmax, sampled)              # [n]


