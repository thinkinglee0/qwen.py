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
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

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

        if self.frequency_penalty is not None:
            if not (-2. <= self.frequency_penalty <= 2.):
                raise ValueError(f"frequency_penalty must be in [-2., 2.]")

        if self.presence_penalty is not None:
            if not (-2. <= self.presence_penalty <= 2.):
                raise ValueError(f"presence_penalty must be in [-2., 2.]")

def extract_info_from_temperature(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    greedy_host = tensor <= EPS
    skip_scale_host = greedy_host | (tensor == 1.0)
    return skip_scale_host, greedy_host

# resident sampling params on device, one column per sequence, for async H2D copy
class SamplingParamTable:
    FLOAT32_FIELDS = ("temperature", "top_p", "repetition_penalty", "frequency_penalty", "presence_penalty")

    def __init__(self, config: ModelConfig):
        '''Called once in the constructor of Scheduler'''

        capacity = config.max_num_req_slots
        self.fields_device = torch.empty(len(self.FLOAT32_FIELDS), capacity, dtype=torch.float32, device=config.device)
        self.fields_stage = torch.empty(len(self.FLOAT32_FIELDS), capacity, dtype=torch.float32, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(len(self.FLOAT32_FIELDS), capacity, dtype=torch.float32)

        self.top_k_device = torch.empty(capacity, dtype=torch.int64, device=config.device)
        self.top_k_stage = torch.empty(capacity, dtype=torch.int64, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(capacity, dtype=torch.int64)

        self.top_k_cutoff = min(MAX_EFFECTIVE_TOP_K, config.vocab_size)

    def set_slot(self, slot: int, user_sampling: SamplingParams | None, config: ModelConfig):
        """Called once in _alloc_resources_on_admission when a request is admitted, not once per step."""

        for i, name in enumerate(self.FLOAT32_FIELDS):
            v = getattr(user_sampling, name, None) if user_sampling is not None else None
            self.fields_stage[i, slot] = v if v is not None else getattr(config, name)
        self.fields_device[:, slot].copy_(self.fields_stage[:, slot], non_blocking=True)   # async H2D copy

        # clamp top-k to [1, top_k_cutoff]
        top_k_eff = user_sampling.top_k if user_sampling is not None and user_sampling.top_k is not None else config.top_k
        if top_k_eff <= 0 or top_k_eff >= self.top_k_cutoff:
            top_k_eff = self.top_k_cutoff
        self.top_k_stage[slot] = top_k_eff
        self.top_k_device[slot].copy_(self.top_k_stage[slot], non_blocking=True)     # async H2D copy

    def gather(self, slot_idx: torch.Tensor, slot_idx_stage: torch.Tensor):
        """Per step: one index_select on device, zero Python iteration."""
        float_fields_device = self.fields_device.index_select(1, slot_idx)

        # calculate switches from fields_stage
        float_fields_stage = self.fields_stage.index_select(1, slot_idx_stage)
        skip_scale_host, greedy_host = extract_info_from_temperature(float_fields_stage[0])
        rep_switch_on = bool((float_fields_stage[2] != 1.0).any().item())
        freq_switch_on = bool((float_fields_stage[3] != 0.0).any().item())
        pres_switch_on = bool((float_fields_stage[4] != 0.0).any().item())

        top_k_stage = self.top_k_stage.index_select(0, slot_idx_stage)
        top_k = self.top_k_device.index_select(0, slot_idx)
        max_k = int(top_k_stage.max().item())

        # (temperature, top_p, repetition_penalty, frequency_penalty, presence_penalty), top_k, max_k
        return float_fields_device.unbind(0), top_k, max_k, (skip_scale_host, greedy_host, rep_switch_on, freq_switch_on, pres_switch_on)

@dataclass
class SamplingTensors:
    temperature: torch.Tensor
    top_k: torch.Tensor
    max_k: int
    top_p: torch.Tensor
    repetition_penalty: torch.Tensor
    frequency_penalty: torch.Tensor
    presence_penalty: torch.Tensor
    prompt_mask: torch.Tensor
    output_counts: torch.Tensor
    output_mask: torch.Tensor
    seen_toks: torch.Tensor
    skip_scale_host: torch.Tensor
    greedy_host: torch.Tensor
    rep_switch_on: bool
    freq_switch_on: bool
    pres_switch_on: bool

    @classmethod
    def from_table(cls, sampling_param_tab: SamplingParamTable, slot_idx: torch.Tensor, slot_idx_stage: torch.Tensor,
                   prompt_mask: torch.Tensor, output_counts: torch.Tensor, output_mask: torch.Tensor,
                   seen_toks: torch.Tensor):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"from_table, {slot_idx}")
        (temperature, top_p, repetition_penalty, frequency_penalty, presence_penalty), \
            top_k, max_k, \
            (skip_scale_host, greedy_host, rep_switch_on, freq_switch_on, pres_switch_on) = \
                sampling_param_tab.gather(slot_idx=slot_idx, slot_idx_stage=slot_idx_stage)

        return cls(
            temperature=temperature, top_k=top_k, max_k=max_k,top_p=top_p,
            repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            prompt_mask=prompt_mask,
            output_counts=output_counts, output_mask=output_mask,
            seen_toks=seen_toks,
            skip_scale_host=skip_scale_host, greedy_host=greedy_host, rep_switch_on=rep_switch_on,
            freq_switch_on=freq_switch_on, pres_switch_on=pres_switch_on,
        )

def apply_rep_penalty(logits, sampling_tensors: SamplingTensors):
    rep_mask = sampling_tensors.prompt_mask | sampling_tensors.output_mask  # [bsz, vocab], bool
    rep = torch.where(rep_mask, sampling_tensors.repetition_penalty[:, None], 1.0)  # unseen set to 1.0
    ori_dtype = logits.dtype
    logits = torch.where(logits > 0, logits / rep, logits * rep)    # ensure new <= old while rep>1.0, no matter the signedness of ligits
    return logits.to(ori_dtype) # keep dtype the same with apply_rep_penalty_sparse for oracle test.

def apply_rep_penalty_sparse(logits, sampling_tensors: SamplingTensors):
    # in place: logits is modified; padding in seen_toks repeats a valid token, so duplicate writes carry the same value
    # logits [bsz, vocab], bf16
    gathered = logits.gather(1, sampling_tensors.seen_toks)     # [B, L]
    rep = sampling_tensors.repetition_penalty[:, None]          # [B, 1], fp32
    vals = torch.where(gathered > 0, gathered / rep, gathered * rep)        # [B, L], fp32

    logits.scatter_(1, sampling_tensors.seen_toks, vals.to(logits.dtype))   # cast back to bf16
    return logits   # bf16

def apply_penalties(logits: torch.Tensor, sampling_tensors: SamplingTensors):
    # logits [bsz, vocab], bf16
    # [B, V], fp32

    if sampling_tensors.rep_switch_on:
        # logits = apply_rep_penalty(logits=logits, sampling_tensors=sampling_tensors)
        logits = apply_rep_penalty_sparse(logits=logits, sampling_tensors=sampling_tensors) # bf16

    # frequency/presence based on output token
    if sampling_tensors.freq_switch_on:
        # bf16.sub_(float32)
        logits = logits.sub_(sampling_tensors.frequency_penalty[:, None] * sampling_tensors.output_counts)

    if sampling_tensors.pres_switch_on:
        # bf16.sub_(float32)
        logits = logits.sub_(sampling_tensors.presence_penalty[:, None] * sampling_tensors.output_mask)

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

    top_vals, indices = torch.topk(logits, max_k, dim=-1)          # [n, max_k] descending, top-max_k of all lines
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

def apply_fused_top_k_and_p(logits: torch.Tensor, top_k: torch.Tensor, max_k: int, top_p: torch.Tensor):
    # logits [bsz, vocab], bf16
    # top_k/top_p [bsz], fp32
    bsz, vocab = logits.shape
    assert 1 <= max_k <= vocab, f"max_k out of range: {max_k}"

    top_vals, indices = torch.topk(logits, max_k, dim=-1, sorted=True)    # [bsz, max_k], bf16, descending, top-max_k of all lines

    # mask out elements beyond top-k
    beyond = torch.arange(max_k, device=logits.device)[None, :] >= top_k[:, None]
    top_vals = top_vals.masked_fill_(beyond, float("-inf"))  # in-place write

    # # promote to fp32 for precision before top-p
    # top_vals = top_vals.to(torch.float)

    # top-p on sorted top_vals
    probs = top_vals.softmax(dim=-1)    # [bsz, max_k]
    cu_probs = probs.cumsum(dim=-1)     # [bsz, max_k]
    remove = (cu_probs - probs) > top_p[:, None]     # [bsz, max_k], bool tensor

    # mask out elements whose cu_probs exceeds top-p
    top_vals = top_vals.masked_fill(remove, float("-inf"))

    out = torch.full_like(logits, float("-inf"))    # full of -inf
    logger.info(f"dtype, out: {out.dtype}, logits: {logits.dtype}, top_vals: {top_vals.dtype}")
    out.scatter_(1, indices, top_vals)      # scatter sruvial elements back to logits
    return out


def sample(logits: torch.Tensor, sampling_tensors: SamplingTensors):
    assert sampling_tensors.temperature is not None and sampling_tensors.top_k is not None and sampling_tensors.top_p is not None
    return sample_topk_window(logits, 
                  skip_scale_host=sampling_tensors.skip_scale_host, greedy_host=sampling_tensors.greedy_host,
                  temperature=sampling_tensors.temperature,
                  top_k=sampling_tensors.top_k, max_k=sampling_tensors.max_k, top_p=sampling_tensors.top_p)

def scaling(logits: torch.Tensor, skip_scale_host: torch.Tensor, greedy_host: torch.Tensor, temperature: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
    greedy = temperature <= EPS if greedy_host.any() else None  # has greedy rows, but not all rows are greedy

    if skip_scale_host.all():
        return logits, greedy   # no scaling

    # NOTE: no not use where() here, because it will create a new tensor
    scale = temperature if greedy is None else temperature.masked_fill(greedy, 1.0)

    # NOTE: div_ do not change the dtype of logits, however logits/scale[:, None] will return a tensor with float32
    return logits.div_(scale[:, None]), greedy  # temperature; greedy rows unscaled

def sample_oracle(logits: torch.Tensor, skip_scale_host: torch.Tensor, greedy_host: torch.Tensor, temperature: torch.Tensor,
           top_k: torch.Tensor, max_k:int, top_p: torch.Tensor):
    # logits: [n, vocab] (penalties already applied), bf16; all three params are [n]

    if greedy_host.all():
        return logits.argmax(dim=-1)  # all greedy, no scaling, no sampling

    logits, greedy = scaling(logits, skip_scale_host, greedy_host, temperature)

    # argmax should be computed before top-k/p
    greedy_tokens = None
    if greedy is not None:
        greedy_tokens = logits.argmax(dim=-1)                           # greedy rows take argmax

    logits = apply_fused_top_k_and_p(logits=logits, top_k=top_k, max_k=max_k, top_p=top_p)
    probs = logits.softmax(dim=-1)      # [n, vocab]

    row_sum = probs.sum(dim=-1, keepdim=True)                # [n, 1]
    dead = ~torch.isfinite(row_sum) | (row_sum <= 0)         # [n, 1] bool, broadcasts over vocab
    probs = probs.masked_fill(dead, 1.0)

    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)    # select by probability

    if greedy is not None:
        assert greedy_tokens is not None
        return torch.where(greedy, greedy_tokens, sampled)              # [n]

    return sampled

# top-k window
def sample_topk_window(logits: torch.Tensor, skip_scale_host: torch.Tensor, greedy_host: torch.Tensor, temperature: torch.Tensor,
           top_k: torch.Tensor, max_k:int, top_p: torch.Tensor):
    # logits [B, V] bf16, penalties applied
    if greedy_host.all():
        return logits.argmax(dim=-1)    # [B]

    # NOTE: keep in mind that indices are token ids.
    top_vals, indices = torch.topk(logits, max_k, dim=-1, sorted=True)  # bf16, ranking is exact
    greedy_tokens = indices[:, 0]                                       # [B], free argmax

    w = top_vals.float()                                                # cast only the window
    if not skip_scale_host.all():                                       # temperature on the window
        scale = temperature if not greedy_host.any() else temperature.masked_fill(temperature <= EPS, 1.0)    # [B], fp32
        w = w / scale[:, None]  # resides on L2 due to its small size.

    # tok-k
    beyond = torch.arange(max_k, device=w.device)[None, :] >= top_k[:, None]    # [B, max_k], bool
    w = w.masked_fill(beyond, float("-inf"))

    # top-p
    probs = w.softmax(dim=-1)
    remove = (probs.cumsum(dim=-1) - probs) > top_p[:, None]
    w = w.masked_fill(remove, float("-inf"))                            # top-p in the window

    # replace sfotmax+multinomial with Gumbel-max
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(w).clamp_(min=1e-20)))
    local = (w + gumbel_noise).argmax(dim=-1, keepdim=True)     # column index of maximal values
    sampled = indices.gather(1, local).squeeze(1)               # [B], tok id of the maximal vlue, that is, tok id with the max probility per row

    if greedy_host.any():
        return torch.where(temperature <= EPS, greedy_tokens, sampled)  # [B]
    return sampled


