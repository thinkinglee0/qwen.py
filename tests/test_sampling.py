import torch
import pytest
import logging

from qwen.sampling import apply_top_p, apply_top_k, apply_penalties, sample, SamplingTensors
from qwen.utils import pad_token_ids
from constants import *
from qwen.utils import resolve_device, default_dtype

logger = logging.getLogger(__name__)

def _make_sampling_tensors(prompts, outputs, vocab_size, repetition_penalty, freq_pen, pres_pen, device):
    """Build the token-history tensors the scheduler normally derives from the
    resident token-id table (see TokenIdTable.bin_count_and_mask)."""
    bsz = len(prompts)
    prompt_mask = torch.zeros(bsz, vocab_size, dtype=torch.bool, device=device)
    output_mask = torch.zeros(bsz, vocab_size, dtype=torch.bool, device=device)
    output_counts = torch.zeros(bsz, vocab_size, dtype=torch.int32, device=device)
    for i, (p, o) in enumerate(zip(prompts, outputs)):
        if p:
            prompt_mask[i, torch.tensor(p, device=device)] = True
        if o:
            o_t = torch.tensor(o, device=device)
            output_mask[i, o_t] = True
            output_counts[i].scatter_add_(0, o_t, torch.ones_like(o_t, dtype=torch.int32))
    return SamplingTensors(
        temperature=torch.ones(bsz, device=device),
        top_k=torch.zeros(bsz, dtype=torch.int64, device=device), max_k=0,
        top_p=torch.ones(bsz, device=device),
        repetition_penalty=repetition_penalty, freq_pen=freq_pen, pres_pen=pres_pen,
        prompt_mask=prompt_mask, output_counts=output_counts, output_mask=output_mask,
    )


def test_apply_penalties():
    torch.manual_seed(0)
    device = resolve_device()
    vocab_size = 151936          # Actual value for Qwen2.5
    bsz = 4
    logits = torch.randn(bsz, vocab_size, device=device, dtype=torch.float32)

    prompts = [[101, 202, 303], [55, 66],      [777], [11, 12]]
    outputs = [[202],           [66, 66, 66],  [],    [12, 12]]   # seq2 not yet generated

    # four reqs: 0 repetition only; 1 frequency only; 2 all closed; 3 presence only
    repetition_penalty  = torch.tensor([1.15, 1.0, 1.0, 1.0], device=device)
    freq_pen = torch.tensor([0.0,  0.5, 0.0, 0.0], device=device)
    pres_pen = torch.tensor([0.0,  0.0, 0.0, 0.7], device=device)

    sampling_tensors = _make_sampling_tensors(prompts, outputs, vocab_size,
                                              repetition_penalty, freq_pen, pres_pen, device)
    new_logits = apply_penalties(logits, sampling_tensors, vocab_size)

    assert new_logits.shape == torch.Size([bsz, vocab_size])
    assert torch.sum(new_logits[0] != logits[0], dtype=torch.float32) == 3.0    # because of three distinct elements (101, 202, 303)
    assert (new_logits[0]-logits[0]).min().item() == min(new_logits[0, 101]-logits[0, 101],
                                                  new_logits[0, 202]-logits[0, 202],
                                                  new_logits[0, 303]-logits[0, 303])    # differences on other positions are zero.
    for tok in (101, 202, 303):     # repetition_penalty>1 must never raise a logit, whatever its sign
        assert new_logits[0, tok] <= logits[0, tok]

    assert torch.sum(new_logits[1] != logits[1], dtype=torch.float32) == 1.0    # only one element (66) occurs in the output
    torch.testing.assert_close((new_logits[1]-logits[1]).min().item(), -1.5, rtol=0, atol=1e-3)     # token_id=66 occurs three times, so frequency penalty = 0.5*3 = 1.5
    torch.testing.assert_close((new_logits[1,66]-logits[1,66]).item(), -1.5, rtol=0, atol=1e-3)              # same
    assert new_logits[1, 55] == logits[1, 55]       # prompt-only token is untouched by freq penalty

    assert torch.equal(new_logits[2], logits[2])        # do nothing

    # presence penalty: a flat -pres_pen on every token seen in the output, count-independent
    assert torch.sum(new_logits[3] != logits[3], dtype=torch.float32) == 1.0    # only token 12 is in the output
    torch.testing.assert_close((new_logits[3, 12]-logits[3, 12]).item(), -0.7, rtol=0, atol=1e-3)
    assert new_logits[3, 11] == logits[3, 11]       # prompt-only token is untouched by pres penalty


# ---------- apply_top_k ----------

def test_top_k_basic_threshold():
    logits = torch.tensor([[5., 4., 3., 2., 1.]])
    out = apply_top_k(logits, torch.tensor([2]), max_k=2)
    expected = torch.tensor([[5., 4., NINF, NINF, NINF]])
    assert torch.equal(out, expected)


def test_top_k_keeps_ties_at_threshold():
    logits = torch.tensor([[5., 4., 4., 1.]])
    out = apply_top_k(logits, torch.tensor([2]), max_k=2)
    expected = torch.tensor([[5., 4., 4., NINF]])   # elements whose value is greater than the 2-nd element
    assert torch.equal(out, expected)


def test_top_k_disabled_rows_untouched():
    # no-op while top_k<=0 or >=vocab
    logits = torch.tensor([[5., 4., 3.],
                           [5., 4., 3.],
                           [5., 4., 3.]])
    out = apply_top_k(logits, torch.tensor([0, 3, -1]), max_k=3)  # vocab=3
    assert torch.equal(out, logits)


def test_top_k_mixed_batch():
    # one line valid, one line disabled, which should not be affected by the valid line
    logits = torch.tensor([[9., 8., 7., 6.],
                           [9., 8., 7., 6.]])
    out = apply_top_k(logits, torch.tensor([1, 0]), max_k=1)  # line 0 top-1, line 1 keeps intact
    expected = torch.tensor([[9., NINF, NINF, NINF],
                             [9., 8., 7., 6.]])
    assert torch.equal(out, expected)


def test_top_k_disabled_does_not_inflate_max_k(monkeypatch):
    logits = torch.randn(2, 1000)
    captured = {}
    real_topk = torch.topk
    def spy(inp, k, *a, **kw):
        captured["k"] = k
        return real_topk(inp, k, *a, **kw)
    monkeypatch.setattr(torch, "topk", spy)

    apply_top_k(logits, torch.tensor([5, 5000]), max_k=5000)  # line 1 disabled, top_k>>vocab
    ''' valid line only needs top5, but prevent from the synchronous operation (as followings) between device and host 
    and computing disabled from top_k_host, so introduce a constant variable MAX_EFFECTIVE_TOP_K to clamp the top_k range. 

    max_k = int(top_k[~disabled].max().item())
    '''
    assert captured["k"] == 1000


# ---------- apply_top_p ----------

def test_top_p_basic_cumulative():
    p = torch.tensor([0.5, 0.25, 0.125, 0.125])     # sorted input
    logits = p.log().unsqueeze(0)  # softmax(log(p)) == p, shape [1, 4]
    # Removal rule: (cumulative_prob_before_this) > p -> remove
    # After sorting cumulative: before = [0, .5, .75, .875]
    #   token0: before=0    -> keep
    #   token1: before=.5   -> keep
    #   token2: before=.75  -> remove
    #   token3: before=.875 -> remove
    out = apply_top_p(logits, torch.tensor([0.7]))
    keep = out[0] != NINF
    assert keep.tolist() == [True, True, False, False]


def test_top_p_always_keeps_top1():
    # Even an extremely small top_p must keep the highest-probability token
    # (otherwise multinomial sampling has no valid candidates)
    p = torch.tensor([0.9, 0.05, 0.05])
    logits = p.log().unsqueeze(0)
    out = apply_top_p(logits, torch.tensor([0.01]))     # extremely small
    # before[token0]=0, 0>0.01 false -> first token must be kept
    assert out[0, 0] != NINF
    assert (out[0, 1:] == NINF).all()


def test_top_p_scatter_back_to_original_order():
    # Key: input is unsorted; verify mask is correctly scattered back to original vocab order
    # Probabilities [0.1, 0.7, 0.2] correspond to indices [0,1,2], descending order [1,2,0]
    p = torch.tensor([0.1, 0.7, 0.2])
    logits = p.log().unsqueeze(0)
    # For top_p=0.7: after sorting cumulative before=[0(idx1), .7(idx2), .9(idx0)]
    #   idx1: before=0    -> keep
    #   idx2: before=.7   -> keep (strict > is required to remove)
    #   idx0: before=.9   -> remove
    out = apply_top_p(logits, torch.tensor([0.7]))
    keep = (out[0] != NINF).tolist()
    assert keep == [False, True, True]  # In original index order: idx0 is pruned


# ---------- sample ----------

def test_sample_greedy_takes_argmax():
    # temperature <= EPS -> greedy; result must be argmax and not random
    logits = torch.tensor([[1., 9., 3.]])
    out = sample(logits, temperature=torch.tensor([0.0]),
                 top_k=torch.tensor([0]), max_k=0,
                 top_p=torch.tensor([1.0]))  # top_k/p no-op
    assert out.item() == 1


def test_sample_greedy_deterministic_across_seeds():
    logits = torch.tensor([[1., 9., 3., 2.]])
    res = []
    for seed in range(5):
        torch.manual_seed(seed)
        res.append(sample(logits, temperature=torch.tensor([0.0]),
                 top_k=torch.tensor([0]), max_k=0,
                 top_p=torch.tensor([1.0])).item())  # top_k/p no-op
    assert len(set(res)) == 1 and res[0] == 1


def test_sample_all_neg_inf_row_does_not_crash():
    # Regression test for bug #3: a row with all -inf should not make multinomial crash
    logits = torch.full((1, 5), NINF)
    torch.manual_seed(0)
    out = sample(logits, temperature=torch.tensor([1.0]),
                 top_k=torch.tensor([0]), max_k=0, top_p=torch.tensor([1.0]))
    assert out.shape == (1,)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Sampled index: {out.item()}")
    assert 0 <= out.item() < 5  # valid index, not out of bounds


def test_sample_mixed_greedy_and_random_rows():
    # Mixed batch with greedy and sampled rows: greedy row is deterministic, sampled row is valid
    logits = torch.tensor([[1., 9., 2.],   # greedy -> 1
                           [3., 1., 2.]])  # sample
    torch.manual_seed(0)
    out = sample(logits, temperature=torch.tensor([0.0, 1.0]),
                 top_k=torch.tensor([0, 0]), max_k=0, top_p=torch.tensor([1.0, 1.0]))
    assert out[0].item() == 1
    assert 0 <= out[1].item() < 3


def test_sample_top_k_1_equals_argmax_even_when_sampling():
    # When top_k=1, even sampling rows have a single candidate = argmax, so result is deterministic
    logits = torch.tensor([[2., 8., 3.]])
    torch.manual_seed(123)
    out = sample(logits, temperature=torch.tensor([1.0]),
                 top_k=torch.tensor([1]), max_k=1, top_p=torch.tensor([1.0]))
    assert out.item() == 1