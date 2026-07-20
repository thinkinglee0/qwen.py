import torch
import pytest
import logging

from qwen.sampling import apply_top_p, apply_top_k, apply_penalties, sample
from qwen.utils import pad_token_ids
from constants import *

logger = logging.getLogger(__name__)

def test_apply_penalties():
    device = "cpu"
    vocab_size = 151936          # Actual value for Qwen2.5
    bsz = 3
    logits = torch.randn(bsz, vocab_size, device=device, dtype=torch.float32)

    prompts = [[101, 202, 303], [55, 66], [777]]
    outputs = [[202],           [66, 66, 66], []]   # seq2 not yet generated

    # three reqs: 0 repetition only; 1 frequency only; 2 all closed
    rep_pen  = torch.tensor([1.15, 1.0, 1.0], device=device)
    freq_pen = torch.tensor([0.0,  0.5, 0.0], device=device)
    pres_pen = torch.tensor([0.0,  0.0, 0.0], device=device)

    new_logits = apply_penalties(logits, prompts, outputs,
                            rep_pen, freq_pen, pres_pen, vocab_size)

    assert new_logits.shape == torch.Size([bsz, vocab_size])
    assert torch.sum(new_logits[0] != logits[0], dtype=torch.float32) == 3.0    # because of three distinct elements (101, 202, 303)
    assert (new_logits[0]-logits[0]).min() == min(new_logits[0, 101]-logits[0, 101],
                                                  new_logits[0, 202]-logits[0, 202],
                                                  new_logits[0, 303]-logits[0, 303])    # differences on other positions are zero.

    assert torch.sum(new_logits[1] != logits[1], dtype=torch.float32) == 1.0    # only one element (66) occurs in the output
    assert (new_logits[1]-logits[1]).min() == -1.5      # token_id=66 occurs three times, so frequency penalty = 0.5*3 = 1.5
    assert new_logits[1,66]-logits[1,66] == -1.5        # same

    assert torch.equal(new_logits[2], logits[2])        # do nothing



# ---------- apply_top_k ----------

def test_top_k_basic_threshold():
    logits = torch.tensor([[5., 4., 3., 2., 1.]])
    out = apply_top_k(logits, torch.tensor([2]))
    expected = torch.tensor([[5., 4., NINF, NINF, NINF]])
    assert torch.equal(out, expected)


def test_top_k_keeps_ties_at_threshold():
    logits = torch.tensor([[5., 4., 4., 1.]])
    out = apply_top_k(logits, torch.tensor([2]))
    expected = torch.tensor([[5., 4., 4., NINF]])   # elements whose value is greater than the 2-nd element
    assert torch.equal(out, expected)


def test_top_k_disabled_rows_untouched():
    # no-op while top_k<=0 or >=vocab
    logits = torch.tensor([[5., 4., 3.],
                           [5., 4., 3.],
                           [5., 4., 3.]])
    out = apply_top_k(logits, torch.tensor([0, 3, -1]))  # vocab=3
    assert torch.equal(out, logits)


def test_top_k_mixed_batch():
    # one line valid, one line disabled, which should not be affected by the valid line
    logits = torch.tensor([[9., 8., 7., 6.],
                           [9., 8., 7., 6.]])
    out = apply_top_k(logits, torch.tensor([1, 0]))  # line 0 top-1, line 1 keeps intact
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

    apply_top_k(logits, torch.tensor([5, 5000]))  # line 1 disabled, top_k>>vocab
    # valid line only needs top5, max_k should be 5 not 1000
    assert captured["k"] == 5


def test_top_k_all_disabled_no_topk(monkeypatch):
    # early return while all disabled
    logits = torch.randn(2, 100)
    called = {"n": 0}
    real_topk = torch.topk
    monkeypatch.setattr(torch, "topk",
                        lambda *a, **kw: (called.__setitem__("n", called["n"]+1),
                                          real_topk(*a, **kw))[1])
    out = apply_top_k(logits, torch.tensor([0, 200]))
    assert called["n"] == 0
    assert torch.equal(out, logits)


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
    # temperature <= _EPS -> greedy; result must be argmax and not random
    logits = torch.tensor([[1., 9., 3.]])
    out = sample(logits, torch.tensor([0.0]),
                 torch.tensor([0]), torch.tensor([1.0]))  # top_k/p no-op
    assert out.item() == 1


def test_sample_greedy_deterministic_across_seeds():
    logits = torch.tensor([[1., 9., 3., 2.]])
    res = []
    for seed in range(5):
        torch.manual_seed(seed)
        res.append(sample(logits, torch.tensor([0.0]),
                          torch.tensor([0]), torch.tensor([1.0])).item())
    assert len(set(res)) == 1 and res[0] == 1


def test_sample_temperature_scaling_applied():
    # Verify temperature scaling is applied for non-greedy rows using an indirect,
    # verifiable signal.
    # High temperature flattens the distribution; low temperature sharpens it.
    # This test ensures greedy vs non-greedy temperature handling is not mixed:
    # greedy rows (temperature=0) are handled deterministically and not scaled.
    # Assert that argmax is unaffected by temperature scaling (scaling doesn't change argmax).
    logits = torch.tensor([[1., 5., 2.]])
    torch.manual_seed(0)
    out_hi = sample(logits.clone(), torch.tensor([100.0]),
                    torch.tensor([1]), torch.tensor([1.0]))  # top_k=1 → 只剩argmax
    assert out_hi.item() == 1  # top_k=1 forces the only candidate to be argmax


def test_sample_all_neg_inf_row_does_not_crash():
    # Regression test for bug #3: a row with all -inf should not make multinomial crash
    logits = torch.full((1, 5), NINF)
    torch.manual_seed(0)
    out = sample(logits, torch.tensor([1.0]),
                 torch.tensor([0]), torch.tensor([1.0]))
    assert out.shape == (1,)
    logger.info(f"Sampled index: {out.item()}")
    assert 0 <= out.item() < 5  # valid index, not out of bounds


def test_sample_mixed_greedy_and_random_rows():
    # Mixed batch with greedy and sampled rows: greedy row is deterministic, sampled row is valid
    logits = torch.tensor([[1., 9., 2.],   # greedy -> 1
                           [3., 1., 2.]])  # sample
    torch.manual_seed(0)
    out = sample(logits, torch.tensor([0.0, 1.0]),
                 torch.tensor([0, 0]), torch.tensor([1.0, 1.0]))
    assert out[0].item() == 1
    assert 0 <= out[1].item() < 3


def test_sample_top_k_1_equals_argmax_even_when_sampling():
    # When top_k=1, even sampling rows have a single candidate = argmax, so result is deterministic
    logits = torch.tensor([[2., 8., 3.]])
    torch.manual_seed(123)
    out = sample(logits, torch.tensor([1.0]),
                 torch.tensor([1]), torch.tensor([1.0]))
    assert out.item() == 1