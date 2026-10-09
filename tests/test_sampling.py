import torch
import pytest
import logging
from types import SimpleNamespace

from qwen.sampling import (apply_top_p, apply_top_k, apply_fused_top_k_and_p, apply_penalties, sample,
                          SamplingParams, SamplingParamTable, SamplingTensors)
from qwen.constants import MAX_EFFECTIVE_TOP_K, DEFAULT_PIPELINE_DEPTH
from qwen.utils import pad_token_ids
from constants import *
from qwen.utils import resolve_device, default_dtype

logger = logging.getLogger(__name__)

def _make_sampling_tensors(prompts, outputs, vocab_size, repetition_penalty, frequency_penalty, presence_penalty, device):
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
        # the param table only ever emits effective top_k in [1, cutoff]; at the cutoff it cuts nothing
        top_k=torch.full((bsz,), min(MAX_EFFECTIVE_TOP_K, vocab_size), dtype=torch.int64, device=device),
        max_k=min(MAX_EFFECTIVE_TOP_K, vocab_size),
        top_p=torch.ones(bsz, device=device),
        repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
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
    frequency_penalty = torch.tensor([0.0,  0.5, 0.0, 0.0], device=device)
    presence_penalty = torch.tensor([0.0,  0.0, 0.0, 0.7], device=device)

    sampling_tensors = _make_sampling_tensors(prompts, outputs, vocab_size,
                                              repetition_penalty, frequency_penalty, presence_penalty, device)
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

    # presence penalty: a flat -presence_penalty on every token seen in the output, count-independent
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

# ---------- apply_fused_top_k_and_p ----------

def _effective_top_k(k_list: list[int], vocab: int):
    """Mirror SamplingParamTable.set_slot, which normalises top_k once at admission: a k that is
    disabled (<=0) or wider than useful (>=cutoff) both collapse to cutoff, so everything
    downstream -- top_k_device, max_k, apply_fused_top_k_and_p -- only ever sees k in [1, cutoff].

    Returns (top_k, max_k) exactly as gather() hands them to the sampler: since k is already
    effective, max_k is a plain max(), with no disabled rows left to special-case."""
    cutoff = min(MAX_EFFECTIVE_TOP_K, vocab)
    eff = [cutoff if (k <= 0 or k >= cutoff) else k for k in k_list]
    return torch.tensor(eff), max(eff)


def test_fused_top_k_only():
    # top_p=1.0 is a no-op, so the fused path must degenerate to plain top-k
    logits = torch.tensor([[5., 4., 3., 2., 1.]])
    out = apply_fused_top_k_and_p(logits, torch.tensor([2]), max_k=2, top_p=torch.tensor([1.0]))
    expected = torch.tensor([[5., 4., NINF, NINF, NINF]])
    assert torch.equal(out, expected)


def test_fused_top_p_only():
    # k at the cutoff (== vocab here) cuts nothing, so the fused path must degenerate to plain top-p
    p = torch.tensor([0.5, 0.25, 0.125, 0.125])
    logits = p.log().unsqueeze(0)
    top_k, max_k = _effective_top_k([0], vocab=4)            # user asked for no top-k
    out = apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=torch.tensor([0.7]))
    assert (out[0] != NINF).tolist() == [True, True, False, False]   # same cut as test_top_p_basic_cumulative


def test_fused_renormalises_probs_within_each_row_k():
    # The reason top_vals must be masked *before* softmax: probabilities have to be renormalised
    # over that row's own k, not over the max_k slab shared by the whole batch.
    #   row0 (k=2): kept mass .4+.3 -> [.571, .429], before=[0, .571] > .5 -> only token0 survives
    #   row1 (k=4): no cut, mass over all four -> before=[0, .4, .7, .9] > .5 -> tokens 0,1 survive
    # Softmaxing over the full max_k=4 slab would have left row0 with before=[0, .4] and kept
    # token1 too -- i.e. fused top-p looser than sequential top-k -> top-p.
    p = torch.tensor([0.4, 0.3, 0.2, 0.1])
    logits = p.log().unsqueeze(0).repeat(2, 1)
    top_k, max_k = _effective_top_k([2, 0], vocab=4)
    out = apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=torch.tensor([0.5, 0.5]))
    assert (out[0] != NINF).tolist() == [True, False, False, False]
    assert (out[1] != NINF).tolist() == [True, True, False, False]


def test_fused_k_at_cutoff_keeps_whole_vocab():
    # vocab <= MAX_EFFECTIVE_TOP_K -> cutoff == vocab, so every way of spelling "no top-k"
    # (0, negative, >= vocab) keeps the row bit-for-bit intact once top_p is a no-op too
    logits = torch.tensor([[5., 4., 3.],
                           [5., 4., 3.],
                           [5., 4., 3.]])
    top_k, max_k = _effective_top_k([0, 3, -1], vocab=3)
    assert top_k.tolist() == [3, 3, 3] and max_k == 3
    out = apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=torch.ones(3))
    assert torch.equal(out, logits)


def test_fused_narrow_row_does_not_cap_wide_row():
    # Regression guard for the max_k bug: max_k is a candidate budget, not just topk()'s width --
    # everything outside the top-max_k columns is hard -inf'd. Deriving max_k from raw top_k
    # (top_k=[1, 0] -> max_k=1) would have truncated the wide row to a single token.
    logits = torch.tensor([[9., 8., 7., 6.],
                           [9., 8., 7., 6.]])
    top_k, max_k = _effective_top_k([1, 0], vocab=4)
    assert max_k == 4
    out = apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=torch.ones(2))
    expected = torch.tensor([[9., NINF, NINF, NINF],
                             [9., 8., 7., 6.]])
    assert torch.equal(out, expected)


def test_fused_always_keeps_top1():
    # Even an absurd top_p must leave one candidate, or multinomial has nothing to draw from
    p = torch.tensor([0.9, 0.05, 0.05])
    logits = p.log().unsqueeze(0)
    out = apply_fused_top_k_and_p(logits, torch.tensor([2]), max_k=2, top_p=torch.tensor([1e-6]))
    assert out[0, 0] != NINF
    assert (out[0, 1:] == NINF).all()


def test_fused_scatter_back_to_original_order():
    # Input unsorted: the mask lives in topk order, so verify it lands on the right vocab ids
    p = torch.tensor([0.1, 0.7, 0.2])
    logits = p.log().unsqueeze(0)
    out = apply_fused_top_k_and_p(logits, torch.tensor([3]), max_k=3, top_p=torch.tensor([0.7]))
    assert (out[0] != NINF).tolist() == [False, True, True]      # idx0 (0.1) pruned
    torch.testing.assert_close(out[0, 1], logits[0, 1])          # survivors keep their own value
    torch.testing.assert_close(out[0, 2], logits[0, 2])


def test_fused_rejects_unclamped_max_k():
    # max_k=0 used to slip through: topk() returns nothing, every row stays -inf, and sample()'s
    # dead-row guard then flattens it into uniform sampling over the whole vocab -- silently.
    logits = torch.randn(1, 8)
    with pytest.raises(AssertionError):
        apply_fused_top_k_and_p(logits, torch.tensor([0]), max_k=0, top_p=torch.tensor([1.0]))
    with pytest.raises(AssertionError):
        apply_fused_top_k_and_p(logits, torch.tensor([9]), max_k=9, top_p=torch.tensor([1.0]))


def test_fused_does_not_widen_topk_beyond_the_batch_max(monkeypatch):
    # The whole point of fusing: no full-vocab sort. apply_top_p would have sorted all 1000.
    logits = torch.randn(2, 1000)
    captured = {}
    real_topk = torch.topk
    def spy(inp, k, *a, **kw):
        captured["k"] = k
        return real_topk(inp, k, *a, **kw)
    monkeypatch.setattr(torch, "topk", spy)

    top_k, max_k = _effective_top_k([5, 8], vocab=1000)          # both narrow
    apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=torch.tensor([0.9, 0.9]))
    assert captured["k"] == 8

    # one row asking for no top-k is enough to force the cutoff width: it must see cutoff candidates
    top_k, max_k = _effective_top_k([5, 5000], vocab=1000)
    apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=torch.tensor([0.9, 0.9]))
    assert captured["k"] == 1000                                 # cutoff = min(MAX_EFFECTIVE_TOP_K, vocab)


def test_fused_truncates_at_cutoff_where_sequential_keeps_the_tail():
    # Once vocab > MAX_EFFECTIVE_TOP_K the two paths deliberately part: a "no top-k" row becomes
    # k=cutoff, which apply_top_k reads as disabled (whole row kept) while the fused path can
    # never scatter back more than cutoff candidates.
    torch.manual_seed(0)
    vocab = MAX_EFFECTIVE_TOP_K + 400
    logits = torch.randn(1, vocab) * 4          # std 4: real lm_head logits are peaky, not N(0,1)
    top_k, max_k = _effective_top_k([0], vocab=vocab)             # -> cutoff, below vocab this time
    assert max_k == MAX_EFFECTIVE_TOP_K

    assert (apply_top_k(logits, top_k, max_k=max_k) != NINF).sum().item() == vocab    # tail kept

    fused = apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=torch.ones(1))
    kept = fused != NINF
    # <= rather than ==: at top_p=1.0 a float32 cumsum overshoots 1.0, so (cu - p) > 1.0 also
    # prunes the last tokens of the slab, whose own probability is ~1e-30. apply_top_p does the
    # same thing on the full vocab -- harmless, but it means counts here are not exactly cutoff.
    assert kept.sum().item() <= max_k
    assert logits[~kept].max() < logits[kept].min()      # what survives is exactly the largest ones
    torch.testing.assert_close(fused[kept], logits[kept])    # survivors keep their own value

    # What the cutoff actually costs -- the justification for MAX_EFFECTIVE_TOP_K. It rides on the
    # logits being sharp: redo this on plain N(0,1) and the cutoff drops ~5% of the mass instead,
    # so the constant is a bet on real logit sharpness, not a property of the cutoff alone.
    dropped = logits.softmax(dim=-1)[~kept].sum().item()
    assert dropped < 1e-5, f"cutoff dropped {dropped:.2e} of the mass"


@pytest.mark.parametrize("k_list,p_list", [
    ([0, 3, 17, 64, 5000, -1], [1.0, 0.9, 0.5, 0.2, 1.0, 0.85]),    # mixed, incl. every "no top-k" spelling
    ([1, 2, 5, 8, 3, 8],       [1.0, 0.95, 0.6, 0.3, 0.75, 1.0]),   # all narrow: max_k=8 << vocab
])
def test_fused_equivalent_to_sequential_reference(k_list, p_list):
    # vocab <= MAX_EFFECTIVE_TOP_K, so cutoff == vocab and even the k==cutoff rows agree: the
    # fused path keeps top-vocab, i.e. everything, which is what apply_top_k's disabled branch
    # does too (see test_fused_truncates_at_cutoff_... for where that stops holding).
    # Random logits: no ties at the k-th value, where sequential's ">= kth" threshold keeps every
    # tied element while the fused path keeps exactly k.
    torch.manual_seed(0)
    vocab = 64
    logits = torch.randn(len(k_list), vocab)
    top_k, max_k = _effective_top_k(k_list, vocab)
    top_p = torch.tensor(p_list)

    fused = apply_fused_top_k_and_p(logits, top_k, max_k=max_k, top_p=top_p)
    ref = apply_top_p(apply_top_k(logits, top_k, max_k=max_k), top_p)       # the unfused reference
    assert torch.equal(fused != NINF, ref != NINF)      # same survivors
    torch.testing.assert_close(fused, ref)              # and the same logit values


# ---------- SamplingParamTable ----------

def _stub_config(vocab_size=151936, max_num_seqs=8, top_k=20):
    """SamplingParamTable only reads these few fields, so the test stays free of real weights."""
    return SimpleNamespace(vocab_size=vocab_size, max_num_seqs=max_num_seqs,
                           max_num_req_slots=max_num_seqs * DEFAULT_PIPELINE_DEPTH, device=resolve_device(),
                           temperature=1.0, top_p=1.0, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                           top_k=top_k)


def test_param_table_clamps_top_k_at_admission():
    # The clamp lives here, once per admitted request, so that neither the per-step gather nor the
    # sampler has to know about "disabled" rows at all.
    cfg = _stub_config()
    cutoff = min(MAX_EFFECTIVE_TOP_K, cfg.vocab_size)
    tab = SamplingParamTable(cfg)

    tab.set_slot(0, SamplingParams(top_k=0), cfg)          # disabled -> cutoff
    tab.set_slot(1, SamplingParams(top_k=5000), cfg)       # wider than useful -> cutoff
    tab.set_slot(2, SamplingParams(top_k=7), cfg)          # in range -> kept as is
    tab.set_slot(3, None, cfg)                             # no user params -> config default
    assert tab.top_k_stage[:4].tolist() == [cutoff, cutoff, 7, cfg.top_k]


def test_param_table_gather_max_k_is_plain_max_of_effective_k():
    # Because set_slot already folded the disabled rows into cutoff, a plain max() is the correct
    # candidate budget. Taken over *raw* top_k it would have been 7 here, capping the two
    # no-top-k rows at 7 candidates out of 151936.
    cfg = _stub_config()
    cutoff = min(MAX_EFFECTIVE_TOP_K, cfg.vocab_size)
    tab = SamplingParamTable(cfg)
    for slot, k in enumerate((0, 7, 5000)):
        tab.set_slot(slot, SamplingParams(top_k=k), cfg)

    idx_host = torch.tensor([0, 1, 2])
    _, top_k, max_k = tab.gather(slot_idx=idx_host.to(cfg.device), slot_idx_stage=idx_host)
    assert top_k.tolist() == [cutoff, 7, cutoff]
    assert max_k == cutoff


# ---------- sample ----------

def test_sample_greedy_takes_argmax():
    # temperature <= EPS -> greedy; result must be argmax and not random
    logits = torch.tensor([[1., 9., 3.]])
    # "no top-k" is spelled k=vocab now: the param table clamps into [1, cutoff] and never emits 0
    out = sample(logits, temperature=torch.tensor([0.0]),
                 top_k=torch.tensor([3]), max_k=3,
                 top_p=torch.tensor([1.0]))  # top_k/p no-op
    assert out.item() == 1


def test_sample_greedy_deterministic_across_seeds():
    logits = torch.tensor([[1., 9., 3., 2.]])
    res = []
    for seed in range(5):
        torch.manual_seed(seed)
        res.append(sample(logits, temperature=torch.tensor([0.0]),
                 top_k=torch.tensor([4]), max_k=4,
                 top_p=torch.tensor([1.0])).item())  # top_k/p no-op
    assert len(set(res)) == 1 and res[0] == 1


def test_sample_all_neg_inf_row_does_not_crash():
    # Regression test for bug #3: a row with all -inf should not make multinomial crash
    logits = torch.full((1, 5), NINF)
    torch.manual_seed(0)
    out = sample(logits, temperature=torch.tensor([1.0]),
                 top_k=torch.tensor([5]), max_k=5, top_p=torch.tensor([1.0]))
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
                 top_k=torch.tensor([3, 3]), max_k=3, top_p=torch.tensor([1.0, 1.0]))
    assert out[0].item() == 1
    assert 0 <= out[1].item() < 3


def test_sample_top_k_1_equals_argmax_even_when_sampling():
    # When top_k=1, even sampling rows have a single candidate = argmax, so result is deterministic
    logits = torch.tensor([[2., 8., 3.]])
    torch.manual_seed(123)
    out = sample(logits, temperature=torch.tensor([1.0]),
                 top_k=torch.tensor([1]), max_k=1, top_p=torch.tensor([1.0]))
    assert out.item() == 1