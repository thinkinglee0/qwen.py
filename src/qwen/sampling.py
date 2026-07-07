import logging
import torch

_EPS = 1e-5


logger = logging.getLogger(__name__)

def bin_counts_and_mask(tokens, vocab_size, bsz):
    # tokens: [bsz, max_len], fill it with vocab_size for len(seq)<max_len while padding
    bc = torch.zeros((bsz, vocab_size + 1),    # padding dimension for padded tokens
                     dtype=torch.long, device=tokens.device)
    bc.scatter_add_(1, tokens, torch.ones_like(tokens))     # to count
    bc = bc[:, :vocab_size]     # remove padding dimension
    return bc, bc > 0

def apply_penalties(logits, prompt_tokens, output_tokens,
                    rep_pen, freq_pen, pres_pen, vocab_size):
    # logits [bsz, vocab_size]
    bsz = logits.shape[0]

    # mask now means presence/existing, shape [bsz, vocab_size]
    _,            prompt_mask  = bin_counts_and_mask(prompt_tokens, vocab_size, bsz)  # ditch the count of prompt
    out_counts,   output_mask  = bin_counts_and_mask(output_tokens, vocab_size, bsz)

    # -> shape [bsz, vocab_size]
    rep = rep_pen[:, None].repeat(1, vocab_size)

    # repetition penalty based on prompt and output
    rep[~(prompt_mask | output_mask)] = 1.0     # unseen set to 1.0
    logits = torch.where(logits > 0, logits / rep, logits * rep)    # ensure new <= old while rep>1.0, no matter the signedness of ligits

    # frequency/presence based on output token
    logits = logits - freq_pen[:, None] * out_counts
    logits = logits - pres_pen[:, None] * output_mask
    return logits

def apply_top_k(logits, top_k):
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

def apply_top_p(logits, top_p):
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

def sample(logits, temperature, top_k, top_p):
    # logits: [n, vocab] (penalties already applied); all three params are [n]
    greedy = temperature <= _EPS
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


