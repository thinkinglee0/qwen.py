import pytest
import torch

from rope import BaseRoPE

def test_inv_freq_values():
    rope = BaseRoPE(dim=8, max_seq_len=16, base=10000.0)
    inv_freq = rope._compute_inv_freq(10000.0)
    expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    torch.testing.assert_close(inv_freq, expected)
    assert inv_freq.shape == (4,)  # dim//2

def test_cache_shapes():
    rope = BaseRoPE(dim=8, max_seq_len=16)
    assert rope.cos_cached.shape == (1, 1, 16, 4)
    assert rope.sin_cached.shape == (1, 1, 16, 4)

def test_position_zero_is_identity():
    # t=0 → freqs=0 → cos=1, sin=0 → remain equal
    rope = BaseRoPE(dim=8, max_seq_len=16)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    q_out, k_out = rope(q, k, offset=0)
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)

def test_relative_position_invariance():
    # RoPE core property: <q_m, k_n> depends only on the relative displacement (m-n)
    # That is, the dot product of q at position m and k at position n is equal to the dot product of each translated by Δ.
    rope = BaseRoPE(dim=64, max_seq_len=256)
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)

    def dot_at(m, n):
        qm, _ = rope(q, q, offset=m)
        kn, _ = rope(k, k, offset=n)
        return (qm * kn).sum().item()

    base = dot_at(10, 5)        # Δ = 5
    shifted = dot_at(20, 15)    # Δ = 5
    assert abs(base - shifted) < 1e-4