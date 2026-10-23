# Tiled online-softmax attention and the reference implementation must agree with each
# other and with PyTorch's fused kernel across shapes, block sizes and causal offsets.

import math

import pytest
import torch

from triallens.attention import causal_mask, reference_attention, sdpa_attention, tiled_attention

SHAPES = [
    # B, H, Tq, Tk, D, q_offset
    (1, 1, 1, 1, 8, 0),
    (1, 2, 7, 7, 16, 0),        # prompt shorter than one block
    (2, 4, 130, 130, 32, 0),    # ragged final block
    (1, 3, 1, 50, 64, 49),      # single decode step on top of a cache
    (1, 2, 5, 69, 32, 64),      # speculative verification: 5 queries after 64 cached keys
    (1, 2, 96, 96, 128, 0),     # Qwen head_dim
]


def naive(q, k, v, causal, q_offset):
    # Textbook formula with the library softmax, used as an independent check.
    s = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device), float("-inf"))
    return (torch.softmax(s, dim=-1) @ v.double()).to(q.dtype)


def make(B, H, Tq, Tk, D, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, T, D, generator=g) * scale for T in (Tq, Tk, Tk)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_reference_matches_naive(shape, causal):
    *dims, off = shape
    q, k, v = make(*dims)
    torch.testing.assert_close(reference_attention(q, k, v, causal, off), naive(q, k, v, causal, off),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("blocks", [(16, 16), (64, 32), (7, 13)])
def test_tiled_matches_reference(shape, blocks):
    *dims, off = shape
    q, k, v = make(*dims)
    out = tiled_attention(q, k, v, True, off, block_q=blocks[0], block_k=blocks[1])
    torch.testing.assert_close(out, reference_attention(q, k, v, True, off), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_sdpa_wrapper_matches_reference(shape):
    *dims, off = shape
    q, k, v = make(*dims)
    torch.testing.assert_close(sdpa_attention(q, k, v, True, off), reference_attention(q, k, v, True, off),
                               atol=1e-5, rtol=1e-5)


def test_large_scores_stay_finite():
    # Scores around +-1e4 overflow a naive exp; the max-shifted versions must not.
    q, k, v = make(1, 2, 40, 40, 16, scale=60.0)
    ref = reference_attention(q, k, v)
    tiled = tiled_attention(q, k, v, block_q=8, block_k=8)
    assert torch.isfinite(ref).all() and torch.isfinite(tiled).all()
    torch.testing.assert_close(tiled, ref, atol=1e-4, rtol=1e-4)


def test_first_query_only_sees_first_key():
    q, k, v = make(1, 1, 4, 4, 8)
    out = tiled_attention(q, k, v, block_q=2, block_k=2)
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0])


def test_fp16_inputs():
    q, k, v = (t.half() for t in make(1, 2, 33, 33, 32))
    out = tiled_attention(q, k, v, block_q=16, block_k=16)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), reference_attention(q, k, v).float(), atol=2e-3, rtol=2e-3)
