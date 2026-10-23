# Scaled dot-product attention, three ways.
#
#   reference  materialises the full score matrix, masks it, and applies a numerically
#              stable softmax (subtract the row max before exp). Easy to read and check.
#   tiled      FlashAttention-style blocked attention. Keys/values are processed in blocks
#              while each query row keeps a running max m, a running normaliser l and an
#              unnormalised output accumulator. Only a block_q x block_k score tile exists
#              at any time, so memory no longer grows with Tq x Tk.
#   sdpa       torch.nn.functional.scaled_dot_product_attention, PyTorch's fused kernel,
#              used as the speed reference.
#
# All three take q [B, H, Tq, D], k and v [B, H, Tk, D] with the same number of heads
# (grouped-query attention is expanded by the caller). With causal=True, query i sits at
# absolute position q_offset + i and may attend to keys 0 .. q_offset + i. This offset is
# what makes the same function work for a full prompt (q_offset = 0), one decode step
# with a KV cache (Tq = 1, q_offset = cache length) and speculative verification
# (Tq = k + 1 drafted tokens on top of the cache).

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def causal_mask(tq: int, tk: int, q_offset: int, device) -> torch.Tensor:
    q_pos = torch.arange(tq, device=device)[:, None] + q_offset
    k_pos = torch.arange(tk, device=device)[None, :]
    return k_pos <= q_pos


def reference_attention(q, k, v, causal: bool = True, q_offset: int = 0) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    # Scores and softmax in fp32 so fp16 inputs do not overflow inside exp.
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        allowed = causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device)
        scores = scores.masked_fill(~allowed, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)
    weights = scores.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return torch.matmul(weights, v.float()).to(q.dtype)


def tiled_attention(q, k, v, causal: bool = True, q_offset: int = 0,
                    block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    B, H, tq, D = q.shape
    tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, H, tq, D, dtype=q.dtype, device=q.device)
    qf, kf, vf = q.float(), k.float(), v.float()

    for qs in range(0, tq, block_q):
        qe = min(qs + block_q, tq)
        q_blk = qf[:, :, qs:qe]
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, D), device=q.device)
        # With a causal mask, key blocks that start after the last query position in this
        # block are fully masked and are skipped entirely.
        k_end = min(tk, q_offset + qe) if causal else tk

        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            s = torch.matmul(q_blk, kf[:, :, ks:ke].transpose(-2, -1)) * scale
            if causal:
                q_pos = torch.arange(qs, qe, device=q.device)[:, None] + q_offset
                k_pos = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(k_pos > q_pos, float("-inf"))

            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # A row can be fully masked inside one tile; using 0 instead of -inf as the
            # shift keeps exp(-inf - -inf) from turning into NaN.
            shift = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - shift)
            correction = torch.exp(m - shift)
            l = l * correction + p.sum(dim=-1, keepdim=True)
            acc = acc * correction + torch.matmul(p, vf[:, :, ks:ke])
            m = m_new

        out[:, :, qs:qe] = (acc / l).to(q.dtype)
    return out


def sdpa_attention(q, k, v, causal: bool = True, q_offset: int = 0) -> torch.Tensor:
    if not causal:
        return F.scaled_dot_product_attention(q, k, v)
    if q_offset == 0 and q.shape[-2] == k.shape[-2]:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # is_causal assumes the query block starts at key 0, which is wrong once a cache is
    # in front of it, so pass the offset mask explicitly instead.
    mask = causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


ATTENTION = {
    "reference": reference_attention,
    "tiled": tiled_attention,
    "sdpa": sdpa_attention,
}
