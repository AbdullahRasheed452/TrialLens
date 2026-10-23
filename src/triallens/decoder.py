# A hand-written Qwen2 forward pass with an explicit KV cache, plus cached and uncached
# greedy decoding loops.
#
# The Hugging Face model is only used as a container for the trained weights. Embedding,
# RMSNorm, the q/k/v projections, rotary position embeddings, grouped-query attention, the
# SwiGLU MLP and the LM head are all applied here, so every step of the cache-aware path
# is visible and testable.
#
# Positions. A token's RoPE angle depends on its absolute position in the sequence. During
# prefill the prompt gets positions 0 .. P-1. Each decode step feeds one new token at
# position = current cache length, so its query and key are rotated exactly as they would
# be in a full recomputation, and the cached keys never need to be re-rotated.
#
# Masking. The attention functions get q_offset = cache length. A prefill uses a P x P
# causal mask; a single decode step attends to every cached key (nothing is in its
# future); a multi-token step on top of a cache (speculative verification) gets a causal
# mask shifted by the offset.
#
# Cache growth. KVCache preallocates [layers, batch, kv_heads, max_len, head_dim] for keys
# and values and keeps a length pointer. Writing a step is a slice assignment, so there is
# no torch.cat copy per token, and rolling back (speculative decoding) is just moving the
# pointer. Memory per token is 2 * layers * kv_heads * head_dim * bytes, which for
# Qwen2.5-1.5B is 28 KB in fp16 and 56 KB in fp32.

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .attention import ATTENTION


class KVCache:
    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, dtype, device):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0

    # Store this step's keys/values at [start, start + T) and return everything up to there.
    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int):
        end = start + k.shape[-2]
        if end > self.max_len:
            raise RuntimeError(f"KV cache full: need {end} positions, allocated {self.max_len}")
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def crop(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot crop cache of length {self.length} to {length}")
        self.length = length

    def bytes_used(self) -> int:
        per_pos = self.k[:, :, :, :1].numel() * self.k.element_size() * 2
        return per_pos * self.length

    def bytes_allocated(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


def rms_norm(x: torch.Tensor, norm) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class QwenDecoder:
    def __init__(self, hf_model, attention: str = "sdpa"):
        self.model = hf_model.model
        self.lm_head = hf_model.lm_head
        cfg = hf_model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        theta = cfg.rope_parameters["rope_theta"]
        self.device = next(hf_model.parameters()).device
        self.dtype = next(hf_model.parameters()).dtype
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32,
                                                      device=self.device) / self.head_dim))
        self.set_attention(attention)

    def set_attention(self, name: str) -> None:
        self.attention_name = name
        self.attn = ATTENTION[name]

    def new_cache(self, max_len: int, batch: int = 1) -> KVCache:
        return KVCache(self.n_layers, batch, self.n_kv, max_len, self.head_dim, self.dtype, self.device)

    def rope(self, positions: torch.Tensor):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    # One forward pass over `ids` [B, T] whose first token sits at absolute position `start`.
    # With a cache, start must equal cache.length; without one, start must be 0.
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: KVCache | None = None, start: int = 0,
                last_only: bool = False) -> torch.Tensor:
        if cache is not None and start != cache.length:
            raise ValueError(f"start {start} does not match cache length {cache.length}")
        if cache is None and start != 0:
            raise ValueError("an uncached forward always starts at position 0")
        B, T = ids.shape
        cos, sin = self.rope(torch.arange(start, start + T, device=self.device))
        h = self.model.embed_tokens(ids)

        for li, layer in enumerate(self.model.layers):
            a = layer.self_attn
            x = rms_norm(h, layer.input_layernorm)
            q = a.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = a.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = a.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            if cache is not None:
                k, v = cache.write(li, k, v, start)
            group = self.n_heads // self.n_kv
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
            o = self.attn(q, k, v, causal=True, q_offset=start)
            h = h + a.o_proj(o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))

            mlp = layer.mlp
            x = rms_norm(h, layer.post_attention_layernorm)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

        if cache is not None:
            cache.length = start + T
        h = rms_norm(h[:, -1:] if last_only else h, self.model.norm)
        return self.lm_head(h)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


# Greedy pick plus the gap between the best and second-best logit. A small gap marks a
# near tie, which is where tiny numerical differences between two paths can flip a token.
def _greedy(logits: torch.Tensor):
    top = logits[:, -1].float().topk(2, dim=-1)
    return top.indices[:, :1], (top.values[0, 0] - top.values[0, 1]).item()


# Greedy decoding that reuses and extends the KV cache: one prefill over the prompt, then
# one single-token forward per generated token.
def generate_cached(dec: QwenDecoder, prompt_ids: torch.Tensor, max_new_tokens: int,
                    eos_ids=(), cache: KVCache | None = None) -> dict:
    P = prompt_ids.shape[1]
    cache = cache or dec.new_cache(P + max_new_tokens)
    cache.crop(0)
    _sync(dec.device)
    t0 = time.perf_counter()
    nxt, gap = _greedy(dec.forward(prompt_ids, cache, start=0, last_only=True))
    _sync(dec.device)
    t1 = time.perf_counter()

    tokens, margins = [nxt.item()], [gap]
    while len(tokens) < max_new_tokens and tokens[-1] not in eos_ids:
        nxt, gap = _greedy(dec.forward(nxt, cache, start=cache.length, last_only=True))
        tokens.append(nxt.item())
        margins.append(gap)
    _sync(dec.device)
    t2 = time.perf_counter()
    return _timing(tokens, margins, t1 - t0, t2 - t1, cache_bytes=cache.bytes_used())


# Reference path without cache reuse: every step recomputes the whole sequence from scratch.
def generate_uncached(dec: QwenDecoder, prompt_ids: torch.Tensor, max_new_tokens: int,
                      eos_ids=()) -> dict:
    seq = prompt_ids
    _sync(dec.device)
    t0 = time.perf_counter()
    nxt, gap = _greedy(dec.forward(seq, None, last_only=True))
    _sync(dec.device)
    t1 = time.perf_counter()

    tokens, margins = [nxt.item()], [gap]
    while len(tokens) < max_new_tokens and tokens[-1] not in eos_ids:
        seq = torch.cat([seq, nxt], dim=1)
        nxt, gap = _greedy(dec.forward(seq, None, last_only=True))
        tokens.append(nxt.item())
        margins.append(gap)
    _sync(dec.device)
    t2 = time.perf_counter()
    return _timing(tokens, margins, t1 - t0, t2 - t1, cache_bytes=0)


def _timing(tokens, margins, prefill_s, decode_s, cache_bytes) -> dict:
    n_decode = len(tokens) - 1
    return {
        "tokens": tokens,
        "margins": margins,
        "n_tokens": len(tokens),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": prefill_s + decode_s,
        "decode_tok_per_s": n_decode / decode_s if decode_s > 0 and n_decode else None,
        "overall_tok_per_s": len(tokens) / (prefill_s + decode_s),
        "kv_cache_bytes": cache_bytes,
    }
