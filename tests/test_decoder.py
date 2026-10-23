# The hand-written decoder is checked on a tiny randomly initialised Qwen2 model so the
# tests run in seconds without downloading weights. The same code path runs the real
# Qwen2.5 models; scripts/bench_kv_cache.py repeats the token checks on those.

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from triallens.decoder import QwenDecoder, generate_cached, generate_uncached


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).eval()
    return model


def prompt(n, seed=1):
    return torch.randint(0, 300, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("attention", ["reference", "tiled", "sdpa"])
def test_forward_matches_huggingface(tiny, attention):
    ids = prompt(37)
    ours = QwenDecoder(tiny, attention).forward(ids)
    with torch.no_grad():
        theirs = tiny(ids).logits
    torch.testing.assert_close(ours, theirs, atol=1e-4, rtol=1e-4)


def test_cached_step_logits_match_full_recompute(tiny):
    dec = QwenDecoder(tiny, "reference")
    ids = prompt(20)
    cache = dec.new_cache(32)
    dec.forward(ids[:, :15], cache, start=0)
    step = torch.cat([dec.forward(ids[:, i:i + 1], cache, start=i) for i in range(15, 20)], dim=1)
    full = dec.forward(ids)[:, 15:20]
    torch.testing.assert_close(step, full, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("p_len,n_new", [(5, 8), (40, 25), (120, 60)])
@pytest.mark.parametrize("attention", ["reference", "tiled", "sdpa"])
def test_cached_and_uncached_greedy_tokens_agree(tiny, p_len, n_new, attention):
    dec = QwenDecoder(tiny, attention)
    ids = prompt(p_len)
    cached = generate_cached(dec, ids, n_new)
    uncached = generate_uncached(dec, ids, n_new)
    assert cached["tokens"] == uncached["tokens"]
    assert len(cached["tokens"]) == n_new


def test_matches_huggingface_generate(tiny):
    ids = prompt(30)
    ours = generate_cached(QwenDecoder(tiny, "sdpa"), ids, 20)["tokens"]
    hf = tiny.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20, min_new_tokens=20,
                       do_sample=False, pad_token_id=0)[0, 30:].tolist()
    assert ours == hf


def test_multi_token_step_on_cache(tiny):
    # The shape speculative verification uses: several new tokens appended to a cache at once.
    dec = QwenDecoder(tiny, "tiled")
    ids = prompt(30)
    cache = dec.new_cache(40)
    dec.forward(ids[:, :24], cache, start=0)
    chunk = dec.forward(ids[:, 24:], cache, start=24)
    torch.testing.assert_close(chunk, dec.forward(ids)[:, 24:], atol=1e-4, rtol=1e-4)


def test_crop_then_rewrite_gives_same_logits(tiny):
    dec = QwenDecoder(tiny, "sdpa")
    ids = prompt(20)
    other = prompt(5, seed=9)
    cache = dec.new_cache(32)
    dec.forward(ids[:, :15], cache, start=0)
    dec.forward(other, cache, start=15)          # write tokens we will throw away
    cache.crop(15)
    after = dec.forward(ids[:, 15:], cache, start=15)
    torch.testing.assert_close(after, dec.forward(ids)[:, 15:], atol=1e-4, rtol=1e-4)


def test_shared_prefix_reuse_matches_fresh_decode(tiny):
    # The app prefills the patient note once and reuses it for every criterion.
    dec = QwenDecoder(tiny, "sdpa")
    prefix = prompt(25, seed=3)
    cache = dec.new_cache(80)
    dec.forward(prefix, cache, start=0)
    for seed in (4, 5, 6):
        full = torch.cat([prefix, prompt(10, seed=seed)], dim=1)
        reused = generate_cached(dec, full, 12, cache=cache, prefix_len=25)
        assert reused["tokens"] == generate_cached(dec, full, 12)["tokens"]


def test_cache_bookkeeping(tiny):
    dec = QwenDecoder(tiny)
    cache = dec.new_cache(10)
    dec.forward(prompt(6), cache, start=0)
    assert cache.length == 6
    # 2 (k and v) * 3 layers * 2 kv heads * 16 head_dim * 4 bytes per position
    assert cache.bytes_used() == 6 * 2 * 3 * 2 * 16 * 4
    with pytest.raises(ValueError):
        dec.forward(prompt(1), cache, start=3)
    with pytest.raises(RuntimeError):
        dec.forward(prompt(5), cache, start=6)
