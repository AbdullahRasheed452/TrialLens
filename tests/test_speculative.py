# Speculative decoding must reproduce plain greedy decoding of the target token for token,
# whatever the draft proposes. Tiny random Qwen2 models keep this fast; a draft built from
# the target's own weights checks the all-accepted path, and an unrelated draft checks the
# rejection path.

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from triallens.decoder import QwenDecoder, generate_cached
from triallens.speculative import generate_speculative


def tiny(seed, layers, hidden):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=300, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    return QwenDecoder(Qwen2ForCausalLM(cfg).eval(), "sdpa")


@pytest.fixture(scope="module")
def target():
    return tiny(0, 3, 64)


@pytest.fixture(scope="module")
def unrelated_draft():
    return tiny(7, 1, 32)


def prompt(n, seed=1):
    return torch.randint(0, 300, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("k", [1, 2, 4, 7])
@pytest.mark.parametrize("p_len,n_new", [(6, 10), (40, 33), (90, 64)])
def test_matches_greedy_with_unrelated_draft(target, unrelated_draft, k, p_len, n_new):
    ids = prompt(p_len)
    greedy = generate_cached(target, ids, n_new)["tokens"]
    spec = generate_speculative(target, unrelated_draft, ids, n_new, k=k)
    assert spec["tokens"] == greedy
    assert spec["drafted"] >= spec["accepted"]


@pytest.mark.parametrize("k", [1, 3, 5])
def test_self_draft_accepts_almost_everything(target, k):
    ids = prompt(30)
    greedy = generate_cached(target, ids, 40)["tokens"]
    spec = generate_speculative(target, target, ids, 40, k=k)
    assert spec["tokens"] == greedy
    assert spec["acceptance_rate"] > 0.9
    # With every draft accepted each round commits k + 1 tokens from one target forward.
    assert spec["tokens_per_target_forward"] > 1


def test_stops_at_eos(target, unrelated_draft):
    ids = prompt(20)
    greedy = generate_cached(target, ids, 30)["tokens"]
    eos = greedy[11]
    expected = greedy[: greedy.index(eos) + 1]
    assert generate_speculative(target, unrelated_draft, ids, 30, k=4, eos_ids={eos})["tokens"] == expected
    assert generate_cached(target, ids, 30, eos_ids={eos})["tokens"] == expected


def test_accounting(target, unrelated_draft):
    spec = generate_speculative(target, unrelated_draft, prompt(25), 20, k=3)
    assert spec["n_tokens"] == 20
    assert sum(spec["accepted_per_round"]) == spec["accepted"]
    # Prefill plus one verification per round.
    assert spec["target_forwards"] == 1 + spec["rounds"]
