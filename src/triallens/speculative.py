# Greedy speculative decoding with a small draft model and a larger target model.
#
# One round:
#   1. Draft. The draft model proposes k tokens d1..dk greedily, one forward per token.
#   2. Verify. The target runs ONE forward over [last, d1, .., dk] on top of its cache and
#      reads its own greedy choice g0..gk after each of those positions.
#   3. Accept. d(i+1) is accepted while it equals g(i). With n accepted drafts, the round
#      commits d1..dn plus g(n): the target's correction for the first rejected draft, or a
#      free bonus token when all k were accepted. Every committed token is therefore the
#      target's own greedy choice, so the output equals plain greedy decoding of the target.
#   4. Sync caches. Both caches wrote entries for tokens that were later rejected. They are
#      cropped back to the last accepted position (just a length pointer in KVCache), and
#      each model keeps a list of committed tokens it has not processed yet ("pending").
#      The target always has one pending token (g(n)); the draft has one, or two when all
#      drafts were accepted, because it never ran a forward on its own last proposal dk.

from __future__ import annotations

import time

import torch

from .decoder import QwenDecoder, _sync


class _Side:
    # A model plus its cache and the committed tokens it still has to feed.
    def __init__(self, dec: QwenDecoder, max_len: int):
        self.dec = dec
        self.cache = dec.new_cache(max_len)
        self.pending: list[int] = []
        self.forwards = 0

    def feed(self, tokens: list[int], last_only: bool) -> torch.Tensor:
        ids = torch.tensor([tokens], device=self.dec.device)
        self.forwards += 1
        return self.dec.forward(ids, self.cache, start=self.cache.length, last_only=last_only)


def generate_speculative(target: QwenDecoder, draft: QwenDecoder, prompt_ids: torch.Tensor,
                         max_new_tokens: int, k: int = 4, eos_ids=()) -> dict:
    P = prompt_ids.shape[1]
    max_len = P + max_new_tokens + k + 2
    tgt, drf = _Side(target, max_len), _Side(draft, max_len)
    prompt = prompt_ids[0].tolist()

    _sync(target.device)
    t0 = time.perf_counter()
    # Prefill both models on the prompt. The target's prediction is the first output token.
    first = tgt.feed(prompt, last_only=True)[0, -1].argmax().item()
    drf.feed(prompt, last_only=True)
    _sync(target.device)
    t_prefill = time.perf_counter() - t0

    out = [first]
    tgt.pending = [first]
    drf.pending = [first]
    rounds = drafted = accepted = 0
    per_round = []

    while len(out) < max_new_tokens and out[-1] not in eos_ids:
        k_now = min(k, max_new_tokens - len(out))
        draft_base = drf.cache.length

        # 1. Draft k tokens. The first forward also absorbs the draft's pending tokens.
        proposals = []
        logits = drf.feed(drf.pending, last_only=True)
        for i in range(k_now):
            proposals.append(logits[0, -1].argmax().item())
            if i < k_now - 1:
                logits = drf.feed([proposals[-1]], last_only=True)

        # 2. Verify all proposals with a single target forward.
        verify_base = tgt.cache.length
        g = tgt.feed(tgt.pending + proposals, last_only=False)[0].argmax(-1).tolist()
        # g[j] is the target's choice after position j of [pending..., proposals...]; the
        # ones that matter start after the last pending token.
        g = g[len(tgt.pending) - 1:]

        # 3. Longest prefix of proposals the target agrees with, plus one target token.
        n = 0
        while n < k_now and proposals[n] == g[n]:
            n += 1
        new = proposals[:n] + [g[n]]

        # 4. Keep only cache entries for committed tokens.
        tgt.cache.crop(verify_base + len(tgt.pending) + n)
        drf_kept = len(drf.pending) + min(n, k_now - 1)
        drf.cache.crop(draft_base + drf_kept)
        tgt.pending = [g[n]]
        drf.pending = ([proposals[-1]] if n == k_now else []) + [g[n]]

        rounds += 1
        drafted += k_now
        accepted += n
        per_round.append(n)
        out.extend(new)
        if any(t in eos_ids for t in new):
            out = out[: next(i for i, t in enumerate(out) if t in eos_ids) + 1]
            break

    out = out[:max_new_tokens]
    _sync(target.device)
    total = time.perf_counter() - t0
    return {
        "tokens": out,
        "n_tokens": len(out),
        "prefill_s": t_prefill,
        "decode_s": total - t_prefill,
        "total_s": total,
        "overall_tok_per_s": len(out) / total,
        "k": k,
        "rounds": rounds,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
        "mean_accepted_per_round": accepted / rounds if rounds else None,
        "accepted_per_round": per_round,
        "target_forwards": tgt.forwards,
        "draft_forwards": drf.forwards,
        "tokens_per_target_forward": len(out) / tgt.forwards,
    }
