# The screening engine behind the app: turns a free-text patient note and a list of trial
# criteria into criterion-level predictions with evidence, explanations and a rule-based
# overall outcome.
#
# Nothing here is a clinical decision. The overall outcome is an aggregation of the
# criterion-level predictions under the explicit rules in `aggregate`, not a separately
# labelled ground truth.

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import __version__
from .data import LABELS
from .decoder import QwenDecoder, generate_cached, generate_uncached
from .inference import DEFAULT_MODEL, load_model
from .parsing import parse_prediction
from .prompts import build_messages
from .speculative import generate_speculative

ROOT = Path(__file__).resolve().parents[2]
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NOTE_TOKENS = 1500
TRAINED_NOTE_TOKENS = 400
MAX_CRITERIA = 40
MAX_NEW_TOKENS = 64

DECODING_MODES = {
    "kv_prefix": "KV cache + shared patient prefix",
    "kv": "KV cache",
    "speculative": "Speculative (0.5B draft)",
    "uncached": "No cache (reference)",
}

DISCLAIMER = ("Educational screening prototype. TrialLens is a research project and does not make "
              "clinical enrollment decisions. Every result must be checked by a qualified person.")


class InputError(ValueError):
    pass


# ---------------------------------------------------------------- patient note

_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*\S)\s*$")
_ABBREV = r"(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e|approx|St|No|Fig)"


# A note whose lines are already numbered ("0. ...") keeps those IDs, as in the dataset.
# Otherwise it is split into sentences and numbered from 0.
def split_sentences(note: str) -> list[dict]:
    note = (note or "").strip()
    if not note:
        raise InputError("The patient note is empty.")
    lines = [l for l in note.splitlines() if l.strip()]
    numbered = [_NUMBERED.match(l) for l in lines]
    if all(numbered):
        ids = [int(m.group(1)) for m in numbered]
        if len(set(ids)) != len(ids):
            raise InputError("The note's sentence numbers repeat. Remove the numbers or fix them.")
        return [{"id": int(m.group(1)), "text": m.group(2)} for m in numbered]

    text = re.sub(r"\s+", " ", note)
    # Protect abbreviations and decimals so they do not end a sentence.
    text = re.sub(rf"\b({_ABBREV})\.", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return [{"id": i, "text": s} for i, s in enumerate(sentences)]


# ---------------------------------------------------------------- criteria

_HEADER = re.compile(r"^\s*(inclusion|exclusion)(\s+criteria)?\s*:?\s*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s+", re.I)


# Accepts the ClinicalTrials.gov layout: an "Inclusion Criteria:" header, then bulleted or
# numbered items, then "Exclusion Criteria:". Lines that are not bullets are joined to the
# previous item when the section uses bullets.
def parse_criteria(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        raise InputError("No eligibility criteria were given.")
    section, items, uses_bullets = None, [], {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _HEADER.match(raw)
        if m:
            section = m.group(1).lower()
            continue
        if section is None:
            raise InputError("Start the criteria with an 'Inclusion criteria:' or 'Exclusion criteria:' line.")
        is_bullet = bool(_BULLET.match(raw))
        uses_bullets[section] = uses_bullets.get(section, False) or is_bullet
        line = _BULLET.sub("", raw).strip()
        if not is_bullet and uses_bullets[section] and items and items[-1]["type"] == section:
            items[-1]["text"] += " " + line
        else:
            items.append({"type": section, "text": line})
    if not items:
        raise InputError("The headers were found but no criteria were listed under them.")
    if len(items) > MAX_CRITERIA:
        raise InputError(f"{len(items)} criteria given; the limit is {MAX_CRITERIA} per run.")
    return [{"index": i, **c} for i, c in enumerate(items)]


# Public trial metadata, used only to fill the criteria box in the app.
def fetch_trial(nct_id: str, timeout: float = 15.0) -> dict:
    nct_id = (nct_id or "").strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise InputError("Trial IDs look like NCT01234567.")
    url = (f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
           "?fields=protocolSection.identificationModule,protocolSection.eligibilityModule")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}),
                                    timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise InputError(f"ClinicalTrials.gov returned {e.code} for {nct_id}.") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise InputError("Could not reach ClinicalTrials.gov. Paste the criteria instead.") from e
    proto = data.get("protocolSection", {})
    criteria = proto.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    if not criteria:
        raise InputError(f"{nct_id} has no eligibility criteria on ClinicalTrials.gov.")
    return {"nct_id": nct_id, "title": proto.get("identificationModule", {}).get("briefTitle", ""),
            "criteria": criteria.replace("\\>", ">").replace("\\<", "<").replace("\\*", "*")}


# ---------------------------------------------------------------- explanations and rules

def explain(label: str, ctype: str, evidence: list[dict], parse_ok: bool) -> str:
    if label == "invalid":
        return ("The model's answer could not be read as a valid label for this criterion, so no "
                "decision is reported." if parse_ok else
                "The model did not return a readable answer, so no decision is reported.")
    cited = "; ".join(f"[{e['id']}] {e['text']}" for e in evidence)
    if label == "not enough information":
        base = "The note does not state enough to decide this criterion."
        return base + (f" Related sentences: {cited}" if cited else " No sentence addresses it.")
    if label == "not applicable":
        return "The criterion does not apply to this patient" + (f", based on: {cited}" if cited else ".")
    verdict = {
        "included": "The patient appears to meet this inclusion criterion",
        "not included": "The patient appears not to meet this inclusion criterion",
        "excluded": "The patient appears to meet this exclusion criterion and would be excluded",
        "not excluded": "The patient does not appear to meet this exclusion criterion",
    }[label]
    if cited:
        return f"{verdict}, based on: {cited}"
    return (f"{verdict}, but the model cited no sentence. Treat this as unsupported and check "
            "the note manually.")


# Explicit aggregation rules for a preliminary patient-level outcome:
#   1. any inclusion criterion judged "not included" or exclusion judged "excluded"
#      -> "Likely ineligible", listing those criteria
#   2. otherwise any "not enough information" or unreadable answer -> "Needs review"
#   3. otherwise (every criterion met, not triggered, or not applicable) -> "Potentially eligible"
def aggregate(results: list[dict]) -> dict:
    blocking = [r for r in results if r["label"] in ("not included", "excluded")]
    missing = [r for r in results if r["label"] == "not enough information"]
    invalid = [r for r in results if r["label"] == "invalid"]
    unsupported = [r for r in results if r["label"] in ("included", "not excluded", "not included", "excluded")
                   and not r["evidence_ids"]]
    if blocking:
        outcome, rule = "Likely ineligible", "At least one criterion rules the patient out."
    elif missing or invalid:
        outcome, rule = "Needs review", "No criterion rules the patient out, but some could not be decided."
    else:
        outcome, rule = "Potentially eligible", "Every criterion is met, not triggered, or not applicable."
    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return {"outcome": outcome, "rule": rule, "counts": counts,
            "blocking": [r["index"] for r in blocking], "missing_information": [r["index"] for r in missing],
            "invalid": [r["index"] for r in invalid], "unsupported": [r["index"] for r in unsupported],
            "note": "Aggregated from criterion-level predictions by fixed rules; not a ground-truth label."}


# ---------------------------------------------------------------- engine

@dataclass
class EngineConfig:
    model: str = DEFAULT_MODEL
    adapter: str | None = None
    draft: str = DRAFT_MODEL
    draft_adapter: str | None = None


@dataclass
class ScreeningEngine:
    config: EngineConfig = field(default_factory=EngineConfig)

    def __post_init__(self):
        self._target = self._draft = self._tok = None

    def load(self):
        if self._target is None:
            model, self._tok = load_model(self.config.model, self.config.adapter)
            self._target = _warm(QwenDecoder(model, "sdpa"))
            self.eos = {self._tok.convert_tokens_to_ids("<|im_end|>"), self._tok.eos_token_id}
        return self

    def load_draft(self):
        if self._draft is None:
            model, _ = load_model(self.config.draft, self.config.draft_adapter)
            self._draft = _warm(QwenDecoder(model, "sdpa"))
        return self._draft

    @property
    def tok(self):
        return self.load()._tok

    @property
    def target(self):
        return self.load()._target

    def _ids(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.target.device)

    # Prompt for one criterion, split into the patient prefix shared by all criteria and the
    # criterion-specific suffix (checked to tokenize identically to the joined prompt).
    def _prompt_parts(self, sentences, criterion):
        full = self.tok.apply_chat_template(build_messages(sentences, criterion["type"], criterion["text"]),
                                            add_generation_prompt=True, tokenize=False)
        cut = full.index("Criterion type:")
        return full[:cut], full[cut:]

    def check_note(self, sentences) -> dict:
        n = len(self.tok("\n".join(s["text"] for s in sentences), add_special_tokens=False)["input_ids"])
        if n > MAX_NOTE_TOKENS:
            raise InputError(f"The note is {n} tokens long; the limit is {MAX_NOTE_TOKENS}. Shorten it.")
        warning = (f"The note is {n} tokens, longer than any note seen in training (max about "
                   f"{TRAINED_NOTE_TOKENS}). Results may be less reliable.") if n > TRAINED_NOTE_TOKENS else None
        return {"note_tokens": n, "warning": warning}

    # Yields one result per criterion so the app can show progress as it goes.
    def screen(self, sentences, criteria, mode: str = "kv_prefix", attention: str = "sdpa", k: int = 4):
        if mode not in DECODING_MODES:
            raise InputError(f"Unknown decoding mode {mode!r}.")
        target = self.target
        target.set_attention(attention)
        draft = self.load_draft() if mode == "speculative" else None
        if draft is not None:
            draft.set_attention(attention)

        prefix_text, _ = self._prompt_parts(sentences, criteria[0])
        prefix_ids = self._ids(prefix_text)
        P = prefix_ids.shape[1]
        cache, prefix_s = None, 0.0
        if mode == "kv_prefix":
            cache = target.new_cache(P + 160 + MAX_NEW_TOKENS)
            t = time.perf_counter()
            target.forward(prefix_ids, cache, start=0)
            prefix_s = time.perf_counter() - t

        valid_ids = {s["id"] for s in sentences}
        by_id = {s["id"]: s["text"] for s in sentences}
        for c in criteria:
            p_text, s_text = self._prompt_parts(sentences, c)
            if p_text != prefix_text:
                raise RuntimeError("prompt prefix changed between criteria")
            ids = torch.cat([prefix_ids, self._ids(s_text)], dim=1)
            if mode == "kv_prefix":
                if ids.shape[1] + MAX_NEW_TOKENS > cache.max_len:
                    raise InputError(f"Criterion {c['index'] + 1} is too long. Shorten or split it.")
                run = generate_cached(target, ids, MAX_NEW_TOKENS, self.eos, cache=cache, prefix_len=P)
            elif mode == "kv":
                run = generate_cached(target, ids, MAX_NEW_TOKENS, self.eos)
            elif mode == "speculative":
                run = generate_speculative(target, draft, ids, MAX_NEW_TOKENS, k=k, eos_ids=self.eos)
            else:
                run = generate_uncached(target, ids, MAX_NEW_TOKENS, self.eos)

            text = self.tok.decode(run["tokens"], skip_special_tokens=True)
            pred = parse_prediction(text, c["type"], valid_ids)
            evidence = [{"id": i, "text": by_id[i]} for i in pred["evidence_ids"]]
            yield {
                "index": c["index"], "type": c["type"], "criterion": c["text"],
                "label": pred["label"], "evidence_ids": pred["evidence_ids"], "evidence": evidence,
                "explanation": explain(pred["label"], c["type"], evidence, pred["parse_ok"]),
                "raw_output": text, "parse_ok": pred["parse_ok"], "dropped_ids": pred["dropped_ids"],
                "prompt_tokens": int(ids.shape[1]), "new_tokens": run["n_tokens"],
                "seconds": round(run["total_s"] + (prefix_s if c is criteria[0] else 0.0), 3),
                "acceptance_rate": run.get("acceptance_rate"),
                "shared_prefix_tokens": P if mode == "kv_prefix" else 0,
            }

    def provenance(self, mode: str, attention: str, k: int) -> dict:
        return {
            "app": f"TrialLens {__version__}",
            "git_commit": _git_commit(),
            "model": self.config.model,
            "adapter": self.config.adapter,
            "adapter_sha256": _file_hash(self.config.adapter),
            "draft": self.config.draft if mode == "speculative" else None,
            "decoding": {"strategy": "greedy", "mode": DECODING_MODES[mode], "attention": attention,
                         "max_new_tokens": MAX_NEW_TOKENS, "draft_k": k if mode == "speculative" else None},
            "device": str(self.target.device),
            "dtype": str(self.target.dtype),
        }


# Weights are memory-mapped and only paged in on first use, so one throwaway forward right
# after loading keeps that one-off cost out of the first timed request.
def _warm(dec: QwenDecoder) -> QwenDecoder:
    cache = dec.new_cache(8)
    dec.forward(torch.zeros(1, 4, dtype=torch.long, device=dec.device), cache, start=0)
    return dec


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _file_hash(adapter: str | None) -> str | None:
    if not adapter:
        return None
    f = Path(adapter) / "adapter_model.safetensors"
    if not f.exists():
        return None
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16]


def build_report(note: str, sentences, criteria_text: str, results, provenance: dict, trial: dict | None = None) -> dict:
    return {
        "disclaimer": DISCLAIMER,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trial": trial,
        "patient_note_sha256": hashlib.sha256(note.encode("utf-8")).hexdigest(),
        "sentences": sentences,
        "criteria_text": criteria_text,
        "results": results,
        "summary": aggregate(results),
        "provenance": provenance,
    }
