# Turns raw model text into a validated prediction.
#
# The model is asked for {"evidence_ids": [...], "label": "..."} but a base model often adds
# prose or markdown fences around it. We take the first balanced {...} block, parse it, and
# validate it against the criterion type and the note's real sentence IDs. Anything that
# cannot be trusted is marked instead of silently fixed.

from __future__ import annotations

import json
import re

from .data import LABELS
from .metrics import INVALID


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _normalise_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower().replace("_", " "))


def parse_prediction(text: str, criterion_type: str, valid_ids: set[int]) -> dict:
    result = {
        "label": INVALID,
        "evidence_ids": [],
        "parse_ok": False,
        "label_ok": False,
        "dropped_ids": [],
        "raw": text,
    }
    blob = _first_json_object(text)
    if blob is None:
        return result
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return result
    if not isinstance(obj, dict):
        return result
    result["parse_ok"] = True

    label = _normalise_label(obj.get("label", ""))
    if label in LABELS[criterion_type]:
        result["label"] = label
        result["label_ok"] = True

    ids = obj.get("evidence_ids", [])
    if not isinstance(ids, list):
        ids = [ids]
    kept, dropped = [], []
    for x in ids:
        try:
            i = int(x)
        except (TypeError, ValueError):
            dropped.append(x)
            continue
        (kept if i in valid_ids else dropped).append(i)
    result["evidence_ids"] = sorted(set(kept))
    result["dropped_ids"] = dropped
    return result
