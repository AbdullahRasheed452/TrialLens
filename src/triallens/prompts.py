"""Prompt construction and output schema.

The patient note comes before the criterion so that every criterion for the same
patient shares one prompt prefix, which lets the app reuse that prefix's KV cache.
"""

from __future__ import annotations

import json

from .data import LABELS

SYSTEM_PROMPT = (
    "You are a clinical trial screening assistant. Judge one eligibility criterion against "
    "a patient note. Use only facts stated in the note. If the note does not mention what "
    "the criterion needs, answer \"not enough information\"; missing facts are never proof "
    "of eligibility or ineligibility. Answer \"not applicable\" when the criterion cannot "
    "apply to this patient (for example, a pregnancy criterion for a male patient). "
    "Reply with a single JSON object and nothing else."
)


def format_note(sentences: list[dict]) -> str:
    return "\n".join(f"{s['id']}. {s['text']}" for s in sentences)


def patient_block(sentences: list[dict]) -> str:
    return f"Patient note (sentences are numbered):\n{format_note(sentences)}\n\n"


def criterion_block(criterion_type: str, criterion_text: str) -> str:
    labels = ", ".join(f'"{l}"' for l in LABELS[criterion_type])
    return (
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        f"Allowed labels: {labels}\n"
        'Return {"evidence_ids": [sentence numbers that support your decision], "label": one allowed label}.'
    )


def build_messages(sentences: list[dict], criterion_type: str, criterion_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_block(sentences) + criterion_block(criterion_type, criterion_text)},
    ]


def target_json(record: dict) -> str:
    """Training target. Evidence comes before the label so the label is conditioned on it."""
    return json.dumps({"evidence_ids": record["evidence_ids"], "label": record["label"]})
