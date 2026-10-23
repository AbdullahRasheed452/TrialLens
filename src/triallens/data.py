"""Loading, splitting and preprocessing the TrialGPT criterion annotations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
SEED = 42
SPLIT_SIZES = {"train": 37, "val": 8, "test": 8}

# Only these columns ever reach the model or the evaluator. The gpt4_* columns,
# explanation_correctness and the dataset's own `training` flag are dropped on purpose.
ALLOWED_COLUMNS = [
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "trial_title",
    "criterion_type",
    "criterion_text",
    "expert_eligibility",
    "expert_sentences",
]
FORBIDDEN_PREFIXES = ("gpt4_",)

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = [
    "included",
    "not included",
    "excluded",
    "not excluded",
    "not enough information",
    "not applicable",
]

_SENT_RE = re.compile(r"^(\d+)\.\s(.*)$")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    if path is not None and Path(path).exists():
        return pd.read_parquet(path)
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split="train").to_pandas()


def make_patient_split(patient_ids) -> dict[str, list[str]]:
    """Sort unique IDs, shuffle with default_rng(42), then take 37/8/8."""
    ids = np.array(sorted(set(patient_ids)))
    if len(ids) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"expected 53 patients, found {len(ids)}")
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    n_train, n_val = SPLIT_SIZES["train"], SPLIT_SIZES["val"]
    return {
        "train": ids[:n_train].tolist(),
        "val": ids[n_train : n_train + n_val].tolist(),
        "test": ids[n_train + n_val :].tolist(),
    }


def parse_note(note: str) -> list[dict]:
    """Split a numbered note ("0. ...\\n1. ...") into [{id, text}], keeping original IDs."""
    sentences = []
    for line in note.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _SENT_RE.match(line)
        if m is None:
            raise ValueError(f"unnumbered sentence in note: {line[:80]!r}")
        sentences.append({"id": int(m.group(1)), "text": m.group(2).strip()})
    return sentences


def to_record(row: pd.Series) -> dict:
    sentences = parse_note(row["note"])
    valid_ids = {s["id"] for s in sentences}
    evidence = sorted(set(json.loads(row["expert_sentences"])))
    if not set(evidence) <= valid_ids:
        raise ValueError(f"annotation {row['annotation_id']} cites missing sentence IDs")
    if row["expert_eligibility"] not in LABELS[row["criterion_type"]]:
        raise ValueError(f"annotation {row['annotation_id']} has label outside its criterion type")
    return {
        "annotation_id": int(row["annotation_id"]),
        "patient_id": row["patient_id"],
        "trial_id": row["trial_id"],
        "trial_title": row["trial_title"],
        "criterion_type": row["criterion_type"],
        "criterion_text": row["criterion_text"].strip(),
        "sentences": sentences,
        "label": row["expert_eligibility"],
        "evidence_ids": evidence,
    }


def find_unusable(df: pd.DataFrame) -> list[dict]:
    """Rows that cannot be posed as a criterion question (e.g. empty criterion text)."""
    text = df["criterion_text"].fillna("").str.strip()
    return [
        {"annotation_id": int(a), "patient_id": p, "reason": "empty criterion_text"}
        for a, p in zip(df.loc[text == "", "annotation_id"], df.loc[text == "", "patient_id"])
    ]


def build_splits(df: pd.DataFrame) -> tuple[dict[str, list[dict]], dict]:
    df = df[ALLOWED_COLUMNS]
    # The split is computed over all 53 patients before any row is dropped.
    split_ids = make_patient_split(df["patient_id"])
    dropped = find_unusable(df)
    df = df[~df["annotation_id"].isin([d["annotation_id"] for d in dropped])]
    records = {}
    for name, ids in split_ids.items():
        part = df[df["patient_id"].isin(ids)].sort_values("annotation_id")
        records[name] = [to_record(r) for _, r in part.iterrows()]

    manifest = {
        "dataset": DATASET_ID,
        "seed": SEED,
        "procedure": "sorted unique patient_id -> numpy default_rng(42).shuffle -> first 37 train, next 8 val, last 8 test",
        "dropped_annotations": dropped,
        "splits": {},
    }
    for name, recs in records.items():
        labels = pd.Series([r["label"] for r in recs])
        ctype = pd.Series([r["criterion_type"] for r in recs])
        manifest["splits"][name] = {
            "patient_ids": split_ids[name],
            "n_patients": len(split_ids[name]),
            "n_annotations": len(recs),
            "n_inclusion": int((ctype == "inclusion").sum()),
            "n_exclusion": int((ctype == "exclusion").sum()),
            "n_empty_evidence": sum(not r["evidence_ids"] for r in recs),
            "label_counts": {k: int(v) for k, v in labels.value_counts().items()},
            "annotation_ids": [r["annotation_id"] for r in recs],
        }
    return records, manifest


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
