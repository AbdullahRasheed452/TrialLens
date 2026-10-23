"""Checks that the split follows the assignment protocol and nothing leaks between parts."""

import json
from pathlib import Path

import numpy as np
import pytest

from triallens.data import ALLOWED_COLUMNS, build_splits, load_raw, make_patient_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/criterion_annotations.parquet"


@pytest.fixture(scope="module")
def built():
    df = load_raw(RAW)
    return df, *build_splits(df)


def test_sizes_are_37_8_8(built):
    _, _, manifest = built
    assert [manifest["splits"][k]["n_patients"] for k in ("train", "val", "test")] == [37, 8, 8]


def test_no_patient_leakage(built):
    _, records, _ = built
    owners = {}
    for name, recs in records.items():
        for r in recs:
            assert owners.setdefault(r["patient_id"], name) == name


def test_every_annotation_used_exactly_once(built):
    df, records, manifest = built
    ids = [r["annotation_id"] for recs in records.values() for r in recs]
    dropped = {d["annotation_id"] for d in manifest["dropped_annotations"]}
    assert len(ids) == len(set(ids))
    assert set(ids) | dropped == set(df["annotation_id"])
    assert not set(ids) & dropped


def test_split_matches_independent_permutation(built):
    # Recompute with rng.permutation to show the result does not depend on shuffle vs permutation.
    df, _, manifest = built
    ids = np.array(sorted(df["patient_id"].unique()))
    perm = np.random.default_rng(42).permutation(ids).tolist()
    assert manifest["splits"]["train"]["patient_ids"] == perm[:37]
    assert manifest["splits"]["val"]["patient_ids"] == perm[37:45]
    assert manifest["splits"]["test"]["patient_ids"] == perm[45:]


def test_split_is_deterministic(built):
    df, _, manifest = built
    again = make_patient_split(df["patient_id"])
    assert again["test"] == manifest["splits"]["test"]["patient_ids"]


def test_no_gpt4_fields_in_records(built):
    _, records, _ = built
    for recs in records.values():
        for r in recs:
            blob = json.dumps(r)
            assert "gpt4" not in blob and "explanation_correctness" not in blob
    assert not any(c.startswith("gpt4_") for c in ALLOWED_COLUMNS)


def test_saved_manifest_matches(built):
    _, _, manifest = built
    saved = ROOT / "data/splits/split_manifest.json"
    if not saved.exists():
        pytest.skip("run scripts/make_split.py first")
    assert json.loads(saved.read_text(encoding="utf-8")) == manifest
