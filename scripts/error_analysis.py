# Group the selected model's held-out test errors into categories and dump each case with
# the full note, gold and predicted labels and evidence, for the written error analysis.
#
# Usage: python scripts/error_analysis.py [--run os40] [--split test]
#
# Categories
#   missing_info_as_negative   gold "not enough information" predicted as a definite answer:
#                              the model treated absent evidence as proof
#   missed_negative            gold "not included"/"excluded" predicted as anything else
#   over_abstained             gold definite answer predicted "not enough information"
#   applicability              "not applicable" confused with another label
#   other_label                any other wrong label
#   unsupported_correct        right label, the experts cited evidence, the model cited nothing
#   evidence_miss              right label, the experts cited evidence, the model cited only
#                              other sentences
# A right label with no evidence on either side is not counted: experts often marked
# "not excluded" without citing a sentence, so the model is following the annotation.

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triallens.data import read_jsonl  # noqa: E402

DEFINITE = {"included", "not included", "excluded", "not excluded"}


def categorise(gold: dict, pred: dict) -> str | None:
    g, p = gold["label"], pred["label"]
    if g != p:
        if g == "not enough information" and p in DEFINITE:
            return "missing_info_as_negative" if p in ("not excluded", "not included") else "missing_info_as_positive"
        if g in ("not included", "excluded"):
            return "missed_negative"
        if p == "not enough information":
            return "over_abstained"
        if "not applicable" in (g, p):
            return "applicability"
        return "other_label"
    if gold["evidence_ids"] and not set(gold["evidence_ids"]) & set(pred["evidence_ids"]):
        return "unsupported_correct" if not pred["evidence_ids"] else "evidence_miss"
    return None


# Annotation conventions on the train and validation splits that explain recurring test errors.
def conventions() -> dict:
    recs = read_jsonl(ROOT / "data/processed/train.jsonl") + read_jsonl(ROOT / "data/processed/val.jsonl")
    no_ev = [r for r in recs if r["criterion_type"] == "exclusion" and not r["evidence_ids"]]
    male = re.compile(r"\b(man|male|boy|gentleman)\b", re.I)
    female = re.compile(r"\b(woman|female|girl|she)\b", re.I)
    preg = [r for r in recs if re.search(r"pregnan|lactat|breast.?feed|child.?bearing", r["criterion_text"], re.I)
            and male.search(r["sentences"][0]["text"]) and not female.search(r["sentences"][0]["text"])]
    return {
        "exclusion_without_gold_evidence": {"n": len(no_ev), "labels": dict(Counter(r["label"] for r in no_ev))},
        "pregnancy_criteria_for_male_patients": {"n": len(preg), "labels": dict(Counter(r["label"] for r in preg))},
        "scope": "train + validation splits",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="os40")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    records = {r["annotation_id"]: r for r in read_jsonl(ROOT / f"data/processed/{args.split}.jsonl")}
    preds = read_jsonl(ROOT / f"results/eval/{args.run}_{args.split}_predictions.jsonl")
    cases = []
    for p in preds:
        r = records[p["annotation_id"]]
        cat = categorise(r, p)
        if cat is None:
            continue
        by_id = {s["id"]: s["text"] for s in r["sentences"]}
        cases.append({
            "category": cat, "annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
            "trial_id": r["trial_id"], "criterion_type": r["criterion_type"], "criterion": r["criterion_text"],
            "gold_label": r["label"], "pred_label": p["label"],
            "gold_evidence": {i: by_id[i] for i in r["evidence_ids"]},
            "pred_evidence": {i: by_id[i] for i in p["evidence_ids"]},
            "raw_output": p["raw"], "note": [f"{s['id']}. {s['text']}" for s in r["sentences"]],
        })

    counts = Counter(c["category"] for c in cases)
    conv = conventions()
    out_dir = ROOT / "results/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.run}_{args.split}_error_cases.json").write_text(
        json.dumps({"counts": dict(counts), "n_records": len(preds), "annotation_conventions": conv, "cases": cases},
                   indent=2), encoding="utf-8")
    print("annotation conventions:", json.dumps(conv))
    print(f"{len(preds)} records, {sum(c['category'] not in ('unsupported_correct', 'evidence_miss') for c in cases)} "
          f"label errors, {counts['unsupported_correct'] + counts['evidence_miss']} evidence-only issues")
    for k, v in counts.most_common():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
