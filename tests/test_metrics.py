# Metric and parser checks against values worked out by hand.
# (scikit-learn is not used as a reference because Windows Smart App Control blocks its DLLs.)

import pytest

from triallens.metrics import INVALID, classification_report, evidence_report, evaluate
from triallens.parsing import parse_prediction

LABELS = ["included", "not included", "not enough information", "not applicable"]


def test_classification_hand_computed():
    gold = ["included", "included", "not included", "not enough information", "not applicable", "included"]
    pred = ["included", "not included", "not included", "included", "not applicable", INVALID]
    rep = classification_report(gold, pred, LABELS)
    # included: TP 1, FP 1, FN 2 | not included: TP 1, FP 1, FN 0
    # not enough information: TP 0, FN 1 | not applicable: TP 1
    expected = {
        "included": (1 / 2, 1 / 3, 0.4),
        "not included": (1 / 2, 1.0, 2 / 3),
        "not enough information": (0.0, 0.0, 0.0),
        "not applicable": (1.0, 1.0, 1.0),
    }
    for l, (p, r, f) in expected.items():
        assert rep["per_class"][l]["precision"] == pytest.approx(p)
        assert rep["per_class"][l]["recall"] == pytest.approx(r)
        assert rep["per_class"][l]["f1"] == pytest.approx(f)
    assert rep["accuracy"] == pytest.approx(0.5)
    assert rep["macro_f1"] == pytest.approx((0.4 + 2 / 3 + 0.0 + 1.0) / 4)
    assert rep["invalid_rate"] == pytest.approx(1 / 6)
    assert rep["confusion_matrix"]["matrix"][0] == [1, 1, 0, 0, 1]


def test_absent_gold_class_is_left_out_of_macro():
    gold = ["included", "not included"]
    pred = ["included", "not applicable"]
    rep = classification_report(gold, pred, LABELS)
    assert rep["macro_f1_labels"] == ["included", "not included"]
    assert rep["per_class"]["not applicable"]["f1"] is None
    assert rep["macro_f1"] == pytest.approx(0.5)


def test_evidence_empty_set_rules():
    gold = [[1, 2], [], [], [3]]
    pred = [[1], [], [4], []]
    rep = evidence_report(gold, pred)
    assert (rep["micro"]["tp"], rep["micro"]["fp"], rep["micro"]["fn"]) == (1, 1, 2)
    assert rep["macro"]["n_records"] == 2
    assert rep["macro"]["precision"] == pytest.approx((1.0 + 0.0) / 2)
    assert rep["macro"]["recall"] == pytest.approx((0.5 + 0.0) / 2)
    assert rep["empty_gold"] == {"n_records": 2, "abstention_accuracy": 0.5}
    assert evidence_report([[1]], [[1]])["empty_gold"]["abstention_accuracy"] is None


def test_evaluate_splits_by_type():
    records = [
        {"annotation_id": 1, "criterion_type": "inclusion", "label": "included", "evidence_ids": [0]},
        {"annotation_id": 2, "criterion_type": "exclusion", "label": "not excluded", "evidence_ids": []},
    ]
    preds = [
        {"annotation_id": 1, "label": "included", "evidence_ids": [0]},
        {"annotation_id": 2, "label": "excluded", "evidence_ids": []},
    ]
    rep = evaluate(records, preds)
    assert rep["overall"]["classification"]["accuracy"] == 0.5
    assert rep["inclusion"]["classification"]["accuracy"] == 1.0
    assert rep["exclusion"]["classification"]["accuracy"] == 0.0


@pytest.mark.parametrize("text,label,ids", [
    ('{"evidence_ids": [0, 2], "label": "included"}', "included", [0, 2]),
    ('Sure!\n```json\n{"evidence_ids": [2], "label": "Not Included"}\n```', "not included", [2]),
    ('{"evidence_ids": [1, 99, "x"], "label": "included"}', "included", [1]),
    ('{"evidence_ids": [], "label": "excluded"}', INVALID, []),
    ("I think the patient is included.", INVALID, []),
    ('{"evidence_ids": [1], "label": ', INVALID, []),
])
def test_parser(text, label, ids):
    out = parse_prediction(text, "inclusion", valid_ids={0, 1, 2})
    assert out["label"] == label
    assert out["evidence_ids"] == ids
