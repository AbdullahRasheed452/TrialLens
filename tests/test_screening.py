# Input handling and the aggregation rules of the screening engine (no model needed).

import pytest

from triallens.screening import InputError, aggregate, explain, parse_criteria, split_sentences


def test_numbered_note_keeps_ids():
    s = split_sentences("0. A 58-year-old woman.\n1. She has chest pain.\n3. No diabetes.")
    assert [x["id"] for x in s] == [0, 1, 3]
    assert s[2]["text"] == "No diabetes."


def test_free_text_note_is_split_and_numbered():
    s = split_sentences("Pt is a 45 yo male seen by Dr. Smith. BMI 31.5 today! History of e.g. asthma. "
                        "No smoking.")
    assert [x["text"] for x in s] == ["Pt is a 45 yo male seen by Dr. Smith.", "BMI 31.5 today!",
                                      "History of e.g. asthma.", "No smoking."]
    assert [x["id"] for x in s] == [0, 1, 2, 3]


@pytest.mark.parametrize("note", ["", "   \n  "])
def test_empty_note_rejected(note):
    with pytest.raises(InputError):
        split_sentences(note)


def test_duplicate_numbers_rejected():
    with pytest.raises(InputError):
        split_sentences("1. One.\n1. Two.")


def test_clinicaltrials_layout():
    text = """Inclusion Criteria:

* Age 18 years or older
* Confirmed type 2 diabetes
  with HbA1c above 7%

Exclusion Criteria:

* Pregnancy
* eGFR < 30"""
    c = parse_criteria(text)
    assert [(x["type"], x["text"]) for x in c] == [
        ("inclusion", "Age 18 years or older"),
        ("inclusion", "Confirmed type 2 diabetes with HbA1c above 7%"),
        ("exclusion", "Pregnancy"),
        ("exclusion", "eGFR < 30"),
    ]
    assert [x["index"] for x in c] == [0, 1, 2, 3]


def test_plain_lines_without_bullets():
    c = parse_criteria("inclusion\nAdults\nChest pain\nexclusion:\nPrior MI")
    assert [x["text"] for x in c] == ["Adults", "Chest pain", "Prior MI"]


@pytest.mark.parametrize("text", ["", "Adults over 18", "Inclusion criteria:\n\nExclusion criteria:"])
def test_bad_criteria_rejected(text):
    with pytest.raises(InputError):
        parse_criteria(text)


def r(i, label, ev=(0,)):
    return {"index": i, "label": label, "evidence_ids": list(ev)}


def test_any_blocking_criterion_makes_ineligible():
    out = aggregate([r(0, "included"), r(1, "excluded"), r(2, "not enough information")])
    assert out["outcome"] == "Likely ineligible" and out["blocking"] == [1]


def test_missing_information_needs_review():
    out = aggregate([r(0, "included"), r(1, "not enough information", ()), r(2, "not excluded")])
    assert out["outcome"] == "Needs review" and out["missing_information"] == [1]


def test_invalid_answer_needs_review():
    assert aggregate([r(0, "included"), r(1, "invalid", ())])["outcome"] == "Needs review"


def test_all_clear_is_potentially_eligible():
    out = aggregate([r(0, "included"), r(1, "not excluded", ()), r(2, "not applicable", ())])
    assert out["outcome"] == "Potentially eligible"
    assert out["unsupported"] == [1]


def test_explanation_flags_missing_evidence():
    assert "cited no sentence" in explain("included", "inclusion", [], True)
    assert "No sentence addresses it" in explain("not enough information", "inclusion", [], True)
    assert "[2] Has asthma." in explain("excluded", "exclusion", [{"id": 2, "text": "Has asthma."}], True)
