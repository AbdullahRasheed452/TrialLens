# Criterion classification metrics and evidence sentence retrieval metrics.
#
# Classification
#   Predictions that fail to parse, or name a label outside the criterion type's label set,
#   become INVALID. INVALID is always counted as wrong and gets its own confusion matrix
#   column, so a model cannot look better by producing garbage.
#   Macro-F1 averages over the gold label set of the scope (overall: all six labels that
#   occur in gold; per type: that type's four labels that occur in gold). A class with no
#   gold examples has undefined recall, so it is left out of the macro average and reported
#   as such in the per-class table.
#
# Evidence
#   micro      TP, FP, FN summed over every record. Records with empty gold only add FP.
#   macro      per-record P/R/F1 averaged over records whose gold set is non-empty. An empty
#              prediction on such a record scores P = R = F1 = 0.
#   empty-gold records are scored separately as "abstention accuracy": the share of
#              records with no gold evidence where the model also cited nothing.

from __future__ import annotations

from collections import Counter

from .data import ALL_LABELS, LABELS

INVALID = "invalid"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


def confusion_matrix(gold: list[str], pred: list[str], labels: list[str]) -> list[list[int]]:
    cols = labels + [INVALID]
    index = {l: i for i, l in enumerate(cols)}
    m = [[0] * len(cols) for _ in labels]
    for g, p in zip(gold, pred):
        m[labels.index(g)][index.get(p, index[INVALID])] += 1
    return m


def classification_report(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    present = [l for l in labels if l in set(gold)]
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1

    per_class = {}
    for l in labels:
        p = _safe_div(tp[l], tp[l] + fp[l])
        r = _safe_div(tp[l], tp[l] + fn[l])
        per_class[l] = {
            "precision": p,
            "recall": r if l in present else None,
            "f1": _f1(p, r) if l in present else None,
            "support": tp[l] + fn[l],
            "predicted": tp[l] + fp[l],
        }

    n = len(gold)
    return {
        "n": n,
        "accuracy": _safe_div(sum(tp.values()), n),
        "macro_f1": _safe_div(sum(per_class[l]["f1"] for l in present), len(present)),
        "macro_f1_labels": present,
        "invalid_rate": _safe_div(sum(p == INVALID for p in pred), n),
        "per_class": per_class,
        "confusion_matrix": {
            "rows_gold": labels,
            "cols_pred": labels + [INVALID],
            "matrix": confusion_matrix(gold, pred, labels),
        },
    }


def evidence_report(gold_sets: list[list[int]], pred_sets: list[list[int]]) -> dict:
    tp = fp = fn = 0
    macro_p, macro_r, macro_f = [], [], []
    empty_gold = empty_agree = 0
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        hit = len(g & p)
        tp, fp, fn = tp + hit, fp + len(p - g), fn + len(g - p)
        if g:
            pr, rc = _safe_div(hit, len(p)), _safe_div(hit, len(g))
            macro_p.append(pr)
            macro_r.append(rc)
            macro_f.append(_f1(pr, rc))
        else:
            empty_gold += 1
            empty_agree += not p

    micro_p, micro_r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    mean = lambda xs: _safe_div(sum(xs), len(xs))
    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": _f1(micro_p, micro_r),
                  "tp": tp, "fp": fp, "fn": fn},
        "macro": {"precision": mean(macro_p), "recall": mean(macro_r), "f1": mean(macro_f),
                  "n_records": len(macro_f)},
        "empty_gold": {"n_records": empty_gold,
                       "abstention_accuracy": empty_agree / empty_gold if empty_gold else None},
    }


# Full report: overall plus separate inclusion and exclusion breakdowns.
def evaluate(records: list[dict], predictions: list[dict]) -> dict:
    by_id = {p["annotation_id"]: p for p in predictions}
    rows = [(r, by_id[r["annotation_id"]]) for r in records]

    def scope(items, labels):
        gold = [r["label"] for r, _ in items]
        pred = [p["label"] for _, p in items]
        return {
            "classification": classification_report(gold, pred, labels),
            "evidence": evidence_report([r["evidence_ids"] for r, _ in items],
                                        [p["evidence_ids"] for _, p in items]),
        }

    out = {"overall": scope(rows, ALL_LABELS)}
    for ctype, labels in LABELS.items():
        out[ctype] = scope([(r, p) for r, p in rows if r["criterion_type"] == ctype], labels)
    return out
