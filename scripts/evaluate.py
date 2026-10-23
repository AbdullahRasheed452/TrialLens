# Run a model over a split and save predictions plus metrics.
#
# Usage:
#   python scripts/evaluate.py --split val --name base
#   python scripts/evaluate.py --split val --name lora --adapter checkpoints/lora-best
#   python scripts/evaluate.py --split test --name lora --adapter ... --final
#
# The test split is locked behind --final so it is only touched once, at final evaluation.

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from triallens.data import read_jsonl  # noqa: E402
from triallens.inference import DECODING, DEFAULT_MODEL, load_model, predict  # noqa: E402
from triallens.metrics import evaluate  # noqa: E402


def summary_line(rep: dict) -> str:
    c, e = rep["classification"], rep["evidence"]
    abstain = e["empty_gold"]["abstention_accuracy"]
    return (f"acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f} invalid={c['invalid_rate']:.3f} "
            f"| evidence microF1={e['micro']['f1']:.3f} macroF1={e['macro']['f1']:.3f} "
            f"abstain={'n/a' if abstain is None else f'{abstain:.3f}'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--name", required=True, help="run name used in output file names")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N records (smoke tests)")
    ap.add_argument("--max-new-tokens", type=int, default=DECODING["max_new_tokens"])
    ap.add_argument("--final", action="store_true", help="required to evaluate on the test split")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        sys.exit("The test split is held out. Pass --final only for the final evaluation.")

    records = read_jsonl(ROOT / f"data/processed/{args.split}.jsonl")
    if args.limit:
        records = records[: args.limit]

    model, tok = load_model(args.model, args.adapter)
    device = str(next(model.parameters()).device)
    print(f"{len(records)} records on {device}")

    out_dir = ROOT / "results/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.name}_{args.split}" + (f"_n{args.limit}" if args.limit else "")
    preds = []
    start = time.perf_counter()
    with (out_dir / f"{tag}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            p = predict(model, tok, r, args.max_new_tokens)
            preds.append(p)
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(records):
                print(f"  {i}/{len(records)}  {time.perf_counter() - start:.0f}s")

    report = evaluate(records, preds)
    report["config"] = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "n_records": len(records),
        "decoding": {**DECODING, "max_new_tokens": args.max_new_tokens},
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "total_seconds": round(time.perf_counter() - start, 1),
        "mean_seconds_per_record": round(sum(p["seconds"] for p in preds) / len(preds), 3),
        "parse_failure_rate": sum(not p["parse_ok"] for p in preds) / len(preds),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for scope in ("overall", "inclusion", "exclusion"):
        print(f"{scope:9s} {summary_line(report[scope])}")
    print(f"saved results/eval/{tag}_*")


if __name__ == "__main__":
    main()
