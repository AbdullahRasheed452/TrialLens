# Measure prompt and target token lengths to decide the max sequence length and check truncation.
#
# Usage: python scripts/analyze_lengths.py [--model Qwen/Qwen2.5-1.5B-Instruct] [--max-len 1024]

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from triallens.data import read_jsonl  # noqa: E402
from triallens.prompts import build_messages, format_note, target_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    report = {"model": args.model, "max_len": args.max_len, "splits": {}}
    for split in ("train", "val", "test"):
        recs = read_jsonl(ROOT / f"data/processed/{split}.jsonl")
        prompt_len, target_len, note_len = [], [], []
        for r in recs:
            msgs = build_messages(r["sentences"], r["criterion_type"], r["criterion_text"])
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompt_len.append(len(tok(text, add_special_tokens=False)["input_ids"]))
            target_len.append(len(tok(target_json(r) + tok.eos_token)["input_ids"]))
            note_len.append(len(tok(format_note(r["sentences"]))["input_ids"]))
        total = np.array(prompt_len) + np.array(target_len)
        stats = lambda a: {"min": int(np.min(a)), "mean": round(float(np.mean(a)), 1),
                           "p95": int(np.percentile(a, 95)), "max": int(np.max(a))}
        report["splits"][split] = {
            "prompt_tokens": stats(prompt_len),
            "target_tokens": stats(target_len),
            "note_tokens": stats(note_len),
            "total_tokens": stats(total),
            "n_over_max_len": int((total > args.max_len).sum()),
        }
        print(split, json.dumps(report["splits"][split]))

    out = ROOT / "results/data/token_lengths.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
