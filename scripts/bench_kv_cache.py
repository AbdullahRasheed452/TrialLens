# Task 3 benchmark: KV-cached vs uncached greedy decoding on the real model.
#
# Usage:
#   python scripts/bench_kv_cache.py                                   # GPU defaults
#   python scripts/bench_kv_cache.py --inputs 128 256 --outputs 16 32 --repeats 2   # CPU sized
#
# For every (input length, output length) pair both paths decode the same prompt with
# EOS ignored, so they produce exactly the requested number of tokens. Each pair gets
# warm-up runs that are not timed, then repeated timed runs with device synchronisation
# around every timer. Tokens from the two paths are compared position by position; on a
# mismatch the first differing step and the top-1/top-2 logit gap there are logged.

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from triallens.data import read_jsonl  # noqa: E402
from triallens.decoder import QwenDecoder, generate_cached, generate_uncached  # noqa: E402
from triallens.inference import DEFAULT_MODEL, load_model, prompt_ids  # noqa: E402
from triallens.profiling import PeakMemory  # noqa: E402


# Real prompts from the validation split, concatenated and cut to an exact token length.
def make_prompt(tok, n_tokens: int, device) -> torch.Tensor:
    pool = []
    for r in read_jsonl(ROOT / "data/processed/val.jsonl"):
        pool.extend(prompt_ids(tok, r["sentences"], r["criterion_type"], r["criterion_text"], "cpu")[0].tolist())
        if len(pool) >= n_tokens:
            break
    return torch.tensor([pool[:n_tokens]], device=device)


def timed(fn, device, repeats: int) -> dict:
    runs, peaks = [], []
    for _ in range(repeats):
        with PeakMemory(device) as mem:
            out = fn()
        runs.append(out)
        peaks.append(mem.peak_bytes)
    med = lambda key: statistics.median(r[key] for r in runs)
    return {
        "tokens": runs[0]["tokens"],
        "margins": runs[0]["margins"],
        "prefill_s": med("prefill_s"),
        "decode_s": med("decode_s"),
        "total_s": med("total_s"),
        "decode_tok_per_s": med("decode_tok_per_s"),
        "overall_tok_per_s": med("overall_tok_per_s"),
        "total_s_runs": [round(r["total_s"], 4) for r in runs],
        "peak_mem_mb": max(peaks) / 2**20,
        "kv_cache_mb": runs[0]["kv_cache_bytes"] / 2**20,
        "same_tokens_every_repeat": all(r["tokens"] == runs[0]["tokens"] for r in runs),
    }


def compare(a: list[int], b: list[int], margins: list[float]) -> dict:
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return {
        "match": a == b,
        "agreement": sum(x == y for x, y in zip(a, b)) / max(len(a), 1),
        "first_mismatch": first,
        "gap_at_mismatch": None if first is None else margins[first],
        "min_gap_overall": min(margins),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--attention", default="sdpa", choices=["sdpa", "reference", "tiled"])
    ap.add_argument("--inputs", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--outputs", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = load_model(args.model, args.adapter)
    dec = QwenDecoder(model, args.attention)
    device = dec.device
    print(f"device={device} dtype={dec.dtype} attention={args.attention}")

    rows = []
    for n_in in args.inputs:
        ids = make_prompt(tok, n_in, device)
        for n_out in args.outputs:
            cached_fn = lambda: generate_cached(dec, ids, n_out)
            uncached_fn = lambda: generate_uncached(dec, ids, n_out)
            for _ in range(args.warmup):
                cached_fn()
                uncached_fn()
            c = timed(cached_fn, device, args.repeats)
            u = timed(uncached_fn, device, args.repeats)
            check = compare(c["tokens"], u["tokens"], c["margins"])
            row = {"input_tokens": n_in, "output_tokens": n_out, "cached": c, "uncached": u,
                   "tokens_check": check, "speedup_total": u["total_s"] / c["total_s"]}
            rows.append(row)
            print(f"in={n_in:5d} out={n_out:4d} | prefill {c['prefill_s']*1e3:7.1f} vs {u['prefill_s']*1e3:7.1f} ms"
                  f" | decode {c['decode_tok_per_s']:7.1f} vs {u['decode_tok_per_s']:7.1f} tok/s"
                  f" | peak {c['peak_mem_mb']:7.1f} vs {u['peak_mem_mb']:7.1f} MB"
                  f" | speedup {row['speedup_total']:5.2f}x | tokens match={check['match']}"
                  + ("" if check["match"] else f" first diff at {check['first_mismatch']}"
                                               f" (gap {check['gap_at_mismatch']:.4f})"))

    for row in rows:
        for path in ("cached", "uncached"):
            row[path].pop("margins")
    report = {
        "config": {"model": args.model, "adapter": args.adapter, "attention": args.attention,
                   "device": str(device), "dtype": str(dec.dtype), "warmup": args.warmup,
                   "repeats": args.repeats, "timing": "median of repeats, device synchronised",
                   "memory": PeakMemory(device).method,
                   "gpu": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                   "torch": torch.__version__,
                   "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "rows": rows,
    }
    out = Path(args.out) if args.out else ROOT / f"results/bench/kv_cache_{device.type}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
