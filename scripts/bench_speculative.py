# Task 5 benchmark: greedy speculative decoding (Qwen2.5-0.5B draft, Qwen2.5-1.5B target)
# against ordinary KV-cached greedy decoding of the target alone.
#
# Usage:
#   python scripts/bench_speculative.py --adapter checkpoints/r16/lora-best
#   python scripts/bench_speculative.py --n-prompts 3 --ks 2 4 --summary-tokens 48   # CPU sized
#
# Two workloads use the same prompts for both decoders:
#   criterion  the real task: a criterion prompt and the short JSON answer, stopping at EOS
#   summary    a free-text summary of the patient note, the longer generation the app also
#              produces, where drafting has more room to pay off
# Every run checks that the speculative tokens equal the greedy tokens.

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from triallens.data import read_jsonl  # noqa: E402
from triallens.decoder import QwenDecoder, generate_cached  # noqa: E402
from triallens.inference import DEFAULT_MODEL, load_model, prompt_ids  # noqa: E402
from triallens.profiling import PeakMemory  # noqa: E402
from triallens.prompts import format_note  # noqa: E402
from triallens.speculative import generate_speculative  # noqa: E402

DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def summary_ids(tok, record, device):
    msgs = [{"role": "user", "content": "Summarise this patient note in a few sentences.\n\n"
             + format_note(record["sentences"])}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def run(fn, device, repeats):
    outs, peak = [], 0
    for _ in range(repeats):
        with PeakMemory(device) as mem:
            outs.append(fn())
        peak = max(peak, mem.peak_bytes)
    out = outs[0]
    out["total_s"] = statistics.median(o["total_s"] for o in outs)
    out["overall_tok_per_s"] = out["n_tokens"] / out["total_s"]
    out["peak_mem_mb"] = peak / 2**20
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--draft", default=DRAFT_MODEL)
    ap.add_argument("--draft-adapter", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--criterion-tokens", type=int, default=64)
    ap.add_argument("--summary-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t_model, tok = load_model(args.model, args.adapter)
    d_model, d_tok = load_model(args.draft, args.draft_adapter)
    # Speculative decoding compares token IDs directly, so the vocabularies must be identical.
    assert tok.get_vocab() == d_tok.get_vocab(), "draft and target tokenizers differ"
    assert t_model.config.vocab_size == d_model.config.vocab_size
    target, draft = QwenDecoder(t_model, "sdpa"), QwenDecoder(d_model, "sdpa")
    device = target.device
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    params = lambda m: sum(p.numel() for p in m.parameters())
    print(f"device={device} target={params(t_model)/1e9:.2f}B draft={params(d_model)/1e9:.2f}B")

    records = read_jsonl(ROOT / "data/processed/val.jsonl")[: args.n_prompts]
    workloads = {
        "criterion": ([prompt_ids(tok, r["sentences"], r["criterion_type"], r["criterion_text"], device)
                       for r in records], args.criterion_tokens),
        "summary": ([summary_ids(tok, r, device) for r in records], args.summary_tokens),
    }

    results = []
    for wname, (prompts, max_new) in workloads.items():
        for _ in range(args.warmup):
            generate_cached(target, prompts[0], max_new, eos)
            generate_speculative(target, draft, prompts[0], max_new, k=args.ks[0], eos_ids=eos)
        greedy = [run(lambda p=p: generate_cached(target, p, max_new, eos), device, args.repeats) for p in prompts]
        g_time = sum(g["total_s"] for g in greedy)
        g_tokens = sum(g["n_tokens"] for g in greedy)
        row = {"workload": wname, "n_prompts": len(prompts), "max_new_tokens": max_new,
               "greedy": {"total_s": g_time, "tokens": g_tokens, "tok_per_s": g_tokens / g_time,
                          "peak_mem_mb": max(g["peak_mem_mb"] for g in greedy)},
               "speculative": []}
        print(f"[{wname}] greedy: {g_tokens} tokens in {g_time:.2f}s = {g_tokens / g_time:.1f} tok/s")
        for k in args.ks:
            spec = [run(lambda p=p: generate_speculative(target, draft, p, max_new, k=k, eos_ids=eos),
                        device, args.repeats) for p in prompts]
            s_time = sum(s["total_s"] for s in spec)
            drafted = sum(s["drafted"] for s in spec)
            accepted = sum(s["accepted"] for s in spec)
            matches = [s["tokens"] == g["tokens"] for s, g in zip(spec, greedy)]
            entry = {"k": k, "total_s": s_time, "tok_per_s": g_tokens / s_time, "speedup": g_time / s_time,
                     "acceptance_rate": accepted / drafted if drafted else None,
                     "mean_accepted_per_round": accepted / sum(s["rounds"] for s in spec),
                     "tokens_per_target_forward": g_tokens / sum(s["target_forwards"] for s in spec),
                     "peak_mem_mb": max(s["peak_mem_mb"] for s in spec),
                     "all_tokens_match_greedy": all(matches), "n_match": sum(matches)}
            row["speculative"].append(entry)
            print(f"  k={k}: {s_time:.2f}s speedup {entry['speedup']:.2f}x accept {entry['acceptance_rate']:.2f}"
                  f" tok/target-fwd {entry['tokens_per_target_forward']:.2f}"
                  f" peak {entry['peak_mem_mb']:.0f} MB match {sum(matches)}/{len(matches)}")
        results.append(row)

    report = {
        "config": {"target": args.model, "adapter": args.adapter, "draft": args.draft,
                   "draft_adapter": args.draft_adapter, "device": str(device), "dtype": str(target.dtype),
                   "target_params": params(t_model), "draft_params": params(d_model),
                   "warmup": args.warmup, "repeats": args.repeats, "decoding": "greedy, attention=sdpa",
                   "memory": PeakMemory(device).method, "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                   "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "results": results,
    }
    out = Path(args.out) if args.out else ROOT / f"results/bench/speculative_{device.type}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
