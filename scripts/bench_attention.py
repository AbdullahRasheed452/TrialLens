# Task 4 benchmark: reference attention vs tiled online-softmax attention vs PyTorch SDPA.
#
# Usage:
#   python scripts/bench_attention.py                       # defaults sized for a GPU
#   python scripts/bench_attention.py --seqs 128 512 1024   # smaller set for a CPU
#
# Numerical check: every method is compared with a float64 softmax(QK^T / sqrt(d)) V
# computed independently, reporting max and mean absolute error, on causal masks with
# and without a cache offset. Shapes use Qwen2.5-1.5B's 12 heads of dimension 128.
#
# Performance: median of repeated, synchronised runs after warm-up, plus peak memory of
# the call. The analytic size of the score matrix each method keeps alive is reported
# next to it, because that is the quantity tiling is designed to shrink.

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from triallens.attention import causal_mask, reference_attention, sdpa_attention, tiled_attention  # noqa: E402
from triallens.profiling import PeakMemory  # noqa: E402


def exact(q, k, v, q_offset):
    s = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q_offset, q.device), float("-inf"))
    return torch.softmax(s, dim=-1) @ v.double()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    times = []
    with PeakMemory(device) as mem:
        for _ in range(repeats):
            sync(device)
            t = time.perf_counter()
            fn()
            sync(device)
            times.append(time.perf_counter() - t)
    return statistics.median(times), mem.peak_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--dtype", default=None, choices=["float16", "float32"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float16 if device.type == "cuda" else torch.float32)
    g = torch.Generator(device="cpu").manual_seed(0)
    H, D = args.heads, args.head_dim
    print(f"device={device} dtype={dtype} heads={H} head_dim={D}")

    # Correctness on controlled shapes: full causal prompts, a decode step, and a verify chunk.
    accuracy = []
    for tq, tk, off in [(1, 1, 0), (17, 17, 0), (64, 64, 0), (100, 100, 0), (257, 257, 0),
                        (1, 300, 299), (5, 133, 128)]:
        q, k, v = (torch.randn(1, H, t, D, generator=g).to(device, dtype) for t in (tq, tk, tk))
        truth = exact(q, k, v, off)
        for name, fn in [("reference", lambda: reference_attention(q, k, v, True, off)),
                         ("sdpa", lambda: sdpa_attention(q, k, v, True, off))] + \
                        [(f"tiled_b{b}", lambda b=b: tiled_attention(q, k, v, True, off, b, b)) for b in args.blocks]:
            err = (fn().double() - truth).abs()
            accuracy.append({"tq": tq, "tk": tk, "q_offset": off, "method": name,
                             "max_abs_err": err.max().item(), "mean_abs_err": err.mean().item()})
    worst = {}
    for a in accuracy:
        worst[a["method"]] = max(worst.get(a["method"], 0), a["max_abs_err"])
    print("worst max |error| vs float64:", {k: f"{v:.2e}" for k, v in worst.items()})

    # Speed and memory on prompt-sized causal attention.
    perf = []
    bytes_per = torch.finfo(torch.float32).bits // 8
    for T in args.seqs:
        q, k, v = (torch.randn(1, H, T, D, generator=g).to(device, dtype) for _ in range(3))
        methods = [("reference", lambda: reference_attention(q, k, v), H * T * T * bytes_per),
                   ("sdpa", lambda: sdpa_attention(q, k, v), None)]
        methods += [(f"tiled_b{b}", lambda b=b: tiled_attention(q, k, v, True, 0, b, b), H * b * b * bytes_per)
                    for b in args.blocks]
        for name, fn, score_bytes in methods:
            try:
                sec, peak = bench(fn, device, args.warmup, args.repeats)
            except torch.OutOfMemoryError:
                perf.append({"seq": T, "method": name, "error": "out of memory"})
                print(f"T={T:5d} {name:11s} out of memory")
                continue
            perf.append({"seq": T, "method": name, "ms": sec * 1e3, "peak_mem_mb": peak / 2**20,
                         "score_matrix_mb": None if score_bytes is None else score_bytes / 2**20})
            print(f"T={T:5d} {name:11s} {sec*1e3:9.2f} ms  peak {peak/2**20:8.1f} MB"
                  + ("" if score_bytes is None else f"  scores held {score_bytes/2**20:8.2f} MB"))

    report = {
        "config": {"device": str(device), "dtype": str(dtype), "heads": H, "head_dim": D,
                   "blocks": args.blocks, "warmup": args.warmup, "repeats": args.repeats,
                   "memory": PeakMemory(device).method, "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                   "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "accuracy": accuracy,
        "worst_max_abs_err": worst,
        "performance": perf,
    }
    out = Path(args.out) if args.out else ROOT / f"results/bench/attention_{device.type}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
