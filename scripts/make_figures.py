# Build every figure in the paper from the committed result files.
#
# Usage: python scripts/make_figures.py      -> paper/figures/*.pdf

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"

# Validated categorical order (blue, orange, aqua) and one-hue blue ramp for magnitude.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.4,
    "lines.markersize": 4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = 3.45  # IEEE single column width in inches


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def training_curves():
    runs = [("r16", "r16 (rank 16)", C1), ("os40", "os40 (oversampled)", C2), ("r8", "r8 (rank 8)", C3)]
    fig, axes = plt.subplots(1, 3, figsize=(COL * 2, 1.9))
    for key, label, color in runs:
        log = load(f"results/train/{key}/train_log.json")
        steps = [s["step"] for s in log["steps"]]
        loss = [s["train_loss"] for s in log["steps"]]
        smooth = [sum(loss[max(0, i - 4):i + 1]) / len(loss[max(0, i - 4):i + 1]) for i in range(len(loss))]
        axes[0].plot(steps, smooth, color=color, label=label)
        ep = [e["epoch"] for e in log["epochs"]]
        axes[1].plot(ep, [e["val_loss"] for e in log["epochs"]], color=color, marker="o", label=label)
        axes[2].plot(ep, [e["val_macro_f1"] for e in log["epochs"]], color=color, marker="o", label=label)
        best = log["best_epoch"]
        axes[2].plot([best], [log["epochs"][best - 1]["val_macro_f1"]], marker="*", markersize=9, color=color,
                     markeredgecolor="white", markeredgewidth=0.6, linestyle="none")
    axes[0].set(title="Training loss (5-step mean)", xlabel="optimizer step", ylabel="cross-entropy")
    axes[1].set(title="Validation loss", xlabel="epoch", xticks=[1, 2, 3, 4])
    axes[2].set(title="Validation macro-F1 (star = kept)", xlabel="epoch", xticks=[1, 2, 3, 4])
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "training_curves")


def before_after():
    base = load("results/eval/base_test_metrics.json")["overall"]["classification"]["per_class"]
    tuned = load("results/eval/os40_test_metrics.json")["overall"]["classification"]["per_class"]
    labels = [l for l in base if base[l]["support"] > 0]
    short = {"included": "incl.", "not included": "not incl.", "not excluded": "not excl.",
             "not enough information": "NEI", "not applicable": "N/A"}
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], [base[l]["f1"] for l in labels], w, color=C1, label="Base (zero-shot)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], [tuned[l]["f1"] for l in labels], w, color=C2, label="LoRA os40")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=5.8, color=INK2)
    ax.set_xticks(list(x), [short[l] + f"\n(n={base[l]['support']})" for l in labels])
    ax.set(ylabel="F1 on test", ylim=(0, 1.0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2)
    save(fig, "before_after_f1")


def confusion():
    cm = load("results/eval/os40_test_metrics.json")["overall"]["classification"]["confusion_matrix"]
    rows = [i for i, r in enumerate(cm["matrix"]) if sum(r) > 0]
    cols = [j for j, c in enumerate(cm["cols_pred"]) if c not in ("excluded", "invalid")]
    M = [[cm["matrix"][i][j] for j in cols] for i in rows]
    abbr = {"included": "incl.", "not included": "not incl.", "excluded": "excl.", "not excluded": "not excl.",
            "not enough information": "NEI", "not applicable": "N/A"}
    fig, ax = plt.subplots(figsize=(COL * 0.8, 2.2))
    vmax = max(max(r) for r in M)
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=vmax)
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > vmax * 0.55 else INK)
    ax.set_xticks(range(len(cols)), [abbr[cm["cols_pred"][j]] for j in cols], rotation=30, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(rows)), [abbr[cm["rows_gold"][i]] for i in rows])
    ax.set(xlabel="predicted", ylabel="expert label")
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    save(fig, "confusion_test")


def kv_cache():
    data = load("results/bench/kv_cache_cuda.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2, 1.8))
    for n_out, ax in zip((32, 128), axes):
        rows = [r for r in data if r["output_tokens"] == n_out]
        x = [r["input_tokens"] for r in rows]
        ax.plot(x, [r["cached"]["decode_tok_per_s"] for r in rows], color=C1, marker="o", label="KV cache")
        ax.plot(x, [r["uncached"]["decode_tok_per_s"] for r in rows], color=C2, marker="s", label="no cache")
        for r in rows:
            mark = "" if r["tokens_check"]["match"] else " *"
            ax.annotate(f"{r['speedup_total']:.1f}x{mark}", (r["input_tokens"], r["cached"]["decode_tok_per_s"]),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color=INK2)
        ax.set(title=f"{n_out} generated tokens", xlabel="prompt tokens", xscale="log", ylim=(0, 34),
               xticks=x, xticklabels=[str(v) for v in x])
        ax.minorticks_off()
    axes[0].set_ylabel("decode tokens / s")
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "kv_cache")


def attention():
    perf = load("results/bench/attention_cuda.json")["performance"]
    series = [("reference", "reference", C1, "o"), ("tiled_b128", "tiled (block 128)", C2, "s"),
              ("sdpa", "PyTorch SDPA", C3, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2, 1.8))
    for key, label, color, m in series:
        rows = [p for p in perf if p["method"] == key and "ms" in p]
        x = [p["seq"] for p in rows]
        axes[0].plot(x, [p["ms"] for p in rows], color=color, marker=m, label=label)
        axes[1].plot(x, [p["peak_mem_mb"] for p in rows], color=color, marker=m, label=label)
    for ax, title, ylabel in [(axes[0], "Runtime", "ms (log)"), (axes[1], "Peak extra GPU memory", "MB (log)")]:
        ax.set(title=title, xlabel="sequence length", ylabel=ylabel, xscale="log", yscale="log",
               xticks=[128, 512, 1024, 2048, 4096], xticklabels=["128", "512", "1k", "2k", "4k"])
        ax.minorticks_off()
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "attention")


def speculative():
    stock = load("results/bench/speculative_cuda_stock_draft.json")["results"]
    tuned = load("results/bench/speculative_cuda_tuned_draft.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2, 1.8))
    for res, label, color, m in [(stock, "stock 0.5B draft", C1, "o"), (tuned, "fine-tuned 0.5B draft", C2, "s")]:
        w = next(r for r in res if r["workload"] == "criterion")
        ks = [s["k"] for s in w["speculative"]]
        axes[0].plot(ks, [s["acceptance_rate"] for s in w["speculative"]], color=color, marker=m, label=label)
        axes[1].plot(ks, [s["speedup"] for s in w["speculative"]], color=color, marker=m, label=label)
    axes[1].axhline(1.0, color=INK2, linewidth=0.8, linestyle="--")
    axes[1].text(1, 1.02, "plain greedy", va="bottom", fontsize=6, color=INK2)
    axes[0].set(title="Draft acceptance (criterion task)", xlabel="draft tokens k", ylabel="accepted / drafted",
                ylim=(0.5, 1.03))
    axes[1].set(title="Speedup over KV-cached greedy (T4)", xlabel="draft tokens k", ylabel="speedup (x)", ylim=(0.5, 1.15))
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "speculative")


if __name__ == "__main__":
    training_curves()
    before_after()
    confusion()
    kv_cache()
    attention()
    speculative()
