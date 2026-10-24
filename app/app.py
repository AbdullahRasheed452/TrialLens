# TrialLens browser application.
#
# Start with start.bat (Windows) or:  python app/app.py [--adapter PATH] [--port 7860]
# then open http://127.0.0.1:7860 in a browser.

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from render import CSS, disclaimer_html, note_html, report_html, results_html, summary_html  # noqa: E402
from triallens.attention import causal_mask, reference_attention, sdpa_attention, tiled_attention  # noqa: E402
from triallens.data import read_jsonl  # noqa: E402
from triallens.decoder import generate_cached, generate_uncached  # noqa: E402
from triallens.profiling import PeakMemory  # noqa: E402
from triallens.prompts import format_note  # noqa: E402
from triallens.screening import (DECODING_MODES, DISCLAIMER, EngineConfig, InputError, ScreeningEngine,  # noqa: E402
                                 build_report, fetch_trial, parse_criteria, split_sentences)
from triallens.speculative import generate_speculative  # noqa: E402

ENGINE: ScreeningEngine | None = None
EXPORT_DIR = Path(tempfile.mkdtemp(prefix="triallens_"))


# ---------------------------------------------------------------- examples

# Validation patients only; the test patients never appear in the app.
def load_examples() -> dict:
    path = ROOT / "data/processed/val.jsonl"
    if not path.exists():
        return {}
    groups = {}
    for r in read_jsonl(path):
        groups.setdefault((r["patient_id"], r["trial_id"]), []).append(r)
    examples = {}
    for (pid, tid), recs in groups.items():
        lines = []
        for ctype in ("inclusion", "exclusion"):
            items = [r for r in recs if r["criterion_type"] == ctype]
            if items:
                lines.append(f"{ctype.capitalize()} criteria:")
                lines += [f"* {r['criterion_text']}" for r in items]
                lines.append("")
        name = f"{pid} · {tid} ({len(recs)} criteria)"
        examples[name] = {
            "note": format_note(recs[0]["sentences"]),
            "criteria": "\n".join(lines).strip(),
            "trial": {"nct_id": tid, "title": recs[0]["trial_title"]},
            "gold": {r["criterion_text"]: {"label": r["label"], "evidence_ids": r["evidence_ids"]} for r in recs},
        }
    return examples


EXAMPLES = load_examples()


def pick_example(name):
    ex = EXAMPLES.get(name)
    if not ex:
        return gr.update(), gr.update(), gr.update(), None
    return ex["note"], ex["criteria"], ex["trial"]["nct_id"], ex


def fetch(nct_id):
    try:
        t = fetch_trial(nct_id)
    except InputError as e:
        gr.Warning(str(e))
        return gr.update(), f"⚠️ {e}", None
    n = len(parse_criteria(t["criteria"]))
    return t["criteria"], f"Loaded **{t['nct_id']}**: {t['title']} ({n} criteria).", {"nct_id": t["nct_id"], "title": t["title"]}


# ---------------------------------------------------------------- screening

def screen(note, criteria_text, mode_label, attention, k, example, trial):
    mode = next(m for m, lbl in DECODING_MODES.items() if lbl == mode_label)
    empty = ("", "", "", gr.update(visible=False), gr.update(visible=False))
    try:
        sentences = split_sentences(note)
        criteria = parse_criteria(criteria_text)
        info = ENGINE.check_note(sentences)
    except InputError as e:
        gr.Warning(str(e))
        yield (f"⚠️ **Input problem:** {e}", *empty)
        return

    gold = example["gold"] if example and example.get("note") == note else {}
    if trial is None and example and example.get("note") == note:
        trial = example["trial"]
    head = f"Screening {len(criteria)} criteria against {len(sentences)} sentences ({info['note_tokens']} tokens) with **{mode_label}**, attention `{attention}`."
    if info["warning"]:
        head += f"\n\n⚠️ {info['warning']}"
    results, start = [], time.perf_counter()
    yield (head + "\n\nLoading models and preparing the patient prefix…", "", "", note_html(sentences, []),
           gr.update(visible=False), gr.update(visible=False))
    try:
        for r in ENGINE.screen(sentences, criteria, mode=mode, attention=attention, k=int(k)):
            results.append(r)
            status = f"{head}\n\nDone {len(results)} of {len(criteria)} · {time.perf_counter() - start:.1f}s elapsed"
            yield (status, "", results_html(results, gold), note_html(sentences, results),
                   gr.update(visible=False), gr.update(visible=False))
    except InputError as e:
        gr.Warning(str(e))
        yield (f"⚠️ **Input problem:** {e}", "", results_html(results, gold), note_html(sentences, results),
               gr.update(visible=False), gr.update(visible=False))
        return
    except Exception as e:  # keep the app alive and show what went wrong
        traceback.print_exc()
        yield (f"❌ **Unexpected error:** {type(e).__name__}: {e}. The app is still running; try again.",
               "", results_html(results, gold), note_html(sentences, results),
               gr.update(visible=False), gr.update(visible=False))
        return

    report = build_report(note, sentences, criteria_text, results, ENGINE.provenance(mode, attention, int(k)), trial)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    html_path = EXPORT_DIR / f"triallens_report_{stamp}.html"
    json_path = EXPORT_DIR / f"triallens_report_{stamp}.json"
    html_path.write_text(report_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    total = time.perf_counter() - start
    status = f"{head}\n\n✅ Finished {len(results)} criteria in {total:.1f}s ({total / len(results):.1f}s per criterion)."
    if gold:
        hits = sum(gold.get(r["criterion"], {}).get("label") == r["label"] for r in results)
        status += f" Agreement with the expert labels on this validation example: {hits}/{len(results)}."
    yield (status, summary_html(report["summary"], results), results_html(results, gold), note_html(sentences, results),
           gr.update(value=str(html_path), visible=True), gr.update(value=str(json_path), visible=True))


# ---------------------------------------------------------------- inference lab

def _summary_prompt(sentences):
    msgs = [{"role": "user", "content": "Summarise this patient note in a few sentences.\n\n" + format_note(sentences)}]
    return ENGINE.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def compare_decoding(note, criteria_text, workload, max_new, modes, k, ignore_eos):
    try:
        sentences = split_sentences(note)
        if workload == "Criterion answer":
            c = parse_criteria(criteria_text)[0]
            prefix, suffix = ENGINE._prompt_parts(sentences, c)
            text = prefix + suffix
        else:
            text = _summary_prompt(sentences)
            prefix = text
        if not modes:
            raise InputError("Pick at least one decoding mode.")
    except InputError as e:
        gr.Warning(str(e))
        return f"⚠️ {e}", [], ""

    dec = ENGINE.target
    dec.set_attention("sdpa")
    ids = ENGINE._ids(text)
    eos = () if ignore_eos else ENGINE.eos
    max_new = int(max_new)
    runs = {}
    with PeakMemory(dec.device) as m:
        runs["KV cache"] = generate_cached(dec, ids, max_new, eos)
    runs["KV cache"]["peak"] = m.peak_bytes
    if "No cache (reference)" in modes:
        with PeakMemory(dec.device) as m:
            runs["No cache (reference)"] = generate_uncached(dec, ids, max_new, eos)
        runs["No cache (reference)"]["peak"] = m.peak_bytes
    if "KV cache + shared prefix" in modes and workload == "Criterion answer":
        P = ENGINE._ids(prefix).shape[1]
        cache = dec.new_cache(ids.shape[1] + max_new)
        dec.forward(ids[:, :P], cache, start=0)
        with PeakMemory(dec.device) as m:
            runs["KV cache + shared prefix"] = generate_cached(dec, ids, max_new, eos, cache=cache, prefix_len=P)
        runs["KV cache + shared prefix"]["peak"] = m.peak_bytes
    if "Speculative" in modes:
        draft = ENGINE.load_draft()
        draft.set_attention("sdpa")
        with PeakMemory(dec.device) as m:
            runs["Speculative"] = generate_speculative(dec, draft, ids, max_new, k=int(k), eos_ids=eos)
        runs["Speculative"]["peak"] = m.peak_bytes

    base = runs["KV cache"]
    order = ["No cache (reference)", "KV cache", "KV cache + shared prefix", "Speculative"]
    rows = []
    for name in order:
        if name not in runs or (name != "KV cache" and name not in modes):
            continue
        r = runs[name]
        rows.append([
            name, round(r["prefill_s"] * 1e3, 1), round(r["decode_s"] * 1e3, 1), round(r["total_s"] * 1e3, 1),
            r["n_tokens"], round(r["n_tokens"] / r["total_s"], 2),
            "✓ identical" if r["tokens"] == base["tokens"] else "✗ differs",
            f'{r["acceptance_rate"]:.0%} (k={k})' if r.get("acceptance_rate") is not None else "",
            round(r.get("kv_cache_bytes", 0) / 2**20, 2) if name != "Speculative" else "",
            round(r["peak"] / 2**20, 1),
        ])
    msg = (f"Prompt: {ids.shape[1]} tokens · device `{dec.device}` · `{dec.dtype}` · greedy decoding. "
           f"Every mode is checked token by token against the KV-cache greedy output. "
           f"Peak memory is {PeakMemory(dec.device).method} during the call.")
    if "KV cache + shared prefix" in modes and workload != "Criterion answer":
        msg += " The shared-prefix mode only applies to the criterion workload and was skipped."
    return msg, rows, ENGINE.tok.decode(base["tokens"], skip_special_tokens=True)


def attention_check(seq_len, block, offset_case):
    T, b, H, D = int(seq_len), int(block), 12, 128
    g = torch.Generator().manual_seed(0)
    if offset_case == "Decode step on a cache (1 query)":
        tq, off = 1, T - 1
    elif offset_case == "Verification chunk (5 queries on a cache)":
        tq, off = 5, T - 5
    else:
        tq, off = T, 0
    q = torch.randn(1, H, tq, D, generator=g)
    k, v = (torch.randn(1, H, T, D, generator=g) for _ in range(2))
    s = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(D)
    s = s.masked_fill(~causal_mask(tq, T, off, q.device), float("-inf"))
    truth = torch.softmax(s, -1) @ v.double()
    rows = []
    for name, fn in [("Reference (full score matrix)", lambda: reference_attention(q, k, v, True, off)),
                     (f"Tiled online softmax (block {b})", lambda: tiled_attention(q, k, v, True, off, b, b)),
                     ("PyTorch SDPA (fused)", lambda: sdpa_attention(q, k, v, True, off))]:
        fn()
        times = []
        for _ in range(3):
            t = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - t)
        err = (out.double() - truth).abs()
        rows.append([name, f"{err.max().item():.2e}", f"{err.mean().item():.2e}", round(statistics.median(times) * 1e3, 2)])
    score_mb = H * tq * T * 4 / 2**20
    tile_mb = H * min(b, tq) * b * 4 / 2**20
    note = (f"q: [1, {H}, {tq}, {D}], k/v: [1, {H}, {T}, {D}], causal with query offset {off}, fp32 on CPU, median of 3 runs. "
            f"The reference keeps a {score_mb:.2f} MB score matrix; the tiled version holds at most {tile_mb:.3f} MB of scores "
            f"at a time. Errors are against a float64 computation.")
    return note, rows


def recorded_benchmarks():
    bench = ROOT / "results/bench"
    parts = []
    for path in sorted(bench.glob("*.json")) if bench.exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        where = cfg.get("gpu") or cfg.get("device")
        parts.append(f"#### {path.name}\nRecorded on **{where}**, {cfg.get('timestamp', '')}")
        if "rows" in data:
            parts.append("| input | output | cached tok/s | uncached tok/s | speedup | tokens match |\n|---|---|---|---|---|---|")
            for r in data["rows"]:
                parts.append(f"| {r['input_tokens']} | {r['output_tokens']} | {r['cached']['decode_tok_per_s']:.1f} | "
                             f"{r['uncached']['decode_tok_per_s']:.1f} | {r['speedup_total']:.2f}x | {r['tokens_check']['match']} |")
        elif "performance" in data:
            parts.append(f"Worst max error vs float64: {', '.join(f'{k} {v:.1e}' for k, v in data['worst_max_abs_err'].items())}")
            parts.append("| seq | method | ms | peak MB |\n|---|---|---|---|")
            for r in data["performance"]:
                parts.append(f"| {r['seq']} | {r['method']} | {r.get('ms', float('nan')):.2f} | {r.get('peak_mem_mb', float('nan')):.1f} |")
        elif "results" in data:
            for w in data["results"]:
                parts.append(f"**{w['workload']}**: greedy {w['greedy']['tok_per_s']:.1f} tok/s")
                parts.append("| k | speedup | acceptance | tokens/target pass | matches greedy |\n|---|---|---|---|---|")
                for s in w["speculative"]:
                    parts.append(f"| {s['k']} | {s['speedup']:.2f}x | {s['acceptance_rate']:.0%} | "
                                 f"{s['tokens_per_target_forward']:.2f} | {s['n_match']}/{w['n_prompts']} |")
        parts.append("")
    return "\n".join(parts) or "No benchmark logs found in results/bench yet."


# ---------------------------------------------------------------- layout

ABOUT = f"""
### What TrialLens does
For every criterion it asks a fine-tuned **Qwen2.5-1.5B-Instruct** (LoRA adapter) for a label and the numbers of
the patient-note sentences that support it. Labels follow the TrialGPT expert annotation scheme:

* inclusion criteria: included, not included, not enough information, not applicable
* exclusion criteria: excluded, not excluded, not enough information, not applicable

The explanation is built only from the sentences the model cites. Missing evidence is reported as missing and is
never treated as proof either way.

### How the preliminary outcome is aggregated
1. Any inclusion criterion judged *not included*, or exclusion criterion judged *excluded*, gives **Likely ineligible**.
2. Otherwise any *not enough information* or unreadable answer gives **Needs review**.
3. Otherwise the outcome is **Potentially eligible**.

These rules combine criterion-level predictions. The dataset has no patient-level labels, so this outcome is not
validated against ground truth.

### Decoding modes
* **No cache (reference)**: every new token recomputes the whole sequence.
* **KV cache**: keys and values of earlier tokens are stored once and reused.
* **KV cache + shared patient prefix**: the system prompt and patient note are computed once and reused for every criterion.
* **Speculative**: Qwen2.5-0.5B drafts k tokens, the 1.5B model verifies them in one pass; output is identical to greedy.

{DISCLAIMER}
"""


def build_ui():
    with gr.Blocks(title="TrialLens") as ui:
        gr.HTML(f'<h1 style="margin:6px 0 0">TrialLens</h1><p style="margin:0 0 4px;opacity:.75">'
                f'Evidence-grounded clinical trial eligibility screening</p>{disclaimer_html(DISCLAIMER)}')
        example_state, trial_state = gr.State(None), gr.State(None)

        with gr.Tab("Screen a patient"):
            with gr.Row():
                with gr.Column(scale=5):
                    example = gr.Dropdown(list(EXAMPLES), label="Load a validation example (optional)", value=None)
                    note = gr.Textbox(label="Patient note", lines=10,
                                      placeholder="Paste a clinical note. Numbered lines (0. ..., 1. ...) keep their numbers; "
                                                  "otherwise the note is split into sentences automatically.")
                    with gr.Row():
                        nct = gr.Textbox(label="ClinicalTrials.gov ID (optional)", placeholder="NCT01397994", scale=3)
                        fetch_btn = gr.Button("Fetch criteria", scale=1)
                    trial_info = gr.Markdown()
                    criteria = gr.Textbox(label="Eligibility criteria", lines=10,
                                          placeholder="Inclusion criteria:\n* ...\n\nExclusion criteria:\n* ...")
                    with gr.Accordion("Inference settings", open=False):
                        mode = gr.Radio(list(DECODING_MODES.values()), value=DECODING_MODES["kv_prefix"], label="Decoding")
                        attention = gr.Radio(["sdpa", "reference", "tiled"], value="sdpa",
                                             label="Attention implementation inside the decoder")
                        k = gr.Slider(1, 8, value=4, step=1, label="Draft tokens per round (speculative only)")
                    run = gr.Button("Run screening", variant="primary")
                with gr.Column(scale=6):
                    status = gr.Markdown("Fill in a note and criteria, or load an example, then press **Run screening**.")
                    summary = gr.HTML()
                    with gr.Row():
                        dl_html = gr.DownloadButton("Download report (HTML)", visible=False)
                        dl_json = gr.DownloadButton("Download report (JSON)", visible=False)
                    with gr.Tab("Criteria"):
                        results = gr.HTML()
                    with gr.Tab("Patient note with evidence"):
                        note_view = gr.HTML()

            example.change(pick_example, example, [note, criteria, nct, example_state])
            fetch_btn.click(fetch, nct, [criteria, trial_info, trial_state])
            run.click(screen, [note, criteria, mode, attention, k, example_state, trial_state],
                      [status, summary, results, note_view, dl_html, dl_json])

        with gr.Tab("Inference lab"):
            gr.Markdown("Live measurements on this machine using the note and criteria from the first tab. "
                        "The uncached reference is slow on a CPU; keep the token count small.")
            with gr.Row():
                workload = gr.Radio(["Criterion answer", "Patient summary (longer output)"], value="Criterion answer",
                                    label="Workload")
                max_new = gr.Slider(4, 96, value=16, step=4, label="Tokens to generate")
            with gr.Row():
                modes = gr.CheckboxGroup(["No cache (reference)", "KV cache + shared prefix", "Speculative"],
                                         value=["No cache (reference)", "KV cache + shared prefix", "Speculative"],
                                         label="Compare against the KV-cache baseline")
                lab_k = gr.Slider(1, 8, value=4, step=1, label="Draft tokens per round")
                ignore_eos = gr.Checkbox(value=True, label="Ignore end-of-sequence (fixed length)")
            cmp_btn = gr.Button("Run comparison", variant="primary")
            cmp_msg = gr.Markdown()
            cmp_table = gr.Dataframe(headers=["Mode", "Prefill ms", "Decode ms", "Total ms", "Tokens", "Tokens/s",
                                              "Same tokens as KV greedy", "Draft acceptance", "KV cache MB", "Peak MB"],
                                     interactive=False)
            cmp_text = gr.Textbox(label="Generated text (KV cache)", lines=3)
            cmp_btn.click(compare_decoding, [note, criteria, workload, max_new, modes, lab_k, ignore_eos],
                          [cmp_msg, cmp_table, cmp_text])

            gr.Markdown("### Attention implementations")
            with gr.Row():
                seq = gr.Slider(16, 1024, value=256, step=16, label="Sequence length")
                block = gr.Slider(8, 256, value=64, step=8, label="Tile size")
                case = gr.Radio(["Full causal prompt", "Decode step on a cache (1 query)",
                                 "Verification chunk (5 queries on a cache)"], value="Full causal prompt", label="Shape")
            att_btn = gr.Button("Check attention")
            att_msg = gr.Markdown()
            att_table = gr.Dataframe(headers=["Implementation", "Max |error|", "Mean |error|", "ms"], interactive=False)
            att_btn.click(attention_check, [seq, block, case], [att_msg, att_table])

            with gr.Accordion("Recorded GPU benchmark logs (results/bench)", open=False):
                gr.Markdown(recorded_benchmarks())

        with gr.Tab("About"):
            gr.Markdown(ABOUT)
            gr.JSON(value=lambda: ENGINE.provenance("kv_prefix", "sdpa", 4), label="Loaded model")
    return ui


def _pointer(name: str) -> str | None:
    pointer = ROOT / "checkpoints" / name
    if pointer.exists():
        path = ROOT / pointer.read_text(encoding="utf-8").strip()
        if (path / "adapter_config.json").exists():
            return str(path)
    return None


def default_adapter() -> str | None:
    if found := _pointer("BEST"):
        return found
    for path in sorted((ROOT / "checkpoints").glob("*/lora-best")):
        cfg = path / "adapter_config.json"
        if cfg.exists() and "1.5B" in cfg.read_text(encoding="utf-8"):
            return str(path)
    return None


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder (defaults to checkpoints/BEST)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    adapter = args.adapter or default_adapter()
    if adapter is None:
        print("WARNING: no fine-tuned adapter found, running the base model. Run scripts/download_adapter.py first.")
    draft_adapter = _pointer("DRAFT")
    print(f"Loading {EngineConfig().model} with adapter {adapter} (speculative draft adapter: {draft_adapter}) ...")
    ENGINE = ScreeningEngine(EngineConfig(adapter=adapter, draft_adapter=draft_adapter)).load()
    print(f"Model ready on {ENGINE.target.device}. Opening http://127.0.0.1:{args.port}")
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser,
        css=CSS, theme=gr.themes.Soft(), allowed_paths=[str(EXPORT_DIR)])


if __name__ == "__main__":
    main()
