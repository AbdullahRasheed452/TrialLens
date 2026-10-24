# HTML rendering for the app and the downloadable report. Everything user supplied is
# escaped before it reaches the page.
#
# Colour is used only where it carries meaning: one teal accent for interaction and
# provenance, and four muted semantic tones for the label categories. Everything else is
# slate. Light and dark values are defined as tokens so both modes stay consistent.

from __future__ import annotations

import json
from html import escape

LABEL_TONE = {
    "included": "met", "not excluded": "met",
    "not included": "out", "excluded": "out",
    "not enough information": "nei", "not applicable": "na", "invalid": "bad",
}
OUTCOME_TONE = {"Potentially eligible": "met", "Needs review": "nei", "Likely ineligible": "out"}

TOKENS = """
:root{
  --tl-ink:#0f172a;--tl-ink-2:#475569;--tl-ink-3:#64748b;--tl-line:#e2e8f0;--tl-line-2:#cbd5e1;
  --tl-surface:#ffffff;--tl-surface-2:#f8fafc;--tl-accent:#0f766e;--tl-accent-soft:#f0fdfa;--tl-accent-line:#99f6e4;
  --tl-met:#166534;--tl-met-bg:#f0fdf4;--tl-met-line:#bbf7d0;
  --tl-out:#991b1b;--tl-out-bg:#fef2f2;--tl-out-line:#fecaca;
  --tl-nei:#92400e;--tl-nei-bg:#fffbeb;--tl-nei-line:#fde68a;
  --tl-na:#475569;--tl-na-bg:#f1f5f9;--tl-na-line:#cbd5e1;
  --tl-bad:#6b21a8;--tl-bad-bg:#faf5ff;--tl-bad-line:#e9d5ff;
  --tl-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;
}
.dark{
  --tl-ink:#e2e8f0;--tl-ink-2:#94a3b8;--tl-ink-3:#7c8aa0;--tl-line:#1e293b;--tl-line-2:#334155;
  --tl-surface:#0f172a;--tl-surface-2:#111c30;--tl-accent:#2dd4bf;--tl-accent-soft:rgba(45,212,191,.08);--tl-accent-line:rgba(45,212,191,.35);
  --tl-met:#86efac;--tl-met-bg:rgba(34,197,94,.10);--tl-met-line:rgba(134,239,172,.28);
  --tl-out:#fca5a5;--tl-out-bg:rgba(239,68,68,.10);--tl-out-line:rgba(252,165,165,.28);
  --tl-nei:#fcd34d;--tl-nei-bg:rgba(245,158,11,.10);--tl-nei-line:rgba(252,211,77,.28);
  --tl-na:#cbd5e1;--tl-na-bg:rgba(148,163,184,.10);--tl-na-line:rgba(203,213,225,.22);
  --tl-bad:#d8b4fe;--tl-bad-bg:rgba(168,85,247,.10);--tl-bad-line:rgba(216,180,254,.28);
}
"""

CSS = TOKENS + """
footer{display:none !important}
.gradio-container button.secondary{border:1px solid var(--tl-line-2) !important;font-weight:500 !important;font-size:14px !important;box-shadow:none !important}
.gradio-container button.secondary:hover{border-color:var(--tl-accent) !important;color:var(--tl-accent) !important}
.tl-tight{min-height:0 !important;padding:0 !important;border:none !important;background:transparent !important;box-shadow:none !important}
.tl-tight:empty, .tl-tight > div:empty{display:none}
.tl-header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;padding:6px 0 14px;border-bottom:1px solid var(--tl-line);margin-bottom:12px}
.tl-brand{font-size:24px;font-weight:650;letter-spacing:-.02em;color:var(--tl-ink)}
.tl-brand span{color:var(--tl-accent)}
.tl-sub{font-size:14px;color:var(--tl-ink-2);margin-top:2px}
.tl-badges{display:flex;gap:6px;flex-wrap:wrap}
.tl-badge{font-family:var(--tl-mono);font-size:11.5px;color:var(--tl-ink-2);border:1px solid var(--tl-line-2);border-radius:4px;padding:2px 7px;background:var(--tl-surface-2)}
.tl-notice{font-size:13px;color:var(--tl-ink-2);border-left:3px solid var(--tl-line-2);padding:4px 0 4px 10px;margin:0 0 6px}
.tl-notice strong{color:var(--tl-ink);font-weight:600}

.tl-status{display:flex;gap:10px;align-items:flex-start;border:1px solid var(--tl-line);border-radius:6px;padding:10px 12px;background:var(--tl-surface-2);font-size:13.5px;color:var(--tl-ink-2)}
.tl-status-title{color:var(--tl-ink);font-weight:550}
.tl-status-detail{margin-top:2px}
.tl-dot{width:8px;height:8px;border-radius:50%;margin-top:6px;flex:none;background:var(--tl-ink-3)}
.tl-status-running .tl-dot{background:var(--tl-accent);animation:tl-pulse 1.2s ease-in-out infinite}
.tl-status-done .tl-dot{background:var(--tl-met)}
.tl-status-error{border-color:var(--tl-out-line);background:var(--tl-out-bg)}
.tl-status-error .tl-dot{background:var(--tl-out)}
.tl-status-warn{border-color:var(--tl-nei-line)}
.tl-status-warn .tl-dot{background:var(--tl-nei)}
@keyframes tl-pulse{0%,100%{opacity:1}50%{opacity:.35}}

.tl-outcome{border:1px solid var(--tl-line);border-left:4px solid var(--tl-ink-3);border-radius:6px;padding:12px 14px;margin:8px 0;background:var(--tl-surface)}
.tl-outcome.met{border-left-color:var(--tl-met)}.tl-outcome.out{border-left-color:var(--tl-out)}.tl-outcome.nei{border-left-color:var(--tl-nei)}
.tl-eyebrow{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--tl-ink-3)}
.tl-outcome-title{font-size:20px;font-weight:650;color:var(--tl-ink);margin:2px 0 4px;letter-spacing:-.01em}
.tl-outcome p{margin:2px 0;font-size:13.5px;color:var(--tl-ink-2)}
.tl-facts{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;margin-top:10px;font-size:13px}
.tl-facts dt{color:var(--tl-ink-3)}
.tl-facts dd{margin:0;color:var(--tl-ink)}
.tl-counts{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}

.tl-label{display:inline-flex;align-items:center;gap:6px;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;border:1px solid}
.tl-label::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.tl-label.met{color:var(--tl-met);background:var(--tl-met-bg);border-color:var(--tl-met-line)}
.tl-label.out{color:var(--tl-out);background:var(--tl-out-bg);border-color:var(--tl-out-line)}
.tl-label.nei{color:var(--tl-nei);background:var(--tl-nei-bg);border-color:var(--tl-nei-line)}
.tl-label.na{color:var(--tl-na);background:var(--tl-na-bg);border-color:var(--tl-na-line)}
.tl-label.bad{color:var(--tl-bad);background:var(--tl-bad-bg);border-color:var(--tl-bad-line)}
.tl-label .n{font-family:var(--tl-mono);font-weight:500;opacity:.85}

.tl-card{border:1px solid var(--tl-line);border-radius:6px;padding:12px 14px;margin:8px 0;background:var(--tl-surface)}
.tl-card-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tl-idx{font-family:var(--tl-mono);font-size:12px;color:var(--tl-ink-3)}
.tl-kind{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--tl-ink-3)}
.tl-crit{font-size:15px;line-height:1.45;color:var(--tl-ink);margin:6px 0 8px}
.tl-ev{display:flex;gap:8px;align-items:baseline;font-size:13.5px;line-height:1.5;color:var(--tl-ink);padding:5px 8px;margin:3px 0;background:var(--tl-surface-2);border-radius:4px}
.tl-ev.none{color:var(--tl-ink-3);font-style:italic}
.tl-sid{font-family:var(--tl-mono);font-size:11.5px;color:var(--tl-accent);flex:none}
.tl-expl{font-size:13px;line-height:1.5;color:var(--tl-ink-2);margin-top:8px}
.tl-flag{font-size:12.5px;color:var(--tl-nei);margin-top:6px}
.tl-foot{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px;padding-top:8px;border-top:1px solid var(--tl-line)}
.tl-gold{font-size:12.5px;color:var(--tl-ink-2)}
.tl-gold b{color:var(--tl-ink);font-weight:600}
.tl-agree{font-size:11px;font-weight:600;border-radius:3px;padding:1px 6px;margin-left:6px}
.tl-agree.yes{color:var(--tl-met);background:var(--tl-met-bg)}
.tl-agree.no{color:var(--tl-out);background:var(--tl-out-bg)}
.tl-meta{font-family:var(--tl-mono);font-size:11px;color:var(--tl-ink-3)}
details.tl-raw{margin-top:6px}
details.tl-raw summary{cursor:pointer;font-size:12px;color:var(--tl-ink-3)}
details.tl-raw pre{font-family:var(--tl-mono);white-space:pre-wrap;font-size:12px;margin:6px 0 0;padding:8px;background:var(--tl-surface-2);border-radius:4px;color:var(--tl-ink)}

.tl-note{font-size:14px;line-height:1.55;color:var(--tl-ink)}
.tl-note .row{display:grid;grid-template-columns:28px 1fr;gap:8px;padding:5px 8px;border-left:3px solid transparent;border-radius:0 4px 4px 0}
.tl-note .row.cited{border-left-color:var(--tl-accent);background:var(--tl-accent-soft)}
.tl-note .sid{font-family:var(--tl-mono);font-size:12px;color:var(--tl-ink-3);padding-top:2px}
.tl-note .by{display:block;font-size:11.5px;color:var(--tl-accent);margin-top:2px}
.tl-empty{font-size:13.5px;color:var(--tl-ink-3);padding:24px 4px}
"""


def label_badge(label: str, count: int | None = None) -> str:
    n = f' <span class="n">{count}</span>' if count is not None else ""
    return f'<span class="tl-label {LABEL_TONE.get(label, "na")}">{escape(label)}{n}</span>'


def header_html(model: str, adapter: str | None, device: str) -> str:
    badges = [model.split("/")[-1], "LoRA adapter" if adapter else "base weights (no adapter)", device]
    return (
        '<div class="tl-header"><div><div class="tl-brand">Trial<span>Lens</span></div>'
        '<div class="tl-sub">Criterion-level clinical trial eligibility screening with sentence-level evidence</div></div>'
        f'<div class="tl-badges">{"".join(f"<span class=tl-badge>{escape(b)}</span>" for b in badges)}</div></div>'
    )


def notice_html(text: str) -> str:
    head, _, rest = text.partition(". ")
    return f'<div class="tl-notice"><strong>{escape(head)}.</strong> {escape(rest)}</div>'


def status_html(kind: str, title: str, detail: str = "") -> str:
    d = f'<div class="tl-status-detail">{detail}</div>' if detail else ""
    return (f'<div class="tl-status tl-status-{kind}"><span class="tl-dot"></span>'
            f'<div><div class="tl-status-title">{escape(title)}</div>{d}</div></div>')


def summary_html(summary: dict, results: list[dict]) -> str:
    tone = OUTCOME_TONE.get(summary["outcome"], "na")
    refs = lambda idx: ", ".join(f"#{i + 1}" for i in idx) or "none"
    counts = "".join(label_badge(k, v) for k, v in sorted(summary["counts"].items(), key=lambda kv: -kv[1]))
    return (
        f'<div class="tl-outcome {tone}"><div class="tl-eyebrow">Preliminary outcome</div>'
        f'<div class="tl-outcome-title">{escape(summary["outcome"])}</div>'
        f'<p>{escape(summary["rule"])} {escape(summary["note"])}</p>'
        f'<div class="tl-counts">{counts}</div>'
        '<dl class="tl-facts">'
        f'<dt>Rules the patient out</dt><dd>{refs(summary["blocking"])}</dd>'
        f'<dt>Missing information</dt><dd>{refs(summary["missing_information"])}</dd>'
        f'<dt>No evidence cited</dt><dd>{refs(summary["unsupported"])}</dd>'
        f'<dt>Unreadable answers</dt><dd>{refs(summary["invalid"])}</dd>'
        "</dl></div>"
    )


def card_html(r: dict, gold: dict | None = None) -> str:
    ev = "".join(f'<div class="tl-ev"><span class="tl-sid">S{e["id"]}</span><span>{escape(e["text"])}</span></div>'
                 for e in r["evidence"]) or '<div class="tl-ev none">No sentence cited</div>'
    unsupported = r["label"] in ("included", "not included", "excluded", "not excluded") and not r["evidence"]
    flag = '<div class="tl-flag">Decision without cited evidence. Check the note manually.</div>' if unsupported else ""
    meta = f'{r["prompt_tokens"]} prompt tok · {r["new_tokens"]} generated · {r["seconds"]:.2f} s'
    if r.get("acceptance_rate") is not None:
        meta += f' · draft acceptance {r["acceptance_rate"]:.0%}'
    if r["dropped_ids"]:
        meta += f' · dropped IDs {r["dropped_ids"]}'
    gold_html = ""
    if gold:
        same = gold["label"] == r["label"]
        ev_ids = ", ".join(f"S{i}" for i in gold["evidence_ids"]) or "none"
        gold_html = (f'<div class="tl-gold">Expert: <b>{escape(gold["label"])}</b> · evidence {ev_ids}'
                     f'<span class="tl-agree {"yes" if same else "no"}">{"Agrees" if same else "Differs"}</span></div>')
    # The cited sentences are already shown above, so the card keeps only the verdict sentence.
    explanation = (r["explanation"].split(", based on:")[0].split(" Related sentences:")[0]
                   .split(", but the model cited no sentence")[0])
    if not explanation.endswith("."):
        explanation += "."
    return (
        f'<div class="tl-card"><div class="tl-card-head"><span class="tl-idx">{r["index"] + 1:02d}</span>'
        f'<span class="tl-kind">{escape(r["type"])}</span>{label_badge(r["label"])}</div>'
        f'<div class="tl-crit">{escape(r["criterion"])}</div>{ev}'
        f'<div class="tl-expl">{escape(explanation)}</div>{flag}'
        f'<div class="tl-foot">{gold_html or "<span></span>"}<span class="tl-meta">{meta}</span></div>'
        f'<details class="tl-raw"><summary>Raw model output</summary><pre>{escape(r["raw_output"])}</pre></details></div>'
    )


def results_html(results: list[dict], gold_by_text: dict | None = None) -> str:
    if not results:
        return '<div class="tl-empty">Criterion results appear here, one card per criterion.</div>'
    gold_by_text = gold_by_text or {}
    return "".join(card_html(r, gold_by_text.get(r["criterion"])) for r in results)


def note_html(sentences: list[dict], results: list[dict]) -> str:
    cited = {}
    for r in results:
        for i in r["evidence_ids"]:
            cited.setdefault(i, []).append(r["index"] + 1)
    rows = []
    for s in sentences:
        ids = cited.get(s["id"])
        by = f'<span class="by">Cited by {", ".join(f"#{i}" for i in ids)}</span>' if ids else ""
        rows.append(f'<div class="row{" cited" if ids else ""}"><span class="sid">S{s["id"]}</span>'
                    f'<span>{escape(s["text"])}{by}</span></div>')
    return f'<div class="tl-note">{"".join(rows)}</div>'


def report_html(report: dict) -> str:
    p = report["provenance"]
    trial = report.get("trial") or {}
    prov_rows = "".join(f"<tr><th>{escape(k)}</th><td>{escape(json.dumps(v) if isinstance(v, dict) else str(v))}</td></tr>"
                        for k, v in p.items())
    title = f'{escape(trial.get("nct_id", ""))} {escape(trial.get("title", ""))}'.strip()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>TrialLens screening report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
{CSS}
body{{font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:var(--tl-ink);background:#fff}}
h1{{font-size:24px;font-weight:650;letter-spacing:-.02em;margin:0}}
h2{{font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--tl-ink-3);margin:28px 0 8px;border-bottom:1px solid var(--tl-line);padding-bottom:6px}}
.sub{{color:var(--tl-ink-2);font-size:14px;margin:4px 0 14px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{text-align:left;border-bottom:1px solid var(--tl-line);padding:5px 8px;vertical-align:top}}
th{{color:var(--tl-ink-3);font-weight:500;width:28%}}
td{{font-family:var(--tl-mono);font-size:11.5px;word-break:break-all}}
details.tl-raw{{display:none}}
@media print{{.tl-card{{break-inside:avoid}}}}
</style></head><body>
<h1>Trial<span style="color:var(--tl-accent)">Lens</span> screening report</h1>
<div class="sub">{title or "Criteria entered manually"} · generated {escape(report["created"])}</div>
{notice_html(report["disclaimer"])}
{summary_html(report["summary"], report["results"])}
<h2>Criteria</h2>
{results_html(report["results"])}
<h2>Patient note</h2>
<p style="font-size:12px;color:var(--tl-ink-3)">SHA-256 of the note as entered: <span style="font-family:var(--tl-mono)">{escape(report["patient_note_sha256"])}</span></p>
{note_html(report["sentences"], report["results"])}
<h2>Provenance</h2>
<table>{prov_rows}</table>
</body></html>"""
