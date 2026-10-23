# HTML rendering for the app and the downloadable report. Everything user supplied is
# escaped before it reaches the page.

from __future__ import annotations

import json
from html import escape

LABEL_STYLE = {
    "included": "ok", "not excluded": "ok",
    "not included": "bad", "excluded": "bad",
    "not enough information": "warn", "not applicable": "muted", "invalid": "err",
}
OUTCOME_STYLE = {"Potentially eligible": "ok", "Needs review": "warn", "Likely ineligible": "bad"}

CSS = """
.tl-banner{border:1px solid #d97706;background:rgba(217,119,6,.10);border-radius:8px;padding:10px 14px;margin:4px 0 8px;font-size:14px}
.tl-summary{border-radius:10px;padding:14px 16px;margin:6px 0;border:1px solid var(--border-color-primary)}
.tl-summary h3{margin:0 0 4px;font-size:20px}
.tl-summary p{margin:2px 0;font-size:14px}
.tl-card{border:1px solid var(--border-color-primary);border-radius:10px;padding:12px 14px;margin:8px 0}
.tl-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.tl-num{font-weight:600;opacity:.7}
.tl-chip{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;border:1px solid transparent}
.tl-type{border-color:var(--border-color-primary);opacity:.85}
.ok{background:rgba(22,163,74,.15);color:#15803d;border-color:rgba(22,163,74,.4)}
.bad{background:rgba(220,38,38,.13);color:#b91c1c;border-color:rgba(220,38,38,.4)}
.warn{background:rgba(217,119,6,.15);color:#b45309;border-color:rgba(217,119,6,.4)}
.muted{background:rgba(100,116,139,.15);color:#475569;border-color:rgba(100,116,139,.4)}
.err{background:rgba(147,51,234,.13);color:#7e22ce;border-color:rgba(147,51,234,.4)}
@media (prefers-color-scheme: dark){.ok{color:#4ade80}.bad{color:#f87171}.warn{color:#fbbf24}.muted{color:#cbd5e1}.err{color:#c084fc}}
.dark .ok{color:#4ade80}.dark .bad{color:#f87171}.dark .warn{color:#fbbf24}.dark .muted{color:#cbd5e1}.dark .err{color:#c084fc}
.tl-crit{font-size:15px;margin:4px 0 8px}
.tl-ev{border-left:3px solid var(--border-color-primary);padding:2px 10px;margin:4px 0;font-size:14px}
.tl-ev b{opacity:.75}
.tl-expl{font-size:13.5px;opacity:.9;margin-top:6px}
.tl-meta{font-size:12px;opacity:.65;margin-top:6px}
.tl-gold{font-size:12.5px;margin-top:4px}
.tl-note{font-size:14px;line-height:1.55}
.tl-note div{padding:3px 8px;border-radius:6px;margin:1px 0}
.tl-note .cited{background:rgba(37,99,235,.13);border-left:3px solid #2563eb}
.tl-note .ids{font-size:11.5px;opacity:.7;margin-left:6px}
details.tl-raw summary{cursor:pointer;font-size:12px;opacity:.7}
details.tl-raw pre{white-space:pre-wrap;font-size:12px;margin:4px 0}
"""


def chip(text: str, style: str) -> str:
    return f'<span class="tl-chip {style}">{escape(text)}</span>'


def disclaimer_html(text: str) -> str:
    return f'<div class="tl-banner">⚠️ <b>{escape(text)}</b></div>'


def summary_html(summary: dict, results: list[dict]) -> str:
    style = OUTCOME_STYLE.get(summary["outcome"], "muted")
    counts = " · ".join(f"{escape(k)}: {v}" for k, v in sorted(summary["counts"].items()))
    def refs(idx):
        return ", ".join(f"#{i + 1}" for i in idx) or "none"
    lines = [
        f'<div class="tl-summary {style}"><h3>Preliminary outcome: {escape(summary["outcome"])}</h3>',
        f'<p>{escape(summary["rule"])} <i>{escape(summary["note"])}</i></p>',
        f"<p>{counts}</p>",
        f"<p>Ruling out: {refs(summary['blocking'])} · Missing information: {refs(summary['missing_information'])}"
        f" · Unreadable answers: {refs(summary['invalid'])} · Decisions without cited evidence: "
        f"{refs(summary['unsupported'])}</p></div>",
    ]
    return "".join(lines)


def card_html(r: dict, gold: dict | None = None) -> str:
    ev = "".join(f'<div class="tl-ev"><b>[{e["id"]}]</b> {escape(e["text"])}</div>' for e in r["evidence"])
    if not ev:
        ev = '<div class="tl-ev"><i>No sentence cited.</i></div>'
    meta = (f'{r["prompt_tokens"]} prompt tokens · {r["new_tokens"]} generated · {r["seconds"]:.2f}s'
            + (f' · draft acceptance {r["acceptance_rate"]:.0%}' if r.get("acceptance_rate") is not None else "")
            + (f' · dropped invalid IDs {r["dropped_ids"]}' if r["dropped_ids"] else ""))
    gold_line = ""
    if gold:
        same = gold["label"] == r["label"]
        gold_line = (f'<div class="tl-gold">Expert annotation (validation example): <b>{escape(gold["label"])}</b>, '
                     f'evidence {gold["evidence_ids"] or "none"} {"✓" if same else "✗ label differs"}</div>')
    return (
        f'<div class="tl-card"><div class="tl-head"><span class="tl-num">#{r["index"] + 1}</span>'
        f'{chip(r["type"], "tl-type")}{chip(r["label"], LABEL_STYLE.get(r["label"], "muted"))}</div>'
        f'<div class="tl-crit">{escape(r["criterion"])}</div>{ev}'
        f'<div class="tl-expl">{escape(r["explanation"])}</div>{gold_line}'
        f'<div class="tl-meta">{meta}</div>'
        f'<details class="tl-raw"><summary>Raw model output</summary><pre>{escape(r["raw_output"])}</pre></details></div>'
    )


def results_html(results: list[dict], gold_by_text: dict | None = None) -> str:
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
        tag = f'<span class="ids">cited by {", ".join(f"#{i}" for i in ids)}</span>' if ids else ""
        rows.append(f'<div class="{"cited" if ids else ""}"><b>{s["id"]}.</b> {escape(s["text"])}{tag}</div>')
    return f'<div class="tl-note">{"".join(rows)}</div>'


def report_html(report: dict) -> str:
    p = report["provenance"]
    trial = report.get("trial") or {}
    prov_rows = "".join(f"<tr><th>{escape(k)}</th><td>{escape(json.dumps(v) if isinstance(v, dict) else str(v))}</td></tr>"
                        for k, v in p.items())
    title = f'{escape(trial.get("nct_id", ""))} {escape(trial.get("title", ""))}'.strip()
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>TrialLens screening report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--border-color-primary:#d4d4d8}}
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#18181b;background:#fff}}
h1{{font-size:24px;margin-bottom:4px}} h2{{font-size:18px;margin-top:28px;border-bottom:1px solid #e4e4e7;padding-bottom:4px}}
table{{border-collapse:collapse;font-size:13px;width:100%}} th,td{{text-align:left;border-bottom:1px solid #e4e4e7;padding:4px 6px;vertical-align:top}}
details.tl-raw{{display:none}}
@media print{{.tl-card{{break-inside:avoid}}}}
{CSS}
</style></head><body>
<h1>TrialLens screening report</h1>
<p>{title or "Trial criteria entered manually"} · created {escape(report["created"])}</p>
{disclaimer_html(report["disclaimer"])}
{summary_html(report["summary"], report["results"])}
<h2>Criterion results</h2>
{results_html(report["results"])}
<h2>Patient note</h2>
<p style="font-size:12px;opacity:.7">SHA-256 of the note as entered: {escape(report["patient_note_sha256"])}</p>
{note_html(report["sentences"], report["results"])}
<h2>Provenance</h2>
<table>{prov_rows}</table>
</body></html>"""
