# Build the fixed 37/8/8 patient-grouped split and write the manifest.
#
# Usage: python scripts/make_split.py

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triallens.data import build_splits, load_raw, write_jsonl  # noqa: E402


def main() -> None:
    df = load_raw(ROOT / "data/raw/criterion_annotations.parquet")
    records, manifest = build_splits(df)
    for name, recs in records.items():
        write_jsonl(recs, ROOT / f"data/processed/{name}.jsonl")
    out = ROOT / "data/splits/split_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, s in manifest["splits"].items():
        print(f"{name:5s} patients={s['n_patients']:2d} annotations={s['n_annotations']:4d} "
              f"incl={s['n_inclusion']:3d} excl={s['n_exclusion']:3d} empty_evidence={s['n_empty_evidence']}")
        print("      ", s["label_counts"])
    print(f"manifest written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
