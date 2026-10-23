# TrialLens

Evidence-grounded clinical trial eligibility screening with a fine-tuned open-weight LLM, featuring hand-built KV-cached decoding, tiled online-softmax attention, and greedy speculative decoding.

> **Educational screening prototype.** TrialLens is a research project and is not a clinical enrollment decision tool.

TrialLens reads a patient note, a criterion type (inclusion or exclusion) and the criterion text, and returns one of the expert label categories along with the IDs of the patient sentences that support the decision.

## Status

| Component | State |
|---|---|
| Fixed 37/8/8 patient-grouped split and manifest | done |
| Token length and truncation analysis | done |
| Fine-tuning (Qwen2.5-1.5B-Instruct) | in progress |
| Evaluation pipeline | planned |
| KV-cached decoding | planned |
| Tiled online-softmax attention | planned |
| Speculative decoding (Qwen2.5-0.5B draft) | planned |
| Browser application | planned |

## Dataset and split

All training and evaluation use [ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) (1,015 criterion annotations over 53 patients). Unique patient IDs are sorted, shuffled with `numpy.random.default_rng(42)`, and assigned 37 to train, 8 to validation and 8 to test, so every annotation of a patient lives in exactly one split.

| Split | Patients | Annotations | Inclusion | Exclusion |
|---|---|---|---|---|
| Train | 37 | 602 | 215 | 387 |
| Validation | 8 | 206 | 66 | 140 |
| Test | 8 | 206 | 75 | 131 |

The full manifest with patient IDs, annotation IDs and label counts is in [`data/splits/split_manifest.json`](data/splits/split_manifest.json). One annotation (ID 883) has an empty criterion text and is excluded from every split; this is recorded in the manifest.

Only the patient note, criterion type, criterion text, expert label and expert evidence sentences are used. The `gpt4_*` prediction columns, `explanation_correctness` and the dataset's own `training` flag are dropped before any processing, and `tests/test_split.py` checks that they never appear in the processed records.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
```

Rebuild the split and run the checks:

```bash
python scripts/make_split.py
python -m pytest
python scripts/analyze_lengths.py
```

## Repository layout

```
src/triallens/     library code (data, prompts, decoding, attention, app)
scripts/           command line entry points
tests/             correctness tests
data/splits/       published split manifest
results/           metrics, benchmark logs and figures
```
