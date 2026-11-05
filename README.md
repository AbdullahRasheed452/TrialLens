# TrialLens

Evidence-grounded clinical trial eligibility screening with a fine-tuned Qwen2.5 LLM, featuring hand-built KV caching, tiled online-softmax attention, and speculative decoding.

> **Educational screening prototype.** TrialLens is a research project and does not make clinical enrollment decisions. Every result must be checked by a qualified person.

**Paper:** [TrialLens_paper.pdf](paper/TrialLens_paper.pdf) (IEEE format, LaTeX source in [`paper/`](paper))

**Demo video:** [youtu.be/StV4EAdW9mc](https://youtu.be/StV4EAdW9mc) (5.5 minutes, recorded on a laptop CPU)

TrialLens reads a patient note, a criterion type (inclusion or exclusion) and the criterion text, and returns one of the expert label categories together with the IDs of the patient sentences that support it. A browser app runs this over every criterion of a trial, highlights the evidence, gives a rule-based preliminary outcome and exports a report.

## Results on the held-out test patients

Same model, same prompt, greedy decoding, 206 criteria from 8 unseen patients.

| | Accuracy | Macro-F1 | Evidence P | Evidence R | Evidence F1 |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct, zero-shot | 0.150 | 0.163 | 0.211 | 0.202 | 0.207 |
| **+ LoRA (os40), this project** | **0.723** | **0.561** | **0.846** | **0.351** | **0.496** |

| Mechanism | Verified | Measured |
|---|---|---|
| KV-cached decoding | logits identical to Hugging Face; tokens equal to the uncached path | up to 6.2x faster on a T4, 8.1x on a laptop CPU |
| Tiled online-softmax attention | max error 1.7e-6 vs float64 (fp32) | 26x less peak memory than the reference at 4k tokens |
| Greedy speculative decoding | tokens equal to greedy in 200/200 GPU runs | 1.51x faster for long outputs on CPU; slower on GPU (explained in the paper) |

Full numbers, plots and the error analysis are in the [paper](paper/main.tex) and [`results/`](results).

## Quick start (Windows)

1. Install Python 3.10 or newer (or [uv](https://docs.astral.sh/uv/)).
2. Double-click **`start.bat`**.

The first run creates `.venv`, installs the pinned packages, rebuilds the dataset split and downloads the fine-tuned adapter, which takes a few minutes. Every later run opens the app at **http://127.0.0.1:7860** in about 20 seconds. Nothing else has to be run by hand.

Requirements: 16 GB RAM (the model needs about 6 GB in fp32, the optional speculative draft another 2 GB), about 6 GB of disk for the model download. No GPU is needed; a criterion takes about 6 to 8 seconds on a 4-core laptop CPU.

### Other platforms

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_split.py
python scripts/download_adapter.py
python app/app.py                         # add --port 7861 or --adapter PATH if needed
```

## Using the app

**Screen a patient.** Paste a note (numbered lines `0. ...` keep their numbers, free text is split into sentences automatically), then type the criteria under `Inclusion criteria:` and `Exclusion criteria:` headers or fetch them with a ClinicalTrials.gov ID such as `NCT01397994`. The dropdown loads validation patients with their expert labels for comparison; test patients never appear in the app. Results stream in one criterion at a time. Each card shows the label, cited sentences, an explanation built only from those sentences, and a warning when a decision has no cited evidence. Download the report as HTML or JSON; it records the model, adapter hash, decoding settings, git commit and a hash of the note.

**Preliminary outcome.** Aggregated by fixed rules, not a ground-truth label: any *not included* or *excluded* gives **Likely ineligible**; otherwise any *not enough information* or unreadable answer gives **Needs review**; otherwise **Potentially eligible**.

**Inference lab.** Runs no-cache, KV-cache, shared-prefix and speculative decoding live on your machine, checks that they produce identical tokens, and compares the three attention implementations against a float64 result. The recorded GPU benchmark logs are listed underneath.

**Invalid input** (empty note, criteria without headers, a malformed trial ID, an over-long note) shows a message and the app keeps running.

## Repository map

| Assignment part | Code | Results |
|---|---|---|
| Dataset, fixed split, leakage checks | [`src/triallens/data.py`](src/triallens/data.py), [`scripts/make_split.py`](scripts/make_split.py), [`tests/test_split.py`](tests/test_split.py) | [`data/splits/split_manifest.json`](data/splits/split_manifest.json) |
| Prompt, output schema, parsing | [`src/triallens/prompts.py`](src/triallens/prompts.py), [`src/triallens/parsing.py`](src/triallens/parsing.py) | [`results/data/token_lengths.json`](results/data/token_lengths.json) |
| Fine-tuning (LoRA) | [`scripts/train_lora.py`](scripts/train_lora.py), [`notebooks/`](notebooks) | [`results/train/`](results/train), [`model_selection.json`](results/train/model_selection.json) |
| Evaluation and metrics | [`scripts/evaluate.py`](scripts/evaluate.py), [`src/triallens/metrics.py`](src/triallens/metrics.py) | [`results/eval/`](results/eval) |
| Error analysis | [`scripts/error_analysis.py`](scripts/error_analysis.py) | [`results/analysis/`](results/analysis) |
| KV-cached decoder (Task 3) | [`src/triallens/decoder.py`](src/triallens/decoder.py), [`tests/test_decoder.py`](tests/test_decoder.py), [`scripts/bench_kv_cache.py`](scripts/bench_kv_cache.py) | [`results/bench/kv_cache_*.json`](results/bench) |
| Tiled attention (Task 4) | [`src/triallens/attention.py`](src/triallens/attention.py), [`tests/test_attention.py`](tests/test_attention.py), [`scripts/bench_attention.py`](scripts/bench_attention.py) | [`results/bench/attention_*.json`](results/bench) |
| Speculative decoding (Task 5) | [`src/triallens/speculative.py`](src/triallens/speculative.py), [`tests/test_speculative.py`](tests/test_speculative.py), [`scripts/bench_speculative.py`](scripts/bench_speculative.py) | [`results/bench/speculative_*.json`](results/bench) |
| Application (Task 6) | [`app/`](app), [`src/triallens/screening.py`](src/triallens/screening.py), [`start.bat`](start.bat) | |
| Paper | [`paper/`](paper) | [`paper/figures/`](paper/figures) |

## Dataset and split

All training and evaluation use [ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) (1,015 criterion annotations, 53 patients). Unique patient IDs are sorted, shuffled with `numpy.random.default_rng(42)` and assigned 37 to train, 8 to validation and 8 to test, so every annotation of a patient lives in exactly one split.

| Split | Patients | Annotations | Inclusion | Exclusion |
|---|---|---|---|---|
| Train | 37 | 602 | 215 | 387 |
| Validation | 8 | 206 | 66 | 140 |
| Test | 8 | 206 | 75 | 131 |

Only the note, criterion type, criterion text, expert label and expert evidence are used. The `gpt4_*` columns, `explanation_correctness` and the dataset's own `training` flag are dropped through an explicit allow-list, and the tests check they never reach a processed record. Annotation 883 has an empty criterion text and is excluded; this is recorded in the manifest. The test split was evaluated once, after the model was chosen on validation; `scripts/evaluate.py` refuses the test split unless `--final` is passed.

## Model and weights

| | |
|---|---|
| Base model | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.54B parameters, Apache-2.0, 32k context) |
| Adapter | LoRA r=16, alpha=32 on all attention and MLP projections, 18.5M trainable parameters, rare labels oversampled to 40 |
| Selected by | best validation macro-F1 (rule fixed before training) |
| Download | `python scripts/download_adapter.py`, or [Abdullah452/triallens-qwen2.5-1.5b-lora](https://huggingface.co/Abdullah452/triallens-qwen2.5-1.5b-lora) on the Hugging Face Hub |
| Draft model | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) with the fine-tuned [draft adapter](https://huggingface.co/Abdullah452/triallens-qwen2.5-0.5b-draft-lora), also fetched by the download script |

## Reproducing the results

Everything below runs from the repository root inside the virtual environment.

```bash
python scripts/make_split.py          # split + manifest (identical on every machine)
python -m pytest                      # 106 tests: split, metrics, parser, attention, decoder, speculative, app logic
python scripts/analyze_lengths.py     # token lengths and truncation check
```

Training and the GPU measurements ran on free Kaggle T4 GPUs through three notebooks. Each clones this repository and calls the scripts, so the notebooks contain no logic of their own.

| Notebook | Runs | Time on Kaggle |
|---|---|---|
| [`kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) | zero-shot baseline on validation, LoRA r16, fine-tuned evaluation | about 45 min |
| [`kaggle_experiments.ipynb`](notebooks/kaggle_experiments.ipynb) | LoRA os40 and r8 in parallel on two T4s | about 45 min |
| [`kaggle_bench.ipynb`](notebooks/kaggle_bench.ipynb) | attention, KV-cache and speculative benchmarks, draft fine-tuning, final test evaluation | about 1 to 1.5 h |

On a CUDA machine the same commands work directly, for example:

```bash
python scripts/train_lora.py --run-name os40 --epochs 4 --oversample-floor 40
python scripts/evaluate.py --split val --name os40 --adapter checkpoints/os40/lora-best
python scripts/bench_kv_cache.py --adapter checkpoints/os40/lora-best
python scripts/bench_attention.py
python scripts/bench_speculative.py --adapter checkpoints/os40/lora-best
```

Figures and the error analysis are generated from the committed result files:

```bash
python scripts/make_figures.py        # paper/figures/*.pdf
python scripts/error_analysis.py      # results/analysis/os40_test_error_cases.json
```

The compiled paper is [`paper/TrialLens_paper.pdf`](paper/TrialLens_paper.pdf). To rebuild it, run `pdflatex main && bibtex main && pdflatex main && pdflatex main` inside `paper/` (MiKTeX or TeX Live), or upload the folder to Overleaf.

## Troubleshooting

| Problem | Fix |
|---|---|
| `start.bat` closes immediately | Run it from a terminal (`cmd`, then `start.bat`) to see the message. |
| `running scripts is disabled` when activating the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in PowerShell, or skip activation and call `.venv\Scripts\python.exe` directly. |
| `An Application Control policy has blocked this file` | Windows Smart App Control blocked an unsigned file. Its verdicts come from an online reputation check and can change, so first simply run `start.bat` again. If `.venv\Scripts\python.exe` itself is blocked, replace that small `uv` launcher with the real interpreter: copy `python.exe`, `pythonw.exe`, `python3.dll`, `python311.dll`, `vcruntime140.dll` and `vcruntime140_1.dll` from the folder named in `.venv\pyvenv.cfg` (`home = ...`) into `.venv\Scripts`. Turning Smart App Control off (Windows Security, App & browser control) removes the problem entirely. The project avoids scikit-learn because its DLLs were blocked. |
| Port 7860 already in use | `start.bat --port 7861` |
| Out of memory | Close other programs; avoid the speculative mode, which loads a second model. |
| Adapter download fails | The app falls back to any adapter in `checkpoints/` and otherwise runs the base model with a warning. |

## AI use

Claude Code (Anthropic) was used as a programming and writing assistant. The design decisions, experiments and verification are the author's; see the AI-use appendix of the paper.
