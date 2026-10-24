---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
datasets:
- ncbi/TrialGPT-Criterion-Annotations
language:
- en
pipeline_tag: text-generation
tags:
- lora
- clinical-trials
- eligibility-screening
- evidence-attribution
---

# TrialLens LoRA adapter for Qwen2.5-1.5B-Instruct

Criterion-level clinical trial eligibility screening with sentence-level evidence. Given a numbered patient note, a criterion type and the criterion text, the model answers with one JSON object:

```json
{"evidence_ids": [0, 2], "label": "included"}
```

Labels follow the TrialGPT expert scheme: *included / not included / not enough information / not applicable* for inclusion criteria and *excluded / not excluded / not enough information / not applicable* for exclusion criteria.

> **Educational screening prototype.** Not a clinical enrollment tool. Every output must be checked by a qualified person.

Code, prompt format, application and paper: https://github.com/AbdullahRasheed452/TrialLens

## Results

Held-out test split (8 patients, 206 criteria), greedy decoding, same prompt before and after fine-tuning.

| | Accuracy | Macro-F1 | Evidence precision | Evidence recall | Evidence F1 |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct, zero-shot | 0.150 | 0.163 | 0.211 | 0.202 | 0.207 |
| + this adapter | 0.723 | 0.561 | 0.846 | 0.351 | 0.496 |

The model never predicts *not included* correctly (7 test cases) and tends to answer *not excluded* when an exclusion condition is simply not mentioned, following the majority annotation convention. See the paper's error analysis.

## Training

| | |
|---|---|
| Data | `ncbi/TrialGPT-Criterion-Annotations`, 37 training patients (602 criteria) from a sorted, `default_rng(42)`-shuffled 37/8/8 patient split; GPT-4 columns never used |
| Method | LoRA r=16, alpha=32, dropout 0.05 on q, k, v, o, gate, up, down projections (18.5M parameters) |
| Objective | cross-entropy on the answer tokens only |
| Imbalance | labels with fewer than 40 training examples oversampled to 40 per epoch |
| Optimiser | AdamW, lr 2e-4, cosine schedule, 5% warm-up, effective batch 16 |
| Selection | epoch 2 of 4, best validation macro-F1 (0.564) |
| Hardware | one Tesla T4, about 36 minutes |

## Use

The easiest way is the TrialLens app (`start.bat` downloads this adapter). Directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "Abdullah452/triallens-qwen2.5-1.5b-lora").merge_and_unload()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

Build the prompt with `triallens.prompts.build_messages` from the repository so the system message and format match training.
