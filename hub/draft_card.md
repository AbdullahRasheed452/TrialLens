---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
datasets:
- ncbi/TrialGPT-Criterion-Annotations
language:
- en
tags:
- lora
- speculative-decoding
- draft-model
---

# TrialLens draft adapter for Qwen2.5-0.5B-Instruct

The draft model for TrialLens' greedy speculative decoding. It was fine-tuned on the same training split and with the same recipe as the [target adapter](https://huggingface.co/Abdullah452/triallens-qwen2.5-1.5b-lora) (LoRA r=16, 8.8M parameters, epoch 3 of 4, validation accuracy 0.738), so its proposals match the target's answer format.

On the criterion task it raised draft acceptance from 60 to 83% (stock 0.5B) to 97 to 100%. Speculative decoding never changes the output: every committed token is verified by the 1.5B target.

> Educational prototype, not for clinical use. Code: https://github.com/AbdullahRasheed452/TrialLens
