# Model loading and criterion-level prediction.
#
# Evaluation uses plain greedy decoding (do_sample=False) so runs are deterministic and
# the before/after fine-tuning comparison only changes the weights, never the decoding.

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .parsing import parse_prediction
from .prompts import build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DECODING = {"do_sample": False, "num_beams": 1, "max_new_tokens": 64}


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# fp16 on GPU (T4/P100 have no fast bf16), fp32 on CPU where half precision is slow.
def load_model(model_id: str = DEFAULT_MODEL, adapter: str | None = None, device: str | None = None):
    device = device or pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def prompt_ids(tok, sentences, criterion_type, criterion_text, device) -> torch.Tensor:
    text = tok.apply_chat_template(
        build_messages(sentences, criterion_type, criterion_text), add_generation_prompt=True, tokenize=False
    )
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


@torch.inference_mode()
def predict(model, tok, record: dict, max_new_tokens: int = DECODING["max_new_tokens"]) -> dict:
    device = next(model.parameters()).device
    ids = prompt_ids(tok, record["sentences"], record["criterion_type"], record["criterion_text"], device)
    start = time.perf_counter()
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    elapsed = time.perf_counter() - start
    new_tokens = out[0, ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    valid_ids = {s["id"] for s in record["sentences"]}
    pred = parse_prediction(text, record["criterion_type"], valid_ids)
    pred.update(
        annotation_id=record["annotation_id"],
        prompt_tokens=int(ids.shape[1]),
        new_tokens=int(new_tokens.shape[0]),
        seconds=round(elapsed, 3),
    )
    return pred
