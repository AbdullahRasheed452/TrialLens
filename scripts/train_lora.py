# LoRA fine-tuning of Qwen2.5-1.5B-Instruct on the training split.
#
# Usage (Kaggle / Colab GPU):
#   python scripts/train_lora.py --run-name r16 --epochs 4
#
# Objective: causal LM cross-entropy on the answer tokens only. Prompt tokens (system,
# patient note, criterion) get label -100, so the model learns to produce the JSON answer
# and is never trained to reproduce the patient note.
#
# Checkpoint selection: after every epoch the adapter is run with greedy decoding on the
# validation split, and the epoch with the best validation macro-F1 is kept as lora-best.
# The test split is never read here.

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

from triallens.data import read_jsonl  # noqa: E402
from triallens.inference import DEFAULT_MODEL, predict  # noqa: E402
from triallens.metrics import evaluate  # noqa: E402
from triallens.prompts import build_messages, target_json  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Tokenise prompt and answer separately so the loss mask lines up exactly with the answer.
def encode(tok, record: dict, max_len: int) -> dict:
    prompt = tok.apply_chat_template(
        build_messages(record["sentences"], record["criterion_type"], record["criterion_text"]),
        add_generation_prompt=True,
        tokenize=False,
    )
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target_json(record) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        raise ValueError(f"annotation {record['annotation_id']} exceeds max_len; raise --max-len")
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def collate(batch, pad_id: int) -> dict:
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# Repeat rare-label records so each label reaches at least `floor` examples per epoch.
def oversample(records: list[dict], floor: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    out = list(records)
    for recs in by_label.values():
        while sum(1 for r in out if r["label"] == recs[0]["label"]) < floor:
            out.append(rng.choice(recs))
    return out


@torch.no_grad()
def val_loss(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = (batch["labels"] != -100).sum().item()
        total += model(**batch).loss.item() * n
        count += n
    return total / count


def val_generation(model, tok, records) -> dict:
    model.eval()
    preds = [predict(model, tok, r) for r in records]
    return evaluate(records, preds), preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--oversample-floor", type=int, default=0, help="0 disables oversampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
    ))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    train = read_jsonl(ROOT / "data/processed/train.jsonl")
    val = read_jsonl(ROOT / "data/processed/val.jsonl")
    if args.oversample_floor:
        train = oversample(train, args.oversample_floor, args.seed)
    train_enc = [encode(tok, r, args.max_len) for r in train]
    val_enc = [encode(tok, r, args.max_len) for r in val]
    pad = lambda b: collate(b, tok.pad_token_id)
    train_loader = DataLoader(train_enc, batch_size=args.batch_size, shuffle=True, collate_fn=pad,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_enc, batch_size=args.batch_size, collate_fn=pad)

    steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)
    # Master weights stay fp32; the forward pass runs in fp16 autocast with loss scaling.
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    log = {"config": vars(args) | {"device": device, "n_train": len(train), "n_val": len(val),
                                   "trainable_params": trainable, "total_params": total,
                                   "optimizer_steps": steps},
           "steps": [], "epochs": []}
    best_f1, step = -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
            if i % args.grad_accum == 0 or i == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                log["steps"].append({"step": step, "epoch": epoch, "train_loss": running,
                                     "lr": sched.get_last_lr()[0]})
                if step % 5 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {running:.4f}")
                running = 0.0

        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            vloss = val_loss(model, val_loader, device)
            report, preds = val_generation(model, tok, val)
        c, e = report["overall"]["classification"], report["overall"]["evidence"]
        row = {"epoch": epoch, "step": step, "val_loss": vloss, "val_accuracy": c["accuracy"],
               "val_macro_f1": c["macro_f1"], "val_invalid_rate": c["invalid_rate"],
               "val_evidence_micro_f1": e["micro"]["f1"], "val_evidence_macro_f1": e["macro"]["f1"],
               "elapsed_s": round(time.perf_counter() - start, 1)}
        log["epochs"].append(row)
        print(json.dumps(row))
        model.save_pretrained(out_dir / f"epoch{epoch}")
        if c["macro_f1"] > best_f1:
            best_f1 = c["macro_f1"]
            model.save_pretrained(out_dir / "lora-best")
            (out_dir / "lora-best" / "val_metrics.json").write_text(json.dumps(report, indent=2))
            with (out_dir / "lora-best" / "val_predictions.jsonl").open("w") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
            log["best_epoch"] = epoch
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    if device == "cuda":
        log["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    log["total_seconds"] = round(time.perf_counter() - start, 1)
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"best epoch {log['best_epoch']} val macro-F1 {best_f1:.3f}; saved to {out_dir}")


if __name__ == "__main__":
    main()
