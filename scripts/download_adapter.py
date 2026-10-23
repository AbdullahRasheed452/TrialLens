# Download the fine-tuned LoRA adapter from the Hugging Face Hub and mark it as the one the
# app should load.
#
# Usage: python scripts/download_adapter.py [--repo HF_REPO_ID] [--name RUN_NAME]

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_REPO = "AbdullahRasheed452/triallens-qwen2.5-1.5b-lora"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ADAPTER_REPO)
    ap.add_argument("--name", default="hub")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    target = ROOT / "checkpoints" / args.name / "lora-best"
    try:
        snapshot_download(repo_id=args.repo, local_dir=target)
    except Exception as e:
        print(f"Could not download {args.repo}: {e}")
        print("The app will fall back to any adapter already in checkpoints/, or the base model.")
        sys.exit(1)
    if not (target / "adapter_config.json").exists():
        sys.exit(f"{args.repo} does not contain adapter_config.json")
    (ROOT / "checkpoints/BEST").write_text(target.relative_to(ROOT).as_posix(), encoding="utf-8")
    print(f"Adapter saved to {target} and set as the default.")


if __name__ == "__main__":
    main()
