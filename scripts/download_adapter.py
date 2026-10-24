# Download the fine-tuned LoRA adapters from the Hugging Face Hub and mark them as the ones
# the app should load: the 1.5B target (checkpoints/BEST) and the 0.5B speculative draft
# (checkpoints/DRAFT).
#
# Usage: python scripts/download_adapter.py [--repo HF_REPO_ID] [--draft-repo HF_REPO_ID]

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_REPO = "Abdullah452/triallens-qwen2.5-1.5b-lora"
DRAFT_REPO = "Abdullah452/triallens-qwen2.5-0.5b-draft-lora"


def fetch(repo: str, name: str, pointer: str) -> bool:
    from huggingface_hub import snapshot_download

    target = ROOT / "checkpoints" / name / "lora-best"
    try:
        snapshot_download(repo_id=repo, local_dir=target, allow_patterns=["adapter_*", "*.json", "README.md"])
    except Exception as e:
        print(f"Could not download {repo}: {e}")
        return False
    if not (target / "adapter_config.json").exists():
        print(f"{repo} does not contain adapter_config.json")
        return False
    (ROOT / "checkpoints" / pointer).write_text(target.relative_to(ROOT).as_posix(), encoding="utf-8")
    print(f"{repo} saved to {target}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ADAPTER_REPO)
    ap.add_argument("--draft-repo", default=DRAFT_REPO)
    args = ap.parse_args()

    ok = fetch(args.repo, "hub", "BEST")
    if not fetch(args.draft_repo, "hub_draft", "DRAFT"):
        print("Speculative mode will use the stock Qwen2.5-0.5B draft.")
    if not ok:
        print("The app will fall back to any adapter already in checkpoints/, or the base model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
