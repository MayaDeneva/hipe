# hipe/kaggle/metadata.py
import json
import os
from pathlib import Path

DATASET_SLUG = "hipe-code"


def kaggle_username() -> str:
    env = os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8"))["username"]
    raise RuntimeError(
        "No Kaggle username: set $KAGGLE_USERNAME or create ~/.kaggle/kaggle.json")


def dataset_metadata(username: str, slug: str = DATASET_SLUG) -> dict:
    return {"title": slug, "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}]}


def kernel_slug(config_path) -> str:
    return ("hipe-run-" + Path(config_path).stem).replace("_", "-")


def kernel_metadata(username, k_slug, code_file, enable_gpu=True, enable_internet=True) -> dict:
    return {
        "id": f"{username}/{k_slug}",
        "title": k_slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
