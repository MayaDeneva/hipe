# hipe/kaggle/metadata.py
import json
import os
import subprocess
from pathlib import Path

DATASET_SLUG = "hipe-code"


def _username_from_cli():
    """Read the username the kaggle CLI is authenticated as.

    Handles the modern ACCESS_TOKEN auth (~/.kaggle/access_token), which stores
    no username/key json — but `kaggle config view` still reports the username.
    """
    try:
        out = subprocess.run(["kaggle", "config", "view"],
                             capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.stdout.splitlines():
        if "username:" in line:
            name = line.split("username:", 1)[1].strip()
            if name and name.lower() != "none":
                return name
    return None


def kaggle_username() -> str:
    env = os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    cfg = Path.home() / ".kaggle" / "kaggle.json"      # legacy username/key file
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding="utf-8"))["username"]
        except (json.JSONDecodeError, KeyError):
            pass
    name = _username_from_cli()                          # modern ACCESS_TOKEN auth
    if name:
        return name
    raise RuntimeError(
        "No Kaggle username: confirm `kaggle config view` shows a username, "
        "or set $KAGGLE_USERNAME")


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
