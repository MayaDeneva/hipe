import csv
import hashlib
import json
from pathlib import Path

LEADERBOARD_COLUMNS = ["run_id", "timestamp", "model", "config_hash", "data",
                       "at_recall", "isAt_recall", "global", "n_dev", "notes"]


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def new_run_dir(model_name, cfg_hash, root, now) -> Path:
    run_dir = Path(root) / f"{now}_{model_name}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir, manifest: dict) -> None:
    (Path(run_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def append_leaderboard(root, row: dict) -> None:
    path = Path(root) / "leaderboard.csv"
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEADERBOARD_COLUMNS})
