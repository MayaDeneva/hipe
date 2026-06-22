import csv
import shutil
from pathlib import Path
from hipe.runs.registry import LEADERBOARD_COLUMNS


def ingest_output(download_dir, runs_root) -> list:
    download = Path(download_dir)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    src_runs = download / "runs"
    copied = []
    if not src_runs.is_dir():
        return copied
    for child in sorted(src_runs.iterdir()):
        if child.is_dir():
            dst = runs_root / child.name
            if not dst.exists():
                shutil.copytree(child, dst)
                copied.append(child.name)   # only newly-copied runs
    _merge_leaderboard(src_runs / "leaderboard.csv", runs_root / "leaderboard.csv")
    return copied


def _merge_leaderboard(src_csv, dst_csv):
    src_csv = Path(src_csv)
    dst_csv = Path(dst_csv)
    if not src_csv.exists():
        return
    rows = {}
    if dst_csv.exists():
        for r in csv.DictReader(dst_csv.open(encoding="utf-8")):
            rows[r["run_id"]] = r
    for r in csv.DictReader(src_csv.open(encoding="utf-8")):
        rows[r["run_id"]] = r              # newest wins, deduped by run_id
    with dst_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in LEADERBOARD_COLUMNS})
