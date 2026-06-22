import csv
from pathlib import Path
from hipe.kaggle.ingest import ingest_output


def _make_download(root, run_id, global_score):
    runs = root / "runs" / run_id
    runs.mkdir(parents=True)
    (runs / "manifest.json").write_text("{}")
    lb = root / "runs" / "leaderboard.csv"
    with lb.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "timestamp", "model", "config_hash", "data",
                    "at_recall", "isAt_recall", "global", "n_dev", "notes"])
        w.writerow([run_id, "t", "xlmr", "h", "d", 0.5, 0.6, global_score, 100, ""])


def test_ingest_copies_run_and_merges_leaderboard(tmp_path):
    dl = tmp_path / "dl"
    _make_download(dl, "2026-06-22_120000_xlmr_abcd1234", 0.55)
    runs_root = tmp_path / "runs"
    copied = ingest_output(dl, runs_root)
    assert copied == ["2026-06-22_120000_xlmr_abcd1234"]
    assert (runs_root / "2026-06-22_120000_xlmr_abcd1234" / "manifest.json").exists()
    rows = list(csv.DictReader(open(runs_root / "leaderboard.csv")))
    assert rows[0]["model"] == "xlmr" and rows[0]["global"] == "0.55"


def test_ingest_is_idempotent_by_run_id(tmp_path):
    dl = tmp_path / "dl"
    _make_download(dl, "2026-06-22_120000_xlmr_abcd1234", 0.55)
    runs_root = tmp_path / "runs"
    ingest_output(dl, runs_root)
    ingest_output(dl, runs_root)               # second time: no duplicate row
    rows = list(csv.DictReader(open(runs_root / "leaderboard.csv")))
    assert len(rows) == 1


def test_ingest_returns_only_newly_copied(tmp_path):
    dl = tmp_path / "dl"
    _make_download(dl, "2026-06-22_120000_xlmr_abcd1234", 0.55)
    runs_root = tmp_path / "runs"
    assert ingest_output(dl, runs_root) == ["2026-06-22_120000_xlmr_abcd1234"]
    assert ingest_output(dl, runs_root) == []   # already present -> nothing newly copied
