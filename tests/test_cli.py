# tests/test_cli.py
from pathlib import Path
import yaml
from hipe.cli import main

FIX = Path(__file__).resolve().parent / "fixtures" / "mini.jsonl"


def test_cli_run_creates_leaderboard(tmp_path, monkeypatch, capsys):
    cfg = {"data": {"train": str(FIX), "dev_frac": 0.5, "seed": 0},
           "model": {"name": "majority"}}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HIPE_RUNS_DIR", str(tmp_path / "runs"))
    rc = main(["run", str(cfg_path)])
    assert rc == 0
    assert (tmp_path / "runs" / "leaderboard.csv").exists()
    assert "global" in capsys.readouterr().out


def test_cli_score(capsys):
    fixdir = Path(__file__).resolve().parent / "fixtures"
    rc = main(["score", str(fixdir / "gold.jsonl"), str(fixdir / "pred.jsonl")])
    assert rc == 0
    assert "0.75" in capsys.readouterr().out
