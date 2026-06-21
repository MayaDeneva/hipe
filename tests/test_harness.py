# tests/test_harness.py
from pathlib import Path
from hipe.harness import run_experiment

FIX = Path(__file__).resolve().parent / "fixtures" / "mini.jsonl"


def test_run_experiment_majority_end_to_end(tmp_path):
    config = {
        "data": {"train": str(FIX), "dev_frac": 0.5, "seed": 0},
        "model": {"name": "majority"},
    }
    result = run_experiment(config, now="2026-06-21_120000", runs_root=tmp_path)
    run_dir = Path(result["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "predictions" / "dev.jsonl").exists()
    assert 0.0 <= result["global"] <= 1.0
    # leaderboard row written
    lb = (tmp_path / "leaderboard.csv").read_text()
    assert "majority" in lb


def test_run_experiment_applies_consistency(tmp_path):
    # A model whose isAt=TRUE must yield at=TRUE in the written submission.
    from hipe.models import registry
    from hipe.models.base import RelationModel
    from hipe.data.load import read_jsonl

    @registry.register("force_isat_true")
    class ForceIsat(RelationModel):
        name = "force_isat_true"
        def fit(self, train, dev=None): pass
        def predict(self, pairs):
            return [{"at": "FALSE", "isAt": "TRUE"} for _ in pairs]

    config = {"data": {"train": str(FIX), "dev_frac": 1.0, "seed": 0},
              "model": {"name": "force_isat_true"}}
    result = run_experiment(config, now="2026-06-21_120001", runs_root=tmp_path)
    rows = read_jsonl(Path(result["run_dir"]) / "predictions" / "dev.jsonl")
    for r in rows:
        for sp in r["sampled_pairs"]:
            if sp["isAt"] == "TRUE":
                assert sp["at"] == "TRUE"


def test_harness_leaderboard_matches_official_scorer_when_pred_class_absent_from_gold(tmp_path):
    # A model that predicts PROBABLE (a class that may be absent from a gold slice)
    # must yield the SAME global the official scorer would, not the in-house metric.
    from hipe.models import registry
    from hipe.models.base import RelationModel
    from hipe.eval.scorer import score_files
    from pathlib import Path

    @registry.register("always_probable")
    class AlwaysProbable(RelationModel):
        name = "always_probable"
        def fit(self, train, dev=None): pass
        def predict(self, pairs):
            return [{"at": "PROBABLE", "isAt": "FALSE"} for _ in pairs]

    config = {"data": {"train": str(FIX), "dev_frac": 1.0, "seed": 0},
              "model": {"name": "always_probable"}}
    result = run_experiment(config, now="2026-06-21_120002", runs_root=tmp_path)
    pred_dir = Path(result["run_dir"]) / "predictions"
    official = score_files(pred_dir / "dev_gold.jsonl", pred_dir / "dev.jsonl")
    assert round(result["global"], 6) == round(official["global"]["macro_recall"], 6)
    assert round(result["at_recall"], 6) == round(official["at"]["macro_recall"], 6)
