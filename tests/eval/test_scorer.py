# tests/eval/test_scorer.py
from pathlib import Path
from hipe.eval.scorer import score_files

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_score_files_matches_hand_computed():
    metrics = score_files(FIX / "gold.jsonl", FIX / "pred.jsonl")
    # at:   gold=[TRUE,FALSE] pred=[TRUE,FALSE] -> macro recall 1.0
    # isAt: gold=[TRUE,FALSE] pred=[FALSE,FALSE] -> recall FALSE=1, TRUE=0 -> 0.5
    assert round(metrics["at"]["macro_recall"], 4) == 1.0
    assert round(metrics["isAt"]["macro_recall"], 4) == 0.5
    assert round(metrics["global"]["macro_recall"], 4) == 0.75
