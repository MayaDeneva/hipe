# hipe/eval/scorer.py
from pathlib import Path
from hipe.eval.official import evaluation_utils as eu


def score_files(gold_path, pred_path) -> dict:
    """Score a prediction file against gold using the vendored official scorer."""
    gold = eu.load_jsonl_to_reshaped_dict(Path(gold_path))
    sub = eu.load_jsonl_to_reshaped_dict(Path(pred_path))
    sub = eu.impute_missing_submission_data(gold, sub)
    labels = eu.flatten_predictions(gold, sub)
    return eu.calculate_metrics(labels)
