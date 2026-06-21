# hipe/eval/scorer.py
import warnings
from pathlib import Path

from sklearn.exceptions import UndefinedMetricWarning

from hipe.eval.official import evaluation_utils as eu


def score_files(gold_path, pred_path) -> dict:
    """Score a prediction file against gold using the vendored official scorer."""
    # The vendored official scorer averages recall over the gold∪pred union; when a
    # model predicts a class absent from the gold slice, sklearn raises an (expected)
    # UndefinedMetricWarning. Suppress that one warning here without editing the
    # vendored file — the 0-recall contribution it warns about is the intended metric.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        gold = eu.load_jsonl_to_reshaped_dict(Path(gold_path))
        sub = eu.load_jsonl_to_reshaped_dict(Path(pred_path))
        sub = eu.impute_missing_submission_data(gold, sub)
        labels = eu.flatten_predictions(gold, sub)
        return eu.calculate_metrics(labels)
