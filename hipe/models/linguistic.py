# hipe/models/linguistic.py
import numpy as np
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.linguistic import linguistic_features, FEATURE_KEYS


def _vectorize(pairs):
    return np.array([[linguistic_features(p).get(k, 0) for k in FEATURE_KEYS]
                     for p in pairs], dtype=float)


def _sample_weights(labels, label_list):
    """Balanced per-example weights (inverse class frequency)."""
    from collections import Counter
    counts = Counter(labels)
    n, k = len(labels), len(label_list)
    w = {cl: n / (k * max(1, counts[cl])) for cl in label_list}
    return np.array([w[l] for l in labels], dtype=float)


class _Head:
    """One XGBoost classifier for a target; constant fallback when <2 classes."""

    def __init__(self, label_list):
        self.label_list = label_list
        self.lab2id = {l: i for i, l in enumerate(label_list)}
        self.clf = None
        self.const = None

    def fit(self, X, y):
        from xgboost import XGBClassifier
        if len(set(y)) < 2:
            self.const = y[0] if y else "FALSE"
            return
        self.clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                 subsample=0.8, eval_metric="mlogloss", n_jobs=4)
        self.clf.fit(X, [self.lab2id[l] for l in y],
                     sample_weight=_sample_weights(y, self.label_list))

    def predict(self, X):
        if self.clf is None:
            proba = {l: (1.0 if l == self.const else 0.0) for l in self.label_list}
            return [self.const] * len(X), [dict(proba) for _ in range(len(X))]
        raw = self.clf.predict_proba(X)
        classes = self.clf.classes_          # integer label ids present in training
        labels, probas = [], []
        for row in raw:
            full = {l: 0.0 for l in self.label_list}
            for col, cls_id in enumerate(classes):
                full[self.label_list[int(cls_id)]] = float(row[col])
            labels.append(max(full, key=full.get))
            probas.append(full)
        return labels, probas


@registry.register("linguistic")
class LinguisticModel(RelationModel):
    """spaCy linguistic features (verbs-between-entities, tense, negation,
    distance) -> XGBoost, one head per target. Interpretable, CPU-only, tiny."""
    name = "linguistic"

    def __init__(self):
        self._at = _Head(cfg.AT_LABELS)
        self._isat = _Head(cfg.ISAT_LABELS)

    def fit(self, train, dev=None):
        X = _vectorize(train)
        self._at.fit(X, [p.gold_at for p in train])
        self._isat.fit(X, [p.gold_isat for p in train])

    def predict(self, pairs):
        X = _vectorize(pairs)
        at, at_p = self._at.predict(X)
        isat, isat_p = self._isat.predict(X)
        return [{"at": a, "isAt": i, "at_proba": ap, "isAt_proba": ip}
                for a, i, ap, ip in zip(at, isat, at_p, isat_p)]
