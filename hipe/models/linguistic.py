# hipe/models/linguistic.py
import os
# xgboost's libomp can collide with another OpenMP runtime (sklearn/torch) in the
# same process on macOS -> segfault. Allow the duplicate before xgboost loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from collections import Counter
import numpy as np
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.linguistic import linguistic_features, relation_span, STRUCT_KEYS
from hipe.features.embeddings import EmbeddingEncoder, DEFAULT_MODEL


def _sample_weights(labels, label_list):
    """Balanced per-example weights (inverse class frequency)."""
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
        self.clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="mlogloss", n_jobs=1)   # n_jobs=1: avoid OpenMP clash
        self.clf.fit(X, [self.lab2id[l] for l in y],
                     sample_weight=_sample_weights(y, self.label_list))

    def predict(self, X):
        if self.clf is None:
            proba = {l: (1.0 if l == self.const else 0.0) for l in self.label_list}
            return [self.const] * X.shape[0], [dict(proba) for _ in range(X.shape[0])]
        raw = self.clf.predict_proba(X)
        classes = self.clf.classes_
        labels, probas = [], []
        for row in raw:
            full = {l: 0.0 for l in self.label_list}
            for col, cid in enumerate(classes):
                full[self.label_list[int(cid)]] = float(row[col])
            labels.append(max(full, key=full.get))
            probas.append(full)
        return labels, probas


@registry.register("linguistic")
class LinguisticModel(RelationModel):
    """Structural spaCy features (tense, negation, distance, order) + a dense
    multilingual embedding of the relation region (verbs-in-context) -> XGBoost.
    Embeddings replace the raw verb-lemma bag: robust to OCR noise and unified
    across languages."""
    name = "linguistic"

    def __init__(self, model_name=DEFAULT_MODEL, cache_path=None):
        slug = model_name.replace("/", "_")
        self.encoder = EmbeddingEncoder(
            model_name, cache_path=cache_path or (cfg.CACHE_DIR / f"emb_{slug}.pkl"))
        self._at = _Head(cfg.AT_LABELS)
        self._isat = _Head(cfg.ISAT_LABELS)

    def _features(self, pairs):
        struct = np.array([[linguistic_features(p).get(k, 0) for k in STRUCT_KEYS]
                           for p in pairs], dtype=float)
        emb = self.encoder.encode([relation_span(p) for p in pairs])
        return np.hstack([struct, emb])

    def fit(self, train, dev=None):
        X = self._features(train)
        self._at.fit(X, [p.gold_at for p in train])
        self._isat.fit(X, [p.gold_isat for p in train])

    def predict(self, pairs):
        X = self._features(pairs)
        at, at_p = self._at.predict(X)
        isat, isat_p = self._isat.predict(X)
        return [{"at": a, "isAt": i, "at_proba": ap, "isAt_proba": ip}
                for a, i, ap, ip in zip(at, isat, at_p, isat_p)]
