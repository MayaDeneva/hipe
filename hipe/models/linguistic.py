# hipe/models/linguistic.py
from collections import Counter
import numpy as np
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.linguistic import linguistic_features


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
        self.clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                 subsample=0.8, eval_metric="mlogloss", n_jobs=4)
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
    """spaCy linguistic features (tense, negation, distance, order) + a learned
    bag-of-verb-lemmas -> XGBoost, one head per target. No hand-written verb
    lexicon: the classifier learns which verbs matter from the labels."""
    name = "linguistic"

    def __init__(self, min_verb_df=3):
        self.min_verb_df = min_verb_df      # prune verbs seen in <N docs (OCR noise)
        self.vec = None
        self._at = _Head(cfg.AT_LABELS)
        self._isat = _Head(cfg.ISAT_LABELS)

    def _prune(self, feats, keep_verbs):
        return [{k: v for k, v in f.items()
                 if not k.startswith("vb_") or k in keep_verbs} for f in feats]

    def fit(self, train, dev=None):
        from sklearn.feature_extraction import DictVectorizer
        feats = [linguistic_features(p) for p in train]
        df = Counter(k for f in feats for k in f if k.startswith("vb_"))
        keep = {k for k, c in df.items() if c >= self.min_verb_df}
        self._keep = keep
        self.vec = DictVectorizer(sparse=True)
        X = self.vec.fit_transform(self._prune(feats, keep))
        self._at.fit(X, [p.gold_at for p in train])
        self._isat.fit(X, [p.gold_isat for p in train])

    def predict(self, pairs):
        feats = self._prune([linguistic_features(p) for p in pairs], self._keep)
        X = self.vec.transform(feats)        # unseen verbs dropped automatically
        at, at_p = self._at.predict(X)
        isat, isat_p = self._isat.predict(X)
        return [{"at": a, "isAt": i, "at_proba": ap, "isAt_proba": ip}
                for a, i, ap, ip in zip(at, isat, at_p, isat_p)]
