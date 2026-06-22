# hipe/models/embedding_svm.py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.embeddings import EmbeddingEncoder, DEFAULT_MODEL
from hipe.features.text import pair_text


class _Head:
    """One target classifier; constant fallback when <2 classes are present."""

    def __init__(self, C):
        self.C = C
        self.clf = None
        self.const = "FALSE"

    def fit(self, X, y):
        classes = sorted(set(y))
        if len(classes) < 2:
            self.const = classes[0] if classes else "FALSE"
            self.clf = None
        else:
            self.clf = LinearSVC(C=self.C, class_weight="balanced")
            self.clf.fit(X, y)

    def predict(self, X):
        if self.clf is None:
            return [self.const] * X.shape[0]
        return list(self.clf.predict(X))


def _cache_path(model_name):
    slug = model_name.replace("/", "_")
    return cfg.CACHE_DIR / f"emb_{slug}.pkl"


@registry.register("embedding_svm")
class EmbeddingSVM(RelationModel):
    """LinearSVC over a pair-specific context embedding. With use_structural, it
    also appends the spaCy structural features (tense, negation, distance, order)
    and standard-scales the combined vector (an SVM needs scaling once embeddings
    and a 0..999 distance feature are mixed)."""
    name = "embedding_svm"

    def __init__(self, model_name=DEFAULT_MODEL, C=1.0, use_structural=False,
                 use_kb=False, cache_path=None, _encoder=None):
        if _encoder is not None:
            self.encoder = _encoder
        else:
            self.encoder = EmbeddingEncoder(
                model_name, cache_path=cache_path or _cache_path(model_name))
        self.use_structural = use_structural
        self.use_kb = use_kb
        self.scaler = None
        self._at = _Head(C)
        self._isat = _Head(C)

    def _features(self, pairs, fit_scaler=False):
        X = np.asarray(self.encoder.encode([pair_text(p) for p in pairs]))
        extra = []
        if self.use_structural:
            from hipe.features.linguistic import linguistic_features, STRUCT_KEYS
            extra.append(np.array([[linguistic_features(p).get(k, 0)
                                    for k in STRUCT_KEYS] for p in pairs], dtype=float))
        if self.use_kb:
            from hipe.features.kb import kb_features, KB_KEYS
            kk = [k for k in KB_KEYS if k != "kb_min_dist_km"]  # log_dist is scale-friendly
            extra.append(np.array([[kb_features(p).get(k, 0) for k in kk]
                                   for p in pairs], dtype=float))
        if not extra:
            return X
        X = np.hstack([X] + extra)
        if fit_scaler:
            self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)

    def fit(self, train, dev=None):
        X = self._features(train, fit_scaler=True)
        self._at.fit(X, [p.gold_at for p in train])
        self._isat.fit(X, [p.gold_isat for p in train])

    def predict(self, pairs):
        X = self._features(pairs)
        at = self._at.predict(X)
        isat = self._isat.predict(X)
        return [{"at": a, "isAt": i, "at_proba": None, "isAt_proba": None}
                for a, i in zip(at, isat)]
