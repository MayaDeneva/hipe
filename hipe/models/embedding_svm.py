# hipe/models/embedding_svm.py
from sklearn.svm import LinearSVC
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
            return [self.const] * len(X)
        return list(self.clf.predict(X))


def _cache_path(model_name):
    slug = model_name.replace("/", "_")
    return cfg.CACHE_DIR / f"emb_{slug}.pkl"


@registry.register("embedding_svm")
class EmbeddingSVM(RelationModel):
    name = "embedding_svm"

    def __init__(self, model_name=DEFAULT_MODEL, C=1.0, cache_path=None, _encoder=None):
        if _encoder is not None:
            self.encoder = _encoder
        else:
            self.encoder = EmbeddingEncoder(
                model_name, cache_path=cache_path or _cache_path(model_name))
        self._at = _Head(C)
        self._isat = _Head(C)

    def fit(self, train, dev=None):
        X = self.encoder.encode([pair_text(p) for p in train])
        self._at.fit(X, [p.gold_at for p in train])
        self._isat.fit(X, [p.gold_isat for p in train])

    def predict(self, pairs):
        X = self.encoder.encode([pair_text(p) for p in pairs])
        at = self._at.predict(X)
        isat = self._isat.predict(X)
        return [{"at": a, "isAt": i, "at_proba": None, "isAt_proba": None}
                for a, i in zip(at, isat)]
