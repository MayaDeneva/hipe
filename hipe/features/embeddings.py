import hashlib
import pickle
from pathlib import Path
import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _default_encode_fn(model_name):
    from sentence_transformers import SentenceTransformer  # lazy: only on real use
    model = SentenceTransformer(model_name)

    def fn(texts):
        return np.asarray(model.encode(list(texts), show_progress_bar=False))

    return fn


class EmbeddingEncoder:
    """Encode texts to vectors with a disk cache keyed by sha1(text)."""

    def __init__(self, model_name=DEFAULT_MODEL, cache_path=None, encode_fn=None):
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self._encode_fn = encode_fn
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = pickle.loads(self.cache_path.read_bytes())

    def _fn(self):
        if self._encode_fn is None:
            self._encode_fn = _default_encode_fn(self.model_name)
        return self._encode_fn

    @staticmethod
    def _key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts):
        texts = list(texts)
        unseen = [t for t in texts if self._key(t) not in self._cache]
        unique_unseen = list(dict.fromkeys(unseen))
        if unique_unseen:
            vecs = self._fn()(unique_unseen)
            for t, v in zip(unique_unseen, vecs):
                self._cache[self._key(t)] = np.asarray(v, dtype=float)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_bytes(pickle.dumps(self._cache))
        return np.vstack([self._cache[self._key(t)] for t in texts])
