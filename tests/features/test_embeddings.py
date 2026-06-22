import numpy as np
from hipe.features.embeddings import EmbeddingEncoder, DEFAULT_MODEL


def _fake_encode_factory():
    calls = {"n": 0, "texts": []}
    def fn(texts):
        calls["n"] += 1
        calls["texts"].extend(texts)
        # deterministic 3-dim vector from text length
        return np.array([[len(t), len(t) % 7, 1.0] for t in texts], dtype=float)
    return fn, calls


def test_default_model_name():
    assert DEFAULT_MODEL == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def test_encode_returns_row_per_text_in_order():
    fn, _ = _fake_encode_factory()
    enc = EmbeddingEncoder(encode_fn=fn)
    out = enc.encode(["ab", "cde"])
    assert out.shape == (2, 3)
    assert out[0, 0] == 2 and out[1, 0] == 3


def test_encode_caches_and_dedupes(tmp_path):
    fn, calls = _fake_encode_factory()
    cache = tmp_path / "emb.pkl"
    enc = EmbeddingEncoder(cache_path=cache, encode_fn=fn)
    enc.encode(["x", "x", "yy"])          # only "x","yy" are unique -> 2 encoded
    assert sorted(calls["texts"]) == ["x", "yy"]
    # a second encoder loading the same cache encodes nothing new for seen text
    fn2, calls2 = _fake_encode_factory()
    enc2 = EmbeddingEncoder(cache_path=cache, encode_fn=fn2)
    enc2.encode(["x", "yy"])
    assert calls2["texts"] == []          # served entirely from disk cache
    assert cache.exists()
