# tests/models/test_embedding_svm.py
import numpy as np
from hipe.data.schema import Entity, Pair
from hipe.models import registry
import hipe.models.embedding_svm  # noqa: F401  (registers the model)


class _FakeEncoder:
    """Maps text -> a separable vector by first label keyword present."""
    def encode(self, texts):
        rows = []
        for t in texts:
            # encode a strong signal: 'TRUEISH' vs 'FALSEISH' marker in context
            rows.append([1.0, 0.0] if "TRUEISH" in t else [0.0, 1.0])
        return np.asarray(rows, dtype=float)


def _pair(at, isat, marker):
    person = Entity("p", "person", ["X"])
    place = Entity("l", "place", ["Y"])
    return Pair(doc_id="d", person=person, place=place,
                context=marker, language="en", pub_date=None,
                gold_at=at, gold_isat=isat)


def test_embedding_svm_learns_separable_signal():
    train = ([_pair("TRUE", "TRUE", "TRUEISH") for _ in range(6)] +
             [_pair("FALSE", "FALSE", "FALSEISH") for _ in range(6)])
    m = registry.get_model("embedding_svm", _encoder=_FakeEncoder())
    m.fit(train)
    preds = m.predict([_pair("?", "?", "TRUEISH"), _pair("?", "?", "FALSEISH")])
    assert preds[0]["at"] == "TRUE" and preds[0]["isAt"] == "TRUE"
    assert preds[1]["at"] == "FALSE" and preds[1]["isAt"] == "FALSE"
    assert preds[0]["at_proba"] is None


def test_embedding_svm_single_class_fallback():
    train = [_pair("FALSE", "FALSE", "FALSEISH") for _ in range(4)]
    m = registry.get_model("embedding_svm", _encoder=_FakeEncoder())
    m.fit(train)                      # only one class present per target
    preds = m.predict([_pair("?", "?", "TRUEISH")])
    assert preds[0]["at"] == "FALSE" and preds[0]["isAt"] == "FALSE"
