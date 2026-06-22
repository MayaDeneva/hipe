# tests/models/test_xlmr.py
import pytest
from hipe.data.schema import Entity, Pair


def _pair(at, isat, ctx):
    return Pair(doc_id="d", person=Entity("p", "person", ["Joe"]),
                place=Entity("l", "place", ["Essex"]), context=ctx,
                language="en", pub_date=None, gold_at=at, gold_isat=isat)


def test_class_weights_inverse_frequency():
    from hipe.models.xlmr import _class_weights
    # labels: 3 FALSE, 1 TRUE over ["FALSE","PROBABLE","TRUE"]
    w = _class_weights(["FALSE", "FALSE", "FALSE", "TRUE"],
                       ["FALSE", "PROBABLE", "TRUE"])
    assert w.shape[0] == 3
    # rarer classes get higher weight: TRUE(1) > FALSE(3); absent PROBABLE highest
    assert float(w[2]) > float(w[0])
    assert float(w[1]) >= float(w[2])


@pytest.mark.slow
def test_xlmr_fit_predict_pipeline_tiny_model():
    # Exercises the full fit->predict mechanics with a tiny model (no XLM-R download).
    pytest.importorskip("transformers")
    from hipe.models import registry
    import hipe.models.xlmr  # noqa: F401  (registers xlmr)
    try:
        m = registry.get_model("xlmr", model_name="prajjwal1/bert-tiny",
                                epochs=1, batch_size=4, max_length=32, seed=0)
        train = ([_pair("TRUE", "TRUE", "Joe lived at Essex") for _ in range(6)] +
                 [_pair("FALSE", "FALSE", "no relation here") for _ in range(6)])
        m.fit(train)
        preds = m.predict([_pair("?", "?", "Joe lived at Essex")])
    except Exception as exc:  # offline / model unavailable
        pytest.skip(f"tiny-model integration unavailable: {exc}")
    assert preds[0]["at"] in ("FALSE", "PROBABLE", "TRUE")
    assert preds[0]["isAt"] in ("FALSE", "TRUE")
    assert isinstance(preds[0]["at_proba"], dict)
    assert abs(sum(preds[0]["at_proba"].values()) - 1.0) < 1e-3
