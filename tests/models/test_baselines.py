# tests/models/test_baselines.py
from hipe.data.schema import Entity, Pair
from hipe.models import registry
import hipe.models.baselines  # noqa: F401  (registers the models)


def _pair(at, isat):
    return Pair(doc_id="d", person=Entity("p", "person", ["x"]),
                place=Entity("l", "place", ["y"]), context="", language="en",
                pub_date=None, gold_at=at, gold_isat=isat)


def test_majority_predicts_modal_label():
    train = [_pair("FALSE", "FALSE"), _pair("FALSE", "FALSE"), _pair("TRUE", "TRUE")]
    m = registry.get_model("majority")
    m.fit(train)
    preds = m.predict([_pair("TRUE", "TRUE")])
    assert preds[0]["at"] == "FALSE"
    assert preds[0]["isAt"] == "FALSE"


def test_random_is_deterministic_with_seed():
    train = [_pair("TRUE", "TRUE"), _pair("FALSE", "FALSE")]
    pairs = [_pair("FALSE", "FALSE") for _ in range(5)]
    a = registry.get_model("random", seed=42); a.fit(train)
    b = registry.get_model("random", seed=42); b.fit(train)
    assert a.predict(pairs) == b.predict(pairs)


def test_random_labels_are_valid():
    from hipe import config
    m = registry.get_model("random", seed=0); m.fit([_pair("TRUE", "TRUE")])
    for p in m.predict([_pair("FALSE", "FALSE") for _ in range(10)]):
        assert p["at"] in config.AT_LABELS
        assert p["isAt"] in config.ISAT_LABELS
