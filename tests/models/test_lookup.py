import numpy as np  # noqa: F401
from pathlib import Path
from hipe.data.schema import Entity, Pair
from hipe.models import registry
import hipe.models.lookup  # noqa: F401  (registers llm_lookup)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini.jsonl"


def _pair(doc, pers_id, loc_id):
    return Pair(doc_id=doc, person=Entity(pers_id, "person", ["X"]),
                place=Entity(loc_id, "place", ["Y"]), context="", language="en",
                pub_date=None)


def test_lookup_returns_source_label_and_false_fallback():
    m = registry.get_model("llm_lookup", label_source=str(FIX))
    m.fit([])  # train is ignored; lookup is built from label_source
    # mini.jsonl d1 has pair (d1-joe, d1-essex) labelled at=TRUE isAt=TRUE
    known = _pair("d1", "d1-joe", "d1-essex")
    unknown = _pair("zzz", "no-pers", "no-loc")
    preds = m.predict([known, unknown])
    assert preds[0]["at"] == "TRUE" and preds[0]["isAt"] == "TRUE"
    assert preds[1]["at"] == "FALSE" and preds[1]["isAt"] == "FALSE"
    assert preds[0]["at_proba"] is None


def test_lookup_normalizes_null_source_labels():
    # mini.jsonl d2 (d2-marie, d2-paris) has at=null isAt=null -> FALSE/FALSE
    m = registry.get_model("llm_lookup", label_source=str(FIX))
    m.fit([])
    preds = m.predict([_pair("d2", "d2-marie", "d2-paris")])
    assert preds[0]["at"] == "FALSE" and preds[0]["isAt"] == "FALSE"
