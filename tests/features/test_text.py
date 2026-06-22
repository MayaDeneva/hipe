from hipe.data.schema import Entity, Pair
from hipe.features.text import pair_text


def test_pair_text_is_pair_specific():
    person = Entity("p1", "person", ["Joe"])
    place = Entity("l1", "place", ["Essex"])
    p = Pair(doc_id="d1", person=person, place=place,
             context="Joe was at Essex county.", language="en", pub_date=None)
    assert pair_text(p) == "Joe [SEP] Essex [SEP] Joe was at Essex county."


def test_pair_text_handles_empty_mentions():
    person = Entity("p1", "person", [])
    place = Entity("l1", "place", ["Paris"])
    p = Pair(doc_id="d", person=person, place=place, context="ctx",
             language="fr", pub_date=None)
    assert pair_text(p) == " [SEP] Paris [SEP] ctx"
