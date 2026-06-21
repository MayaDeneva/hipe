from hipe.data.schema import Entity, Pair


def test_entity_surface_is_first_mention():
    e = Entity(entity_id="p1", etype="person", mentions=["Joe", "J. Joe"])
    assert e.surface == "Joe"


def test_entity_surface_empty_when_no_mentions():
    e = Entity(entity_id="p1", etype="person", mentions=[])
    assert e.surface == ""


def test_pair_defaults_to_false_labels():
    person = Entity("p1", "person", ["Joe"])
    place = Entity("l1", "place", ["Essex"])
    p = Pair(doc_id="d1", person=person, place=place, context="...",
             language="en", pub_date=None)
    assert p.gold_at == "FALSE"
    assert p.gold_isat == "FALSE"
    assert p.features == {}
