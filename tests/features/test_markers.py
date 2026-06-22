from hipe.data.schema import Entity, Pair
from hipe.features.markers import marked_text, MARKER_TOKENS


def _pair(context, pers_mentions, loc_mentions):
    return Pair(doc_id="d", person=Entity("p", "person", pers_mentions),
                place=Entity("l", "place", loc_mentions), context=context,
                language="en", pub_date=None)


def test_marker_tokens():
    assert MARKER_TOKENS == ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def test_marks_both_mentions_in_place():
    p = _pair("Joe was at Essex.", ["Joe"], ["Essex"])
    assert marked_text(p) == "[E1]Joe[/E1] was at [E2]Essex[/E2]."


def test_fallback_prepends_when_not_found():
    p = _pair("nothing relevant here", ["Zzz"], ["Qqq"])
    out = marked_text(p)
    assert out.startswith("[E1]Zzz[/E1] [E2]Qqq[/E2] ")
    assert out.endswith("nothing relevant here")
