from hipe.data.preprocess import fuzzy_find

E1_START, E1_END = "[E1]", "[/E1]"
E2_START, E2_END = "[E2]", "[/E2]"
MARKER_TOKENS = [E1_START, E1_END, E2_START, E2_END]


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _insert(text, span, start_tok, end_tok) -> str:
    s, e = span
    return text[:s] + start_tok + text[s:e] + end_tok + text[e:]


def _locate(text, mentions):
    for m in mentions:
        span = fuzzy_find(text, m)
        if span is not None:
            return span
    return None


def marked_text(pair) -> str:
    """Context with [E1]..[/E1] around the person mention and [E2]..[/E2] around
    the place mention. Falls back to prepending the marked surfaces when a
    mention can't be located or the two spans overlap."""
    text = pair.context
    pspan = _locate(text, pair.person.mentions)
    lspan = _locate(text, pair.place.mentions)
    if pspan is not None and lspan is not None and not _overlap(pspan, lspan):
        # insert the later span first so the earlier span's offsets stay valid
        for span, st, en in sorted(
                [(pspan, E1_START, E1_END), (lspan, E2_START, E2_END)],
                key=lambda x: x[0][0], reverse=True):
            text = _insert(text, span, st, en)
        return text
    return (f"{E1_START}{pair.person.surface}{E1_END} "
            f"{E2_START}{pair.place.surface}{E2_END} {text}")
