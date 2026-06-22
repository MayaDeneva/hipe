from hipe.data.preprocess import fuzzy_find

# entity-representation schemes: (person_start, person_end, place_start, place_end, mask)
#   plain      — generic markers, entity string kept ([E1]John[/E1])      (baseline)
#   typed      — type markers, string kept ([PER]John[/PER])              (Zhong & Chen 2021)
#   typed_mask — type markers, string masked ([PER][/PER])                (entity-mask)
SCHEMES = {
    "plain":      ("[E1]", "[/E1]", "[E2]", "[/E2]", False),
    "typed":      ("[PER]", "[/PER]", "[LOC]", "[/LOC]", False),
    "typed_mask": ("[PER]", "[/PER]", "[LOC]", "[/LOC]", True),
}
DATE_TOKEN = "[DATE]"

# kept for backward-compat (default plain scheme markers)
E1_START, E1_END = "[E1]", "[/E1]"
E2_START, E2_END = "[E2]", "[/E2]"
MARKER_TOKENS = [E1_START, E1_END, E2_START, E2_END]


def scheme_marker_tokens(scheme="plain", add_date=False):
    """special tokens to add to the tokenizer for a scheme."""
    ps, pe, ls, le, _ = SCHEMES[scheme]
    toks = [ps, pe, ls, le]
    if add_date:
        toks.append(DATE_TOKEN)
    return toks


def scheme_pool_markers(scheme="plain"):
    """(person_start, person_end, place_start, place_end) for R-BERT pooling."""
    ps, pe, ls, le, _ = SCHEMES[scheme]
    return ps, pe, ls, le


def year_of(pub_date):
    if not pub_date:
        return None
    try:
        return int(str(pub_date).lstrip("+")[:4])
    except ValueError:
        return None


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _locate(text, mentions):
    for m in mentions:
        span = fuzzy_find(text, m)
        if span is not None:
            return span
    return None


def marked_text(pair, scheme="plain", add_date=False) -> str:
    """Context with the two entities wrapped per `scheme`; optionally prefixed
    with the publication year ([DATE] 1850) so the model can reason about 'now'
    for isAt. Falls back to prepending the marked surfaces when a mention can't
    be located or the two spans overlap."""
    ps, pe, ls, le, mask = SCHEMES[scheme]
    text = pair.context
    pspan = _locate(text, pair.person.mentions)
    lspan = _locate(text, pair.place.mentions)
    if pspan is not None and lspan is not None and not _overlap(pspan, lspan):
        # insert the later span first so the earlier span's offsets stay valid
        for span, st, en in sorted(
                [(pspan, ps, pe), (lspan, ls, le)],
                key=lambda x: x[0][0], reverse=True):
            s, e = span
            content = "" if mask else text[s:e]
            text = text[:s] + st + content + en + text[e:]
        out = text
    else:
        pc = "" if mask else pair.person.surface
        lc = "" if mask else pair.place.surface
        out = f"{ps}{pc}{pe} {ls}{lc}{le} {text}"
    if add_date:
        y = year_of(pair.pub_date)
        if y is not None:
            out = f"{DATE_TOKEN} {y} {out}"
    return out
