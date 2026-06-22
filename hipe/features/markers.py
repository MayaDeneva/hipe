from hipe.data.preprocess import fuzzy_find

# Structural markers are ALWAYS [E1]/[E2] (so pooling is scheme-independent).
# The scheme only changes the TEXT between them:
#   plain      — entity string only          ([E1]John[/E1])              (baseline)
#   typed      — readable type word + string  ([E1] person John [/E1])    (Zhong & Chen 2021)
#   typed_mask — readable type word, string masked ([E1] person [/E1])
# Type info must be a REAL word (subword-tokenized) — an opaque special token
# [PER] gets the same id/embedding as [E1] and conveys nothing.
SCHEMES = {                          # name -> (use_type_word, mask_entity)
    "plain":      (False, False),
    "typed":      (True, False),
    "typed_mask": (True, True),
}
PERSON_MARK = ("[E1]", "[/E1]")
PLACE_MARK = ("[E2]", "[/E2]")
DATE_TOKEN = "[DATE]"
MARKER_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]   # backward-compat

TYPE_WORDS = {
    "person": {"en": "person", "de": "Person", "fr": "personne"},
    "place":  {"en": "location", "de": "Ort", "fr": "lieu"},
}


def _type_word(etype, lang):
    d = TYPE_WORDS[etype]
    return d.get(lang if lang in d else "en", d["en"])


def scheme_marker_tokens(scheme="plain", add_date=False):
    toks = list(MARKER_TOKENS)
    if add_date:
        toks.append(DATE_TOKEN)
    return toks


def scheme_pool_markers(scheme="plain"):
    return PERSON_MARK[0], PERSON_MARK[1], PLACE_MARK[0], PLACE_MARK[1]


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


def _wrap(markers, type_word, entity_text, mask):
    st, en = markers
    parts = [p for p in (type_word, "" if mask else entity_text) if p]
    return f"{st} {' '.join(parts)} {en}"


def marked_text(pair, scheme="plain", add_date=False) -> str:
    """Context with the two entities wrapped in [E1]/[E2]; `typed` schemes inject
    a readable type word, optionally masking the entity string. Optional [DATE]
    <year> prefix so the model can reason about 'now' for isAt."""
    use_type, mask = SCHEMES[scheme]
    lang = pair.language
    ptw = _type_word("person", lang) if use_type else ""
    ltw = _type_word("place", lang) if use_type else ""
    text = pair.context
    pspan = _locate(text, pair.person.mentions)
    lspan = _locate(text, pair.place.mentions)
    if pspan is not None and lspan is not None and not _overlap(pspan, lspan):
        for span, mark, tw in sorted(
                [(pspan, PERSON_MARK, ptw), (lspan, PLACE_MARK, ltw)],
                key=lambda x: x[0][0], reverse=True):
            s, e = span
            text = text[:s] + _wrap(mark, tw, text[s:e], mask) + text[e:]
        out = text
    else:
        out = (f"{_wrap(PERSON_MARK, ptw, pair.person.surface, mask)} "
               f"{_wrap(PLACE_MARK, ltw, pair.place.surface, mask)} {text}")
    if add_date:
        y = year_of(pair.pub_date)
        if y is not None:
            out = f"{DATE_TOKEN} {y} {out}"
    return out
