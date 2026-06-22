"""spaCy structural features + the relation-region text for embedding.

The hypothesis (tense / verbs / proximity signal the relation) is kept, but the
LEXICAL part is no longer a hand-counted bag of verb lemmas (which drowns in OCR
noise and fragments across languages). Instead `linguistic_features` returns the
robust STRUCTURAL signals (tense, negation, distance, order) and `relation_span`
returns the text region spanning both entities — which a multilingual embedding
turns into dense, OCR- and language-robust verb/context features.
"""
from functools import lru_cache
from hipe.data.preprocess import fuzzy_find

_NEG = {"not", "no", "never", "ne", "pas", "nicht", "kein", "keine", "nie"}
_MODEL = {"en": "en_core_web_sm", "de": "de_core_news_sm", "fr": "fr_core_news_sm"}

STRUCT_KEYS = ["person_found", "place_found", "dist_chars", "person_first",
               "n_verbs", "has_past", "has_pres", "has_negation"]


@lru_cache(maxsize=4)
def _nlp(lang):
    import spacy
    name = _MODEL.get(lang, "en_core_web_sm")
    try:
        return spacy.load(name, disable=["ner"])
    except OSError:
        return spacy.blank(lang if lang in _MODEL else "en")


def _span(text, mentions):
    for m in mentions:
        s = fuzzy_find(text, m)
        if s is not None:
            return s
    return None


def relation_span(pair) -> str:
    """Text region covering both entities (+ margin) — contains the linking
    verbs in context; what the embedding encodes."""
    text = pair.context
    ps = _span(text, pair.person.mentions)
    ls = _span(text, pair.place.mentions)
    if ps is None or ls is None:
        return text[:300]
    lo, hi = min(ps[0], ls[0]), max(ps[1], ls[1])
    return text[max(0, lo - 30):min(len(text), hi + 30)]


def linguistic_features(pair) -> dict:
    """Robust structural signals only (no lexical bag)."""
    text = pair.context
    lang = pair.language if pair.language in _MODEL else "en"
    ps = _span(text, pair.person.mentions)
    ls = _span(text, pair.place.mentions)
    f = {k: 0 for k in STRUCT_KEYS}
    f["person_found"] = int(ps is not None)
    f["place_found"] = int(ls is not None)
    f["dist_chars"] = 999
    if ps is None or ls is None:
        return f

    lo, hi = (ps, ls) if ps[0] <= ls[0] else (ls, ps)
    f["person_first"] = int(ps[0] <= ls[0])
    f["dist_chars"] = min(hi[0] - lo[1], 999)
    window = text[max(0, lo[0] - 40):min(len(text), hi[1] + 40)]
    doc = _nlp(lang)(window)

    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    f["n_verbs"] = len(verbs)
    tenses = set()
    for t in verbs:
        tenses |= set(t.morph.get("Tense"))
    f["has_past"] = int("Past" in tenses)
    f["has_pres"] = int("Pres" in tenses)
    f["has_negation"] = int(any(t.dep_ == "neg" or t.lower_ in _NEG for t in doc))
    return f
