"""spaCy linguistic features for person-place relation classification.

Encodes the hypothesis that the relation is signalled by the verbs between the
two entities, their tense (past -> was ever there; present -> there now),
negation, and the entities' proximity/order — but WITHOUT a hand-written verb
lexicon. The actual verb lemmas between the entities are emitted as features
(`vb_<lemma>`) so the classifier learns which verbs matter from the labels.
Uses POS + lemma + morphology (robust on noisy OCR); not the dependency parse.
"""
from functools import lru_cache
from hipe.data.preprocess import fuzzy_find

_NEG = {"not", "no", "never", "ne", "pas", "nicht", "kein", "keine", "nie"}
_MODEL = {"en": "en_core_web_sm", "de": "de_core_news_sm", "fr": "fr_core_news_sm"}


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


def linguistic_features(pair) -> dict:
    """Return a feature dict mixing structural signals with learned per-verb
    features (`vb_<lemma>`: 1). The classifier's vectorizer turns the verb keys
    into a data-driven bag-of-verb-lemmas; unseen verbs are simply dropped."""
    text = pair.context
    lang = pair.language if pair.language in _MODEL else "en"
    ps = _span(text, pair.person.mentions)
    ls = _span(text, pair.place.mentions)
    f = {"person_found": int(ps is not None), "place_found": int(ls is not None),
         "dist_chars": 999, "person_first": 0, "n_verbs": 0,
         "has_past": 0, "has_pres": 0, "has_negation": 0}
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
        lemma = t.lemma_.lower().strip()
        if lemma:
            f["vb_" + lemma] = 1                 # learned bag-of-verb-lemmas
    f["has_past"] = int("Past" in tenses)
    f["has_pres"] = int("Pres" in tenses)
    f["has_negation"] = int(any(t.dep_ == "neg" or t.lower_ in _NEG for t in doc))
    return f
