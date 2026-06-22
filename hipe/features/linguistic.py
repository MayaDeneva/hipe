"""spaCy linguistic features for person-place relation classification.

Encodes the hypothesis that the relation is signalled by the verbs between the
two entities, their tense (past -> was ever there; present -> there now),
negation, and the entities' proximity/order. Uses POS + lemma + morphology
(robust on noisy historical OCR); does NOT rely on the full dependency parse.
"""
from functools import lru_cache
from hipe.data.preprocess import fuzzy_find

# verb lemmas (lowercased) signalling movement-to vs. staying-at, per language
MOVEMENT = {
    "en": {"go", "travel", "move", "arrive", "come", "return", "flee", "march",
           "sail", "ride", "depart", "leave", "enter", "reach", "journey", "cross"},
    "de": {"gehen", "reisen", "kommen", "ankommen", "ziehen", "fliehen", "fahren",
           "marschieren", "zurückkehren", "abreisen", "verlassen", "erreichen"},
    "fr": {"aller", "voyager", "venir", "arriver", "partir", "fuir", "retourner",
           "marcher", "naviguer", "quitter", "traverser", "rejoindre"},
}
STAY = {
    "en": {"live", "reside", "stay", "remain", "dwell", "settle", "be", "locate",
           "bear", "die", "inhabit"},
    "de": {"leben", "wohnen", "bleiben", "aufhalten", "sein", "sterben", "verweilen"},
    "fr": {"vivre", "habiter", "rester", "demeurer", "être", "naître", "mourir",
           "séjourner", "résider"},
}
_NEG = {"not", "no", "never", "ne", "pas", "nicht", "kein", "keine", "nie"}
_MODEL = {"en": "en_core_web_sm", "de": "de_core_news_sm", "fr": "fr_core_news_sm"}

FEATURE_KEYS = ["person_found", "place_found", "dist_chars", "person_first",
                "n_verbs", "has_movement", "has_stay", "has_past", "has_pres",
                "has_negation"]


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
    text = pair.context
    lang = pair.language if pair.language in _MODEL else "en"
    ps = _span(text, pair.person.mentions)
    ls = _span(text, pair.place.mentions)
    base = {k: 0 for k in FEATURE_KEYS}
    base["person_found"] = int(ps is not None)
    base["place_found"] = int(ls is not None)
    base["dist_chars"] = 999
    if ps is None or ls is None:
        return base

    lo, hi = (ps, ls) if ps[0] <= ls[0] else (ls, ps)
    base["person_first"] = int(ps[0] <= ls[0])
    base["dist_chars"] = min(hi[0] - lo[1], 999)
    # parse a window spanning both mentions (+ a little context on each side)
    window = text[max(0, lo[0] - 40):min(len(text), hi[1] + 40)]
    doc = _nlp(lang)(window)

    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    lemmas = {t.lemma_.lower() for t in verbs}
    tenses = set()
    for t in verbs:
        tenses |= set(t.morph.get("Tense"))
    base["n_verbs"] = len(verbs)
    base["has_movement"] = int(bool(lemmas & MOVEMENT.get(lang, MOVEMENT["en"])))
    base["has_stay"] = int(bool(lemmas & STAY.get(lang, STAY["en"])))
    base["has_past"] = int("Past" in tenses)
    base["has_pres"] = int("Pres" in tenses)
    base["has_negation"] = int(any(t.dep_ == "neg" or t.lower_ in _NEG for t in doc))
    return base
