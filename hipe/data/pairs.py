from hipe import config
from hipe.data.load import read_jsonl
from hipe.data.preprocess import normalize_text, fuzzy_find
from hipe.data.schema import Entity, Pair


def context_for(text: str, mentions: list[str], margin: int = 200) -> str:
    """Window around the first matching mention; whole text if none found."""
    span = None
    for m in mentions:
        found = fuzzy_find(text, m)
        if found is not None:
            span = found
            break
    if span is None:
        return text
    lo = max(0, span[0] - margin)
    hi = min(len(text), span[1] + margin)
    return text[lo:hi]


def _first_span(text, mentions):
    for m in mentions:
        s = fuzzy_find(text, m)
        if s is not None:
            return s
    return None


def context_for_pair(text: str, person_mentions, place_mentions,
                     margin: int = 150, max_span: int = 600) -> str:
    """Window guaranteeing BOTH entities are present. When they're within
    max_span chars, span both (capturing the connecting text); when far apart,
    take a dual window around each (joined by ' [...] '). Fixes the ~43% of pairs
    where one entity fell outside a single window anchored on the other."""
    ps = _first_span(text, person_mentions)
    ls = _first_span(text, place_mentions)
    if ps is None and ls is None:
        return text[:2 * margin + 300]
    if ps is None or ls is None:
        s = ps or ls
        return text[max(0, s[0] - margin):min(len(text), s[1] + margin)]
    lo, hi = (ps, ls) if ps[0] <= ls[0] else (ls, ps)
    if hi[1] - lo[0] <= max_span:
        return text[max(0, lo[0] - margin):min(len(text), hi[1] + margin)]
    w1 = text[max(0, ps[0] - margin):min(len(text), ps[1] + margin)]
    w2 = text[max(0, ls[0] - margin):min(len(text), ls[1] + margin)]
    return w1 + " [...] " + w2


def _intern_entity(registry, entity_id, etype, mentions, qid) -> Entity:
    if entity_id not in registry:
        registry[entity_id] = Entity(entity_id=entity_id, etype=etype,
                                     mentions=list(mentions), qid=qid)
    else:
        existing = registry[entity_id]
        seen = set(existing.mentions)
        for m in mentions:
            if m not in seen:
                existing.mentions.append(m)
                seen.add(m)
    return registry[entity_id]


def pair_key(pair) -> str:
    """Match the official scorer key: pers_entity_id concatenated with loc_entity_id."""
    return f"{pair.person.entity_id}{pair.place.entity_id}"


def load_pairs(path) -> list[Pair]:
    pairs = []
    # registry persists across documents; relies on entity IDs being globally unique
    # (true for HIPE-2026: entity IDs are document-prefixed, so collisions cannot occur).
    registry: dict = {}
    for raw in read_jsonl(path):
        text = normalize_text(raw.get("text", ""))
        for sp in raw.get("sampled_pairs", []):
            person = _intern_entity(registry, sp["pers_entity_id"], "person",
                                    sp.get("pers_mentions_list", []),
                                    sp.get("pers_wikidata_QID"))
            place = _intern_entity(registry, sp["loc_entity_id"], "place",
                                   sp.get("loc_mentions_list", []),
                                   sp.get("loc_wikidata_QID"))
            pairs.append(Pair(
                doc_id=str(raw["document_id"]), person=person, place=place,
                context=context_for_pair(text, person.mentions, place.mentions),
                language=raw.get("language", ""), pub_date=raw.get("date"),
                gold_at=config.norm_label(sp.get("at"), "at"),
                gold_isat=config.norm_label(sp.get("isAt"), "isAt"),
            ))
    return pairs


def unique_entities(pairs) -> dict:
    seen = {}
    for p in pairs:
        for e in (p.person, p.place):
            seen.setdefault(e.entity_id, e)
    return seen
