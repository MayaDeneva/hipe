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
                context=context_for(text, person.mentions + place.mentions),
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
