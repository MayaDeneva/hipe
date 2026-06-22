"""Temporal-aware NIL entity linking.

For entities the organizers left unlinked (no QID), search Wikidata for the
mention, then pick the candidate that fits BOTH the type (person/place) and the
ARTICLE'S PUBLICATION YEAR — preferring entities whose existence window contains
that year. This avoids the classic historical-linking error of snapping
"Germany, 1850" onto modern Germany (founded 1949) and pulling wrong facts.
"""
import json
import time
from hipe import config as cfg
from hipe.features import kb   # reuse _sparql / _parse_point / UA

API = "https://www.wikidata.org/w/api.php"
SEARCH_CACHE = cfg.CACHE_DIR / "link_search_cache.json"
FACTS_CACHE = cfg.CACHE_DIR / "link_facts_cache.json"
PLACE_TYPES = {"Q515", "Q6256", "Q486972", "Q515", "Q3957", "Q5119", "Q15284",
               "Q1549591", "Q532", "Q748331", "Q35657"}  # city/country/settlement/...

_search = None
_facts = None


def _load(path, ref):
    if ref[0] is None:
        ref[0] = json.load(open(path)) if path.exists() else {}
    return ref[0]


_S, _F = [None], [None]


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"))


def _year(s):
    if not s:
        return None
    try:
        return int(str(s).lstrip("+")[:4])
    except ValueError:
        return None


def wbsearch(mention, lang):
    """top candidate QIDs for a surface form (cached)."""
    c = _load(SEARCH_CACHE, _S)
    key = lang + "|" + mention
    if key in c:
        return c[key]
    import requests
    out = []
    try:
        r = requests.get(API, params={"action": "wbsearchentities", "search": mention,
                                      "language": lang, "uselang": lang, "format": "json",
                                      "limit": 8, "type": "item"},
                         headers=kb.UA, timeout=30)
        if r.status_code == 200:
            out = [x["id"] for x in r.json().get("search", [])]
    except Exception:
        out = []
    c[key] = out
    _save(SEARCH_CACHE, c)
    return out


def fetch_candidate_facts(qids):
    """batch-fetch type/temporal/coord facts for candidate QIDs (cached)."""
    c = _load(FACTS_CACHE, _F)
    todo = [q for q in qids if q not in c]
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        vals = " ".join("wd:" + q for q in batch)
        q = ("SELECT ?e ?type ?inception ?dissolution ?birth ?death ?coord WHERE { "
             "VALUES ?e { %s } "
             "OPTIONAL { ?e wdt:P31 ?type } OPTIONAL { ?e wdt:P571 ?inception } "
             "OPTIONAL { ?e wdt:P576 ?dissolution } OPTIONAL { ?e wdt:P569 ?birth } "
             "OPTIONAL { ?e wdt:P570 ?death } OPTIONAL { ?e wdt:P625 ?coord } }" % vals)
        agg = {qid: {"types": set(), "inception": None, "dissolution": None,
                     "birth": None, "death": None, "has_coord": False} for qid in batch}
        for b in kb._sparql(q):
            e = b["e"]["value"].split("/")[-1]
            if e not in agg:
                continue
            a = agg[e]
            if "type" in b:
                a["types"].add(b["type"]["value"].split("/")[-1])
            for k, p in (("inception", "inception"), ("dissolution", "dissolution"),
                         ("birth", "birth"), ("death", "death")):
                if p in b:
                    y = _year(b[p]["value"])
                    if y is not None and (a[k] is None or y < a[k]):
                        a[k] = y
            if "coord" in b:
                a["has_coord"] = True
        for qid, a in agg.items():
            a["is_human"] = "Q5" in a["types"]
            a["is_place"] = a["has_coord"] or bool(a["types"] & PLACE_TYPES)
            a["types"] = list(a["types"])
            c[qid] = a
        _save(FACTS_CACHE, c)
    return c


def _score(f, pub_year, etype, rank):
    s = (8 - rank) * 0.3                         # prominence (search order)
    if etype == "person":
        if not f.get("is_human"):
            return -1e9
        b, d = f.get("birth"), f.get("death")
        if pub_year and b and pub_year < b:
            s -= 8                                # not born yet at publication
        if pub_year and b and d and b <= pub_year <= d + 80:
            s += 3                                # plausibly contemporary
    else:
        if not f.get("is_place"):
            return -1e9
        inc, dis = f.get("inception"), f.get("dissolution")
        if pub_year and inc and pub_year < inc:
            s -= 8                                # didn't exist yet (modern Germany @1850)
        if pub_year and dis and pub_year > dis:
            s -= 4                                # already dissolved
        if pub_year and inc and pub_year >= inc:
            s += 2
        if f.get("has_coord"):
            s += 1
    return s


def link_entity(mentions, lang, pub_year, etype, facts):
    cands = []
    for m in mentions:
        cands = wbsearch(m, lang if lang in ("en", "de", "fr") else "en")
        if cands:
            break
    best, best_s = None, -1e8
    for rank, qid in enumerate(cands):
        s = _score(facts.get(qid, {}), pub_year, etype, rank)
        if s > best_s:
            best, best_s = qid, s
    return best


def resolve_pairs(pairs):
    """fill missing QIDs in place on pair.person/pair.place via temporal linking."""
    # gather NIL entities + their candidates, then batch-fetch candidate facts
    nils = {}
    for p in pairs:
        for ent in (p.person, p.place):
            if not ent.qid and ent.entity_id not in nils:
                nils[ent.entity_id] = (ent, p.language, _year(p.pub_date))
    all_cands = set()
    for ent, lang, _ in nils.values():
        for q in wbsearch(ent.surface, lang if lang in ("en", "de", "fr") else "en"):
            all_cands.add(q)
    facts = fetch_candidate_facts(sorted(all_cands))
    linked = 0
    for p in pairs:
        for ent in (p.person, p.place):
            if not ent.qid:
                q = link_entity(ent.mentions, p.language, _year(p.pub_date),
                                ent.etype, facts)
                if q:
                    ent.qid = q
                    linked += 1
    return linked
