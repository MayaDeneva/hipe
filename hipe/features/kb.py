"""Wikidata knowledge-base features for the at target.

Idea (maps onto at's 3 classes): pull EVERY geographic entity (anything with
coordinates P625) the person is linked to by ANY Wikidata property, then compare
to the mentioned place's coordinates.
  - person directly linked to the place        -> at=TRUE   signal
  - linked only to a place NEAR it (by coords)  -> at=PROBABLE signal
  - all known places far away                   -> at=FALSE  signal
This is orthogonal to the text (what is *known* vs what the document *says*).
"""
import json
import math
import time
from hipe import config as cfg

ENDPOINT = "https://query.wikidata.org/sparql"
UA = {"User-Agent": "HIPE2026-coursework/0.1 (research; mdeneva@theoremus.com)"}
CACHE_PATH = cfg.CACHE_DIR / "kb_cache.json"

KB_KEYS = ["kb_signal", "kb_direct", "kb_min_dist_km", "kb_near_25", "kb_near_100",
           "kb_near_500", "kb_n_places", "kb_log_dist"]

_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
    return _cache


def _save_cache():
    if _cache is not None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(_cache, open(CACHE_PATH, "w"))


def _sparql(query, retries=3):
    import requests
    for i in range(retries):
        try:
            r = requests.get(ENDPOINT, params={"query": query, "format": "json"},
                             headers=UA, timeout=60)
            if r.status_code == 429:
                time.sleep(2 * (i + 1)); continue
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception:
            if i == retries - 1:
                return []
            time.sleep(1.5 * (i + 1))
    return []


def _parse_point(value):
    # 'Point(2.3522 48.8566)' -> (lat, lon)
    try:
        lon, lat = value.replace("Point(", "").replace(")", "").split()
        return [float(lat), float(lon)]
    except Exception:
        return None


def person_places(qid):
    """[(place_qid, lat, lon)] for every located entity linked to the person."""
    c = _load_cache()
    key = "P:" + qid
    if key in c:
        return c[key]
    rows = _sparql("SELECT ?place ?coord WHERE { wd:%s ?p ?place . "
                   "?place wdt:P625 ?coord . } LIMIT 300" % qid)
    out = []
    for b in rows:
        pt = _parse_point(b["coord"]["value"])
        if pt:
            out.append([b["place"]["value"].split("/")[-1], pt[0], pt[1]])
    c[key] = out
    _save_cache()
    return out


def place_coord(qid):
    c = _load_cache()
    key = "L:" + qid
    if key in c:
        return c[key]
    rows = _sparql("SELECT ?coord WHERE { wd:%s wdt:P625 ?coord . } LIMIT 1" % qid)
    res = _parse_point(rows[0]["coord"]["value"]) if rows else None
    c[key] = res
    _save_cache()
    return res


def _haversine(a, b):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def kb_features(pair) -> dict:
    f = {k: 0 for k in KB_KEYS}
    f["kb_min_dist_km"] = 99999.0
    f["kb_log_dist"] = math.log1p(99999.0)
    pq, lq = pair.person.qid, pair.place.qid
    if not pq or not lq:
        return f
    places = person_places(pq)
    target = place_coord(lq)
    f["kb_n_places"] = len(places)
    if not places or target is None:
        return f
    f["kb_signal"] = 1
    if any(p[0] == lq for p in places):       # exact same entity -> direct
        f["kb_direct"] = 1
        dist = 0.0
    else:
        dist = min(_haversine(target, [p[1], p[2]]) for p in places)
    f["kb_min_dist_km"] = round(dist, 2)
    f["kb_log_dist"] = round(math.log1p(dist), 4)
    f["kb_near_25"] = int(dist <= 25)
    f["kb_near_100"] = int(dist <= 100)
    f["kb_near_500"] = int(dist <= 500)
    return f


# ---- entity glosses (description + type) for KGPool-style text injection ----
GLOSS_PATH = cfg.ROOT / "data" / "kb" / "glosses.json"
_glosses = None


def _find_glosses():
    from pathlib import Path
    # cfg.ROOT works locally; on Kaggle the package installs non-editable (ROOT ->
    # site-packages) while the repo is cloned to ./code, so search that too.
    for p in (GLOSS_PATH,
              Path.cwd() / "code" / "data" / "kb" / "glosses.json",
              Path.cwd() / "data" / "kb" / "glosses.json"):
        if p.exists():
            return p
    return None


def load_glosses():
    global _glosses
    if _glosses is None:
        p = _find_glosses()
        _glosses = json.load(open(p)) if p else {}
    return _glosses


def fetch_glosses(qids, max_len=60):
    """Wikidata English description (+ instance-of fallback) per QID -> glosses.json.
    A short gloss like 'French military leader' that we inject as text so the
    transformer's attention can use (or ignore) the KG fact."""
    g = load_glosses()
    todo = [q for q in qids if q and q not in g]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        vals = " ".join("wd:" + q for q in batch)
        rows = _sparql("SELECT ?e ?desc ?typeLabel WHERE { VALUES ?e { %s } "
                       "OPTIONAL { ?e schema:description ?desc . FILTER(LANG(?desc)='en') } "
                       "OPTIONAL { ?e wdt:P31 ?t . ?t rdfs:label ?typeLabel . "
                       "FILTER(LANG(?typeLabel)='en') } }" % vals)
        agg = {q: {"desc": None, "types": []} for q in batch}
        for b in rows:
            e = b["e"]["value"].split("/")[-1]
            if e not in agg:
                continue
            if "desc" in b and not agg[e]["desc"]:
                agg[e]["desc"] = b["desc"]["value"]
            if "typeLabel" in b:
                t = b["typeLabel"]["value"]
                if t not in agg[e]["types"]:
                    agg[e]["types"].append(t)
        for q, a in agg.items():
            gl = a["desc"] or (", ".join(a["types"][:2]) if a["types"] else "")
            g[q] = gl[:max_len]
        GLOSS_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(g, open(GLOSS_PATH, "w"), ensure_ascii=False)
    return g
