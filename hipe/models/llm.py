# hipe/models/llm.py
import json
import re
from collections import defaultdict
from hipe import config as cfg
from hipe.data.pairs import pair_key
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.markers import year_of

INSTRUCTION = (
    "You annotate a historical person-place relation task. For the given PERSON "
    "and PLACE mentioned in a historical TEXT (an excerpt from a newspaper or "
    "literary work), decide two relations:\n"
    "- \"at\": was the person EVER at this place (anytime in their life)? "
    "One of TRUE / PROBABLE / FALSE. "
    "TRUE = text or knowledge clearly indicates they were/are there; "
    "PROBABLE = plausible or likely but not explicit; "
    "FALSE = no indication, or the place is unrelated to the person.\n"
    "- \"isAt\": is the person AT the place in the specific situation this text "
    "describes (present there, in this context/moment)? One of TRUE / FALSE.\n"
    "For \"at\", use BOTH the text AND your own historical/world knowledge about "
    "the person, AND any provided 'Known places' facts — the text often will NOT "
    "state it explicitly even when the person truly was there. For \"isAt\", rely "
    "on what THIS text describes.\n"
    "Answer with ONLY a JSON object like {\"at\": \"PROBABLE\", \"isAt\": \"FALSE\"}."
)


def _block(pair, gloss_fn, places_fn=None, max_ctx=1200):
    lang = pair.language
    pg = gloss_fn(pair.person.qid, lang) if pair.person.qid else ""
    lg = gloss_fn(pair.place.qid, lang) if pair.place.qid else ""
    y = year_of(pair.pub_date)
    lines = [f"PERSON: {pair.person.surface}" + (f" — {pg}" if pg else ""),
             f"PLACE: {pair.place.surface}" + (f" — {lg}" if lg else "")]
    if places_fn and pair.person.qid:
        kp = places_fn(pair.person.qid, lang)
        if kp:
            lines.append("Known places associated with the person (Wikidata): "
                         + ", ".join(kp))
    if y:
        lines.append(f"DATE: {y}")
    lines.append(f"TEXT: {pair.context[:max_ctx]}")
    return "\n".join(lines)


@registry.register("llm")
class LLMModel(RelationModel):
    """Prompt a local Ollama instruct model per pair (task + entity glosses + date
    + context + few-shot), parse at/isAt. Responses are cached so re-runs are free.
    A genuinely decorrelated signal vs the fine-tuned transformer (reasoning vs
    pattern), for ensembling."""
    name = "llm"

    def __init__(self, model="qwen2.5:7b", endpoint="http://localhost:11434",
                 n_shots=3, cache_path=None, temperature=0.0,
                 prompt_version="v1", use_known_places=False, resolve_nil=False):
        self.model = model
        self.endpoint = endpoint
        self.n_shots = n_shots
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.use_known_places = use_known_places
        self.resolve_nil = resolve_nil
        self.cache_path = cache_path or (cfg.CACHE_DIR / f"llm_{model.replace(':', '_')}.json")
        self._cache = None
        self.shots = []

    def _load_cache(self):
        if self._cache is None:
            self._cache = json.load(open(self.cache_path)) if self.cache_path.exists() else {}
        return self._cache

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self._cache, open(self.cache_path, "w"), ensure_ascii=False)

    def _gloss(self, qid, lang):
        from hipe.features.kb import gloss_for
        return gloss_for(qid, lang)

    def _places(self, qid, lang):
        from hipe.features.kb import person_place_labels
        return person_place_labels(qid, lang)

    def fit(self, train, dev=None):
        # balanced few-shot exemplars: a couple per at-class
        by = defaultdict(list)
        for p in train:
            by[p.gold_at].append(p)
        self.shots = []
        for lab in cfg.AT_LABELS:
            self.shots += by[lab][:max(1, self.n_shots // 3 + 1)]
        self.shots = self.shots[:self.n_shots * 2]

    def _messages(self, pair):
        pf = self._places if self.use_known_places else None
        msgs = [{"role": "system", "content": INSTRUCTION}]
        for s in self.shots:
            msgs.append({"role": "user", "content": _block(s, self._gloss, pf, max_ctx=400)})
            msgs.append({"role": "assistant",
                         "content": json.dumps({"at": s.gold_at, "isAt": s.gold_isat})})
        msgs.append({"role": "user", "content": _block(pair, self._gloss, pf)})
        return msgs

    def _call(self, messages):
        import requests
        r = requests.post(self.endpoint + "/api/chat",
                          json={"model": self.model, "messages": messages,
                                "stream": False, "options": {"temperature": self.temperature}},
                          timeout=120)
        return r.json().get("message", {}).get("content", "")

    @staticmethod
    def _parse(text):
        at, isat = "FALSE", "FALSE"
        m = re.search(r'\{[^}]*\}', text, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                at = cfg.norm_label(str(d.get("at", "")).upper(), "at")
                isat = cfg.norm_label(str(d.get("isAt", d.get("isat", ""))).upper(), "isAt")
                return at, isat
            except Exception:
                pass
        ua = text.upper()
        if "PROBABLE" in ua:
            at = "PROBABLE"
        elif re.search(r'"AT"\s*:\s*"TRUE"|AT.{0,6}TRUE', ua):
            at = "TRUE"
        return at, isat

    def _key(self, p):
        return f"{self.prompt_version}|{self.model}|{p.doc_id}|{pair_key(p)}"

    def predict(self, pairs):
        if self.resolve_nil:
            from hipe.features import linking
            n = linking.resolve_pairs(pairs)
            print(f"[llm] NIL-linked {n} entities", flush=True)
        cache = self._load_cache()
        out, dirty = [], False
        for i, p in enumerate(pairs):
            ck = self._key(p)
            if ck in cache:
                txt = cache[ck]
            else:
                try:
                    txt = self._call(self._messages(p))
                except Exception:
                    txt = ""
                cache[ck] = txt
                dirty = True
                if i % 25 == 0:
                    self._save_cache()
                    print(f"[llm] {i}/{len(pairs)}", flush=True)
        # second pass: parse (kept separate so cache writes are batched)
        if dirty:
            self._save_cache()
        for p in pairs:
            txt = cache.get(self._key(p), "")
            at, isat = self._parse(txt)
            out.append({"at": at, "isAt": isat, "at_proba": None, "isAt_proba": None})
        return out
