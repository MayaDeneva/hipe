"""Official-test soft ensemble: fit the at/isAt metas on ALL in-domain (401) using
the curriculum transformer probas + Claude's in-domain preds, then APPLY to the 638
official impresso-test pairs (curriculum official probas + Claude official preds).
Scores with the vendored official scorer.

  python scripts/official_ensemble.py [hipe_preds_official.json]
"""
import json, tempfile, sys
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from hipe.data.pairs import load_pairs, pair_key
from hipe.data.submission import write_submission
from hipe.eval.scorer import score_files
from hipe.models.base import apply_consistency
from hipe import config as cfg

IN_CURR = "runs/2026-06-23_075753_xlmr_439f756d"                              # in-domain curriculum
OFF_CURR = "runs/official_curr_dl/runs/2026-06-23_175509_xlmr_5934527d"       # official curriculum
IN_LLM = "hipe_preds.json"                                                   # Claude in-domain
OFF_LLM = sys.argv[1] if len(sys.argv) > 1 else "hipe_preds_official.json"   # Claude official
AT, IS = cfg.AT_LABELS, cfg.ISAT_LABELS


def k(p): return (p.doc_id, pair_key(p))
def jk(p): return f"{p.doc_id}|{pair_key(p)}"
def oh(v, labs): return [1. if v == c else 0. for c in labs]


def load(curr_dir, llm_path):
    gold = load_pairs(curr_dir + "/predictions/dev_gold.jsonl")
    keys = [k(p) for p in gold]
    raw = json.load(open(llm_path))
    llm = {k(p): cfg.norm_label(str(raw.get(jk(p), {}).get("at", "FALSE")).upper(), "at") for p in gold}
    lli = {k(p): cfg.norm_label(str(raw.get(jk(p), {}).get("isAt", "FALSE")).upper(), "isAt") for p in gold}
    kb = {k(p): int(bool(p.person.qid and p.place.qid)) for p in gold}
    lang = {k(p): p.language for p in gold}
    cap, isp = {}, {}
    for line in open(curr_dir + "/predictions/probas.jsonl"):
        j = json.loads(line); kk = (j["doc_id"], j["pair_key"])
        cap[kk] = j["at_proba"]; isp[kk] = j["isAt_proba"]
    ga = {k(p): p.gold_at for p in gold}; gi = {k(p): p.gold_isat for p in gold}
    return dict(gold=gold, keys=keys, llm=llm, lli=lli, kb=kb, lang=lang, cap=cap, isp=isp, ga=ga, gi=gi)


def Xat(d, keys): return np.array([[d["cap"].get(x, {}).get(c, 0.) for c in AT] + oh(d["llm"][x], AT) + [d["kb"][x]] for x in keys])
# isAt meta: transformer isAt-proba + Claude isAt + Claude at + transformer at-proba
# (the user's bidirectional cross-help: transformer helps Claude's isAt, Claude helps at)
def Xis(d, keys): return np.array([[d["isp"].get(x, {}).get(c, 0.) for c in IS] + oh(d["lli"][x], IS)
                                   + oh(d["llm"][x], AT) + [d["cap"].get(x, {}).get(c, 0.) for c in AT]
                                   + [d["kb"][x]] for x in keys])


tr = load(IN_CURR, IN_LLM)
te = load(OFF_CURR, OFF_LLM)
n_llm = sum(1 for p in te["gold"] if jk(p) in json.load(open(OFF_LLM)))
print(f"fit on {len(tr['keys'])} in-domain -> apply to {len(te['keys'])} official ({n_llm} have LLM preds)")

at_clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xat(tr, tr["keys"]), [tr["ga"][x] for x in tr["keys"]])
is_clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xis(tr, tr["keys"]), [tr["gi"][x] for x in tr["keys"]])
at_pred = dict(zip(te["keys"], at_clf.predict(Xat(te, te["keys"]))))
is_pred = dict(zip(te["keys"], is_clf.predict(Xis(te, te["keys"]))))

GOLD = OFF_CURR + "/predictions/dev_gold.jsonl"
curr_hard = {k(p): (p.gold_at, p.gold_isat) for p in load_pairs(OFF_CURR + "/predictions/dev.jsonl")}


def score(sub, tag):
    s = {x: apply_consistency(dict(sub(x)), "soft") for x in te["keys"]}
    with tempfile.TemporaryDirectory() as td:
        pp = td + "/p.jsonl"; write_submission(GOLD, s, pp); m = score_files(GOLD, pp)
    print(f"{tag:34} at={m['at']['macro_recall']:.4f} isAt={m['isAt']['macro_recall']:.4f} global={m['global']['macro_recall']:.4f}")
    return m["global"]["macro_recall"]


score(lambda x: {"at": curr_hard[x][0], "isAt": curr_hard[x][1]}, "OFFICIAL transformer-only")
score(lambda x: {"at": at_pred[x], "isAt": is_pred[x]}, "OFFICIAL soft ensemble")
# aggregate over all 638; official overall-test-a is mean-of-per-language (close proxy).
# per-`at`-class recall to confirm the PROBABLE recovery transfers:
from sklearn.metrics import recall_score
yk = te["keys"]
print("ensemble at per-class recall:",
      dict(zip(AT, recall_score([te["ga"][x] for x in yk], [at_pred[x] for x in yk],
                                labels=AT, average=None, zero_division=0).round(3))))
