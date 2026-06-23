"""Soft-ensemble the transformer with a frontier LLM whose per-pair at/isAt come
from a Kaggle Community Benchmarks run (hipe_preds.json: {"doc|pairkey": {at,isAt}}).

  python scripts/router_from_json.py <hipe_preds.json>

isAt always benefits from the transformer; at routes to the LLM where KB-grounded;
the soft isAt meta uses the LLM's at as a feature. Compares to transformer-only."""
import os, json, tempfile, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.linear_model import LogisticRegression
from hipe.data.pairs import load_pairs, pair_key
from hipe.data.submission import write_submission
from hipe.eval.scorer import score_files
from hipe.models.base import apply_consistency
from hipe import config as cfg

ALONE = "runs/2026-06-22_153045_xlmr_07e7698b"   # transformer: best at
CURR = "runs/2026-06-23_075753_xlmr_439f756d"    # transformer: best isAt (curriculum)
PREDS = sys.argv[1] if len(sys.argv) > 1 else "hipe_preds.json"
GOLD = CURR + "/predictions/dev_gold.jsonl"
AT, IS = cfg.AT_LABELS, cfg.ISAT_LABELS
llm_raw = json.load(open(PREDS))


def key(p): return (p.doc_id, pair_key(p))
def jkey(p): return f"{p.doc_id}|{pair_key(p)}"
gold = load_pairs(GOLD); keys = [key(p) for p in gold]
gold_at = {key(p): p.gold_at for p in gold}
gold_is = {key(p): p.gold_isat for p in gold}
kbflag = {key(p): int(bool(p.person.qid and p.place.qid)) for p in gold}
llm = {key(p): (cfg.norm_label(str(llm_raw.get(jkey(p), {}).get("at", "FALSE")).upper(), "at"),
                cfg.norm_label(str(llm_raw.get(jkey(p), {}).get("isAt", "FALSE")).upper(), "isAt"))
       for p in gold}
def hp(d): return {key(p): (p.gold_at, p.gold_isat) for p in load_pairs(d + "/predictions/dev.jsonl")}
alone, curr = hp(ALONE), hp(CURR)
ap, isp, cap = {}, {}, {}
for line in open(ALONE + "/predictions/probas.jsonl"):
    j = json.loads(line); ap[(j["doc_id"], j["pair_key"])] = j["at_proba"]
for line in open(CURR + "/predictions/probas.jsonl"):
    j = json.loads(line); k = (j["doc_id"], j["pair_key"])
    isp[k] = j["isAt_proba"]; cap[k] = j["at_proba"]

from sklearn.metrics import recall_score
g = [k for k in keys if kbflag[k]]
print(f"LLM preds loaded: {sum(1 for p in gold if jkey(p) in llm_raw)}/{len(keys)}  grounded={len(g)}")
for nm, src in (("transformer", alone), ("LLM", llm)):
    r = recall_score([gold_at[k] for k in keys], [src[k][0] for k in keys], labels=AT, average="macro", zero_division=0)
    rg = recall_score([gold_at[k] for k in g], [src[k][0] for k in g], labels=AT, average="macro", zero_division=0)
    print(f"  {nm:12} at(all)={r:.4f}  at(grounded)={rg:.4f}")
print()


def oh(v, labs): return [1. if v == c else 0. for c in labs]
def fold5():
    docs = sorted({k[0] for k in keys}); np.random.RandomState(0).shuffle(docs)
    return np.array([docs.index(k[0]) % 5 for k in keys])
def oof(X, y):
    fold = fold5(); pred = {}
    for f in range(5):
        tr, te = fold != f, fold == f
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        for k, lab in zip([k for k, t in zip(keys, te) if t], clf.predict(X[te])):
            pred[k] = lab
    return pred


Xat = np.array([[ap.get(k, {}).get(c, 0.) for c in AT] + oh(llm[k][0], AT) + [kbflag[k]] for k in keys])
at_meta = oof(Xat, np.array([gold_at[k] for k in keys]))
Xis = np.array([[isp.get(k, {}).get(c, 0.) for c in IS] + oh(llm[k][0], AT)
                + [cap.get(k, {}).get(c, 0.) for c in AT] + [kbflag[k]] for k in keys])
is_meta = oof(Xis, np.array([gold_is[k] for k in keys]))


def score(at_fn, is_fn, tag):
    sub = {k: apply_consistency({"at": at_fn(k), "isAt": is_fn(k)}, "soft") for k in keys}
    with tempfile.TemporaryDirectory() as td:
        pp = td + "/p.jsonl"; write_submission(GOLD, sub, pp); m = score_files(GOLD, pp)
    print(f"{tag:50} at={m['at']['macro_recall']:.4f} isAt={m['isAt']['macro_recall']:.4f} "
          f"global={m['global']['macro_recall']:.4f}")


Ta, Ti = lambda k: alone[k][0], lambda k: curr[k][1]
score(Ta, Ti, "BASELINE transformer-only")
score(lambda k: llm[k][0] if kbflag[k] else alone[k][0], Ti, "at<-LLM-if-grounded | isAt<-transformer")
score(lambda k: at_meta[k], Ti, "at<-META | isAt<-transformer")
score(Ta, lambda k: is_meta[k], "at<-transformer | isAt<-SOFT-META")
score(lambda k: at_meta[k], lambda k: is_meta[k], "at<-META | isAt<-SOFT-META  (full soft ensemble)")
