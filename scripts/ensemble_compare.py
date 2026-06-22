"""Compare ensemble methods (stacking vs confidence-weighted vote) over the
in-domain base models, leakage-free via document-grouped 5-fold OOF CV, scored
with the official scorer. No new model runs: reuses saved predictions/probas.

Members are auto-discovered from run manifests (latest n_dev=401 run per model;
base vs large xlmr disambiguated by config model_name). Stacking uses each
model's class PROBABILITIES when available (probas.jsonl), else one-hot labels.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import glob
import tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from hipe.data.pairs import load_pairs, pair_key
from hipe.data.submission import write_submission
from hipe.eval.scorer import score_files
from hipe.models.base import apply_consistency
from hipe import config as cfg

AT, ISAT = cfg.AT_LABELS, cfg.ISAT_LABELS


def _discover():
    """latest n_dev=401 run per logical member -> {member: run_dir}."""
    cand = {}
    for mpath in glob.glob("runs/*/manifest.json"):
        m = json.load(open(mpath))
        if m.get("n_dev") != 401:
            continue
        name = m["model"]
        if name == "xlmr":   # disambiguate base vs large
            mn = m.get("config", {}).get("model", {}).get("model_name", "")
            name = "xlmr_large" if "large" in mn else "xlmr_base"
        key = (name, m["now"])
        cand.setdefault(name, []).append((m["now"], os.path.dirname(mpath)))
    members = {}
    for name, runs in cand.items():
        members[name] = sorted(runs)[-1][1]   # latest by timestamp
    return members


MEMBERS_ALL = _discover()
# ensemble these (skip majority/random)
ORDER = [m for m in ("llm_lookup", "embedding_svm", "xlmr_base", "xlmr_large")
         if m in MEMBERS_ALL]
print("members:", {m: MEMBERS_ALL[m] for m in ORDER})

GOLD = MEMBERS_ALL[ORDER[0]] + "/predictions/dev_gold.jsonl"


def _key(p):
    return (p.doc_id, pair_key(p))


# gold + per-member hard preds + probas, aligned by pair key
gold = load_pairs(GOLD)
gold_by = {_key(p): p for p in gold}
keys = list(gold_by)

hard, proba = {}, {}
for m in ORDER:
    rd = MEMBERS_ALL[m]
    hard[m] = {_key(p): (p.gold_at, p.gold_isat)
               for p in load_pairs(rd + "/predictions/dev.jsonl")}
    proba[m] = {}
    ppath = rd + "/predictions/probas.jsonl"
    if os.path.exists(ppath):
        for line in open(ppath):
            d = json.loads(line)
            proba[m][(d["doc_id"], d["pair_key"])] = (d["at_proba"], d["isAt_proba"])


def _vec(label, labels):
    return [1.0 if label == c else 0.0 for c in labels]


def feat(k, members):
    """per member: probability vector if available, else one-hot of hard label."""
    v = []
    for m in members:
        pa = proba.get(m, {}).get(k)
        if pa and pa[0] is not None and pa[1] is not None:
            v += [pa[0].get(c, 0.0) for c in AT] + [pa[1].get(c, 0.0) for c in ISAT]
        else:
            at, isat = hard[m][k]
            v += _vec(at, AT) + _vec(isat, ISAT)
    return v


y_at = np.array([gold_by[k].gold_at for k in keys])
y_isat = np.array([gold_by[k].gold_isat for k in keys])

docs = sorted({k[0] for k in keys})
rng = np.random.RandomState(0)
rng.shuffle(docs)
fold_of = {d: i % 5 for i, d in enumerate(docs)}
fold = np.array([fold_of[k[0]] for k in keys])


def base_macro(m, target):
    y = y_at if target == "at" else y_isat
    labels = AT if target == "at" else ISAT
    yp = [hard[m][k][0 if target == "at" else 1] for k in keys]
    return recall_score(y, yp, labels=labels, average="macro", zero_division=0)


def run_method(method, members):
    Xm = np.array([feat(k, members) for k in keys])
    out_at, out_isat = {}, {}
    for f in range(5):
        tr, te = fold != f, fold == f
        te_keys = [k for k, t in zip(keys, te) if t]
        if method == "stack":
            for labels, yv, store in ((AT, y_at, out_at), (ISAT, y_isat, out_isat)):
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(Xm[tr], yv[tr])
                for k, lab in zip(te_keys, clf.predict(Xm[te])):
                    store[k] = lab
        else:  # weighted vote, weights = per-member macro-recall on the train fold
            idx = [i for i, t in enumerate(tr) if t]
            w = {}
            for m in members:
                for target in ("at", "isat"):
                    yv = y_at if target == "at" else y_isat
                    labels = AT if target == "at" else ISAT
                    yp = [hard[m][keys[i]][0 if target == "at" else 1] for i in idx]
                    w[(m, target)] = recall_score(yv[idx], yp, labels=labels,
                                                  average="macro", zero_division=0)
            for k in te_keys:
                for target, labels, store in (("at", AT, out_at), ("isat", ISAT, out_isat)):
                    j = 0 if target == "at" else 1
                    sc = {c: 0.0 for c in labels}
                    for m in members:
                        sc[hard[m][k][j]] += w[(m, target)]
                    store[k] = max(sc, key=sc.get)
    return out_at, out_isat


def score(out_at, out_isat, tag):
    sub = {k: apply_consistency({"at": out_at[k], "isAt": out_isat[k]}, "soft")
           for k in keys}
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pred.jsonl")
        write_submission(GOLD, sub, pp)
        mt = score_files(GOLD, pp)
    print(f"{tag:24} at={mt['at']['macro_recall']:.4f}  "
          f"isAt={mt['isAt']['macro_recall']:.4f}  "
          f"global={mt['global']['macro_recall']:.4f}")
    return mt["global"]["macro_recall"]


print("\nbase-model macro-recall (full 401):")
for m in ORDER:
    has_p = bool(proba.get(m)) and any(v[0] for v in proba[m].values())
    print(f"  {m:14} at={base_macro(m,'at'):.4f}  isAt={base_macro(m,'isat'):.4f}"
          f"  {'[proba]' if has_p else '[hard]'}")

# sweep member subsets (stacking) to find the best combination
SUBSETS = [
    ["llm_lookup", "embedding_svm", "xlmr_base"],
    ["llm_lookup", "embedding_svm", "xlmr_large"],
    ["llm_lookup", "embedding_svm", "xlmr_base", "xlmr_large"],
    ["llm_lookup", "xlmr_base", "xlmr_large"],
    ["llm_lookup", "xlmr_large"],
    ["llm_lookup", "xlmr_base"],
    ["llm_lookup", "embedding_svm"],
]
print("\nstacking sweep over member subsets:")
best = (0, None)
for sub in SUBSETS:
    sub = [m for m in sub if m in MEMBERS_ALL]
    if len(sub) < 2:
        continue
    g = score(*run_method("stack", sub), "+".join(s.replace("_lookup", "").replace("embedding_", "")
                                                   for s in sub))
    if g > best[0]:
        best = (g, sub)
print(f"\nBEST stacking: {best[0]:.4f}  with {best[1]}")
print("weighted-vote (all 4):")
score(*run_method("vote", ORDER), "weighted-vote")
