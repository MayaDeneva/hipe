"""Compare ensemble methods (stacking vs confidence-weighted vote) over the three
in-domain base models, leakage-free via document-grouped 5-fold OOF CV, scored
with the official scorer. No new model runs: reuses the saved predictions."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from hipe.data.pairs import load_pairs, pair_key
from hipe.data.submission import write_submission
from hipe.eval.scorer import score_files
from hipe import config as cfg
from hipe.models.base import apply_consistency

RUNS = {
    "llm":  "runs/2026-06-22_175710_llm_lookup_eb9720af",
    "svm":  "runs/2026-06-22_175716_embedding_svm_afbde9a9",
    "xlmr": "runs/2026-06-22_145951_xlmr_8ead8075",
}
MODELS = list(RUNS)
GOLD = RUNS["xlmr"] + "/predictions/dev_gold.jsonl"
AT, ISAT = cfg.AT_LABELS, cfg.ISAT_LABELS


def _key(p):
    return (p.doc_id, pair_key(p))


def _onehot(label, labels):
    return [1.0 if label == c else 0.0 for c in labels]


# ---- load gold + each model's predictions, aligned by pair key ----
gold = load_pairs(GOLD)
gold_by = {_key(p): p for p in gold}
keys = list(gold_by)
preds = {m: {} for m in MODELS}
for m in MODELS:
    for p in load_pairs(RUNS[m] + "/predictions/dev.jsonl"):
        preds[m][_key(p)] = (p.gold_at, p.gold_isat)   # loader reads pred at/isAt here

# feature matrix: per pair, each model's at+isAt prediction one-hot encoded
def feat(k):
    v = []
    for m in MODELS:
        at, isat = preds[m][k]
        v += _onehot(at, AT) + _onehot(isat, ISAT)
    return v

X = np.array([feat(k) for k in keys])
y_at = np.array([gold_by[k].gold_at for k in keys])
y_isat = np.array([gold_by[k].gold_isat for k in keys])

# ---- document-grouped 5-fold split ----
docs = sorted({k[0] for k in keys})
rng = np.random.RandomState(0)
rng.shuffle(docs)
fold_of_doc = {d: i % 5 for i, d in enumerate(docs)}
fold = np.array([fold_of_doc[k[0]] for k in keys])


def macro_recall(model, target):
    """per-model macro-recall on given indices, used as vote weight."""
    y = y_at if target == "at" else y_isat
    labels = AT if target == "at" else ISAT
    yp = [preds[model][k][0 if target == "at" else 1] for k in keys]
    return recall_score(y, yp, labels=labels, average="macro", zero_division=0)


def run_method(method):
    """return OOF predicted (at, isat) per key."""
    out_at, out_isat = {}, {}
    for f in range(5):
        tr, te = fold != f, fold == f
        tr_keys = [k for k, m in zip(keys, tr) if m]
        te_keys = [k for k, m in zip(keys, te) if m]
        if method == "stack":
            for target, labels, yv, store in (("at", AT, y_at, out_at),
                                              ("isat", ISAT, y_isat, out_isat)):
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(X[tr], yv[tr])
                pr = clf.predict(X[te])
                for k, lab in zip(te_keys, pr):
                    store[k] = lab
        else:  # weighted vote, weights = per-model macro-recall on the TRAIN fold
            w = {}
            for m in MODELS:
                for target in ("at", "isat"):
                    yv = y_at if target == "at" else y_isat
                    labels = AT if target == "at" else ISAT
                    idx = [i for i, t in enumerate(tr) if t]
                    yp = [preds[m][keys[i]][0 if target == "at" else 1] for i in idx]
                    w[(m, target)] = recall_score(yv[idx], yp, labels=labels,
                                                  average="macro", zero_division=0)
            for k in te_keys:
                for target, labels, store in (("at", AT, out_at), ("isat", ISAT, out_isat)):
                    score = {c: 0.0 for c in labels}
                    j = 0 if target == "at" else 1
                    for m in MODELS:
                        score[preds[m][k][j]] += w[(m, target)]
                    store[k] = max(score, key=score.get)
    return out_at, out_isat


def score(out_at, out_isat, tag):
    sub = {}
    for k in keys:
        d = apply_consistency({"at": out_at[k], "isAt": out_isat[k]}, "soft")
        sub[k] = d
    with tempfile.TemporaryDirectory() as td:
        pred_path = os.path.join(td, "pred.jsonl")
        write_submission(GOLD, sub, pred_path)
        mt = score_files(GOLD, pred_path)
    print(f"{tag:22} at={mt['at']['macro_recall']:.4f}  "
          f"isAt={mt['isAt']['macro_recall']:.4f}  "
          f"global={mt['global']['macro_recall']:.4f}")
    return mt["global"]["macro_recall"]


print("base-model macro-recall (full 401):")
for m in MODELS:
    print(f"  {m:6} at={macro_recall(m,'at'):.4f}  isAt={macro_recall(m,'isat'):.4f}")
print()
score(*run_method("stack"), "ENSEMBLE stacking")
score(*run_method("vote"), "ENSEMBLE weighted-vote")
