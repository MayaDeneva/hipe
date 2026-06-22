# hipe/harness.py
import json
from pathlib import Path
from hipe import config as cfg
from hipe.data.load import read_jsonl
from hipe.data.pairs import load_pairs, pair_key
from hipe.data.split import split_by_document
from hipe.data.submission import write_submission
from hipe.models import baselines  # noqa: F401  (registers majority/random)
from hipe.models import embedding_svm  # noqa: F401  (registers embedding_svm)
from hipe.models import lookup  # noqa: F401  (registers llm_lookup)
from hipe.models import linguistic  # noqa: F401  (registers linguistic)
from hipe.models import xlmr  # noqa: F401  (registers xlmr)
from hipe.models import registry
from hipe.models.base import apply_consistency
from hipe.eval.scorer import score_files
from hipe.runs import registry as runs


def _as_paths(spec) -> list:
    return [spec] if isinstance(spec, str) else list(spec)


def _load_pairs_spec(spec) -> list:
    pairs = []
    for path in _as_paths(spec):
        pairs.extend(load_pairs(path))
    return pairs


def run_experiment(config: dict, now: str, runs_root=None) -> dict:
    runs_root = Path(runs_root) if runs_root is not None else cfg.RUNS_DIR
    data = config["data"]
    train_spec = data["train"]
    dev_spec = data.get("dev")

    train = _load_pairs_spec(train_spec)
    if dev_spec is not None:
        dev = _load_pairs_spec(dev_spec)
        # in-domain holdout: document-split the dev files, add the held-IN docs to
        # training and keep the held-OUT slice as the dev (so the model trains on
        # the test domain). Reproducible from the cloned data via the fixed seed.
        holdout = data.get("dev_holdout_frac")
        if holdout:
            dev_in, dev = split_by_document(
                dev, dev_frac=holdout, seed=data.get("dev_holdout_seed", 0))
            train = train + dev_in
        dev_doc_ids = {p.doc_id for p in dev}
        train = [p for p in train if p.doc_id not in dev_doc_ids]
        gold_sources = dev_spec
    else:
        dev_frac = data.get("dev_frac", 0.2)
        seed = data.get("seed", 0)
        train, dev = split_by_document(train, dev_frac=dev_frac, seed=seed)
        gold_sources = train_spec

    n_train = len(train)
    model_cfg = dict(config["model"])
    name = model_cfg.pop("name")
    model = registry.get_model(name, **model_cfg)
    model.fit(train, dev)

    consistency_mode = config.get("consistency", "soft")
    raw_preds = model.predict(dev)
    preds = {}
    for p, pred in zip(dev, raw_preds):
        preds[(p.doc_id, pair_key(p))] = apply_consistency(dict(pred), consistency_mode)

    cfg_hash = runs.config_hash(config)
    run_dir = runs.new_run_dir(name, cfg_hash, runs_root, now)
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    dev_docs = {p.doc_id for p in dev}
    _write_subset(gold_sources, dev_docs, pred_dir / "dev_gold.jsonl")
    write_submission(pred_dir / "dev_gold.jsonl", preds, pred_dir / "dev.jsonl")

    # per-pair class probabilities (null for models that don't emit them) — used
    # as richer stacking features than hard labels.
    with open(pred_dir / "probas.jsonl", "w") as f:
        for p, pred in zip(dev, raw_preds):
            f.write(json.dumps({"doc_id": p.doc_id, "pair_key": pair_key(p),
                                "at_proba": pred.get("at_proba"),
                                "isAt_proba": pred.get("isAt_proba")}) + "\n")

    metrics = score_files(pred_dir / "dev_gold.jsonl", pred_dir / "dev.jsonl")
    at_recall = metrics["at"]["macro_recall"]
    isat_recall = metrics["isAt"]["macro_recall"]
    global_recall = metrics["global"]["macro_recall"]

    manifest = {"model": name, "config": config, "config_hash": cfg_hash,
                "now": now, "at_recall": at_recall, "isAt_recall": isat_recall,
                "global": global_recall, "n_dev": len(dev), "n_train": n_train,
                "consistency": consistency_mode}
    runs.write_manifest(run_dir, manifest)
    runs.append_leaderboard(runs_root, {
        "run_id": run_dir.name, "timestamp": now, "model": name,
        "config_hash": cfg_hash, "data": str(train_spec),
        "at_recall": round(at_recall, 4), "isAt_recall": round(isat_recall, 4),
        "global": round(global_recall, 4), "n_dev": len(dev), "notes": ""})

    return {"run_dir": str(run_dir), "at_recall": at_recall,
            "isAt_recall": isat_recall, "global": global_recall, "n_dev": len(dev),
            "n_train": n_train}


def _write_subset(sources, keep_doc_ids, out_path):
    rows = []
    for src in _as_paths(sources):
        rows.extend(r for r in read_jsonl(src)
                    if str(r["document_id"]) in keep_doc_ids)
    for r in rows:
        for sp in r.get("sampled_pairs", []):
            sp["at"] = cfg.norm_label(sp.get("at"), "at")
            sp["isAt"] = cfg.norm_label(sp.get("isAt"), "isAt")
    with Path(out_path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
