# HIPE-2026 Relation-Extraction Experiment Harness — Design

**Date:** 2026-06-21
**Course:** Откриване на знания в текст (NLP)
**Competition:** CLEF HIPE-2026 — person–place relation extraction in multilingual historical documents
**Author:** Maya Deneva

## 1. Goal

Build a **config-driven experiment harness** that runs, scores, and persists every relation-extraction approach we try, so that:

- **Maxing the competition metric** (global macro-recall) is the primary objective; the harness exists to make iteration fast and results trustworthy.
- **Nothing is ever lost**: every run's config, predictions, metrics, and artifacts are saved and comparable in a committed leaderboard.

This is the project's backbone. Individual models can start as stubs and improve over the semester without changing the harness.

## 2. The task (fixed by the competition)

For each `(person, place)` pair in a historical document, predict two relations:

- **`at`** → `TRUE | PROBABLE | FALSE` — was the person ever at this place?
- **`isAt`** → `TRUE | FALSE` — are they there in this text's context?

**Official metric** (from the shared task's `evaluation_utils.calculate_metrics`): macro-recall computed **per target**, then **global = mean(macro_recall(at), macro_recall(isAt))**. Accuracy is secondary. The competition *also* scores efficiency (model size, inference speed) — so heavier is not automatically better.

**Data**: UTF-8 `.jsonl`, one document per line. Each doc has `document_id`, `media`, `language`, `date`, OCR'd `text` (noisy, historical), and `sampled_pairs[]`. Each pair has `pers_entity_id`, `pers_wikidata_QID` (often `null`), `pers_mentions_list`, the `loc_*` equivalents, and gold `at` / `isAt` (+ explanation fields).

**Splits**: `newspapers/v1.0` (en/de/fr), `sandbox` (LLM-auto-annotated extra training, en/de/fr), and later `literaryworks` (16–18c French — the "surprise" generalization set, our Set B). Labels are **imbalanced** (e.g. en-train `at`: FALSE 239 / PROBABLE 159 / TRUE 98; `isAt`: FALSE 419 / TRUE 77) — which is exactly why macro-recall is the metric and why the rare TRUE/PROBABLE classes matter most.

## 3. Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary objective | Max global macro-recall | User priority; harness serves it |
| Architecture style | Config-driven Python package + CLI + file registry | One mental model, reproducible, offline, git-committable |
| Run tracking | File-based run registry + committed `leaderboard.csv` | Zero deps, works across Mac/Kaggle/Ollama, "nothing lost" |
| Compute | Mac (orchestration, classical ML, eval); Kaggle GPU (transformers); Ollama→Claude (LLM) | Matches available resources |
| LLM client | `litellm` | Single client for both local Ollama and Claude API via a model string |
| `at` / `isAt` | Two **independent** classifiers | Different label sets, different signals |
| Consistency rule | `isAt == TRUE ⇒ at = TRUE` | Logical: present-in-context implies was-there; applied post-prediction to all models |
| KG enrichment | Optional, off-by-default feature family | Empirically adds little (observed in prior `bz` project); keep only as an ablation |
| Reuse | Port `data/`, `enrich/`, `linking/`, `eval/metrics.py` from `bz/hipe2026` | Clean, already macro-recall-aware |
| Data acquisition | `scripts/fetch_data.py` — clone/pull the public `hipe-eval/HIPE-2026-data` repo into `data/raw/`, pinned to a release/commit | Simpler and more transparent than a git submodule (mirrors how `bz` did a manual clone) |

## 4. Repository layout

```
hipe/                      # installable package (pip install -e .)
  data/
    schema.py              # Entity, Document, Pair        (ported from bz)
    load.py                # read_jsonl, load_documents     (ported)
    pairs.py               # pair generation, context window, entity interning (ported)
    submission.py          # write predictions back into official jsonl shape
  features/
    base.py                # FeatureExtractor ABC
    lexical.py             # tokens/words between mentions
    syntactic.py           # token distance, ordering
    contextual.py          # movement / stay verb cues near mentions
    kg.py                  # Wikidata-derived features (OPTIONAL, off by default)
    store.py               # compute-once, parquet cache keyed by content-hash
  enrich/                  # Wikidata REST fetch + disk cache (ported, optional path)
  linking/                 # NIL -> QID resolution (ported, optional path)
  models/
    base.py                # RelationModel ABC
    registry.py            # name -> class
    majority.py random.py  # trivial baselines (scorer-parity sanity)
    sklearn_model.py       # LogReg / RF / XGBoost / SVM over feature store
    transformer.py         # XLM-R / mBERT (train on Kaggle, load weights to predict)
    llm.py                 # litellm few-shot, structured output
    ensemble.py            # voting + stacking over saved OOF/test predictions
  llm/
    client.py              # litellm wrapper: complete(prompt, schema), response cache
  eval/
    metrics.py             # macro_recall, confusion          (ported)
    scorer.py              # wraps official evaluation_utils for parity
  runs/
    registry.py            # create run dir, write manifest, append leaderboard row
  cli.py                   # `hipe run`, `hipe score`, `hipe leaderboard`, `hipe ingest`
configs/                   # one YAML per experiment
  data.yaml
  baseline_majority.yaml
  sklearn_xgb.yaml
  transformer_xlmr.yaml
  llm_ollama.yaml
  ensemble_stack.yaml
runs/                      # OUTPUT (artifacts gitignored, manifests committed)
  2026-06-21_xgb_<hash>/
    config.yaml
    predictions/{en,de,fr}.jsonl
    metrics.json
    oof.parquet
    model.pkl
    log.txt
  leaderboard.csv          # COMMITTED ledger — one row per run
kaggle/
  export_job.py            # package code+config+data+requirements for a GPU notebook
  ingest.py                # pull weights/predictions/metrics into a runs/ folder
  notebook_template.ipynb  # trains transformer, writes artifacts
scripts/
  fetch_data.py            # clone/pull hipe-eval/HIPE-2026-data into data/raw/ (pinned commit/tag)
data/                      # raw HIPE data (downloaded, gitignored) + cache/
tests/
```

The KG/ontology temporal-reasoning core from the prior `bz` project is treated as **one optional approach/feature family**, not the backbone.

## 5. Core contracts

### 5.1 Canonical unit — `Pair`

All approaches operate on `list[Pair]` (ported): `doc_id`, `person`, `place`, `context` (window around mentions), `language`, `pub_date`, `gold_at`, `gold_isat`, `features: dict`.

### 5.2 Model interface

```python
Prediction = TypedDict("Prediction", {
    "at": str,        # TRUE | PROBABLE | FALSE
    "isAt": str,      # TRUE | FALSE
    "at_proba": dict[str, float] | None,    # optional, enables stacking
    "isAt_proba": dict[str, float] | None,
})

class RelationModel(ABC):
    name: str
    def fit(self, train: list[Pair], dev: list[Pair] | None = None) -> None: ...
    def predict(self, pairs: list[Pair]) -> list[Prediction]: ...
    def save(self, dir: Path) -> None: ...      # default: pickle
    @classmethod
    def load(cls, dir: Path) -> "RelationModel": ...
```

- `at` and `isAt` are **two independent classifiers** (jointly or as two heads internally — hidden behind the interface).
- `majority` and `random` implement the same interface → scorer parity is proven before any real model exists.
- Models register by name in `models/registry.py`; configs select by name.

### 5.3 Consistency rule

A single post-prediction normalization, applied by the run protocol to **every** model's output before scoring and submission:

```
if pred["isAt"] == "TRUE": pred["at"] = "TRUE"
```

Centralized so every approach benefits identically and the rule is tested once.

### 5.4 Run protocol — `hipe run config.yaml`

Deterministic, identical for every approach:

1. Resolve data per `config.data` (splits/languages) — cached.
2. Build/load features via the feature store (no-op for raw-text models).
3. `model.fit(train, dev)`; predict on dev **and** test.
4. Apply the consistency rule (5.3).
5. Write predictions into **official `.jsonl`** (round-trip input docs, fill `at`/`isAt`).
6. Score with `eval/scorer.py` — a thin wrapper around the official `evaluation_utils.calculate_metrics` so our local number equals the leaderboard number. Store confusion matrices.
7. Persist `runs/<date>_<model>_<confighash>/`: `config.yaml`, `predictions/*.jsonl`, `metrics.json`, `oof.parquet`, `model.pkl`, `log.txt`.
8. Append one row to **`leaderboard.csv`** (committed): `run_id, timestamp, model, config_hash, data_version, at_recall, isAt_recall, global, size_mb, infer_ms, git_sha, notes`.
9. **Idempotent**: same `config_hash` short-circuits unless `--force` — nothing silently overwritten.

### 5.5 Feature store

Features computed once per `(pair, featureset)` and cached to **parquet keyed by content-hash**. Families (from the proposal): **lexical**, **syntactic**, **contextual**, **kg** (optional). sklearn + ensemble models read the store; transformer/LLM use raw context but may append KG features when explicitly enabled.

### 5.6 Validation strategy

- **Document-grouped, label-stratified split** — no pair from a document leaks across train/dev.
- **Per-language and cross-lingual** evaluation; held-out `literaryworks`-style check for the generalization (Set B) angle.
- Every base model emits **out-of-fold (OOF) predictions** (`oof.parquet`) so stacking/voting train on honest held-out predictions, not training-set leakage.

### 5.7 LLM provider (litellm)

`llm/client.py` wraps **litellm**: `complete(prompt, schema)` targets Ollama locally or Claude via a model string in the config. Prompt templates are versioned; responses cached by `(prompt_hash, model)` so re-runs are free and deterministic.

### 5.8 Kaggle bridge

- `export_job` packages {code, config, train data, requirements} for a GPU notebook (`notebook_template.ipynb`) that trains XLM-R/mBERT and writes `weights/ + predictions.jsonl + metrics.json`.
- `ingest` pulls those into a normal `runs/` folder — a Kaggle-trained transformer is **indistinguishable** from a local run in the leaderboard.

### 5.9 Ensembles as tracked runs

`voting` and `stacking` are `RelationModel`s that consume **other runs' OOF/test predictions by `run_id`**. "ML + Transformer → meta-LogReg" (stacking) and majority voting are themselves `hipe run` invocations with their own leaderboard rows.

## 6. Approaches (all behind the one interface)

**Framing vs. the RE literature** (per nlpprogress.com/relationship_extraction): this is **relation *classification*** — entities are given, we label the pair — not joint entity-relation extraction. So the joint encoder-decoder / RL methods (WDec, HRLRE on NYT/DocRED) are out of scope. The directly-relevant SOTA is the **entity-marker BERT family** (R-BERT, Matching-the-Blanks, LUKE) used in approach 3. Dependency-parse methods (A-GCN, SDP-LSTM) are deprioritized: parsing OCR-noisy multilingual historical text is unreliable; dependency features stay an optional `features/syntactic.py` extra. Classical SVM/gradient-boosting with embeddings is a validated competitive baseline (approach 2).


1. **Trivial baselines** — `majority`, `random` (scorer-parity sanity, leaderboard floor).
2. **Classical ML** — LogReg / RandomForest / XGBoost / SVM over the feature store.
3. **Transformer** — **entity-marker (R-BERT-style) XLM-R / mBERT**: insert special boundary tokens around the person and place mentions in the context and classify from those token positions (concatenated with `[CLS]`), with two heads (`at`, `isAt`). This entity-marker scheme — shared by R-BERT, Matching-the-Blanks, and LUKE — is the SOTA technique for relation *classification* on SemEval-2010/TACRED-style benchmarks and consistently beats plain `[CLS]` fine-tuning. Trained on Kaggle GPU.
4. **LLM** — litellm few-shot / structured-output prompting (Ollama default, Claude fallback).
5. **Ensembles** — voting + stacking over saved OOF/test predictions.
6. **KG-enriched ablation** — classical ML + optional `features/kg.py`, to quantify (likely small) KG contribution.

## 7. Error handling & reproducibility

- Config hashing + idempotent runs; `--force` to recompute.
- Seeds fixed and recorded in each manifest; `git_sha` stored per run.
- Disk caches (features, Wikidata, LLM responses) keyed by content-hash → safe to delete, rebuilt deterministically.
- Official-format prediction files validated against the shared-task JSON schema before scoring.
- Local score == official score guaranteed by wrapping the official scorer.

## 8. Testing

- Port and keep the prior project's data/eval tests.
- Scorer-parity test: harness metrics on a fixed prediction file match the official script's output exactly.
- Consistency-rule test: `isAt=TRUE` forces `at=TRUE` in all paths.
- Round-trip test: load → predict → write official jsonl → re-load yields valid, schema-conformant submissions.
- Split-leakage test: no document appears in both train and dev.

## 9. Out of scope (YAGNI)

- MLflow / W&B dashboards (file registry chosen instead).
- Heavy orchestration frameworks (Hydra/DVC/Snakemake).
- Notebook-per-approach as the backbone (a single Kaggle training notebook is ingested into the registry instead).
- Reliance on KG enrichment in the main path.

## 10. First milestones

1. `scripts/fetch_data.py` (pinned clone/pull); port `data/`, `eval/metrics.py`; wrap official scorer; wire the run registry + `leaderboard.csv`.
2. `majority`/`random` baselines green end-to-end with scorer parity.
3. Feature store + XGBoost over lexical/syntactic/contextual features → first real leaderboard rows.
4. Transformer via Kaggle bridge; LLM via litellm.
5. Voting + stacking ensembles.
