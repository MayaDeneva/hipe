# Embedding-SVM (local) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first real classical model — an SVM over multilingual sentence-embeddings of each person–place pair — and run it locally to produce a leaderboard row, training on sandbox (silver) and validating on the gold, test-domain newspapers set.

**Architecture:** A new `RelationModel` (`embedding_svm`) encodes a *pair-specific* templated string (`"<person> [SEP] <place> [SEP] <context>"`) with a multilingual sentence-transformer (disk-cached), then trains two independent `LinearSVC(class_weight="balanced")` heads (`at`, `isAt`). A small harness enhancement lets configs name explicit `train:`/`dev:` files (and lists, to combine languages) so we validate on the provided newspapers split instead of an internal random split. Everything plugs into the existing run protocol — no scorer/registry/leaderboard changes.

**Tech Stack:** Python 3.12, scikit-learn (`LinearSVC`), sentence-transformers (`paraphrase-multilingual-mpnet-base-v2`), numpy, pytest. Builds on the merged harness foundation.

## Global Constraints

- Python 3.12.
- Label spaces: `at ∈ {FALSE, PROBABLE, TRUE}`, `isAt ∈ {FALSE, TRUE}`. Models read gold via `pair.gold_at` / `pair.gold_isat`.
- The metric and scoring are unchanged: the harness scores via the vendored official `score_files`; do not add a second metric path.
- Class imbalance is severe (EDA: `isAt` 88:12, `at` 59/24/17) → every classifier head uses `class_weight="balanced"`.
- Embedding input is **pair-specific**: `f"{pair.person.surface} [SEP] {pair.place.surface} [SEP] {pair.context}"` (pairs sharing a document must not collapse to identical features).
- Default embedding model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- Encodings are disk-cached, keyed by `(model_name, sha1(text))`, so re-runs and the later Kaggle path are cheap and deterministic.
- Data setup (Option A): train on `sandbox/*-train`, validate on `newspapers/v1.0/*` (gold, same impresso domain as the official test).
- `RelationModel.predict` returns one dict per pair with keys `"at"`, `"isAt"` (and `"at_proba"`/`"isAt_proba"`, here `None` — probability calibration is deferred to the ensembling plan).
- No GPU and no network in unit tests: the sentence-transformer is loaded lazily and only on a real encode; all unit tests inject a fake encoder.

---

### Task 1: Harness support for multi-file train + explicit dev file

**Files:**
- Modify: `hipe/harness.py`
- Create: `tests/fixtures/mini_dev.jsonl`
- Test: `tests/test_harness.py` (add tests)

**Interfaces:**
- Consumes: existing `load_pairs`, `split_by_document`, `read_jsonl`, `score_files`, run registry.
- Produces: `run_experiment` now accepts `config["data"]["train"]` and optional `config["data"]["dev"]` as **either a string path or a list of string paths**. When `dev` is present, it is loaded as the validation set (no internal split) and the gold for scoring is drawn from the `dev` files; when absent, behavior is unchanged (internal document-grouped split of `train`). New module-level helpers `_as_paths(spec) -> list[str]` and `_load_pairs_spec(spec) -> list[Pair]`. `_write_subset` now takes a `sources` spec (str or list) instead of a single path.

- [ ] **Step 1: Create the dev fixture `tests/fixtures/mini_dev.jsonl`** (one line, distinct doc id `e1`):

```json
{"document_id": "e1", "media": {}, "source": "src/e.tsv", "language": "en", "date": "1900", "text": "Anna lived in Berlin for years.", "sampled_pairs": [{"pers_entity_id": "e1-anna", "pers_wikidata_QID": null, "pers_mentions_list": ["Anna"], "loc_entity_id": "e1-berlin", "loc_wikidata_QID": "Q64", "loc_mentions_list": ["Berlin"], "at": "TRUE", "at_explanation": "", "isAt": "TRUE", "isAt_explanation": ""}]}
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_harness.py`):

```python
def test_run_experiment_uses_explicit_dev_file(tmp_path):
    # train on mini fixture, validate on the separate dev fixture
    dev_fix = FIX.parent / "mini_dev.jsonl"
    config = {"data": {"train": str(FIX), "dev": str(dev_fix)},
              "model": {"name": "majority"}}
    result = run_experiment(config, now="2026-06-22_130000", runs_root=tmp_path)
    from hipe.data.load import read_jsonl
    rows = read_jsonl(Path(result["run_dir"]) / "predictions" / "dev_gold.jsonl")
    # dev gold comes from the dev file (doc e1), NOT from the train fixture
    assert [r["document_id"] for r in rows] == ["e1"]
    assert result["n_dev"] == 1


def test_run_experiment_accepts_list_of_train_files(tmp_path):
    dev_fix = FIX.parent / "mini_dev.jsonl"
    config = {"data": {"train": [str(FIX), str(dev_fix)], "dev": str(dev_fix)},
              "model": {"name": "majority"}}
    result = run_experiment(config, now="2026-06-22_130001", runs_root=tmp_path)
    # train pairs = mini (3) + mini_dev (1) = 4; dev pairs = mini_dev (1)
    assert result["n_dev"] == 1
    assert Path(result["run_dir"]).exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_harness.py::test_run_experiment_uses_explicit_dev_file -v`
Expected: FAIL (dev gold currently drawn from the train file / no `dev` handling)

- [ ] **Step 4: Edit `hipe/harness.py`** — replace the top of `run_experiment` (the data-loading + split block, lines that currently read `train_path`, do `load_pairs`, and `split_by_document`) and `_write_subset`. The full new file is:

```python
# hipe/harness.py
import json
from pathlib import Path
from hipe import config as cfg
from hipe.data.load import read_jsonl
from hipe.data.pairs import load_pairs, pair_key
from hipe.data.split import split_by_document
from hipe.data.submission import write_submission
from hipe.models import baselines  # noqa: F401  (registers majority/random)
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
        gold_sources = dev_spec
    else:
        dev_frac = data.get("dev_frac", 0.2)
        seed = data.get("seed", 0)
        train, dev = split_by_document(train, dev_frac=dev_frac, seed=seed)
        gold_sources = train_spec

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

    metrics = score_files(pred_dir / "dev_gold.jsonl", pred_dir / "dev.jsonl")
    at_recall = metrics["at"]["macro_recall"]
    isat_recall = metrics["isAt"]["macro_recall"]
    global_recall = metrics["global"]["macro_recall"]

    manifest = {"model": name, "config": config, "config_hash": cfg_hash,
                "now": now, "at_recall": at_recall, "isAt_recall": isat_recall,
                "global": global_recall, "n_dev": len(dev),
                "consistency": consistency_mode}
    runs.write_manifest(run_dir, manifest)
    runs.append_leaderboard(runs_root, {
        "run_id": run_dir.name, "timestamp": now, "model": name,
        "config_hash": cfg_hash, "data": str(train_spec),
        "at_recall": round(at_recall, 4), "isAt_recall": round(isat_recall, 4),
        "global": round(global_recall, 4), "n_dev": len(dev), "notes": ""})

    return {"run_dir": str(run_dir), "at_recall": at_recall,
            "isAt_recall": isat_recall, "global": global_recall, "n_dev": len(dev)}


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
```

Note: the `embedding_svm` model module does not exist until Task 4, so the harness must NOT import it yet — the code block above deliberately omits that import. Task 4 Step 4 adds the `from hipe.models import embedding_svm` registration line. Implement Tasks 1–4 in order.

- [ ] **Step 5: Run the new tests + the full harness test file**

Run: `python -m pytest tests/test_harness.py -v`
Expected: PASS (all existing harness tests still green + the 2 new ones)

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `python -m pytest -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add hipe/harness.py tests/fixtures/mini_dev.jsonl tests/test_harness.py
git commit -m "feat: harness supports multi-file train + explicit dev file"
```

---

### Task 2: Features package + pair text + ml dependency

**Files:**
- Modify: `pyproject.toml`
- Create: `hipe/features/__init__.py`
- Create: `hipe/features/text.py`
- Test: `tests/features/test_text.py`

**Interfaces:**
- Produces: `hipe.features.text.pair_text(pair) -> str` returning `"<person surface> [SEP] <place surface> [SEP] <context>"`.
- Adds an optional dependency group `ml` to `pyproject.toml` (`sentence-transformers`).

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_text.py
from hipe.data.schema import Entity, Pair
from hipe.features.text import pair_text


def test_pair_text_is_pair_specific():
    person = Entity("p1", "person", ["Joe"])
    place = Entity("l1", "place", ["Essex"])
    p = Pair(doc_id="d1", person=person, place=place,
             context="Joe was at Essex county.", language="en", pub_date=None)
    assert pair_text(p) == "Joe [SEP] Essex [SEP] Joe was at Essex county."


def test_pair_text_handles_empty_mentions():
    person = Entity("p1", "person", [])
    place = Entity("l1", "place", ["Paris"])
    p = Pair(doc_id="d", person=person, place=place, context="ctx",
             language="fr", pub_date=None)
    assert pair_text(p) == " [SEP] Paris [SEP] ctx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.features'`

- [ ] **Step 3: Create `hipe/features/__init__.py`**

```python
# hipe/features/__init__.py
```

- [ ] **Step 4: Write `hipe/features/text.py`**

```python
# hipe/features/text.py
def pair_text(pair) -> str:
    """Pair-specific text for embedding: person, place, then the context window.

    Including the person and place surfaces is essential — many pairs share a
    document context, so context alone would collapse them to identical features.
    """
    return f"{pair.person.surface} [SEP] {pair.place.surface} [SEP] {pair.context}"
```

- [ ] **Step 5: Add the `ml` optional dependency to `pyproject.toml`** — change the `[project.optional-dependencies]` block to:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
ml = ["sentence-transformers>=2.2"]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/features/test_text.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml hipe/features/__init__.py hipe/features/text.py tests/features/test_text.py
git commit -m "feat: features package + pair_text + ml extra (sentence-transformers)"
```

---

### Task 3: Embedding encoder with disk cache

**Files:**
- Create: `hipe/features/embeddings.py`
- Test: `tests/features/test_embeddings.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone, numpy only).
- Produces:
  - `hipe.features.embeddings.DEFAULT_MODEL: str` = `"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"`.
  - `hipe.features.embeddings.EmbeddingEncoder(model_name=DEFAULT_MODEL, cache_path=None, encode_fn=None)`. `encode_fn`, if given, is a callable `list[str] -> np.ndarray` used instead of loading a sentence-transformer (this is the test seam; the real model is loaded lazily only when `encode_fn` is None and a cache miss occurs). Method `encode(texts: list[str]) -> np.ndarray` returns one row per input text (order preserved), using and persisting a disk cache keyed by `sha1(text)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_embeddings.py
import numpy as np
from hipe.features.embeddings import EmbeddingEncoder, DEFAULT_MODEL


def _fake_encode_factory():
    calls = {"n": 0, "texts": []}
    def fn(texts):
        calls["n"] += 1
        calls["texts"].extend(texts)
        # deterministic 3-dim vector from text length
        return np.array([[len(t), len(t) % 7, 1.0] for t in texts], dtype=float)
    return fn, calls


def test_default_model_name():
    assert DEFAULT_MODEL == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def test_encode_returns_row_per_text_in_order():
    fn, _ = _fake_encode_factory()
    enc = EmbeddingEncoder(encode_fn=fn)
    out = enc.encode(["ab", "cde"])
    assert out.shape == (2, 3)
    assert out[0, 0] == 2 and out[1, 0] == 3


def test_encode_caches_and_dedupes(tmp_path):
    fn, calls = _fake_encode_factory()
    cache = tmp_path / "emb.pkl"
    enc = EmbeddingEncoder(cache_path=cache, encode_fn=fn)
    enc.encode(["x", "x", "yy"])          # only "x","yy" are unique -> 2 encoded
    assert sorted(calls["texts"]) == ["x", "yy"]
    # a second encoder loading the same cache encodes nothing new for seen text
    fn2, calls2 = _fake_encode_factory()
    enc2 = EmbeddingEncoder(cache_path=cache, encode_fn=fn2)
    enc2.encode(["x", "yy"])
    assert calls2["texts"] == []          # served entirely from disk cache
    assert cache.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.features.embeddings'`

- [ ] **Step 3: Write `hipe/features/embeddings.py`**

```python
# hipe/features/embeddings.py
import hashlib
import pickle
from pathlib import Path
import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _default_encode_fn(model_name):
    from sentence_transformers import SentenceTransformer  # lazy: only on real use
    model = SentenceTransformer(model_name)

    def fn(texts):
        return np.asarray(model.encode(list(texts), show_progress_bar=False))

    return fn


class EmbeddingEncoder:
    """Encode texts to vectors with a disk cache keyed by sha1(text)."""

    def __init__(self, model_name=DEFAULT_MODEL, cache_path=None, encode_fn=None):
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self._encode_fn = encode_fn
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = pickle.loads(self.cache_path.read_bytes())

    def _fn(self):
        if self._encode_fn is None:
            self._encode_fn = _default_encode_fn(self.model_name)
        return self._encode_fn

    @staticmethod
    def _key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts):
        texts = list(texts)
        unseen = [t for t in texts if self._key(t) not in self._cache]
        unique_unseen = list(dict.fromkeys(unseen))
        if unique_unseen:
            vecs = self._fn()(unique_unseen)
            for t, v in zip(unique_unseen, vecs):
                self._cache[self._key(t)] = np.asarray(v, dtype=float)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_bytes(pickle.dumps(self._cache))
        return np.vstack([self._cache[self._key(t)] for t in texts])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_embeddings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/features/embeddings.py tests/features/test_embeddings.py
git commit -m "feat: sentence-embedding encoder with disk cache"
```

---

### Task 4: EmbeddingSVM model

**Files:**
- Create: `hipe/models/embedding_svm.py`
- Modify: `hipe/harness.py` (add the registration import deferred from Task 1)
- Test: `tests/models/test_embedding_svm.py`

**Interfaces:**
- Consumes: `RelationModel`, `registry.register`, `EmbeddingEncoder`, `DEFAULT_MODEL`, `pair_text`, `config.CACHE_DIR`.
- Produces: `EmbeddingSVM(model_name=DEFAULT_MODEL, C=1.0, cache_path=None, _encoder=None)` registered as `"embedding_svm"`. `fit(train, dev=None)` encodes `pair_text` of each train pair and fits two `LinearSVC(class_weight="balanced")` heads on `gold_at` / `gold_isat`; a head with <2 classes in training falls back to predicting that single (or `"FALSE"`) constant. `predict(pairs)` returns `[{"at","isAt","at_proba":None,"isAt_proba":None}, ...]`. `_encoder` injects a fake encoder for tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_embedding_svm.py
import numpy as np
from hipe.data.schema import Entity, Pair
from hipe.models import registry
import hipe.models.embedding_svm  # noqa: F401  (registers the model)


class _FakeEncoder:
    """Maps text -> a separable vector by first label keyword present."""
    def encode(self, texts):
        rows = []
        for t in texts:
            # encode a strong signal: 'TRUEISH' vs 'FALSEISH' marker in context
            rows.append([1.0, 0.0] if "TRUEISH" in t else [0.0, 1.0])
        return np.asarray(rows, dtype=float)


def _pair(at, isat, marker):
    person = Entity("p", "person", ["X"])
    place = Entity("l", "place", ["Y"])
    return Pair(doc_id="d", person=person, place=place,
                context=marker, language="en", pub_date=None,
                gold_at=at, gold_isat=isat)


def test_embedding_svm_learns_separable_signal():
    train = ([_pair("TRUE", "TRUE", "TRUEISH") for _ in range(6)] +
             [_pair("FALSE", "FALSE", "FALSEISH") for _ in range(6)])
    m = registry.get_model("embedding_svm", _encoder=_FakeEncoder())
    m.fit(train)
    preds = m.predict([_pair("?", "?", "TRUEISH"), _pair("?", "?", "FALSEISH")])
    assert preds[0]["at"] == "TRUE" and preds[0]["isAt"] == "TRUE"
    assert preds[1]["at"] == "FALSE" and preds[1]["isAt"] == "FALSE"
    assert preds[0]["at_proba"] is None


def test_embedding_svm_single_class_fallback():
    train = [_pair("FALSE", "FALSE", "FALSEISH") for _ in range(4)]
    m = registry.get_model("embedding_svm", _encoder=_FakeEncoder())
    m.fit(train)                      # only one class present per target
    preds = m.predict([_pair("?", "?", "TRUEISH")])
    assert preds[0]["at"] == "FALSE" and preds[0]["isAt"] == "FALSE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_embedding_svm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.models.embedding_svm'`

- [ ] **Step 3: Write `hipe/models/embedding_svm.py`**

```python
# hipe/models/embedding_svm.py
from sklearn.svm import LinearSVC
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.embeddings import EmbeddingEncoder, DEFAULT_MODEL
from hipe.features.text import pair_text


class _Head:
    """One target classifier; constant fallback when <2 classes are present."""

    def __init__(self, C):
        self.C = C
        self.clf = None
        self.const = "FALSE"

    def fit(self, X, y):
        classes = sorted(set(y))
        if len(classes) < 2:
            self.const = classes[0] if classes else "FALSE"
            self.clf = None
        else:
            self.clf = LinearSVC(C=self.C, class_weight="balanced")
            self.clf.fit(X, y)

    def predict(self, X):
        if self.clf is None:
            return [self.const] * len(X)
        return list(self.clf.predict(X))


def _cache_path(model_name):
    slug = model_name.replace("/", "_")
    return cfg.CACHE_DIR / f"emb_{slug}.pkl"


@registry.register("embedding_svm")
class EmbeddingSVM(RelationModel):
    name = "embedding_svm"

    def __init__(self, model_name=DEFAULT_MODEL, C=1.0, cache_path=None, _encoder=None):
        if _encoder is not None:
            self.encoder = _encoder
        else:
            self.encoder = EmbeddingEncoder(
                model_name, cache_path=cache_path or _cache_path(model_name))
        self._at = _Head(C)
        self._isat = _Head(C)

    def fit(self, train, dev=None):
        X = self.encoder.encode([pair_text(p) for p in train])
        self._at.fit(X, [p.gold_at for p in train])
        self._isat.fit(X, [p.gold_isat for p in train])

    def predict(self, pairs):
        X = self.encoder.encode([pair_text(p) for p in pairs])
        at = self._at.predict(X)
        isat = self._isat.predict(X)
        return [{"at": a, "isAt": i, "at_proba": None, "isAt_proba": None}
                for a, i in zip(at, isat)]
```

- [ ] **Step 4: Re-add the registration import in `hipe/harness.py`** — add this line to the imports block (right after `from hipe.models import baselines  # noqa: F401  (registers majority/random)`):

```python
from hipe.models import embedding_svm  # noqa: F401  (registers embedding_svm)
```

- [ ] **Step 5: Run the model tests + full suite**

Run: `python -m pytest tests/models/test_embedding_svm.py -v && python -m pytest -q`
Expected: model tests PASS (2) and the full suite PASS (no regression; the harness now imports `embedding_svm` cleanly — sentence-transformers is NOT imported at module load, only lazily on a real encode).

- [ ] **Step 6: Commit**

```bash
git add hipe/models/embedding_svm.py hipe/harness.py tests/models/test_embedding_svm.py
git commit -m "feat: embedding_svm model (LinearSVC over sentence-embeddings, balanced)"
```

---

### Task 5: Config + end-to-end local run on real data

**Files:**
- Create: `configs/embedding_svm.yaml`
- Create: `configs/embedding_svm_sandboxdev.yaml`
- Test: `tests/test_embedding_svm_config.py`

**Interfaces:**
- Consumes: `run_experiment`, the configs, the real fetched data.
- Produces: two runnable configs and a smoke test asserting the config files are valid and select `embedding_svm` with the Option-A data layout.

- [ ] **Step 1: Write `configs/embedding_svm.yaml`** (Option A: train sandbox, validate newspapers/gold)

```yaml
data:
  train:
    - data/raw/HIPE-2026-data/data/sandbox/en-train.jsonl
    - data/raw/HIPE-2026-data/data/sandbox/de-train.jsonl
    - data/raw/HIPE-2026-data/data/sandbox/fr-train.jsonl
  dev:
    - data/raw/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-en.jsonl
    - data/raw/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl
    - data/raw/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-fr.jsonl
consistency: soft
model:
  name: embedding_svm
  model_name: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  C: 1.0
```

- [ ] **Step 2: Write `configs/embedding_svm_sandboxdev.yaml`** (comparison: validate on the silver sandbox dev)

```yaml
data:
  train:
    - data/raw/HIPE-2026-data/data/sandbox/en-train.jsonl
    - data/raw/HIPE-2026-data/data/sandbox/de-train.jsonl
    - data/raw/HIPE-2026-data/data/sandbox/fr-train.jsonl
  dev:
    - data/raw/HIPE-2026-data/data/sandbox/en-dev.jsonl
    - data/raw/HIPE-2026-data/data/sandbox/de-dev.jsonl
    - data/raw/HIPE-2026-data/data/sandbox/fr-dev.jsonl
consistency: soft
model:
  name: embedding_svm
  model_name: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  C: 1.0
```

- [ ] **Step 3: Write the config smoke test**

```python
# tests/test_embedding_svm_config.py
from pathlib import Path
import yaml

CFG = Path(__file__).resolve().parents[1] / "configs"


def test_embedding_svm_config_valid():
    c = yaml.safe_load((CFG / "embedding_svm.yaml").read_text())
    assert c["model"]["name"] == "embedding_svm"
    # Option A: train = sandbox, dev = newspapers
    assert all("sandbox" in p for p in c["data"]["train"])
    assert all("newspapers" in p for p in c["data"]["dev"])
    assert c["consistency"] == "soft"


def test_sandboxdev_config_valid():
    c = yaml.safe_load((CFG / "embedding_svm_sandboxdev.yaml").read_text())
    assert all("sandbox" in p for p in c["data"]["dev"])
```

- [ ] **Step 4: Run the config test + full suite**

Run: `python -m pytest tests/test_embedding_svm_config.py -v && python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Install the ml extra (one-time, downloads torch + sentence-transformers)**

Run: `pip install -e ".[ml,dev]"`
Expected: completes successfully (this pulls torch + transformers; can take a few minutes).

- [ ] **Step 6: Real end-to-end run (manual; downloads the model on first use)**

Run:
```bash
hipe run configs/embedding_svm.yaml
hipe leaderboard
```
Expected: the first command downloads `paraphrase-multilingual-mpnet-base-v2` (~1GB, once) and encodes ~7,400 texts (cached afterward), then prints `at_recall`, `isAt_recall`, `global` with `n_dev=104` (the newspapers docs). `hipe leaderboard` shows the `embedding_svm` row. Record the numbers in your report. (A second run is fast — embeddings are cached.)

- [ ] **Step 7: Optional comparison run**

Run: `hipe run configs/embedding_svm_sandboxdev.yaml && hipe leaderboard`
Expected: a second `embedding_svm` row validated on sandbox dev (`n_dev=156`), for silver-vs-gold comparison.

- [ ] **Step 8: Commit**

```bash
git add configs/embedding_svm.yaml configs/embedding_svm_sandboxdev.yaml tests/test_embedding_svm_config.py
git commit -m "feat: embedding_svm configs (Option A: sandbox train, newspapers gold dev) + smoke test"
```

---

## Self-Review

**Spec coverage (this plan = build step 1, embedding-based classical model):**
- Embedding-based classical model (§6.2b) → Tasks 3, 4 (encoder + EmbeddingSVM).
- Pair-specific embedding input (Global Constraints) → Task 2 (`pair_text`).
- `class_weight="balanced"` for imbalance (EDA) → Task 4 (`_Head`).
- Encoding cache (§5.5 feature store intent; Kaggle-GPU justification) → Task 3.
- Option-A data setup (train sandbox / dev newspapers gold) (§2b, conversation) → Tasks 1, 5.
- Provided dev file instead of internal split (harness enhancement) → Task 1.
- Soft consistency default carried through (§5.3) → unchanged harness path, exercised by configs.
- Official-scorer parity preserved (no new metric path) → Task 1 keeps the single `score_files` call.
- Probabilities for stacking are `None` for now → deferred to the ensembling plan (noted in Global Constraints); not a gap.
- *Deferred to later plans (correctly out of scope here):* Kaggle bridge (next plan), transformer, LLM, ensembles, hand-crafted feature store.

**Placeholder scan:** none — every code/step block is concrete. (Task 1 Step 4 intentionally defers one import to Task 4 Step 4, stated explicitly with the exact line.)

**Type consistency:** `pair_text(pair) -> str` defined in Task 2, used in Task 4. `EmbeddingEncoder(model_name, cache_path, encode_fn)` + `.encode(texts) -> np.ndarray` defined in Task 3, consumed in Task 4 (which injects `_encoder` in tests, or builds a real `EmbeddingEncoder`). `run_experiment(config, now, runs_root)` signature unchanged; `config["data"]["train"]`/`["dev"]` accept str|list per Task 1, matching the configs in Task 5. Model registered name `"embedding_svm"` consistent across Tasks 4 and 5 and the harness import.
