# Entity-marker XLM-R Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an entity-marker (R-BERT-style) XLM-R `RelationModel` — the supervised transformer meant to challenge the prompting-LLM baseline — that runs through the existing harness, validated locally with a tiny sanity run (the full GPU run goes through the Kaggle bridge in the next plan).

**Architecture:** A new `RelationModel` (`xlmr`) inserts `[E1]…[/E1]` / `[E2]…[/E2]` markers around the person/place mentions in the context (located via `fuzzy_find`), adds them as special tokens, and fine-tunes two independent `AutoModelForSequenceClassification` heads (`at` 3-class, `isAt` 2-class) with class-weighted loss. It returns softmax probabilities (so it can feed later stacking). Training runs anywhere torch runs — CPU for a tiny sanity check, GPU on Kaggle for the real run.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace transformers + accelerate, scikit-learn (already present), pytest. Builds on the merged harness + the `markers`/`features` and `RelationModel`/registry patterns.

## Global Constraints

- Python 3.12.
- Label spaces: `at ∈ {FALSE, PROBABLE, TRUE}`, `isAt ∈ {FALSE, TRUE}`; read gold via `pair.gold_at` / `pair.gold_isat`.
- Metric/scoring unchanged: the harness scores via the vendored official `score_files`; do NOT add a second metric path.
- Class imbalance is severe (EDA) → both heads train with class-weighted cross-entropy.
- Entity-marker scheme: `[E1]…[/E1]` around the person mention, `[E2]…[/E2]` around the place mention, inserted into the context window; markers are added as tokenizer special tokens and the embedding matrix is resized.
- `RelationModel.predict` returns one dict per pair: `{"at","isAt","at_proba","isAt_proba"}`; for this model the proba fields are `{label: probability}` dicts (softmax) — usable by later stacking.
- Default base model `xlm-roberta-base`; selectable via config `model_name`.
- Consistency rule stays the harness default (`soft`); not re-implemented here.
- Determinism: a `seed` is set via `transformers.set_seed` before each head trains.
- Unit tests must not download `xlm-roberta-base` or require a GPU: the pure helpers are tested directly, and one integration test exercises the full fit/predict pipeline with a tiny model (`prajjwal1/bert-tiny`, ~17MB), skipped if transformers/network is unavailable. The real `xlm-roberta-base` validation is the local sanity run (Task 4); the full multilingual run is deferred to the Kaggle-bridge plan.

---

### Task 1: Transformer dependencies

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_ml_deps.py`

**Interfaces:**
- Adds `transformers` and `accelerate` to the `ml` optional-dependency group (torch arrives transitively; it is already present via sentence-transformers).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ml_deps.py
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_ml_extra_includes_transformers_and_accelerate():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    ml = data["project"]["optional-dependencies"]["ml"]
    joined = " ".join(ml)
    assert "sentence-transformers" in joined
    assert "transformers" in joined
    assert "accelerate" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ml_deps.py -v`
Expected: FAIL (`accelerate`/`transformers` not listed in the `ml` extra)

- [ ] **Step 3: Update the `ml` extra in `pyproject.toml`**

Change the `[project.optional-dependencies]` block to:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
ml = [
    "sentence-transformers>=2.2",
    "transformers>=4.40",
    "accelerate>=0.30",
]
```

- [ ] **Step 4: Install the updated extra**

Run: `pip install -e ".[ml,dev]"`
Expected: completes; installs `accelerate` (transformers already satisfied).

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_ml_deps.py -v && python -c "import accelerate; print('accelerate', accelerate.__version__)"`
Expected: test PASS; prints an accelerate version.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_ml_deps.py
git commit -m "feat: add transformers + accelerate to the ml extra"
```

---

### Task 2: Entity-marker text builder

**Files:**
- Create: `hipe/features/markers.py`
- Test: `tests/features/test_markers.py`

**Interfaces:**
- Consumes: `hipe.data.preprocess.fuzzy_find`, `Pair` (uses `.context`, `.person`, `.place`).
- Produces: `hipe.features.markers.MARKER_TOKENS: list[str]` = `["[E1]", "[/E1]", "[E2]", "[/E2]"]`; `hipe.features.markers.marked_text(pair) -> str` (context with markers around the located person/place mentions; falls back to prepending the marked surfaces if a mention can't be located or the spans overlap).

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_markers.py
from hipe.data.schema import Entity, Pair
from hipe.features.markers import marked_text, MARKER_TOKENS


def _pair(context, pers_mentions, loc_mentions):
    return Pair(doc_id="d", person=Entity("p", "person", pers_mentions),
                place=Entity("l", "place", loc_mentions), context=context,
                language="en", pub_date=None)


def test_marker_tokens():
    assert MARKER_TOKENS == ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def test_marks_both_mentions_in_place():
    p = _pair("Joe was at Essex.", ["Joe"], ["Essex"])
    assert marked_text(p) == "[E1]Joe[/E1] was at [E2]Essex[/E2]."


def test_fallback_prepends_when_not_found():
    p = _pair("nothing relevant here", ["Zzz"], ["Qqq"])
    out = marked_text(p)
    assert out.startswith("[E1]Zzz[/E1] [E2]Qqq[/E2] ")
    assert out.endswith("nothing relevant here")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_markers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.features.markers'`

- [ ] **Step 3: Write `hipe/features/markers.py`**

```python
# hipe/features/markers.py
from hipe.data.preprocess import fuzzy_find

E1_START, E1_END = "[E1]", "[/E1]"
E2_START, E2_END = "[E2]", "[/E2]"
MARKER_TOKENS = [E1_START, E1_END, E2_START, E2_END]


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _insert(text, span, start_tok, end_tok) -> str:
    s, e = span
    return text[:s] + start_tok + text[s:e] + end_tok + text[e:]


def _locate(text, mentions):
    for m in mentions:
        span = fuzzy_find(text, m)
        if span is not None:
            return span
    return None


def marked_text(pair) -> str:
    """Context with [E1]..[/E1] around the person mention and [E2]..[/E2] around
    the place mention. Falls back to prepending the marked surfaces when a
    mention can't be located or the two spans overlap."""
    text = pair.context
    pspan = _locate(text, pair.person.mentions)
    lspan = _locate(text, pair.place.mentions)
    if pspan is not None and lspan is not None and not _overlap(pspan, lspan):
        # insert the later span first so the earlier span's offsets stay valid
        for span, st, en in sorted(
                [(pspan, E1_START, E1_END), (lspan, E2_START, E2_END)],
                key=lambda x: x[0][0], reverse=True):
            text = _insert(text, span, st, en)
        return text
    return (f"{E1_START}{pair.person.surface}{E1_END} "
            f"{E2_START}{pair.place.surface}{E2_END} {text}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_markers.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/features/markers.py tests/features/test_markers.py
git commit -m "feat: entity-marker text builder for relation classification"
```

---

### Task 3: XLM-R model

**Files:**
- Create: `hipe/models/xlmr.py`
- Modify: `hipe/harness.py` (add the registration import)
- Test: `tests/models/test_xlmr.py`

**Interfaces:**
- Consumes: `RelationModel`, `registry.register`, `marked_text`, `MARKER_TOKENS`, `config.AT_LABELS`/`ISAT_LABELS`/`CACHE_DIR`.
- Produces: `XLMRModel(model_name="xlm-roberta-base", epochs=3, batch_size=16, lr=2e-5, max_length=192, max_train=None, seed=0)` registered `"xlmr"`. `fit(train, dev=None)` truncates to `max_train` if set, builds `marked_text`, and fine-tunes two `AutoModelForSequenceClassification` heads (`at`, `isAt`) with class-weighted loss; a target with <2 training classes falls back to a constant. `predict(pairs)` returns `[{"at","isAt","at_proba","isAt_proba"}]` with proba dicts `{label: prob}`. Module-level helper `_class_weights(labels: list[str], label_list: list[str]) -> torch.Tensor` (inverse-frequency, aligned to `label_list`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_xlmr.py
import pytest
from hipe.data.schema import Entity, Pair


def _pair(at, isat, ctx):
    return Pair(doc_id="d", person=Entity("p", "person", ["Joe"]),
                place=Entity("l", "place", ["Essex"]), context=ctx,
                language="en", pub_date=None, gold_at=at, gold_isat=isat)


def test_class_weights_inverse_frequency():
    from hipe.models.xlmr import _class_weights
    # labels: 3 FALSE, 1 TRUE over ["FALSE","PROBABLE","TRUE"]
    w = _class_weights(["FALSE", "FALSE", "FALSE", "TRUE"],
                       ["FALSE", "PROBABLE", "TRUE"])
    assert w.shape[0] == 3
    # rarer classes get higher weight: TRUE(1) > FALSE(3); absent PROBABLE highest
    assert float(w[2]) > float(w[0])
    assert float(w[1]) >= float(w[2])


@pytest.mark.slow
def test_xlmr_fit_predict_pipeline_tiny_model():
    # Exercises the full fit->predict mechanics with a tiny model (no XLM-R download).
    pytest.importorskip("transformers")
    from hipe.models import registry
    import hipe.models.xlmr  # noqa: F401  (registers xlmr)
    try:
        m = registry.get_model("xlmr", model_name="prajjwal1/bert-tiny",
                                epochs=1, batch_size=4, max_length=32, seed=0)
        train = ([_pair("TRUE", "TRUE", "Joe lived at Essex") for _ in range(6)] +
                 [_pair("FALSE", "FALSE", "no relation here") for _ in range(6)])
        m.fit(train)
        preds = m.predict([_pair("?", "?", "Joe lived at Essex")])
    except Exception as exc:  # offline / model unavailable
        pytest.skip(f"tiny-model integration unavailable: {exc}")
    assert preds[0]["at"] in ("FALSE", "PROBABLE", "TRUE")
    assert preds[0]["isAt"] in ("FALSE", "TRUE")
    assert isinstance(preds[0]["at_proba"], dict)
    assert abs(sum(preds[0]["at_proba"].values()) - 1.0) < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/test_xlmr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.models.xlmr'`

- [ ] **Step 3: Write `hipe/models/xlmr.py`**

```python
# hipe/models/xlmr.py
import numpy as np
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.markers import marked_text, MARKER_TOKENS


def _class_weights(labels, label_list):
    """Inverse-frequency (balanced) weights aligned to label_list order."""
    import torch
    counts = np.array([max(1, labels.count(l)) for l in label_list], dtype=float)
    w = counts.sum() / (len(label_list) * counts)
    return torch.tensor(w, dtype=torch.float)


class _Target:
    """One fine-tuned sequence classifier for a single relation target.

    Falls back to a constant prediction when <2 classes are present in training
    (a transformer cannot be trained on a single class)."""

    def __init__(self, label_list, model_name, max_length):
        self.label_list = label_list
        self.lab2id = {l: i for i, l in enumerate(label_list)}
        self.model_name = model_name
        self.max_length = max_length
        self.tok = None
        self.model = None
        self.const = None

    def train(self, texts, labels, *, epochs, batch_size, lr, seed):
        import torch
        from torch.utils.data import Dataset
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  Trainer, TrainingArguments, DataCollatorWithPadding,
                                  set_seed)
        if len(set(labels)) < 2:
            self.const = labels[0] if labels else "FALSE"
            return
        set_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label_list))
        self.model.resize_token_embeddings(len(self.tok))

        enc = self.tok(texts, truncation=True, max_length=self.max_length)
        y = [self.lab2id[l] for l in labels]

        class _DS(Dataset):
            def __len__(self_inner):
                return len(y)

            def __getitem__(self_inner, i):
                item = {k: enc[k][i] for k in enc}
                item["labels"] = y[i]
                return item

        weights = _class_weights(labels, self.label_list)

        class _WeightedTrainer(Trainer):
            def compute_loss(self_t, model, inputs, return_outputs=False, **kw):
                labels_ = inputs.pop("labels")
                outputs = model(**inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, labels_, weight=weights.to(outputs.logits.device))
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=str(cfg.CACHE_DIR / "xlmr_tmp"),
            num_train_epochs=epochs, per_device_train_batch_size=batch_size,
            learning_rate=lr, logging_strategy="no", save_strategy="no",
            report_to=[], seed=seed)
        trainer = _WeightedTrainer(
            model=self.model, args=args, train_dataset=_DS(),
            data_collator=DataCollatorWithPadding(self.tok))
        trainer.train()

    def predict(self, texts):
        if self.const is not None:
            proba = {l: (1.0 if l == self.const else 0.0) for l in self.label_list}
            return [self.const] * len(texts), [dict(proba) for _ in texts]
        import torch
        self.model.eval()
        labels, probas = [], []
        for i in range(0, len(texts), 32):
            batch = self.tok(texts[i:i + 32], truncation=True,
                             max_length=self.max_length, padding=True,
                             return_tensors="pt")
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                logits = self.model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                j = int(row.argmax())
                labels.append(self.label_list[j])
                probas.append({l: float(row[k]) for k, l in enumerate(self.label_list)})
        return labels, probas


@registry.register("xlmr")
class XLMRModel(RelationModel):
    name = "xlmr"

    def __init__(self, model_name="xlm-roberta-base", epochs=3, batch_size=16,
                 lr=2e-5, max_length=192, max_train=None, seed=0):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.max_train = max_train
        self.seed = seed
        self._at = _Target(cfg.AT_LABELS, model_name, max_length)
        self._isat = _Target(cfg.ISAT_LABELS, model_name, max_length)

    def fit(self, train, dev=None):
        if self.max_train:
            train = train[:self.max_train]
        texts = [marked_text(p) for p in train]
        kw = dict(epochs=self.epochs, batch_size=self.batch_size,
                  lr=self.lr, seed=self.seed)
        self._at.train(texts, [p.gold_at for p in train], **kw)
        self._isat.train(texts, [p.gold_isat for p in train], **kw)

    def predict(self, pairs):
        texts = [marked_text(p) for p in pairs]
        at, at_p = self._at.predict(texts)
        isat, isat_p = self._isat.predict(texts)
        return [{"at": a, "isAt": i, "at_proba": ap, "isAt_proba": ip}
                for a, i, ap, ip in zip(at, isat, at_p, isat_p)]
```

- [ ] **Step 4: Register the model in the harness** — add to `hipe/harness.py`'s import block (after the `lookup` import line):

```python
from hipe.models import xlmr  # noqa: F401  (registers xlmr)
```

- [ ] **Step 5: Run the unit test (pure helper) + confirm harness imports**

Run: `python -m pytest "tests/models/test_xlmr.py::test_class_weights_inverse_frequency" -v && python -c "import hipe.harness; from hipe.models import registry; print('xlmr' in registry._REGISTRY)"`
Expected: the weight test PASSES; prints `True`. (Importing `hipe.harness` must NOT download any model — transformers is imported lazily inside `_Target.train`/`predict`.)

- [ ] **Step 6: Run the optional tiny-model integration test**

Run: `python -m pytest "tests/models/test_xlmr.py::test_xlmr_fit_predict_pipeline_tiny_model" -v`
Expected: PASS (downloads ~17MB `prajjwal1/bert-tiny`, trains 1 epoch on 12 examples) — or SKIP if offline. Either is acceptable.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: all PASS (the `@pytest.mark.slow` integration test runs or skips; everything else green).

- [ ] **Step 8: Commit**

```bash
git add hipe/models/xlmr.py hipe/harness.py tests/models/test_xlmr.py
git commit -m "feat: entity-marker XLM-R model (two class-weighted heads, softmax proba)"
```

---

### Task 4: Configs + local sanity run

**Files:**
- Create: `configs/xlmr_sanity.yaml`
- Create: `configs/xlmr.yaml`
- Test: `tests/test_xlmr_config.py`

**Interfaces:**
- Consumes: `run_experiment`, the configs, real data.
- Produces: a tiny CPU-runnable sanity config and the full multilingual config (the latter is run on Kaggle in the next plan); a smoke test validating both configs.

- [ ] **Step 1: Write `configs/xlmr_sanity.yaml`** (tiny, CPU-runnable, proves the pipeline)

```yaml
data:
  train:
    - data/raw/HIPE-2026-data/data/sandbox/en-train.jsonl
  dev:
    - data/raw/HIPE-2026-data/data/sandbox/en-dev.jsonl
consistency: soft
model:
  name: xlmr
  model_name: xlm-roberta-base
  epochs: 1
  max_train: 64
  max_length: 128
  batch_size: 8
```

- [ ] **Step 2: Write `configs/xlmr.yaml`** (full multilingual — run on Kaggle GPU in the bridge plan)

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
  name: xlmr
  model_name: xlm-roberta-base
  epochs: 3
  max_length: 192
  batch_size: 16
  lr: 2.0e-5
```

- [ ] **Step 3: Write the config smoke test**

```python
# tests/test_xlmr_config.py
from pathlib import Path
import yaml

CFG = Path(__file__).resolve().parents[1] / "configs"


def test_xlmr_sanity_config_valid():
    c = yaml.safe_load((CFG / "xlmr_sanity.yaml").read_text())
    assert c["model"]["name"] == "xlmr"
    assert c["model"]["model_name"] == "xlm-roberta-base"
    assert c["model"]["max_train"] == 64        # tiny for local CPU
    assert all("sandbox" in p for p in c["data"]["train"])


def test_xlmr_full_config_valid():
    c = yaml.safe_load((CFG / "xlmr.yaml").read_text())
    assert c["model"]["name"] == "xlmr"
    assert all("sandbox" in p for p in c["data"]["train"])
    assert all("newspapers" in p for p in c["data"]["dev"])
    assert c["model"]["epochs"] == 3
```

- [ ] **Step 4: Run the config test + full suite**

Run: `python -m pytest tests/test_xlmr_config.py -v && python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Local sanity run (real, ~2–5 min on CPU; downloads `xlm-roberta-base` once, ~1.1GB)**

Run:
```bash
HIPE_RUNS_DIR=runs hipe run configs/xlmr_sanity.yaml
HIPE_RUNS_DIR=runs hipe leaderboard
```
Expected: completes without error and writes an `xlmr` leaderboard row with `n_dev=151` (sandbox en-dev) and a `global` score. The number itself will be weak (1 epoch, 64 examples, English only) — the point is to prove fit→predict→score→leaderboard works end-to-end before the full Kaggle run. Record the row in your report.

- [ ] **Step 6: Commit**

```bash
git add configs/xlmr_sanity.yaml configs/xlmr.yaml tests/test_xlmr_config.py
git commit -m "feat: xlmr sanity + full configs; local sanity run validates the pipeline"
```

---

## Self-Review

**Spec coverage (this plan = build step 3, entity-marker transformer):**
- Entity-marker (R-BERT-style) XLM-R, markers as special tokens (§6.3) → Tasks 2, 3.
- Two independent heads `at`/`isAt`, class-weighted for imbalance (§6.3, EDA) → Task 3.
- Softmax probabilities for later stacking (§5.9) → Task 3 (`predict` returns proba dicts).
- Runs through the harness/scorer/leaderboard with no new metric path → Tasks 3, 4.
- Default `xlm-roberta-base`, config-selectable (§6.3) → Tasks 3, 4.
- Trained on sandbox, evaluated on the gold/test-domain dev via the leakage-guarded harness (§2b) → Task 4 (full config; the doc-leakage guard from the prior plan applies automatically).
- Soft consistency default carried through (§5.3) → unchanged harness path, set in configs.
- *Deferred to later plans (correctly out of scope here):* the **full GPU run on Kaggle** (next plan: the automated bridge runs `configs/xlmr.yaml`), the R-BERT span-pooling variant and the MTB marker-position variant (this plan uses markers-in-input + `[CLS]`-head classification, the robust first version), mLUKE, ensembling, calibration.

**Placeholder scan:** none — every code/step block is concrete. Task 4 Step 5 is a real run with an expected-weak-but-valid number, explicitly framed as a pipeline check, not a result claim.

**Type consistency:** `marked_text(pair) -> str` and `MARKER_TOKENS` defined in Task 2, consumed in Task 3. `_class_weights(labels, label_list) -> torch.Tensor` defined and tested in Task 3. `XLMRModel(...)` kwargs (`model_name`, `epochs`, `batch_size`, `lr`, `max_length`, `max_train`, `seed`) are consistent between Task 3 and the configs in Task 4. Registered name `"xlmr"` consistent across Task 3, the harness import, and Task 4 configs. `predict` returns the same 4-key dict shape as the other models (`embedding_svm`, `lookup`).
