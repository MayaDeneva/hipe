# HIPE-2026 Harness Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the experiment harness foundation — data pipeline, official-scorer parity, file-based run registry, model interface, and trivial baselines — so `hipe run <config>` produces a scored, persisted leaderboard row end-to-end.

**Architecture:** A `pip install -e .` Python package. Every approach implements one `RelationModel` interface selected by a YAML config. One command loads pairs → preprocesses → predicts → applies the `isAt⇒at` consistency rule → writes official-format predictions → scores → persists a run folder + appends `runs/leaderboard.csv`. This plan delivers the harness plus `majority`/`random` baselines; classical-ML, transformer, LLM, and ensemble models are later plans that plug into the same interface.

**Tech Stack:** Python 3.12, scikit-learn, RapidFuzz, PyYAML, pytest. Reuses ported code from the prior `bz/hipe2026` project.

## Global Constraints

- Python 3.12.
- Label spaces are uppercase strings per the official schema: `at ∈ {FALSE, PROBABLE, TRUE}`, `isAt ∈ {FALSE, TRUE}`. `null/None → FALSE`.
- Official metric: macro-recall per target (`at`, `isAt`), then `global = mean(macro_recall(at), macro_recall(isAt))`.
- Consistency rule, applied to every model's output before scoring/submission: `isAt == "TRUE" ⇒ at = "TRUE"`.
- Scoring must use the official `evaluation_utils` code (vendored, pinned) — local score must equal the leaderboard score by construction.
- Run tracking is file-based: one timestamped folder per run + a committed `runs/leaderboard.csv`. No external tracking services.
- The core path makes no network calls (data is fetched separately by `scripts/fetch_data.py`).
- Submission files round-trip the input documents unchanged except for the `at`/`isAt` fields.

---

### Task 1: Package scaffold + config

**Files:**
- Create: `pyproject.toml`
- Create: `hipe/__init__.py`
- Create: `hipe/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `hipe.config.AT_LABELS: list[str]`, `hipe.config.ISAT_LABELS: list[str]`, `hipe.config.norm_label(value, relation: str) -> str`, and path constants `ROOT`, `DATA_RAW`, `CACHE_DIR`, `RUNS_DIR` (all `pathlib.Path`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from hipe import config


def test_label_spaces():
    assert config.AT_LABELS == ["FALSE", "PROBABLE", "TRUE"]
    assert config.ISAT_LABELS == ["FALSE", "TRUE"]


def test_norm_label_null_to_false():
    assert config.norm_label(None, "at") == "FALSE"
    assert config.norm_label(None, "isAt") == "FALSE"


def test_norm_label_uppercases_and_validates():
    assert config.norm_label("true", "at") == "TRUE"
    assert config.norm_label("Probable", "at") == "PROBABLE"
    # PROBABLE is not valid for isAt -> coerced to FALSE
    assert config.norm_label("PROBABLE", "isAt") == "FALSE"
    assert config.norm_label("garbage", "at") == "FALSE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe'`

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "hipe"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "scikit-learn>=1.4",
    "rapidfuzz>=3.6",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
hipe = "hipe.cli:main"

[tool.setuptools.packages.find]
include = ["hipe*"]
```

- [ ] **Step 4: Write `hipe/__init__.py`**

```python
# hipe/__init__.py
```

(Empty file — marks the package.)

- [ ] **Step 5: Write `hipe/config.py`**

```python
# hipe/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root
DATA_RAW = ROOT / "data" / "raw"                    # cloned HIPE-2026-data repo
CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"

# Label spaces (uppercase strings, per the official schema). null -> FALSE.
AT_LABELS = ["FALSE", "PROBABLE", "TRUE"]
ISAT_LABELS = ["FALSE", "TRUE"]


def norm_label(value, relation: str) -> str:
    """null/None -> FALSE; coerce to the relation's allowed set."""
    allowed = AT_LABELS if relation == "at" else ISAT_LABELS
    if value is None:
        return "FALSE"
    v = str(value).upper()
    return v if v in allowed else "FALSE"
```

- [ ] **Step 6: Install the package and run the tests**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml hipe/__init__.py hipe/config.py tests/test_config.py
git commit -m "feat: package scaffold + config (label spaces, norm_label, paths)"
```

---

### Task 2: Data fetch script

**Files:**
- Create: `scripts/fetch_data.py`
- Test: `tests/test_fetch_data.py`

**Interfaces:**
- Produces: `scripts.fetch_data.clone_command(dest: Path) -> list[str]`, `scripts.fetch_data.is_present(dest: Path) -> bool`, `scripts.fetch_data.DATA_REPO_URL: str`, `scripts.fetch_data.PINNED_COMMIT: str`. Running the module clones/pins the HIPE data into `data/raw/HIPE-2026-data`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch_data.py
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.py")
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)


def test_clone_command_uses_repo_url_and_dest(tmp_path):
    dest = tmp_path / "HIPE-2026-data"
    cmd = fetch_data.clone_command(dest)
    assert cmd[0] == "git" and cmd[1] == "clone"
    assert fetch_data.DATA_REPO_URL in cmd
    assert str(dest) in cmd


def test_is_present_false_when_missing(tmp_path):
    assert fetch_data.is_present(tmp_path / "nope") is False


def test_is_present_true_when_git_dir_exists(tmp_path):
    dest = tmp_path / "HIPE-2026-data"
    (dest / ".git").mkdir(parents=True)
    assert fetch_data.is_present(dest) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetch_data.py -v`
Expected: FAIL with `AttributeError`/`FileNotFoundError` (module/functions not defined)

- [ ] **Step 3: Write `scripts/fetch_data.py`**

```python
# scripts/fetch_data.py
"""Clone (or update) the pinned HIPE-2026-data repo into data/raw/.

Usage: python scripts/fetch_data.py
"""
import subprocess
import sys
from pathlib import Path

DATA_REPO_URL = "https://github.com/hipe-eval/HIPE-2026-data.git"
PINNED_COMMIT = "4228562"   # pin for reproducibility; bump deliberately
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "HIPE-2026-data"


def clone_command(dest: Path) -> list[str]:
    return ["git", "clone", DATA_REPO_URL, str(dest)]


def is_present(dest: Path) -> bool:
    return (dest / ".git").is_dir()


def main(dest: Path = DEST) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_present(dest):
        print(f"Updating {dest}")
        subprocess.run(["git", "-C", str(dest), "fetch", "--all"], check=True)
    else:
        print(f"Cloning into {dest}")
        subprocess.run(clone_command(dest), check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", PINNED_COMMIT], check=True)
    print("Data ready at", dest)


if __name__ == "__main__":
    main()
    sys.exit(0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch_data.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Fetch the real data (manual, one-time)**

Run: `python scripts/fetch_data.py`
Expected: prints `Data ready at .../data/raw/HIPE-2026-data`. Confirm with:
`ls data/raw/HIPE-2026-data/data/sandbox/en-train.jsonl`

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_data.py tests/test_fetch_data.py
git commit -m "feat: pinned HIPE-2026-data fetch script"
```

---

### Task 3: Data schema

**Files:**
- Create: `hipe/data/__init__.py`
- Create: `hipe/data/schema.py`
- Test: `tests/data/test_schema.py`

**Interfaces:**
- Produces: dataclasses `Entity(entity_id, etype, mentions, qid=None, link_score=None)` with property `surface`; `Document(doc_id, text, language, pub_date, media, source)`; `Pair(doc_id, person, place, context, language, pub_date, gold_at="FALSE", gold_isat="FALSE", features=dict)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_schema.py
from hipe.data.schema import Entity, Pair


def test_entity_surface_is_first_mention():
    e = Entity(entity_id="p1", etype="person", mentions=["Joe", "J. Joe"])
    assert e.surface == "Joe"


def test_entity_surface_empty_when_no_mentions():
    e = Entity(entity_id="p1", etype="person", mentions=[])
    assert e.surface == ""


def test_pair_defaults_to_false_labels():
    person = Entity("p1", "person", ["Joe"])
    place = Entity("l1", "place", ["Essex"])
    p = Pair(doc_id="d1", person=person, place=place, context="...",
             language="en", pub_date=None)
    assert p.gold_at == "FALSE"
    assert p.gold_isat == "FALSE"
    assert p.features == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.data'`

- [ ] **Step 3: Create `hipe/data/__init__.py`**

```python
# hipe/data/__init__.py
```

- [ ] **Step 4: Write `hipe/data/schema.py`**

```python
# hipe/data/schema.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    entity_id: str                   # pers_entity_id / loc_entity_id
    etype: str                       # "person" | "place"
    mentions: list[str]              # surface forms (mention list)
    qid: Optional[str] = None        # pre-supplied or resolved by linking
    link_score: Optional[float] = None

    @property
    def surface(self) -> str:        # primary mention
        return self.mentions[0] if self.mentions else ""


@dataclass
class Document:
    doc_id: str
    text: str
    language: str
    pub_date: Optional[str]          # ISO date / "YYYY" / None
    media: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class Pair:
    doc_id: str
    person: Entity
    place: Entity
    context: str
    language: str
    pub_date: Optional[str]
    gold_at: str = "FALSE"           # TRUE | PROBABLE | FALSE
    gold_isat: str = "FALSE"         # TRUE | FALSE
    features: dict = field(default_factory=dict)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add hipe/data/__init__.py hipe/data/schema.py tests/data/test_schema.py
git commit -m "feat: data schema (Entity, Document, Pair)"
```

---

### Task 4: JSONL loader

**Files:**
- Create: `hipe/data/load.py`
- Create: `tests/fixtures/mini.jsonl`
- Test: `tests/data/test_load.py`

**Interfaces:**
- Consumes: `hipe.data.schema.Document`.
- Produces: `hipe.data.load.read_jsonl(path) -> list[dict]`, `hipe.data.load.load_documents(path) -> list[Document]`.

- [ ] **Step 1: Create the fixture `tests/fixtures/mini.jsonl`**

Two documents, one line each (exact content — used by later tasks too):

```json
{"document_id": "d1", "media": {"publication_title": "Gazette"}, "source": "src/a.tsv", "language": "en", "date": "1820-01-10", "text": "Joe was committed to the jail at Essex county.", "sampled_pairs": [{"pers_entity_id": "d1-joe", "pers_wikidata_QID": null, "pers_mentions_list": ["Joe"], "loc_entity_id": "d1-essex", "loc_wikidata_QID": "Q1", "loc_mentions_list": ["Essex"], "at": "TRUE", "at_explanation": "", "isAt": "TRUE", "isAt_explanation": ""}, {"pers_entity_id": "d1-joe", "pers_wikidata_QID": null, "pers_mentions_list": ["Joe"], "loc_entity_id": "d1-rapp", "loc_wikidata_QID": "Q2", "loc_mentions_list": ["Rappahannock"], "at": "FALSE", "at_explanation": "", "isAt": "FALSE", "isAt_explanation": ""}]}
{"document_id": "d2", "media": {}, "source": "src/b.tsv", "language": "fr", "date": "1850", "text": "Marie habite a Paris.", "sampled_pairs": [{"pers_entity_id": "d2-marie", "pers_wikidata_QID": null, "pers_mentions_list": ["Marie"], "loc_entity_id": "d2-paris", "loc_wikidata_QID": "Q90", "loc_mentions_list": ["Paris"], "at": null, "at_explanation": "", "isAt": null, "isAt_explanation": ""}]}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/data/test_load.py
from pathlib import Path
from hipe.data.load import read_jsonl, load_documents

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini.jsonl"


def test_read_jsonl_skips_blank_lines():
    rows = read_jsonl(FIX)
    assert len(rows) == 2
    assert rows[0]["document_id"] == "d1"


def test_load_documents_maps_fields():
    docs = load_documents(FIX)
    assert [d.doc_id for d in docs] == ["d1", "d2"]
    assert docs[0].language == "en"
    assert docs[0].pub_date == "1820-01-10"
    assert docs[1].pub_date == "1850"
    assert docs[0].media["publication_title"] == "Gazette"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/data/test_load.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.data.load'`

- [ ] **Step 4: Write `hipe/data/load.py`**

```python
# hipe/data/load.py
import json
from pathlib import Path
from hipe.data.schema import Document


def read_jsonl(path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_documents(path) -> list[Document]:
    docs = []
    for raw in read_jsonl(path):
        docs.append(Document(
            doc_id=str(raw["document_id"]),
            text=raw.get("text", ""),
            language=raw.get("language", ""),
            pub_date=raw.get("date"),
            media=raw.get("media", {}) or {},
            source=raw.get("source", ""),
        ))
    return docs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_load.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add hipe/data/load.py tests/fixtures/mini.jsonl tests/data/test_load.py
git commit -m "feat: jsonl loader + mini fixture"
```

---

### Task 5: Preprocessing (normalization + fuzzy mention location)

**Files:**
- Create: `hipe/data/preprocess.py`
- Test: `tests/data/test_preprocess.py`

**Interfaces:**
- Produces: `hipe.data.preprocess.normalize_text(text: str) -> str`; `hipe.data.preprocess.fuzzy_find(text: str, mention: str, min_score: float = 80.0) -> tuple[int, int] | None` (start/end char span of the best match, or `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_preprocess.py
from hipe.data.preprocess import normalize_text, fuzzy_find


def test_normalize_dehyphenates_linebreaks():
    assert normalize_text("exam-\nple") == "example"


def test_normalize_long_s_and_whitespace():
    assert normalize_text("ſtreet   of\n\nParis") == "street of Paris"


def test_fuzzy_find_exact():
    text = "Joe was at Essex county."
    assert fuzzy_find(text, "Essex") == (11, 16)


def test_fuzzy_find_handles_ocr_noise():
    text = "committed to the /ail at Essex"
    span = fuzzy_find(text, "jail")
    assert span is not None
    assert text[span[0]:span[1]].lower() in ("/ail", "jail")


def test_fuzzy_find_returns_none_below_threshold():
    assert fuzzy_find("totally unrelated words", "Rappahannock") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.data.preprocess'`

- [ ] **Step 3: Write `hipe/data/preprocess.py`**

```python
# hipe/data/preprocess.py
import re
import unicodedata
from rapidfuzz import fuzz

_SOFT_HYPHEN = "­"


def normalize_text(text: str) -> str:
    """Conservative OCR-aware normalization. Does not rewrite words."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_SOFT_HYPHEN, "")
    text = text.replace("ſ", "s")          # long s
    # de-hyphenate line-break splits: "exam-\nple" -> "example"
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)            # collapse whitespace/newlines
    return text.strip()


def fuzzy_find(text: str, mention: str, min_score: float = 80.0):
    """Return (start, end) of the best match for `mention` in `text`, or None."""
    if not text or not mention:
        return None
    idx = text.find(mention)
    if idx >= 0:
        return (idx, idx + len(mention))
    m = len(mention)
    ml = mention.lower()
    best_span = None
    best_score = min_score
    for i in range(0, max(1, len(text) - m + 1)):
        window = text[i:i + m]
        score = fuzz.ratio(window.lower(), ml)
        if score > best_score:
            best_score = score
            best_span = (i, i + m)
    return best_span
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/data/preprocess.py tests/data/test_preprocess.py
git commit -m "feat: conservative OCR normalization + fuzzy mention location"
```

---

### Task 6: Pair generation

**Files:**
- Create: `hipe/data/pairs.py`
- Test: `tests/data/test_pairs.py`

**Interfaces:**
- Consumes: `read_jsonl`, `normalize_text`, `fuzzy_find`, `config.norm_label`, `Entity`, `Pair`.
- Produces: `hipe.data.pairs.context_for(text, mentions, margin=200) -> str`; `hipe.data.pairs.load_pairs(path) -> list[Pair]`; `hipe.data.pairs.unique_entities(pairs) -> dict[str, Entity]`; `hipe.data.pairs.pair_key(pair) -> str` (returns `person.entity_id + place.entity_id`, matching the official scorer key).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_pairs.py
from pathlib import Path
from hipe.data.pairs import load_pairs, unique_entities, pair_key, context_for

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini.jsonl"


def test_load_pairs_count_and_labels():
    pairs = load_pairs(FIX)
    assert len(pairs) == 3                      # 2 from d1, 1 from d2
    p0 = pairs[0]
    assert p0.gold_at == "TRUE"
    assert p0.gold_isat == "TRUE"
    # d2 had null labels -> normalized to FALSE
    d2 = [p for p in pairs if p.doc_id == "d2"][0]
    assert d2.gold_at == "FALSE"
    assert d2.gold_isat == "FALSE"


def test_context_window_is_around_mention():
    pairs = load_pairs(FIX)
    p0 = pairs[0]
    assert "Essex" in p0.context
    assert "Joe" in p0.context


def test_pair_key_matches_official_concatenation():
    pairs = load_pairs(FIX)
    assert pair_key(pairs[0]) == "d1-joed1-essex"


def test_unique_entities_dedupes():
    pairs = load_pairs(FIX)
    ents = unique_entities(pairs)
    # d1-joe appears in two pairs but is one entity
    assert "d1-joe" in ents
    assert ents["d1-joe"].etype == "person"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.data.pairs'`

- [ ] **Step 3: Write `hipe/data/pairs.py`**

```python
# hipe/data/pairs.py
from hipe import config
from hipe.data.load import read_jsonl
from hipe.data.preprocess import normalize_text, fuzzy_find
from hipe.data.schema import Entity, Pair


def context_for(text: str, mentions: list[str], margin: int = 200) -> str:
    """Window around the first matching mention; whole text if none found."""
    span = None
    for m in mentions:
        found = fuzzy_find(text, m)
        if found is not None:
            span = found
            break
    if span is None:
        return text
    lo = max(0, span[0] - margin)
    hi = min(len(text), span[1] + margin)
    return text[lo:hi]


def _intern_entity(registry, entity_id, etype, mentions, qid) -> Entity:
    if entity_id not in registry:
        registry[entity_id] = Entity(entity_id=entity_id, etype=etype,
                                     mentions=list(mentions), qid=qid)
    else:
        existing = registry[entity_id]
        seen = set(existing.mentions)
        for m in mentions:
            if m not in seen:
                existing.mentions.append(m)
                seen.add(m)
    return registry[entity_id]


def pair_key(pair) -> str:
    """Match the official scorer key: pers_entity_id concatenated with loc_entity_id."""
    return f"{pair.person.entity_id}{pair.place.entity_id}"


def load_pairs(path) -> list[Pair]:
    pairs = []
    registry: dict = {}
    for raw in read_jsonl(path):
        text = normalize_text(raw.get("text", ""))
        for sp in raw.get("sampled_pairs", []):
            person = _intern_entity(registry, sp["pers_entity_id"], "person",
                                    sp.get("pers_mentions_list", []),
                                    sp.get("pers_wikidata_QID"))
            place = _intern_entity(registry, sp["loc_entity_id"], "place",
                                   sp.get("loc_mentions_list", []),
                                   sp.get("loc_wikidata_QID"))
            pairs.append(Pair(
                doc_id=str(raw["document_id"]), person=person, place=place,
                context=context_for(text, person.mentions + place.mentions),
                language=raw.get("language", ""), pub_date=raw.get("date"),
                gold_at=config.norm_label(sp.get("at"), "at"),
                gold_isat=config.norm_label(sp.get("isAt"), "isAt"),
            ))
    return pairs


def unique_entities(pairs) -> dict:
    seen = {}
    for p in pairs:
        for e in (p.person, p.place):
            seen.setdefault(e.entity_id, e)
    return seen
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_pairs.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/data/pairs.py tests/data/test_pairs.py
git commit -m "feat: pair generation with fuzzy context windowing + official pair_key"
```

---

### Task 7: Document-grouped split

**Files:**
- Create: `hipe/data/split.py`
- Test: `tests/data/test_split.py`

**Interfaces:**
- Consumes: `Pair` (uses `.doc_id`).
- Produces: `hipe.data.split.split_by_document(pairs, dev_frac=0.2, seed=0) -> tuple[list, list]` (train, dev) with no document appearing in both sides.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_split.py
from hipe.data.schema import Entity, Pair
from hipe.data.split import split_by_document


def _pair(doc_id):
    return Pair(doc_id=doc_id, person=Entity("p", "person", ["x"]),
                place=Entity("l", "place", ["y"]), context="", language="en",
                pub_date=None)


def test_no_document_leakage():
    pairs = [_pair(f"d{i}") for i in range(10) for _ in range(3)]
    train, dev = split_by_document(pairs, dev_frac=0.2, seed=0)
    train_docs = {p.doc_id for p in train}
    dev_docs = {p.doc_id for p in dev}
    assert train_docs.isdisjoint(dev_docs)
    assert len(dev_docs) == 2          # 20% of 10 docs


def test_split_is_deterministic():
    pairs = [_pair(f"d{i}") for i in range(10)]
    a = split_by_document(pairs, seed=0)
    b = split_by_document(pairs, seed=0)
    assert [p.doc_id for p in a[1]] == [p.doc_id for p in b[1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.data.split'`

- [ ] **Step 3: Write `hipe/data/split.py`**

```python
# hipe/data/split.py
import random


def split_by_document(pairs, dev_frac=0.2, seed=0):
    docs = sorted({p.doc_id for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_dev = max(1, int(len(docs) * dev_frac))
    dev_docs = set(docs[:n_dev])
    train = [p for p in pairs if p.doc_id not in dev_docs]
    dev = [p for p in pairs if p.doc_id in dev_docs]
    return train, dev
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_split.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/data/split.py tests/data/test_split.py
git commit -m "feat: document-grouped train/dev split"
```

---

### Task 8: Metrics

**Files:**
- Create: `hipe/eval/__init__.py`
- Create: `hipe/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `hipe.eval.metrics.macro_recall(y_true, y_pred) -> float`; `hipe.eval.metrics.confusion(y_true, y_pred) -> tuple[list, ndarray]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
from hipe.eval.metrics import macro_recall


def test_macro_recall_known_value():
    # 3 classes present in gold; predicting all FALSE.
    y_true = ["FALSE", "TRUE", "FALSE", "PROBABLE"]
    y_pred = ["FALSE", "FALSE", "FALSE", "FALSE"]
    # recall: FALSE=2/2=1, PROBABLE=0/1=0, TRUE=0/1=0 -> macro = 1/3
    assert round(macro_recall(y_true, y_pred), 4) == 0.3333


def test_macro_recall_perfect():
    y = ["TRUE", "FALSE", "FALSE"]
    assert macro_recall(y, y) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.eval'`

- [ ] **Step 3: Create `hipe/eval/__init__.py`**

```python
# hipe/eval/__init__.py
```

- [ ] **Step 4: Write `hipe/eval/metrics.py`**

```python
# hipe/eval/metrics.py
from sklearn.metrics import recall_score, confusion_matrix


def macro_recall(y_true, y_pred) -> float:
    labels = sorted(set(y_true))
    return float(recall_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0))


def confusion(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    return labels, confusion_matrix(y_true, y_pred, labels=labels)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add hipe/eval/__init__.py hipe/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: macro-recall + confusion metrics"
```

---

### Task 9: Vendor + wrap the official scorer

**Files:**
- Create: `hipe/eval/official/__init__.py`
- Create: `hipe/eval/official/evaluation_utils.py` (copied from the fetched data repo)
- Create: `hipe/eval/scorer.py`
- Create: `tests/fixtures/gold.jsonl`, `tests/fixtures/pred.jsonl`
- Test: `tests/eval/test_scorer.py`

**Interfaces:**
- Produces: `hipe.eval.scorer.score_files(gold_path, pred_path) -> dict` returning the official metrics dict (`{"at": {...}, "isAt": {...}, "global": {"macro_recall": float, ...}}`).

- [ ] **Step 1: Create the official-scorer package marker `hipe/eval/official/__init__.py`**

```python
# hipe/eval/official/__init__.py
```

- [ ] **Step 2: Vendor the official scorer module (pinned copy)**

Requires Task 2's data fetch to have run. Copy the file, then prepend a provenance header:

Run:
```bash
cp data/raw/HIPE-2026-data/scripts/evaluation_utils.py hipe/eval/official/evaluation_utils.py
```

Then add this header as the first lines of `hipe/eval/official/evaluation_utils.py` (above the existing imports), leaving the rest of the file unchanged:

```python
# Vendored verbatim from hipe-eval/HIPE-2026-data @ commit 4228562
# Source: scripts/evaluation_utils.py — DO NOT EDIT (bump deliberately when re-vendoring).
```

- [ ] **Step 3: Create the fixtures**

`tests/fixtures/gold.jsonl` (one line):

```json
{"document_id": "g1", "language": "en", "date": "1900", "media": {}, "source": "", "text": "x", "sampled_pairs": [{"pers_entity_id": "P1", "loc_entity_id": "L1", "pers_mentions_list": ["A"], "loc_mentions_list": ["B"], "at": "TRUE", "isAt": "TRUE"}, {"pers_entity_id": "P2", "loc_entity_id": "L2", "pers_mentions_list": ["C"], "loc_mentions_list": ["D"], "at": "FALSE", "isAt": "FALSE"}]}
```

`tests/fixtures/pred.jsonl` (one line — same docs/pairs, predicted labels):

```json
{"document_id": "g1", "language": "en", "date": "1900", "media": {}, "source": "", "text": "x", "sampled_pairs": [{"pers_entity_id": "P1", "loc_entity_id": "L1", "pers_mentions_list": ["A"], "loc_mentions_list": ["B"], "at": "TRUE", "isAt": "FALSE"}, {"pers_entity_id": "P2", "loc_entity_id": "L2", "pers_mentions_list": ["C"], "loc_mentions_list": ["D"], "at": "FALSE", "isAt": "FALSE"}]}
```

- [ ] **Step 4: Write the failing test**

```python
# tests/eval/test_scorer.py
from pathlib import Path
from hipe.eval.scorer import score_files

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_score_files_matches_hand_computed():
    metrics = score_files(FIX / "gold.jsonl", FIX / "pred.jsonl")
    # at:   gold=[TRUE,FALSE] pred=[TRUE,FALSE] -> macro recall 1.0
    # isAt: gold=[TRUE,FALSE] pred=[FALSE,FALSE] -> recall FALSE=1, TRUE=0 -> 0.5
    assert round(metrics["at"]["macro_recall"], 4) == 1.0
    assert round(metrics["isAt"]["macro_recall"], 4) == 0.5
    assert round(metrics["global"]["macro_recall"], 4) == 0.75
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/eval/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.eval.scorer'`

- [ ] **Step 6: Write `hipe/eval/scorer.py`**

```python
# hipe/eval/scorer.py
from pathlib import Path
from hipe.eval.official import evaluation_utils as eu


def score_files(gold_path, pred_path) -> dict:
    """Score a prediction file against gold using the vendored official scorer."""
    gold = eu.load_jsonl_to_reshaped_dict(Path(gold_path))
    sub = eu.load_jsonl_to_reshaped_dict(Path(pred_path))
    sub = eu.impute_missing_submission_data(gold, sub)
    labels = eu.flatten_predictions(gold, sub)
    return eu.calculate_metrics(labels)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/eval/test_scorer.py -v`
Expected: PASS (1 test)

- [ ] **Step 8: Commit**

```bash
git add hipe/eval/official tests/fixtures/gold.jsonl tests/fixtures/pred.jsonl hipe/eval/scorer.py tests/eval/test_scorer.py
git commit -m "feat: vendor + wrap official scorer (parity by construction)"
```

---

### Task 10: Model interface + consistency rule + registry

**Files:**
- Create: `hipe/models/__init__.py`
- Create: `hipe/models/base.py`
- Create: `hipe/models/registry.py`
- Test: `tests/models/test_base.py`

**Interfaces:**
- Produces:
  - `hipe.models.base.RelationModel` (ABC) with `name: str`, `fit(self, train, dev=None) -> None`, `predict(self, pairs) -> list[dict]`. Each prediction dict has keys `"at"`, `"isAt"` (and optional `"at_proba"`, `"isAt_proba"`).
  - `hipe.models.base.apply_consistency(pred: dict) -> dict` enforcing `isAt == "TRUE" ⇒ at = "TRUE"` (returns the same dict, mutated).
  - `hipe.models.registry.register(name)` (decorator) and `hipe.models.registry.get_model(name, **kwargs) -> RelationModel`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_base.py
import pytest
from hipe.models.base import RelationModel, apply_consistency
from hipe.models import registry


def test_apply_consistency_forces_at_true():
    assert apply_consistency({"at": "FALSE", "isAt": "TRUE"})["at"] == "TRUE"
    assert apply_consistency({"at": "PROBABLE", "isAt": "FALSE"})["at"] == "PROBABLE"


def test_registry_roundtrip():
    @registry.register("dummy_test_model")
    class Dummy(RelationModel):
        name = "dummy_test_model"
        def fit(self, train, dev=None):
            pass
        def predict(self, pairs):
            return [{"at": "FALSE", "isAt": "FALSE"} for _ in pairs]

    m = registry.get_model("dummy_test_model")
    assert isinstance(m, RelationModel)
    assert m.predict([1, 2]) == [{"at": "FALSE", "isAt": "FALSE"}] * 2


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        registry.get_model("does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.models'`

- [ ] **Step 3: Create `hipe/models/__init__.py`**

```python
# hipe/models/__init__.py
```

- [ ] **Step 4: Write `hipe/models/base.py`**

```python
# hipe/models/base.py
from abc import ABC, abstractmethod


class RelationModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train, dev=None) -> None:
        ...

    @abstractmethod
    def predict(self, pairs) -> list[dict]:
        """Return one dict per pair with keys 'at' and 'isAt'."""
        ...


def apply_consistency(pred: dict) -> dict:
    """Enforce isAt == TRUE  ==>  at = TRUE."""
    if pred.get("isAt") == "TRUE":
        pred["at"] = "TRUE"
    return pred
```

- [ ] **Step 5: Write `hipe/models/registry.py`**

```python
# hipe/models/registry.py
_REGISTRY: dict = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_model(name, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/models/test_base.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add hipe/models/__init__.py hipe/models/base.py hipe/models/registry.py tests/models/test_base.py
git commit -m "feat: RelationModel interface, consistency rule, model registry"
```

---

### Task 11: Baseline models

**Files:**
- Create: `hipe/models/baselines.py`
- Test: `tests/models/test_baselines.py`

**Interfaces:**
- Consumes: `RelationModel`, `registry.register`, `config.AT_LABELS`, `config.ISAT_LABELS`.
- Produces: `MajorityModel` (registered as `"majority"`) predicts the modal `at`/`isAt` label seen in `fit`; `RandomModel` (registered as `"random"`, accepts `seed=0`) predicts deterministically given the seed. Both read gold via `pair.gold_at` / `pair.gold_isat`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_baselines.py
from hipe.data.schema import Entity, Pair
from hipe.models import registry
import hipe.models.baselines  # noqa: F401  (registers the models)


def _pair(at, isat):
    return Pair(doc_id="d", person=Entity("p", "person", ["x"]),
                place=Entity("l", "place", ["y"]), context="", language="en",
                pub_date=None, gold_at=at, gold_isat=isat)


def test_majority_predicts_modal_label():
    train = [_pair("FALSE", "FALSE"), _pair("FALSE", "FALSE"), _pair("TRUE", "TRUE")]
    m = registry.get_model("majority")
    m.fit(train)
    preds = m.predict([_pair("TRUE", "TRUE")])
    assert preds[0]["at"] == "FALSE"
    assert preds[0]["isAt"] == "FALSE"


def test_random_is_deterministic_with_seed():
    train = [_pair("TRUE", "TRUE"), _pair("FALSE", "FALSE")]
    pairs = [_pair("FALSE", "FALSE") for _ in range(5)]
    a = registry.get_model("random", seed=42); a.fit(train)
    b = registry.get_model("random", seed=42); b.fit(train)
    assert a.predict(pairs) == b.predict(pairs)


def test_random_labels_are_valid():
    from hipe import config
    m = registry.get_model("random", seed=0); m.fit([_pair("TRUE", "TRUE")])
    for p in m.predict([_pair("FALSE", "FALSE") for _ in range(10)]):
        assert p["at"] in config.AT_LABELS
        assert p["isAt"] in config.ISAT_LABELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.models.baselines'`

- [ ] **Step 3: Write `hipe/models/baselines.py`**

```python
# hipe/models/baselines.py
import random
from collections import Counter
from hipe import config
from hipe.models.base import RelationModel
from hipe.models import registry


@registry.register("majority")
class MajorityModel(RelationModel):
    name = "majority"

    def __init__(self):
        self._at = "FALSE"
        self._isat = "FALSE"

    def fit(self, train, dev=None):
        if train:
            self._at = Counter(p.gold_at for p in train).most_common(1)[0][0]
            self._isat = Counter(p.gold_isat for p in train).most_common(1)[0][0]

    def predict(self, pairs):
        return [{"at": self._at, "isAt": self._isat} for _ in pairs]


@registry.register("random")
class RandomModel(RelationModel):
    name = "random"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, train, dev=None):
        pass

    def predict(self, pairs):
        rng = random.Random(self.seed)
        return [{"at": rng.choice(config.AT_LABELS),
                 "isAt": rng.choice(config.ISAT_LABELS)} for _ in pairs]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models/test_baselines.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/models/baselines.py tests/models/test_baselines.py
git commit -m "feat: majority + random baseline models"
```

---

### Task 12: Submission writer

**Files:**
- Create: `hipe/data/submission.py`
- Test: `tests/data/test_submission.py`

**Interfaces:**
- Consumes: `read_jsonl`, `pair_key`.
- Produces: `hipe.data.submission.write_submission(src_path, predictions: dict, out_path) -> None`, where `predictions` maps `(doc_id, pair_key) -> {"at":..., "isAt":...}`. It round-trips every document/pair from `src_path`, overwriting only `at`/`isAt`; pairs absent from `predictions` are written as `FALSE`/`FALSE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_submission.py
from pathlib import Path
from hipe.data.load import read_jsonl
from hipe.data.submission import write_submission

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini.jsonl"


def test_write_submission_roundtrips_and_sets_labels(tmp_path):
    preds = {
        ("d1", "d1-joed1-essex"): {"at": "PROBABLE", "isAt": "FALSE"},
    }
    out = tmp_path / "sub.jsonl"
    write_submission(FIX, preds, out)
    rows = read_jsonl(out)
    # structure preserved
    assert [r["document_id"] for r in rows] == ["d1", "d2"]
    d1 = rows[0]["sampled_pairs"]
    first = [p for p in d1 if p["pers_entity_id"] == "d1-joe"
             and p["loc_entity_id"] == "d1-essex"][0]
    assert first["at"] == "PROBABLE" and first["isAt"] == "FALSE"
    # pair not in preds -> defaulted to FALSE/FALSE
    second = [p for p in d1 if p["loc_entity_id"] == "d1-rapp"][0]
    assert second["at"] == "FALSE" and second["isAt"] == "FALSE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_submission.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.data.submission'`

- [ ] **Step 3: Write `hipe/data/submission.py`**

```python
# hipe/data/submission.py
import json
from pathlib import Path
from hipe.data.load import read_jsonl


def write_submission(src_path, predictions: dict, out_path) -> None:
    """Round-trip src docs, overwriting only at/isAt from `predictions`.

    predictions: {(doc_id, pers_entity_id+loc_entity_id): {"at":..., "isAt":...}}
    """
    rows = read_jsonl(src_path)
    for row in rows:
        doc_id = str(row["document_id"])
        for sp in row.get("sampled_pairs", []):
            key = f"{sp['pers_entity_id']}{sp['loc_entity_id']}"
            pred = predictions.get((doc_id, key), {"at": "FALSE", "isAt": "FALSE"})
            sp["at"] = pred["at"]
            sp["isAt"] = pred["isAt"]
    with Path(out_path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_submission.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add hipe/data/submission.py tests/data/test_submission.py
git commit -m "feat: official-format submission writer"
```

---

### Task 13: Run registry + leaderboard

**Files:**
- Create: `hipe/runs/__init__.py`
- Create: `hipe/runs/registry.py`
- Test: `tests/runs/test_registry.py`

**Interfaces:**
- Produces:
  - `hipe.runs.registry.config_hash(config: dict) -> str` (stable 8-char hex of the canonicalized config).
  - `hipe.runs.registry.new_run_dir(model_name, cfg_hash, root, now) -> Path` — creates `root/<now>_<model>_<hash>/` (where `now` is a `YYYY-MM-DD_HHMMSS` string).
  - `hipe.runs.registry.write_manifest(run_dir, manifest: dict) -> None` (writes `config.yaml`-style `manifest.json`).
  - `hipe.runs.registry.append_leaderboard(root, row: dict) -> None` — appends one CSV row to `root/leaderboard.csv`, writing the header on first use. Columns: `run_id, timestamp, model, config_hash, data, at_recall, isAt_recall, global, n_dev, notes`.

- [ ] **Step 1: Write the failing test**

```python
# tests/runs/test_registry.py
from hipe.runs import registry


def test_config_hash_is_stable_and_order_independent():
    a = registry.config_hash({"model": {"name": "majority"}, "data": {"x": 1}})
    b = registry.config_hash({"data": {"x": 1}, "model": {"name": "majority"}})
    assert a == b
    assert len(a) == 8


def test_new_run_dir_created(tmp_path):
    d = registry.new_run_dir("majority", "abcd1234", root=tmp_path,
                             now="2026-06-21_120000")
    assert d.exists()
    assert d.name == "2026-06-21_120000_majority_abcd1234"


def test_append_leaderboard_writes_header_once(tmp_path):
    row = {"run_id": "r1", "timestamp": "t", "model": "majority",
           "config_hash": "h", "data": "f.jsonl", "at_recall": 0.5,
           "isAt_recall": 0.5, "global": 0.5, "n_dev": 10, "notes": ""}
    registry.append_leaderboard(tmp_path, row)
    registry.append_leaderboard(tmp_path, {**row, "run_id": "r2"})
    text = (tmp_path / "leaderboard.csv").read_text()
    assert text.count("run_id,timestamp") == 1     # header once
    assert "r1" in text and "r2" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runs/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.runs'`

- [ ] **Step 3: Create `hipe/runs/__init__.py`**

```python
# hipe/runs/__init__.py
```

- [ ] **Step 4: Write `hipe/runs/registry.py`**

```python
# hipe/runs/registry.py
import csv
import hashlib
import json
from pathlib import Path

LEADERBOARD_COLUMNS = ["run_id", "timestamp", "model", "config_hash", "data",
                       "at_recall", "isAt_recall", "global", "n_dev", "notes"]


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def new_run_dir(model_name, cfg_hash, root, now) -> Path:
    run_dir = Path(root) / f"{now}_{model_name}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir, manifest: dict) -> None:
    (Path(run_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def append_leaderboard(root, row: dict) -> None:
    path = Path(root) / "leaderboard.csv"
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEADERBOARD_COLUMNS})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/runs/test_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add hipe/runs/__init__.py hipe/runs/registry.py tests/runs/test_registry.py
git commit -m "feat: file-based run registry + leaderboard"
```

---

### Task 14: Harness orchestration

**Files:**
- Create: `hipe/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `load_pairs`, `split_by_document`, `pair_key`, `registry.get_model`, `apply_consistency`, `macro_recall`, `write_submission`, `runs.registry` functions.
- Produces: `hipe.harness.run_experiment(config: dict, now: str, runs_root=None) -> dict` returning `{"run_dir": str, "at_recall": float, "isAt_recall": float, "global": float, "n_dev": int}`. It loads pairs from `config["data"]["train"]`, splits, fits the model named in `config["model"]`, predicts on dev, applies the consistency rule, writes a dev submission + gold file, computes metrics, writes the manifest, and appends a leaderboard row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness.py
from pathlib import Path
from hipe.harness import run_experiment

FIX = Path(__file__).resolve().parent / "fixtures" / "mini.jsonl"


def test_run_experiment_majority_end_to_end(tmp_path):
    config = {
        "data": {"train": str(FIX), "dev_frac": 0.5, "seed": 0},
        "model": {"name": "majority"},
    }
    result = run_experiment(config, now="2026-06-21_120000", runs_root=tmp_path)
    run_dir = Path(result["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "predictions" / "dev.jsonl").exists()
    assert 0.0 <= result["global"] <= 1.0
    # leaderboard row written
    lb = (tmp_path / "leaderboard.csv").read_text()
    assert "majority" in lb


def test_run_experiment_applies_consistency(tmp_path):
    # A model whose isAt=TRUE must yield at=TRUE in the written submission.
    from hipe.models import registry
    from hipe.models.base import RelationModel
    from hipe.data.load import read_jsonl

    @registry.register("force_isat_true")
    class ForceIsat(RelationModel):
        name = "force_isat_true"
        def fit(self, train, dev=None): pass
        def predict(self, pairs):
            return [{"at": "FALSE", "isAt": "TRUE"} for _ in pairs]

    config = {"data": {"train": str(FIX), "dev_frac": 1.0, "seed": 0},
              "model": {"name": "force_isat_true"}}
    result = run_experiment(config, now="2026-06-21_120001", runs_root=tmp_path)
    rows = read_jsonl(Path(result["run_dir"]) / "predictions" / "dev.jsonl")
    for r in rows:
        for sp in r["sampled_pairs"]:
            if sp["isAt"] == "TRUE":
                assert sp["at"] == "TRUE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.harness'`

- [ ] **Step 3: Write `hipe/harness.py`**

```python
# hipe/harness.py
from pathlib import Path
from hipe import config as cfg
from hipe.data.pairs import load_pairs, pair_key
from hipe.data.split import split_by_document
from hipe.data.submission import write_submission
from hipe.models import baselines  # noqa: F401  (registers majority/random)
from hipe.models import registry
from hipe.models.base import apply_consistency
from hipe.eval.metrics import macro_recall
from hipe.eval.scorer import score_files
from hipe.runs import registry as runs


def run_experiment(config: dict, now: str, runs_root=None) -> dict:
    runs_root = Path(runs_root) if runs_root is not None else cfg.RUNS_DIR
    train_path = config["data"]["train"]
    dev_frac = config["data"].get("dev_frac", 0.2)
    seed = config["data"].get("seed", 0)

    pairs = load_pairs(train_path)
    train, dev = split_by_document(pairs, dev_frac=dev_frac, seed=seed)

    model_cfg = dict(config["model"])
    name = model_cfg.pop("name")
    model = registry.get_model(name, **model_cfg)
    model.fit(train, dev)

    raw_preds = model.predict(dev)
    preds = {}
    for p, pred in zip(dev, raw_preds):
        preds[(p.doc_id, pair_key(p))] = apply_consistency(dict(pred))

    cfg_hash = runs.config_hash(config)
    run_dir = runs.new_run_dir(name, cfg_hash, runs_root, now)
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # write a dev gold file (only dev docs) + a dev submission, then score
    dev_docs = {p.doc_id for p in dev}
    _write_subset(train_path, dev_docs, pred_dir / "dev_gold.jsonl")
    write_submission(pred_dir / "dev_gold.jsonl", preds, pred_dir / "dev.jsonl")

    at_true = [p.gold_at for p in dev]
    at_pred = [preds[(p.doc_id, pair_key(p))]["at"] for p in dev]
    isat_true = [p.gold_isat for p in dev]
    isat_pred = [preds[(p.doc_id, pair_key(p))]["isAt"] for p in dev]
    at_recall = macro_recall(at_true, at_pred)
    isat_recall = macro_recall(isat_true, isat_pred)
    global_recall = (at_recall + isat_recall) / 2

    manifest = {"model": name, "config": config, "config_hash": cfg_hash,
                "now": now, "at_recall": at_recall, "isAt_recall": isat_recall,
                "global": global_recall, "n_dev": len(dev)}
    runs.write_manifest(run_dir, manifest)
    runs.append_leaderboard(runs_root, {
        "run_id": run_dir.name, "timestamp": now, "model": name,
        "config_hash": cfg_hash, "data": str(train_path),
        "at_recall": round(at_recall, 4), "isAt_recall": round(isat_recall, 4),
        "global": round(global_recall, 4), "n_dev": len(dev), "notes": ""})

    return {"run_dir": str(run_dir), "at_recall": at_recall,
            "isAt_recall": isat_recall, "global": global_recall, "n_dev": len(dev)}


def _write_subset(src_path, keep_doc_ids, out_path):
    import json
    from hipe.data.load import read_jsonl
    rows = [r for r in read_jsonl(src_path) if str(r["document_id"]) in keep_doc_ids]
    with Path(out_path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/harness.py tests/test_harness.py
git commit -m "feat: run_experiment harness (load->split->fit->predict->score->persist)"
```

---

### Task 15: CLI + configs

**Files:**
- Create: `hipe/cli.py`
- Create: `configs/baseline_majority.yaml`
- Create: `configs/baseline_random.yaml`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `run_experiment`, `score_files`, YAML configs.
- Produces: `hipe.cli.main(argv=None) -> int` with subcommands:
  - `hipe run <config.yaml>` — runs an experiment (stamps `now` from the wall clock).
  - `hipe score <gold.jsonl> <pred.jsonl>` — prints official metrics.
  - `hipe leaderboard` — prints `runs/leaderboard.csv`.

- [ ] **Step 1: Write the configs**

`configs/baseline_majority.yaml`:

```yaml
data:
  train: data/raw/HIPE-2026-data/data/sandbox/en-train.jsonl
  dev_frac: 0.2
  seed: 0
model:
  name: majority
```

`configs/baseline_random.yaml`:

```yaml
data:
  train: data/raw/HIPE-2026-data/data/sandbox/en-train.jsonl
  dev_frac: 0.2
  seed: 0
model:
  name: random
  seed: 0
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_cli.py
from pathlib import Path
import yaml
from hipe.cli import main

FIX = Path(__file__).resolve().parent / "fixtures" / "mini.jsonl"


def test_cli_run_creates_leaderboard(tmp_path, monkeypatch, capsys):
    cfg = {"data": {"train": str(FIX), "dev_frac": 0.5, "seed": 0},
           "model": {"name": "majority"}}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HIPE_RUNS_DIR", str(tmp_path / "runs"))
    rc = main(["run", str(cfg_path)])
    assert rc == 0
    assert (tmp_path / "runs" / "leaderboard.csv").exists()
    assert "global" in capsys.readouterr().out


def test_cli_score(capsys):
    fixdir = Path(__file__).resolve().parent / "fixtures"
    rc = main(["score", str(fixdir / "gold.jsonl"), str(fixdir / "pred.jsonl")])
    assert rc == 0
    assert "0.75" in capsys.readouterr().out
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.cli'`

- [ ] **Step 4: Write `hipe/cli.py`**

```python
# hipe/cli.py
import argparse
import os
from datetime import datetime
from pathlib import Path
import yaml
from hipe import config as cfg
from hipe.harness import run_experiment
from hipe.eval.scorer import score_files


def _runs_root():
    return Path(os.environ.get("HIPE_RUNS_DIR", cfg.RUNS_DIR))


def _cmd_run(args) -> int:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result = run_experiment(config, now=now, runs_root=_runs_root())
    print(f"run_dir: {result['run_dir']}")
    print(f"at_recall: {result['at_recall']:.4f}  "
          f"isAt_recall: {result['isAt_recall']:.4f}  "
          f"global: {result['global']:.4f}  (n_dev={result['n_dev']})")
    return 0


def _cmd_score(args) -> int:
    m = score_files(args.gold, args.pred)
    print(f"at macro_recall:    {m['at']['macro_recall']:.4f}")
    print(f"isAt macro_recall:  {m['isAt']['macro_recall']:.4f}")
    print(f"global macro_recall: {m['global']['macro_recall']:.4f}")
    return 0


def _cmd_leaderboard(args) -> int:
    path = _runs_root() / "leaderboard.csv"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
    else:
        print("(no runs yet)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hipe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run an experiment from a config")
    p_run.add_argument("config")
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser("score", help="score a prediction file vs gold")
    p_score.add_argument("gold")
    p_score.add_argument("pred")
    p_score.set_defaults(func=_cmd_score)

    p_lb = sub.add_parser("leaderboard", help="print the leaderboard")
    p_lb.set_defaults(func=_cmd_leaderboard)

    args = parser.parse_args(argv)
    return args.func(args)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q`
Expected: all tests PASS

- [ ] **Step 7: Smoke-test on real data (manual)**

Run: `hipe run configs/baseline_majority.yaml && hipe run configs/baseline_random.yaml && hipe leaderboard`
Expected: two leaderboard rows printed (majority + random) with `global` scores.

- [ ] **Step 8: Commit**

```bash
git add hipe/cli.py configs/baseline_majority.yaml configs/baseline_random.yaml tests/test_cli.py
git commit -m "feat: CLI (run/score/leaderboard) + baseline configs"
```

---

### Task 16: Update .gitignore for run artifacts + README pointer

**Files:**
- Modify: `.gitignore`
- Create: `README.md`

**Interfaces:** none (housekeeping).

- [ ] **Step 1: Append run-artifact ignores to `.gitignore`**

Add these lines (keep `leaderboard.csv` committed — it is the ledger):

```gitignore
runs/**/predictions/
runs/**/manifest.json
runs/**/*.pkl
!runs/leaderboard.csv
data/raw/
data/cache/
```

- [ ] **Step 2: Write `README.md`**

```markdown
# HIPE-2026 Relation-Extraction Harness

Person–place relation extraction for CLEF HIPE-2026. See the design spec in
`docs/superpowers/specs/` and the plans in `docs/superpowers/plans/`.

## Setup

```bash
pip install -e ".[dev]"
python scripts/fetch_data.py        # clone pinned HIPE-2026-data into data/raw/
```

## Run

```bash
hipe run configs/baseline_majority.yaml   # train -> predict -> score -> leaderboard row
hipe leaderboard                          # show all runs
hipe score <gold.jsonl> <pred.jsonl>      # official-parity score of a file
```

Each run writes a folder under `runs/` (config, predictions, metrics) and appends
one row to the committed `runs/leaderboard.csv` — nothing is lost.
```

- [ ] **Step 3: Verify the suite still passes**

Run: `pytest -q`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add .gitignore README.md
git commit -m "chore: ignore run artifacts (keep leaderboard); add README"
```

---

## Self-Review

**Spec coverage (this plan = build step 0):**
- Shared preprocessing / OCR handling (§5.4b) → Task 5 (normalize + fuzzy_find), used in Task 6.
- Canonical `Pair` unit (§5.1) → Tasks 3, 6.
- Model interface + consistency rule (§5.2, §5.3) → Task 10; consistency enforced in harness Task 14.
- Run protocol (§5.4) → Tasks 13, 14 (load → preprocess → fit → predict → consistency → submission → score → persist → leaderboard, idempotent dir naming via config_hash).
- Official-scorer parity (§7) → Task 9 (vendored official module + wrapper).
- File-based registry + committed `leaderboard.csv` (§3) → Tasks 13, 16.
- Document-grouped split (§5.6) → Task 7.
- Data acquisition via pinned `fetch_data.py` (§3) → Task 2.
- Trivial baselines (§6.1) → Task 11.
- Submission format round-trips input docs (Global Constraints) → Task 12.
- *Deferred to later plans (correctly out of scope here):* feature store + classical ML (step 1), transformer + Kaggle bridge (step 3), LLM via litellm (step 2), ensembles (step 4), OOF predictions for stacking. The `RelationModel` interface + registry make these additive with no harness change.

**Placeholder scan:** none — every code/step block is concrete.

**Type consistency:** `pair_key(pair) -> "person.entity_id+place.entity_id"` is defined in Task 6 and reused identically in Tasks 12 (via string concat of the same ids) and 14. `predictions` dict keyed by `(doc_id, pair_key)` is consistent between Task 12 (writer) and Task 14 (harness). `run_experiment(config, now, runs_root)` signature is consistent between Tasks 14 and 15. Leaderboard columns defined once in Task 13 and used by Task 14.
