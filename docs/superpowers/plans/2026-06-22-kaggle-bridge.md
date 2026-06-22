# Automated Kaggle Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `hipe kaggle-run <config>` command that runs any harness config on Kaggle GPU fully automatically — package the code as a Kaggle Dataset, push a GPU+internet kernel that runs `hipe run`, poll to completion, and ingest the results into the local `runs/` leaderboard.

**Architecture:** A `hipe/kaggle/` package wraps the `kaggle` CLI via subprocess. `export` stages the repo code into a Dataset bundle + a Python script kernel (with `kernel-metadata.json` enabling GPU and internet). The kernel installs the package from the dataset, clones the HIPE data, and runs `hipe run <config>` writing to `/kaggle/working/runs`. `bridge` pushes, polls `kaggle kernels status`, downloads outputs, and `ingest` merges the downloaded run folder + leaderboard row into local `runs/`. Every piece is unit-tested with the subprocess layer mocked; the one live round-trip is a user-run command.

**Tech Stack:** Python 3.12, the `kaggle` CLI/SDK, subprocess, pytest. Builds on the existing run-registry + harness. The model being run on Kaggle (e.g. `xlmr`) is unchanged — the bridge is model-agnostic.

## Global Constraints

- Python 3.12.
- The bridge shells out to the `kaggle` CLI; all CLI calls go through one mockable `_run(cmd)` so unit tests never touch the network or require Kaggle auth.
- Kernel metadata MUST set `enable_gpu: true` and `enable_internet: true` (internet is needed for `pip install` and cloning the HIPE data on Kaggle), `kernel_type: "script"`, `is_private: true`.
- The Kaggle username comes from `$KAGGLE_USERNAME` or `~/.kaggle/kaggle.json`; never hard-coded.
- The kernel runs `hipe run <config>` with `HIPE_RUNS_DIR=/kaggle/working/runs` so outputs land in the kernel's output; the harness/scorer/leaderboard are unchanged.
- Data on Kaggle is cloned from `https://github.com/hipe-eval/HIPE-2026-data.git` at the pinned commit `4228562` (matches `scripts/fetch_data.py`).
- `ingest` merges by `run_id` (idempotent) using the existing `LEADERBOARD_COLUMNS`; it never overwrites a committed local run folder that already exists.
- Live end-to-end runs require a phone-verified Kaggle account + `~/.kaggle/kaggle.json` (user-provided); the implementation itself needs neither.

---

### Task 1: Kaggle dependency + metadata builders

**Files:**
- Modify: `pyproject.toml`
- Create: `hipe/kaggle/__init__.py`
- Create: `hipe/kaggle/metadata.py`
- Test: `tests/kaggle/test_metadata.py`

**Interfaces:**
- Adds a `kaggle` optional-dependency extra (`kaggle>=1.6`).
- Produces: `hipe.kaggle.metadata.DATASET_SLUG="hipe-code"`; `kaggle_username() -> str`; `dataset_metadata(username, slug=DATASET_SLUG) -> dict`; `kernel_slug(config_path) -> str`; `kernel_metadata(username, k_slug, code_file, dataset_slug=DATASET_SLUG, enable_gpu=True, enable_internet=True) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/kaggle/test_metadata.py
from hipe.kaggle import metadata as md


def test_kaggle_username_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    assert md.kaggle_username() == "alice"


def test_dataset_metadata_id(monkeypatch):
    d = md.dataset_metadata("alice")
    assert d["id"] == "alice/hipe-code"
    assert d["title"] == "hipe-code"
    assert d["licenses"][0]["name"]


def test_kernel_slug_from_config_path():
    assert md.kernel_slug("configs/xlmr.yaml") == "hipe-run-xlmr"
    assert md.kernel_slug("configs/embedding_svm.yaml") == "hipe-run-embedding-svm"


def test_kernel_metadata_enables_gpu_and_internet():
    m = md.kernel_metadata("alice", "hipe-run-xlmr", "run_kernel.py")
    assert m["id"] == "alice/hipe-run-xlmr"
    assert m["enable_gpu"] is True
    assert m["enable_internet"] is True
    assert m["kernel_type"] == "script"
    assert m["is_private"] is True
    assert m["dataset_sources"] == ["alice/hipe-code"]
    assert m["code_file"] == "run_kernel.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/kaggle/test_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.kaggle'`

- [ ] **Step 3: Add the `kaggle` extra to `pyproject.toml`** — under `[project.optional-dependencies]` add:

```toml
kaggle = ["kaggle>=1.6"]
```

(Keep the existing `dev`, `ml` extras unchanged.)

- [ ] **Step 4: Create `hipe/kaggle/__init__.py`** (empty) and write `hipe/kaggle/metadata.py`**

```python
# hipe/kaggle/metadata.py
import json
import os
from pathlib import Path

DATASET_SLUG = "hipe-code"


def kaggle_username() -> str:
    env = os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8"))["username"]
    raise RuntimeError(
        "No Kaggle username: set $KAGGLE_USERNAME or create ~/.kaggle/kaggle.json")


def dataset_metadata(username: str, slug: str = DATASET_SLUG) -> dict:
    return {"title": slug, "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}]}


def kernel_slug(config_path) -> str:
    return ("hipe-run-" + Path(config_path).stem).replace("_", "-")


def kernel_metadata(username, k_slug, code_file, dataset_slug=DATASET_SLUG,
                    enable_gpu=True, enable_internet=True) -> dict:
    return {
        "id": f"{username}/{k_slug}",
        "title": k_slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "dataset_sources": [f"{username}/{dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
```

- [ ] **Step 5: Install the kaggle extra**

Run: `pip install -e ".[kaggle]"`
Expected: installs the `kaggle` package; `kaggle --version` prints a version. (The user's `~/.kaggle/kaggle.json` provides auth — not needed for this task's tests.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/kaggle/test_metadata.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml hipe/kaggle/__init__.py hipe/kaggle/metadata.py tests/kaggle/test_metadata.py
git commit -m "feat: kaggle extra + bridge metadata builders"
```

---

### Task 2: Kernel script template

**Files:**
- Create: `hipe/kaggle/kernel.py`
- Test: `tests/kaggle/test_kernel.py`

**Interfaces:**
- Produces: `hipe.kaggle.kernel.DATA_REPO`, `hipe.kaggle.kernel.PINNED_COMMIT`, and `render_kernel_script(config_rel: str, dataset_slug: str = "hipe-code") -> str` — the source of the Python script that runs ON Kaggle.

- [ ] **Step 1: Write the failing test**

```python
# tests/kaggle/test_kernel.py
from hipe.kaggle.kernel import render_kernel_script, PINNED_COMMIT


def test_kernel_script_has_required_steps():
    s = render_kernel_script("configs/xlmr.yaml")
    # installs the package from the input dataset with the ml extra
    assert "/kaggle/input/hipe-code" in s
    assert "pip" in s and "[ml]" in s
    # clones the pinned HIPE data
    assert "HIPE-2026-data" in s
    assert PINNED_COMMIT in s
    # runs the harness with outputs going to the kernel working dir
    assert "HIPE_RUNS_DIR" in s
    assert "/kaggle/working/runs" in s
    assert "configs/xlmr.yaml" in s
    # it is valid Python
    compile(s, "<kernel>", "exec")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/kaggle/test_kernel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.kaggle.kernel'`

- [ ] **Step 3: Write `hipe/kaggle/kernel.py`**

```python
# hipe/kaggle/kernel.py
DATA_REPO = "https://github.com/hipe-eval/HIPE-2026-data.git"
PINNED_COMMIT = "4228562"

_TEMPLATE = '''import os
import subprocess
import sys

WORK = "/kaggle/working"
DATASET = "/kaggle/input/{dataset_slug}"
os.chdir(WORK)

# 1. install the hipe package (with the ml extra) from the read-only input dataset
subprocess.run([sys.executable, "-m", "pip", "install", DATASET + "[ml]"], check=True)

# 2. clone the pinned HIPE data into the writable working dir (internet enabled)
subprocess.run(["git", "clone", "{data_repo}", "data/raw/HIPE-2026-data"], check=True)
subprocess.run(["git", "-C", "data/raw/HIPE-2026-data", "checkout", "{pinned}"], check=True)

# 3. run the harness; outputs (run folder + leaderboard) land in /kaggle/working/runs
os.environ["HIPE_RUNS_DIR"] = WORK + "/runs"
subprocess.run(["hipe", "run", DATASET + "/{config_rel}"], check=True)
'''


def render_kernel_script(config_rel: str, dataset_slug: str = "hipe-code") -> str:
    return _TEMPLATE.format(dataset_slug=dataset_slug, data_repo=DATA_REPO,
                            pinned=PINNED_COMMIT, config_rel=config_rel)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/kaggle/test_kernel.py -v`
Expected: PASS (1 test, including the `compile()` validity check)

- [ ] **Step 5: Commit**

```bash
git add hipe/kaggle/kernel.py tests/kaggle/test_kernel.py
git commit -m "feat: kaggle kernel script template (install, clone data, run harness)"
```

---

### Task 3: Code staging (export)

**Files:**
- Create: `hipe/kaggle/export.py`
- Test: `tests/kaggle/test_export.py`

**Interfaces:**
- Consumes: `metadata`, `kernel.render_kernel_script`.
- Produces: `hipe.kaggle.export.CODE_INCLUDE` (list), `stage_job(config_path, staging_dir, repo_root) -> dict` returning `{"code": <path>, "kernel": <path>, "kernel_id": "<user>/<slug>"}`. It copies the package code into `staging/code/` (with `dataset-metadata.json`), and writes `staging/kernel/run_kernel.py` + `kernel-metadata.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/kaggle/test_export.py
import json
from pathlib import Path
from hipe.kaggle.export import stage_job


def _fake_repo(root):
    (root / "hipe").mkdir()
    (root / "hipe" / "__init__.py").write_text("")
    (root / "configs").mkdir()
    (root / "configs" / "xlmr.yaml").write_text("model:\n  name: xlmr\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "fetch_data.py").write_text("# fetch\n")
    (root / "pyproject.toml").write_text("[project]\nname='hipe'\n")


def test_stage_job_builds_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    repo = tmp_path / "repo"; repo.mkdir(); _fake_repo(repo)
    staging = tmp_path / "stage"
    job = stage_job("configs/xlmr.yaml", staging, repo)

    code = Path(job["code"]); kernel = Path(job["kernel"])
    assert (code / "hipe" / "__init__.py").exists()
    assert (code / "configs" / "xlmr.yaml").exists()
    assert (code / "pyproject.toml").exists()
    dsmeta = json.loads((code / "dataset-metadata.json").read_text())
    assert dsmeta["id"] == "alice/hipe-code"
    assert (kernel / "run_kernel.py").exists()
    kmeta = json.loads((kernel / "kernel-metadata.json").read_text())
    assert kmeta["enable_gpu"] is True and kmeta["enable_internet"] is True
    assert "configs/xlmr.yaml" in (kernel / "run_kernel.py").read_text()
    assert job["kernel_id"] == "alice/hipe-run-xlmr"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/kaggle/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.kaggle.export'`

- [ ] **Step 3: Write `hipe/kaggle/export.py`**

```python
# hipe/kaggle/export.py
import json
import shutil
from pathlib import Path
from hipe.kaggle import metadata
from hipe.kaggle.kernel import render_kernel_script

CODE_INCLUDE = ["hipe", "configs", "scripts", "pyproject.toml"]


def stage_job(config_path, staging_dir, repo_root) -> dict:
    staging = Path(staging_dir)
    code = staging / "code"
    kernel = staging / "kernel"
    code.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root)

    for item in CODE_INCLUDE:
        src = repo_root / item
        dst = code / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            shutil.copy2(src, dst)

    user = metadata.kaggle_username()
    (code / "dataset-metadata.json").write_text(
        json.dumps(metadata.dataset_metadata(user), indent=2), encoding="utf-8")

    k_slug = metadata.kernel_slug(config_path)
    code_file = "run_kernel.py"
    config_rel = f"configs/{Path(config_path).name}"
    (kernel / code_file).write_text(render_kernel_script(config_rel), encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(metadata.kernel_metadata(user, k_slug, code_file), indent=2),
        encoding="utf-8")

    return {"code": str(code), "kernel": str(kernel),
            "kernel_id": f"{user}/{k_slug}"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/kaggle/test_export.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add hipe/kaggle/export.py tests/kaggle/test_export.py
git commit -m "feat: stage repo code + kernel into a Kaggle job bundle"
```

---

### Task 4: Kaggle CLI wrappers (bridge subprocess layer)

**Files:**
- Create: `hipe/kaggle/bridge.py`
- Test: `tests/kaggle/test_bridge.py`

**Interfaces:**
- Produces: `hipe.kaggle.bridge._run(cmd)` (the single subprocess seam); `push_dataset(code_dir)`, `push_kernel(kernel_dir)`, `kernel_status(kernel_id) -> str` (one of `"complete"|"error"|"running"|"unknown"`), `download_output(kernel_id, dest)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/kaggle/test_bridge.py
import subprocess
from hipe.kaggle import bridge


class _Rec:
    def __init__(self, stdout="", fail=False):
        self.calls = []
        self.stdout = stdout
        self.fail = fail

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if self.fail:
            raise subprocess.CalledProcessError(1, cmd)
        return type("R", (), {"stdout": self.stdout, "returncode": 0})()


def test_push_kernel_command(monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(bridge, "_run", rec)
    bridge.push_kernel("/tmp/k")
    assert rec.calls[0] == ["kaggle", "kernels", "push", "-p", "/tmp/k"]


def test_push_dataset_versions_then_creates_on_failure(monkeypatch):
    # version fails (dataset absent) -> falls back to create
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        if cmd[1:3] == ["datasets", "version"]:
            raise subprocess.CalledProcessError(1, cmd)
        return type("R", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(bridge, "_run", fake)
    bridge.push_dataset("/tmp/code")
    assert calls[0][1:3] == ["datasets", "version"]
    assert calls[1][1:3] == ["datasets", "create"]


def test_kernel_status_parses(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='status "complete"'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "complete"
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='has error'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "error"
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='still running'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "running"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/kaggle/test_bridge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.kaggle.bridge'`

- [ ] **Step 3: Write `hipe/kaggle/bridge.py`**

```python
# hipe/kaggle/bridge.py
import subprocess
from pathlib import Path


def _run(cmd, **kw):
    """Single subprocess seam for the kaggle CLI (mocked in tests)."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def push_dataset(code_dir):
    """Create the dataset, or push a new version if it already exists."""
    try:
        return _run(["kaggle", "datasets", "version", "-p", str(code_dir),
                     "-m", "update", "--dir-mode", "zip"])
    except subprocess.CalledProcessError:
        return _run(["kaggle", "datasets", "create", "-p", str(code_dir),
                     "--dir-mode", "zip"])


def push_kernel(kernel_dir):
    return _run(["kaggle", "kernels", "push", "-p", str(kernel_dir)])


def kernel_status(kernel_id) -> str:
    out = _run(["kaggle", "kernels", "status", kernel_id]).stdout.lower()
    if "complete" in out:
        return "complete"
    if "error" in out:
        return "error"
    if "running" in out or "queue" in out:
        return "running"
    return "unknown"


def download_output(kernel_id, dest):
    Path(dest).mkdir(parents=True, exist_ok=True)
    return _run(["kaggle", "kernels", "output", kernel_id, "-p", str(dest)])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/kaggle/test_bridge.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/kaggle/bridge.py tests/kaggle/test_bridge.py
git commit -m "feat: kaggle CLI wrappers (push dataset/kernel, status, output)"
```

---

### Task 5: Ingest Kaggle outputs into the local leaderboard

**Files:**
- Create: `hipe/kaggle/ingest.py`
- Test: `tests/kaggle/test_ingest.py`

**Interfaces:**
- Consumes: `hipe.runs.registry.LEADERBOARD_COLUMNS`.
- Produces: `hipe.kaggle.ingest.ingest_output(download_dir, runs_root) -> list[str]` (names of run folders copied in). It copies `download_dir/runs/<run_id>/` folders into `runs_root/` (skipping any that already exist) and merges `download_dir/runs/leaderboard.csv` into `runs_root/leaderboard.csv` deduped by `run_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/kaggle/test_ingest.py
import csv
from pathlib import Path
from hipe.kaggle.ingest import ingest_output


def _make_download(root, run_id, global_score):
    runs = root / "runs" / run_id
    runs.mkdir(parents=True)
    (runs / "manifest.json").write_text("{}")
    lb = root / "runs" / "leaderboard.csv"
    with lb.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "timestamp", "model", "config_hash", "data",
                    "at_recall", "isAt_recall", "global", "n_dev", "notes"])
        w.writerow([run_id, "t", "xlmr", "h", "d", 0.5, 0.6, global_score, 100, ""])


def test_ingest_copies_run_and_merges_leaderboard(tmp_path):
    dl = tmp_path / "dl"
    _make_download(dl, "2026-06-22_120000_xlmr_abcd1234", 0.55)
    runs_root = tmp_path / "runs"
    copied = ingest_output(dl, runs_root)
    assert copied == ["2026-06-22_120000_xlmr_abcd1234"]
    assert (runs_root / "2026-06-22_120000_xlmr_abcd1234" / "manifest.json").exists()
    rows = list(csv.DictReader(open(runs_root / "leaderboard.csv")))
    assert rows[0]["model"] == "xlmr" and rows[0]["global"] == "0.55"


def test_ingest_is_idempotent_by_run_id(tmp_path):
    dl = tmp_path / "dl"
    _make_download(dl, "2026-06-22_120000_xlmr_abcd1234", 0.55)
    runs_root = tmp_path / "runs"
    ingest_output(dl, runs_root)
    ingest_output(dl, runs_root)               # second time: no duplicate row
    rows = list(csv.DictReader(open(runs_root / "leaderboard.csv")))
    assert len(rows) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/kaggle/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hipe.kaggle.ingest'`

- [ ] **Step 3: Write `hipe/kaggle/ingest.py`**

```python
# hipe/kaggle/ingest.py
import csv
import shutil
from pathlib import Path
from hipe.runs.registry import LEADERBOARD_COLUMNS


def ingest_output(download_dir, runs_root) -> list:
    download = Path(download_dir)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    src_runs = download / "runs"
    copied = []
    if not src_runs.is_dir():
        return copied
    for child in sorted(src_runs.iterdir()):
        if child.is_dir():
            dst = runs_root / child.name
            if not dst.exists():
                shutil.copytree(child, dst)
            copied.append(child.name)
    _merge_leaderboard(src_runs / "leaderboard.csv", runs_root / "leaderboard.csv")
    return copied


def _merge_leaderboard(src_csv, dst_csv):
    src_csv = Path(src_csv)
    dst_csv = Path(dst_csv)
    if not src_csv.exists():
        return
    rows = {}
    if dst_csv.exists():
        for r in csv.DictReader(dst_csv.open(encoding="utf-8")):
            rows[r["run_id"]] = r
    for r in csv.DictReader(src_csv.open(encoding="utf-8")):
        rows[r["run_id"]] = r              # newest wins, deduped by run_id
    with dst_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in LEADERBOARD_COLUMNS})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/kaggle/test_ingest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add hipe/kaggle/ingest.py tests/kaggle/test_ingest.py
git commit -m "feat: ingest Kaggle run outputs into the local leaderboard (idempotent)"
```

---

### Task 6: Orchestration + `hipe kaggle-run` CLI

**Files:**
- Modify: `hipe/kaggle/bridge.py` (add `run_kaggle`)
- Modify: `hipe/cli.py` (add the `kaggle-run` subcommand)
- Create: `README` section note (Modify: `README.md`)
- Test: `tests/kaggle/test_run_kaggle.py`

**Interfaces:**
- Consumes: `export.stage_job`, `push_dataset`, `push_kernel`, `kernel_status`, `download_output`, `ingest.ingest_output`.
- Produces: `hipe.kaggle.bridge.run_kaggle(config_path, repo_root, runs_root, staging_dir, *, poll_interval=30, max_polls=240, sleep=None) -> dict` returning `{"kernel_id": str, "runs": list[str]}`; raises `RuntimeError` on kernel error, `TimeoutError` if it never completes. CLI: `hipe kaggle-run <config>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/kaggle/test_run_kaggle.py
from hipe.kaggle import bridge


def test_run_kaggle_full_flow(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(bridge.export, "stage_job",
                        lambda c, s, r: {"code": "C", "kernel": "K",
                                         "kernel_id": "alice/hipe-run-xlmr"})
    monkeypatch.setattr(bridge, "push_dataset", lambda d: events.append(("ds", d)))
    monkeypatch.setattr(bridge, "push_kernel", lambda d: events.append(("kn", d)))
    statuses = iter(["running", "running", "complete"])
    monkeypatch.setattr(bridge, "kernel_status", lambda kid: next(statuses))
    monkeypatch.setattr(bridge, "download_output",
                        lambda kid, dest: events.append(("dl", kid)))
    monkeypatch.setattr(bridge.ingest, "ingest_output",
                        lambda dl, rr: ["2026-06-22_120000_xlmr_abcd1234"])

    result = bridge.run_kaggle("configs/xlmr.yaml", repo_root=tmp_path,
                               runs_root=tmp_path / "runs",
                               staging_dir=tmp_path / "stage",
                               poll_interval=0, sleep=lambda s: None)
    assert result["kernel_id"] == "alice/hipe-run-xlmr"
    assert result["runs"] == ["2026-06-22_120000_xlmr_abcd1234"]
    assert ("ds", "C") in events and ("kn", "K") in events
    assert ("dl", "alice/hipe-run-xlmr") in events


def test_run_kaggle_raises_on_kernel_error(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(bridge.export, "stage_job",
                        lambda c, s, r: {"code": "C", "kernel": "K",
                                         "kernel_id": "alice/x"})
    monkeypatch.setattr(bridge, "push_dataset", lambda d: None)
    monkeypatch.setattr(bridge, "push_kernel", lambda d: None)
    monkeypatch.setattr(bridge, "kernel_status", lambda kid: "error")
    with pytest.raises(RuntimeError):
        bridge.run_kaggle("configs/xlmr.yaml", repo_root=tmp_path,
                          runs_root=tmp_path / "runs",
                          staging_dir=tmp_path / "stage",
                          poll_interval=0, sleep=lambda s: None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/kaggle/test_run_kaggle.py -v`
Expected: FAIL with `AttributeError: module 'hipe.kaggle.bridge' has no attribute 'run_kaggle'`

- [ ] **Step 3: Add `run_kaggle` to `hipe/kaggle/bridge.py`** — add these imports at the top (below the existing imports) and the function at the end:

```python
import time as _time
from hipe.kaggle import export, ingest


def run_kaggle(config_path, repo_root, runs_root, staging_dir, *,
               poll_interval=30, max_polls=240, sleep=None) -> dict:
    sleep = sleep or _time.sleep
    job = export.stage_job(config_path, staging_dir, repo_root)
    push_dataset(job["code"])
    push_kernel(job["kernel"])
    for _ in range(max_polls):
        status = kernel_status(job["kernel_id"])
        if status == "complete":
            break
        if status == "error":
            raise RuntimeError(f"Kaggle kernel failed: {job['kernel_id']}")
        sleep(poll_interval)
    else:
        raise TimeoutError(f"Kaggle kernel did not complete: {job['kernel_id']}")
    out_dir = Path(staging_dir) / "output"
    download_output(job["kernel_id"], out_dir)
    runs = ingest.ingest_output(out_dir, runs_root)
    return {"kernel_id": job["kernel_id"], "runs": runs}
```

- [ ] **Step 4: Add the `kaggle-run` subcommand to `hipe/cli.py`** — add this command function and register it in `main`'s subparsers (alongside `run`/`score`/`leaderboard`):

```python
def _cmd_kaggle_run(args) -> int:
    import tempfile
    from hipe.kaggle.bridge import run_kaggle
    staging = tempfile.mkdtemp(prefix="hipe_kaggle_")
    result = run_kaggle(args.config, repo_root=cfg.ROOT, runs_root=_runs_root(),
                        staging_dir=staging)
    print(f"kernel: {result['kernel_id']}")
    print(f"ingested runs: {result['runs']}")
    return 0
```

And in `main`, after the `leaderboard` subparser, add:

```python
    p_kaggle = sub.add_parser("kaggle-run",
                              help="run a config on Kaggle GPU and ingest results")
    p_kaggle.add_argument("config")
    p_kaggle.set_defaults(func=_cmd_kaggle_run)
```

- [ ] **Step 5: Run the orchestration tests + full suite**

Run: `python -m pytest tests/kaggle/test_run_kaggle.py -v && python -m pytest -q`
Expected: PASS (2 new + the whole suite).

- [ ] **Step 6: Add a README note**

Add to `README.md` under the Run section:

```markdown
## Run on Kaggle GPU (automated)

Requires a phone-verified Kaggle account and `~/.kaggle/kaggle.json`, plus `pip install -e ".[kaggle]"`.

```bash
hipe kaggle-run configs/xlmr.yaml   # packages code -> Kaggle dataset -> GPU kernel -> ingests results
hipe leaderboard                    # the xlmr row, trained on Kaggle, appears here
```

It pushes the code as a private dataset, runs a GPU+internet kernel that clones the
HIPE data and runs `hipe run`, polls to completion, and merges the run folder +
leaderboard row into local `runs/`.
```

- [ ] **Step 7: Commit**

```bash
git add hipe/kaggle/bridge.py hipe/cli.py README.md tests/kaggle/test_run_kaggle.py
git commit -m "feat: hipe kaggle-run orchestration + CLI"
```

- [ ] **Step 8: LIVE end-to-end run (manual — requires your Kaggle account)**

This is the one step the unit tests can't cover. With `~/.kaggle/kaggle.json` in place and `pip install -e ".[kaggle]"` done:

Run: `hipe kaggle-run configs/xlmr.yaml`
Expected: prints a `kernel:` id, pushes the dataset + kernel, polls (the full multilingual XLM-R fine-tune runs on Kaggle GPU — minutes), then prints `ingested runs: [...]`. Then `hipe leaderboard` shows the Kaggle-trained `xlmr` row on the newspapers-gold dev — directly comparable to the LLM baseline's 0.65 / 0.724. If the kernel errors, inspect it in the Kaggle GUI (the run is a browsable Notebook Version) and iterate. Record the resulting leaderboard numbers.

---

## Self-Review

**Spec coverage (this plan = §5.8 Kaggle bridge):**
- `export_job` packages code + config for a GPU notebook (§5.8) → Tasks 2, 3.
- `kernel-metadata.json` with `enable_gpu`/`enable_internet`, headless commit run via `kaggle kernels push` (§5.8) → Tasks 1, 4.
- Poll `kaggle kernels status`, download via `kaggle kernels output` (§5.8) → Tasks 4, 6.
- `ingest` drops outputs into a normal `runs/` folder so a Kaggle-trained model is indistinguishable from a local run (§5.8) → Tasks 5, 6.
- Data cloned from the pinned HIPE repo on Kaggle; runs land via `HIPE_RUNS_DIR` (no harness change) → Task 2.
- Fully automated via the Kaggle API (user choice) → Task 6 (`hipe kaggle-run`), with the live run as the documented manual step (Task 6 Step 8) since it needs the user's account.
- *Out of scope (correctly):* the QLoRA-on-Kaggle adapter path and the wheel-packaging smoke test of the vendored scorer (separate later plans); the model code (`xlmr`) is unchanged and already merged.

**Placeholder scan:** none — every code/step block is concrete. Task 6 Step 8 is an explicit manual live run, framed as such (the only step requiring Kaggle credentials).

**Type consistency:** `kaggle_username`, `dataset_metadata`, `kernel_slug`, `kernel_metadata` (Task 1) consumed by `export.stage_job` (Task 3). `render_kernel_script(config_rel, dataset_slug)` (Task 2) consumed by `stage_job` (Task 3). `stage_job(config_path, staging_dir, repo_root) -> {"code","kernel","kernel_id"}` (Task 3) consumed by `run_kaggle` (Task 6). `push_dataset`/`push_kernel`/`kernel_status`/`download_output` (Task 4) consumed by `run_kaggle` (Task 6). `ingest_output(download_dir, runs_root) -> list` (Task 5) consumed by `run_kaggle` (Task 6). `LEADERBOARD_COLUMNS` reused from the existing registry. The `_run` seam is the single mock point across all subprocess tests.
