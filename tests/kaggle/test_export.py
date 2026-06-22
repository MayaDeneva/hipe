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
