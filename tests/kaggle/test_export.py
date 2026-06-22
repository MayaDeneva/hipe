import json
from pathlib import Path
from hipe.kaggle.export import stage_job


def test_stage_job_builds_kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    staging = tmp_path / "stage"
    job = stage_job("configs/xlmr.yaml", staging, "https://github.com/u/r.git")

    kernel = Path(job["kernel"])
    assert (kernel / "run_kernel.py").exists()
    script_text = (kernel / "run_kernel.py").read_text()
    assert "https://github.com/u/r.git" in script_text
    assert "configs/xlmr.yaml" in script_text
    kmeta = json.loads((kernel / "kernel-metadata.json").read_text())
    assert kmeta["enable_gpu"] is True
    assert kmeta["enable_internet"] is True
    assert job["kernel_id"] == "alice/hipe-run-xlmr"
