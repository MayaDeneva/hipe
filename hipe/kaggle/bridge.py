import subprocess
import time as _time
from pathlib import Path

from hipe.kaggle import export, ingest


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
