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
