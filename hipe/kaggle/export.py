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
