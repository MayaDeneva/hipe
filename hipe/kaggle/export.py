# hipe/kaggle/export.py
import json
from pathlib import Path
from hipe.kaggle import metadata
from hipe.kaggle.kernel import render_kernel_script


def stage_job(config_path, staging_dir, repo_url) -> dict:
    staging = Path(staging_dir)
    kernel = staging / "kernel"
    kernel.mkdir(parents=True, exist_ok=True)

    user = metadata.kaggle_username()
    k_slug = metadata.kernel_slug(config_path)
    code_file = "run_kernel.py"
    config_rel = f"configs/{Path(config_path).name}"

    (kernel / code_file).write_text(
        render_kernel_script(repo_url, config_rel), encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(metadata.kernel_metadata(user, k_slug, code_file), indent=2),
        encoding="utf-8")
    return {"kernel": str(kernel), "kernel_id": f"{user}/{k_slug}"}
