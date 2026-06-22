import json
import zipfile
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

    with zipfile.ZipFile(code / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for item in CODE_INCLUDE:
            src = repo_root / item
            if src.is_dir():
                for p in src.rglob("*"):
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                        z.write(p, p.relative_to(repo_root))
            elif src.exists():
                z.write(src, src.relative_to(repo_root))

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
    return {"code": str(code), "kernel": str(kernel), "kernel_id": f"{user}/{k_slug}"}
