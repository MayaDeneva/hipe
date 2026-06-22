# hipe/kaggle/kernel.py
DATA_REPO = "https://github.com/hipe-eval/HIPE-2026-data.git"
PINNED_COMMIT = "4228562"

_TEMPLATE = '''import os
import subprocess
import sys

WORK = "/kaggle/working"
CODE = WORK + "/code"
os.chdir(WORK)

# 1. clone the project repo (internet enabled)
subprocess.run(["git", "clone", "{repo_url}", CODE], check=True)

# 2. install the package with the ml extra
subprocess.run([sys.executable, "-m", "pip", "install", CODE + "[ml]"], check=True)

# 3. clone the pinned HIPE data
subprocess.run(["git", "clone", "{data_repo}", "data/raw/HIPE-2026-data"], check=True)
subprocess.run(["git", "-C", "data/raw/HIPE-2026-data", "checkout", "{pinned}"], check=True)

# 4. run the harness; outputs (run folder + leaderboard) land in /kaggle/working/runs
os.environ["HIPE_RUNS_DIR"] = WORK + "/runs"
subprocess.run([sys.executable, "-m", "hipe.cli", "run", CODE + "/{config_rel}"], check=True)
'''


def render_kernel_script(repo_url: str, config_rel: str) -> str:
    return _TEMPLATE.format(repo_url=repo_url, data_repo=DATA_REPO,
                            pinned=PINNED_COMMIT, config_rel=config_rel)
