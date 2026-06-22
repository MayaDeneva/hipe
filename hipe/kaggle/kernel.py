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

# 2. pin Kaggle's pre-installed torch (built for its GPU, e.g. P100/sm_60) so the
#    pip install below does NOT upgrade it to a build that drops older GPU support
ver = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__.split('+')[0])"],
    capture_output=True, text=True, check=True).stdout.strip()
con = WORK + "/constraints.txt"
with open(con, "w") as f:
    f.write("torch==" + ver + "\\n")

# 3. install the package with the ml extra, keeping Kaggle's torch
subprocess.run([sys.executable, "-m", "pip", "install", "-c", con, CODE + "[ml]"], check=True)

# 4. clone the pinned HIPE data
subprocess.run(["git", "clone", "{data_repo}", "data/raw/HIPE-2026-data"], check=True)
subprocess.run(["git", "-C", "data/raw/HIPE-2026-data", "checkout", "{pinned}"], check=True)

# 5. run the harness; outputs (run folder + leaderboard) land in /kaggle/working/runs
os.environ["HIPE_RUNS_DIR"] = WORK + "/runs"
subprocess.run([sys.executable, "-m", "hipe.cli", "run", CODE + "/{config_rel}"], check=True)
'''


def render_kernel_script(repo_url: str, config_rel: str) -> str:
    return _TEMPLATE.format(repo_url=repo_url, data_repo=DATA_REPO,
                            pinned=PINNED_COMMIT, config_rel=config_rel)
