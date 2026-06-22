DATA_REPO = "https://github.com/hipe-eval/HIPE-2026-data.git"
PINNED_COMMIT = "4228562"

_TEMPLATE = '''import os
import subprocess
import sys
import zipfile

WORK = "/kaggle/working"
DATASET = "/kaggle/input/{dataset_slug}"
CODE = WORK + "/code"
os.chdir(WORK)

# 1. extract the code bundle from the read-only input dataset
with zipfile.ZipFile(DATASET + "/bundle.zip") as z:
    z.extractall(CODE)

# 2. install the package (with the ml extra) from the extracted writable code dir
subprocess.run([sys.executable, "-m", "pip", "install", CODE + "[ml]"], check=True)

# 3. clone the pinned HIPE data into the writable working dir (internet enabled)
subprocess.run(["git", "clone", "{data_repo}", "data/raw/HIPE-2026-data"], check=True)
subprocess.run(["git", "-C", "data/raw/HIPE-2026-data", "checkout", "{pinned}"], check=True)

# 4. run the harness; outputs (run folder + leaderboard) land in /kaggle/working/runs
os.environ["HIPE_RUNS_DIR"] = WORK + "/runs"
subprocess.run([sys.executable, "-m", "hipe.cli", "run", CODE + "/{config_rel}"], check=True)
'''


def render_kernel_script(config_rel: str, dataset_slug: str = "hipe-code") -> str:
    return _TEMPLATE.format(dataset_slug=dataset_slug, data_repo=DATA_REPO,
                            pinned=PINNED_COMMIT, config_rel=config_rel)
