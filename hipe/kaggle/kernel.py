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

# 2. install a torch build that supports the assigned GPU's compute capability.
#    Kaggle's default torch (and the latest PyPI torch) dropped Pascal sm_60, so a
#    P100 errors with "no kernel image". The cu121 wheels for torch 2.4.1 cover
#    sm_60..sm_90 (P100/T4/V100/A100). Pin it so the package install keeps it.
#    Install the MATCHED torch/torchvision/torchaudio trio so their ABIs agree
#    (a lone torch downgrade leaves Kaggle's newer torchvision -> "nms does not exist").
subprocess.run([sys.executable, "-m", "pip", "install",
                "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
con = WORK + "/constraints.txt"
with open(con, "w") as f:
    f.write("torch==2.4.1\\ntorchvision==0.19.1\\ntorchaudio==2.4.1\\n")

# 3. install the package with the ml extra, keeping the torch we just installed
subprocess.run([sys.executable, "-m", "pip", "install", "-c", con, CODE + "[ml]"], check=True)

# 4. clone the pinned HIPE data
subprocess.run(["git", "clone", "{data_repo}", "data/raw/HIPE-2026-data"], check=True)
subprocess.run(["git", "-C", "data/raw/HIPE-2026-data", "checkout", "{pinned}"], check=True)

# 5. run the harness; outputs (run folder + leaderboard) land in /kaggle/working/runs
os.environ["HIPE_RUNS_DIR"] = WORK + "/runs"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # Kaggle renders tqdm badly
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
subprocess.run([sys.executable, "-m", "hipe.cli", "run", CODE + "/{config_rel}"], check=True)
'''


def render_kernel_script(repo_url: str, config_rel: str) -> str:
    return _TEMPLATE.format(repo_url=repo_url, data_repo=DATA_REPO,
                            pinned=PINNED_COMMIT, config_rel=config_rel)
