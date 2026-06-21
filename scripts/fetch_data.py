"""Clone (or update) the pinned HIPE-2026-data repo into data/raw/.

Usage: python scripts/fetch_data.py
"""
import subprocess
import sys
from pathlib import Path

DATA_REPO_URL = "https://github.com/hipe-eval/HIPE-2026-data.git"
PINNED_COMMIT = "4228562"   # pin for reproducibility; bump deliberately
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "HIPE-2026-data"


def clone_command(dest: Path) -> list[str]:
    return ["git", "clone", DATA_REPO_URL, str(dest)]


def is_present(dest: Path) -> bool:
    return (dest / ".git").is_dir()


def main(dest: Path = DEST) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_present(dest):
        print(f"Updating {dest}")
        subprocess.run(["git", "-C", str(dest), "fetch", "--all"], check=True)
    else:
        print(f"Cloning into {dest}")
        subprocess.run(clone_command(dest), check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", PINNED_COMMIT], check=True)
    print("Data ready at", dest)


if __name__ == "__main__":
    main()
    sys.exit(0)
