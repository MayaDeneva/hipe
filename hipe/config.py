# hipe/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root
DATA_RAW = ROOT / "data" / "raw"                    # cloned HIPE-2026-data repo
CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"

# Label spaces (uppercase strings, per the official schema). null -> FALSE.
AT_LABELS = ["FALSE", "PROBABLE", "TRUE"]
ISAT_LABELS = ["FALSE", "TRUE"]


def norm_label(value, relation: str) -> str:
    """null/None -> FALSE; coerce to the relation's allowed set."""
    if relation not in ("at", "isAt"):
        raise ValueError(f"Unknown relation {relation!r}; expected 'at' or 'isAt'")
    allowed = AT_LABELS if relation == "at" else ISAT_LABELS
    if value is None:
        return "FALSE"
    v = str(value).upper()
    return v if v in allowed else "FALSE"
