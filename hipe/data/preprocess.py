# hipe/data/preprocess.py
import re
import unicodedata
from rapidfuzz import fuzz

_SOFT_HYPHEN = "­"


def normalize_text(text: str) -> str:
    """Conservative OCR-aware normalization. Does not rewrite words."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_SOFT_HYPHEN, "")
    text = text.replace("ſ", "s")          # long s
    # de-hyphenate line-break splits: "exam-\nple" -> "example"
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)            # collapse whitespace/newlines
    return text.strip()


def fuzzy_find(text: str, mention: str, min_score: float = 75.0):
    """Return (start, end) of the best match for `mention` in `text`, or None."""
    if not text or not mention:
        return None
    idx = text.find(mention)
    if idx >= 0:
        return (idx, idx + len(mention))
    m = len(mention)
    ml = mention.lower()
    best_span = None
    best_score = -1
    for i in range(0, max(1, len(text) - m + 1)):
        window = text[i:i + m]
        score = fuzz.ratio(window.lower(), ml)
        if score > best_score:
            best_score = score
            best_span = (i, i + m)
    return best_span if best_score >= min_score else None
