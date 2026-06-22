"""Build a dev set restricted to the pairs the LLM baseline actually covered.

The competition's LLM baseline = the sandbox ensemble labels. It only covers the
newspapers-gold pairs that also appear in the sandbox. To compare models against
the LLM fairly, every model must be scored on the SAME pairs the LLM predicted —
not on pairs where the LLM has no prediction (and would only get a FALSE fallback).

This writes `data/cache/newspapers_llm_covered.jsonl`: the newspapers gold docs
keeping only the sampled_pairs whose (document_id, pers_entity_id+loc_entity_id)
key is present in the sandbox label source. Docs left with no covered pair are
dropped. Gold labels are preserved verbatim.

Usage: python scripts/make_llm_covered_dev.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "HIPE-2026-data" / "data"
SANDBOX = [DATA / "sandbox" / f"{lg}-{sp}.jsonl"
           for lg in ("en", "de", "fr") for sp in ("train", "dev")]
NEWS = [DATA / "newspapers" / "v1.0" / f"HIPE-2026-v1.0-impresso-train-{lg}.jsonl"
        for lg in ("en", "de", "fr")]
OUT = ROOT / "data" / "cache" / "newspapers_llm_covered.jsonl"


def _read(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _pair_key(sp):
    return f"{sp['pers_entity_id']}{sp['loc_entity_id']}"


def main():
    covered = set()
    for path in SANDBOX:
        for d in _read(path):
            doc_id = str(d["document_id"])
            for sp in d.get("sampled_pairs", []):
                covered.add((doc_id, _pair_key(sp)))

    out_docs, kept_pairs = [], 0
    for path in NEWS:
        for d in _read(path):
            doc_id = str(d["document_id"])
            keep = [sp for sp in d.get("sampled_pairs", [])
                    if (doc_id, _pair_key(sp)) in covered]
            if keep:
                d = dict(d)
                d["sampled_pairs"] = keep
                out_docs.append(d)
                kept_pairs += len(keep)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for d in out_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"LLM-covered dev: {len(out_docs)} docs / {kept_pairs} pairs -> {OUT}")


if __name__ == "__main__":
    main()
