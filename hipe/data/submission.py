# hipe/data/submission.py
import json
from pathlib import Path
from hipe.data.load import read_jsonl


def write_submission(src_path, predictions: dict, out_path) -> None:
    """Round-trip src docs, overwriting only at/isAt from `predictions`.

    predictions: {(doc_id, pers_entity_id+loc_entity_id): {"at":..., "isAt":...}}
    """
    rows = read_jsonl(src_path)
    for row in rows:
        doc_id = str(row["document_id"])
        for sp in row.get("sampled_pairs", []):
            key = f"{sp['pers_entity_id']}{sp['loc_entity_id']}"
            pred = predictions.get((doc_id, key), {"at": "FALSE", "isAt": "FALSE"})
            sp["at"] = pred["at"]
            sp["isAt"] = pred["isAt"]
    with Path(out_path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
