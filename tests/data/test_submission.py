# tests/data/test_submission.py
from pathlib import Path
from hipe.data.load import read_jsonl
from hipe.data.submission import write_submission

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini.jsonl"


def test_write_submission_roundtrips_and_sets_labels(tmp_path):
    preds = {
        ("d1", "d1-joed1-essex"): {"at": "PROBABLE", "isAt": "FALSE"},
    }
    out = tmp_path / "sub.jsonl"
    write_submission(FIX, preds, out)
    rows = read_jsonl(out)
    # structure preserved
    assert [r["document_id"] for r in rows] == ["d1", "d2"]
    d1 = rows[0]["sampled_pairs"]
    first = [p for p in d1 if p["pers_entity_id"] == "d1-joe"
             and p["loc_entity_id"] == "d1-essex"][0]
    assert first["at"] == "PROBABLE" and first["isAt"] == "FALSE"
    # pair not in preds -> defaulted to FALSE/FALSE
    second = [p for p in d1 if p["loc_entity_id"] == "d1-rapp"][0]
    assert second["at"] == "FALSE" and second["isAt"] == "FALSE"
