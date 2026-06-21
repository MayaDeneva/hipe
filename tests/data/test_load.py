from pathlib import Path
from hipe.data.load import read_jsonl, load_documents

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini.jsonl"


def test_read_jsonl_skips_blank_lines():
    rows = read_jsonl(FIX)
    assert len(rows) == 2
    assert rows[0]["document_id"] == "d1"


def test_load_documents_maps_fields():
    docs = load_documents(FIX)
    assert [d.doc_id for d in docs] == ["d1", "d2"]
    assert docs[0].language == "en"
    assert docs[0].pub_date == "1820-01-10"
    assert docs[1].pub_date == "1850"
    assert docs[0].media["publication_title"] == "Gazette"
