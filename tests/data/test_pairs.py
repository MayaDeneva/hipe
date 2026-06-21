from pathlib import Path
from hipe.data.pairs import load_pairs, unique_entities, pair_key, context_for

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "mini.jsonl"


def test_load_pairs_count_and_labels():
    pairs = load_pairs(FIX)
    assert len(pairs) == 3                      # 2 from d1, 1 from d2
    p0 = pairs[0]
    assert p0.gold_at == "TRUE"
    assert p0.gold_isat == "TRUE"
    # d2 had null labels -> normalized to FALSE
    d2 = [p for p in pairs if p.doc_id == "d2"][0]
    assert d2.gold_at == "FALSE"
    assert d2.gold_isat == "FALSE"


def test_context_window_is_around_mention():
    pairs = load_pairs(FIX)
    p0 = pairs[0]
    assert "Essex" in p0.context
    assert "Joe" in p0.context


def test_pair_key_matches_official_concatenation():
    pairs = load_pairs(FIX)
    assert pair_key(pairs[0]) == "d1-joed1-essex"


def test_unique_entities_dedupes():
    pairs = load_pairs(FIX)
    ents = unique_entities(pairs)
    # d1-joe appears in two pairs but is one entity
    assert "d1-joe" in ents
    assert ents["d1-joe"].etype == "person"
