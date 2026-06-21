# tests/data/test_preprocess.py
from hipe.data.preprocess import normalize_text, fuzzy_find


def test_normalize_dehyphenates_linebreaks():
    assert normalize_text("exam-\nple") == "example"


def test_normalize_long_s_and_whitespace():
    assert normalize_text("ſtreet   of\n\nParis") == "street of Paris"


def test_fuzzy_find_exact():
    text = "Joe was at Essex county."
    assert fuzzy_find(text, "Essex") == (11, 16)


def test_fuzzy_find_handles_ocr_noise():
    text = "committed to the /ail at Essex"
    span = fuzzy_find(text, "jail")
    assert span is not None
    assert text[span[0]:span[1]].lower() in ("/ail", "jail")


def test_fuzzy_find_returns_none_below_threshold():
    assert fuzzy_find("totally unrelated words", "Rappahannock") is None
