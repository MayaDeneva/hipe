# tests/test_config.py
from hipe import config


def test_label_spaces():
    assert config.AT_LABELS == ["FALSE", "PROBABLE", "TRUE"]
    assert config.ISAT_LABELS == ["FALSE", "TRUE"]


def test_norm_label_null_to_false():
    assert config.norm_label(None, "at") == "FALSE"
    assert config.norm_label(None, "isAt") == "FALSE"


def test_norm_label_uppercases_and_validates():
    assert config.norm_label("true", "at") == "TRUE"
    assert config.norm_label("Probable", "at") == "PROBABLE"
    # PROBABLE is not valid for isAt -> coerced to FALSE
    assert config.norm_label("PROBABLE", "isAt") == "FALSE"
    assert config.norm_label("garbage", "at") == "FALSE"
