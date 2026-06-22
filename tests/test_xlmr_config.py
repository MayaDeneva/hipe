# tests/test_xlmr_config.py
from pathlib import Path
import yaml

CFG = Path(__file__).resolve().parents[1] / "configs"


def test_xlmr_sanity_config_valid():
    c = yaml.safe_load((CFG / "xlmr_sanity.yaml").read_text())
    assert c["model"]["name"] == "xlmr"
    assert c["model"]["model_name"] == "xlm-roberta-base"
    assert c["model"]["max_train"] == 64        # tiny for local CPU
    assert all("sandbox" in p for p in c["data"]["train"])


def test_xlmr_full_config_valid():
    c = yaml.safe_load((CFG / "xlmr.yaml").read_text())
    assert c["model"]["name"] == "xlmr"
    assert all("sandbox" in p for p in c["data"]["train"])
    assert all("newspapers" in p for p in c["data"]["dev"])
    assert c["model"]["epochs"] == 8
