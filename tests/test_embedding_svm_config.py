# tests/test_embedding_svm_config.py
from pathlib import Path
import yaml

CFG = Path(__file__).resolve().parents[1] / "configs"


def test_embedding_svm_config_valid():
    c = yaml.safe_load((CFG / "embedding_svm.yaml").read_text())
    assert c["model"]["name"] == "embedding_svm"
    # Option A: train = sandbox, dev = newspapers
    assert all("sandbox" in p for p in c["data"]["train"])
    assert all("newspapers" in p for p in c["data"]["dev"])
    assert c["consistency"] == "soft"


def test_sandboxdev_config_valid():
    c = yaml.safe_load((CFG / "embedding_svm_sandboxdev.yaml").read_text())
    assert all("sandbox" in p for p in c["data"]["dev"])
