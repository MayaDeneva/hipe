# tests/kaggle/test_metadata.py
from hipe.kaggle import metadata as md


def test_kaggle_username_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    assert md.kaggle_username() == "alice"


def test_dataset_metadata_id(monkeypatch):
    d = md.dataset_metadata("alice")
    assert d["id"] == "alice/hipe-code"
    assert d["title"] == "hipe-code"
    assert d["licenses"][0]["name"]


def test_kernel_slug_from_config_path():
    assert md.kernel_slug("configs/xlmr.yaml") == "hipe-run-xlmr"
    assert md.kernel_slug("configs/embedding_svm.yaml") == "hipe-run-embedding-svm"


def test_kernel_metadata_enables_gpu_and_internet():
    m = md.kernel_metadata("alice", "hipe-run-xlmr", "run_kernel.py")
    assert m["id"] == "alice/hipe-run-xlmr"
    assert m["enable_gpu"] is True
    assert m["enable_internet"] is True
    assert m["kernel_type"] == "script"
    assert m["is_private"] is True
    assert m["dataset_sources"] == ["alice/hipe-code"]
    assert m["code_file"] == "run_kernel.py"
