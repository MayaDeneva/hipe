# tests/kaggle/test_run_kaggle.py
from hipe.kaggle import bridge


def test_run_kaggle_full_flow(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(bridge.export, "stage_job",
                        lambda c, s, r: {"code": "C", "kernel": "K",
                                         "kernel_id": "alice/hipe-run-xlmr"})
    monkeypatch.setattr(bridge, "push_dataset", lambda d: events.append(("ds", d)))
    monkeypatch.setattr(bridge, "push_kernel", lambda d: events.append(("kn", d)))
    statuses = iter(["running", "running", "complete"])
    monkeypatch.setattr(bridge, "kernel_status", lambda kid: next(statuses))
    monkeypatch.setattr(bridge, "download_output",
                        lambda kid, dest: events.append(("dl", kid)))
    monkeypatch.setattr(bridge.ingest, "ingest_output",
                        lambda dl, rr: ["2026-06-22_120000_xlmr_abcd1234"])

    result = bridge.run_kaggle("configs/xlmr.yaml", repo_root=tmp_path,
                               runs_root=tmp_path / "runs",
                               staging_dir=tmp_path / "stage",
                               poll_interval=0, sleep=lambda s: None)
    assert result["kernel_id"] == "alice/hipe-run-xlmr"
    assert result["runs"] == ["2026-06-22_120000_xlmr_abcd1234"]
    assert ("ds", "C") in events and ("kn", "K") in events
    assert ("dl", "alice/hipe-run-xlmr") in events


def test_run_kaggle_raises_on_kernel_error(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(bridge.export, "stage_job",
                        lambda c, s, r: {"code": "C", "kernel": "K",
                                         "kernel_id": "alice/x"})
    monkeypatch.setattr(bridge, "push_dataset", lambda d: None)
    monkeypatch.setattr(bridge, "push_kernel", lambda d: None)
    monkeypatch.setattr(bridge, "kernel_status", lambda kid: "error")
    with pytest.raises(RuntimeError):
        bridge.run_kaggle("configs/xlmr.yaml", repo_root=tmp_path,
                          runs_root=tmp_path / "runs",
                          staging_dir=tmp_path / "stage",
                          poll_interval=0, sleep=lambda s: None)
