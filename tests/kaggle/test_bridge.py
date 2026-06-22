import subprocess
from hipe.kaggle import bridge


class _Rec:
    def __init__(self, stdout="", fail=False):
        self.calls = []
        self.stdout = stdout
        self.fail = fail

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if self.fail:
            raise subprocess.CalledProcessError(1, cmd)
        return type("R", (), {"stdout": self.stdout, "returncode": 0})()


def test_push_kernel_command(monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(bridge, "_run", rec)
    bridge.push_kernel("/tmp/k")
    assert rec.calls[0] == ["kaggle", "kernels", "push", "-p", "/tmp/k"]


def test_push_dataset_versions_then_creates_on_failure(monkeypatch):
    # version fails (dataset absent) -> falls back to create
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        if cmd[1:3] == ["datasets", "version"]:
            raise subprocess.CalledProcessError(1, cmd)
        return type("R", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(bridge, "_run", fake)
    bridge.push_dataset("/tmp/code")
    assert calls[0][1:3] == ["datasets", "version"]
    assert calls[1][1:3] == ["datasets", "create"]


def test_kernel_status_parses(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='status "complete"'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "complete"
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='has error'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "error"
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='still running'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "running"
