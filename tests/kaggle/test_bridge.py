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


def test_repo_url(monkeypatch):
    fake_result = type("R", (), {"stdout": "https://github.com/u/r.git\n", "returncode": 0})()
    monkeypatch.setattr(bridge, "_run", lambda cmd, **kw: fake_result)
    assert bridge.repo_url("/repo") == "https://github.com/u/r.git"


def test_kernel_status_parses(monkeypatch):
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='status "complete"'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "complete"
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='has error'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "error"
    monkeypatch.setattr(bridge, "_run", _Rec(stdout='still running'))
    assert bridge.kernel_status("alice/hipe-run-xlmr") == "running"
