from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.py")
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)


def test_clone_command_uses_repo_url_and_dest(tmp_path):
    dest = tmp_path / "HIPE-2026-data"
    cmd = fetch_data.clone_command(dest)
    assert cmd[0] == "git" and cmd[1] == "clone"
    assert fetch_data.DATA_REPO_URL in cmd
    assert str(dest) in cmd


def test_is_present_false_when_missing(tmp_path):
    assert fetch_data.is_present(tmp_path / "nope") is False


def test_is_present_true_when_git_dir_exists(tmp_path):
    dest = tmp_path / "HIPE-2026-data"
    (dest / ".git").mkdir(parents=True)
    assert fetch_data.is_present(dest) is True
