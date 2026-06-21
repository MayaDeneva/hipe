from hipe.runs import registry


def test_config_hash_is_stable_and_order_independent():
    a = registry.config_hash({"model": {"name": "majority"}, "data": {"x": 1}})
    b = registry.config_hash({"data": {"x": 1}, "model": {"name": "majority"}})
    assert a == b
    assert len(a) == 8


def test_new_run_dir_created(tmp_path):
    d = registry.new_run_dir("majority", "abcd1234", root=tmp_path,
                             now="2026-06-21_120000")
    assert d.exists()
    assert d.name == "2026-06-21_120000_majority_abcd1234"


def test_append_leaderboard_writes_header_once(tmp_path):
    row = {"run_id": "r1", "timestamp": "t", "model": "majority",
           "config_hash": "h", "data": "f.jsonl", "at_recall": 0.5,
           "isAt_recall": 0.5, "global": 0.5, "n_dev": 10, "notes": ""}
    registry.append_leaderboard(tmp_path, row)
    registry.append_leaderboard(tmp_path, {**row, "run_id": "r2"})
    text = (tmp_path / "leaderboard.csv").read_text()
    assert text.count("run_id,timestamp") == 1     # header once
    assert "r1" in text and "r2" in text
