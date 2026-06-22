import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_ml_extra_includes_transformers_and_accelerate():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    ml = data["project"]["optional-dependencies"]["ml"]
    joined = " ".join(ml)
    assert "sentence-transformers" in joined
    assert "transformers" in joined
    assert "accelerate" in joined
