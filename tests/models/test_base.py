# tests/models/test_base.py
import pytest
from hipe.models.base import RelationModel, apply_consistency
from hipe.models import registry


def test_apply_consistency_forces_at_true():
    assert apply_consistency({"at": "FALSE", "isAt": "TRUE"})["at"] == "TRUE"
    assert apply_consistency({"at": "PROBABLE", "isAt": "FALSE"})["at"] == "PROBABLE"


def test_registry_roundtrip():
    @registry.register("dummy_test_model")
    class Dummy(RelationModel):
        name = "dummy_test_model"
        def fit(self, train, dev=None):
            pass
        def predict(self, pairs):
            return [{"at": "FALSE", "isAt": "FALSE"} for _ in pairs]

    m = registry.get_model("dummy_test_model")
    assert isinstance(m, RelationModel)
    assert m.predict([1, 2]) == [{"at": "FALSE", "isAt": "FALSE"}] * 2


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        registry.get_model("does_not_exist")
