# hipe/models/base.py
from abc import ABC, abstractmethod


class RelationModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train, dev=None) -> None:
        ...

    @abstractmethod
    def predict(self, pairs) -> list[dict]:
        """Return one dict per pair with keys 'at' and 'isAt'."""
        ...


def apply_consistency(pred: dict) -> dict:
    """Enforce isAt == TRUE  ==>  at = TRUE."""
    if pred.get("isAt") == "TRUE":
        pred["at"] = "TRUE"
    return pred
