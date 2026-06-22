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


def apply_consistency(pred: dict, mode: str = "soft") -> dict:
    """Reconcile at/isAt per the relationship observed in gold.

    Gold fact: isAt == "TRUE" never co-occurs with at == "FALSE"
    (at is TRUE or PROBABLE). Modes:
      "soft" (default): if isAt == TRUE and at == FALSE, set at = TRUE
                        (PROBABLE is left intact).
      "hard": if isAt == TRUE, force at = TRUE.
      "off":  no change.
    Mutates and returns the same dict.
    """
    if mode == "off":
        return pred
    if mode not in ("soft", "hard"):
        raise ValueError(f"Unknown consistency mode {mode!r}; expected 'soft', 'hard', or 'off'")
    if pred.get("isAt") == "TRUE":
        if mode == "hard":
            pred["at"] = "TRUE"
        elif pred.get("at") == "FALSE":   # soft
            pred["at"] = "TRUE"
    return pred
