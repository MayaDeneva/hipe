# hipe/models/lookup.py
from hipe import config as cfg
from hipe.data.load import read_jsonl
from hipe.data.pairs import pair_key
from hipe.models.base import RelationModel
from hipe.models import registry


@registry.register("llm_lookup")
class LookupModel(RelationModel):
    """Predict labels from a fixed label-source jsonl (e.g. the sandbox LLM
    annotations), matched by (doc_id, pers_entity_id+loc_entity_id); FALSE/FALSE
    for pairs absent from the source. Ignores `train` — it is a fixed lookup,
    used to score an external system (the competition's LLM baseline)."""
    name = "llm_lookup"

    def __init__(self, label_source):
        self.label_source = ([label_source] if isinstance(label_source, str)
                             else list(label_source))
        self.lookup = {}

    def fit(self, train, dev=None):
        self.lookup = {}
        for path in self.label_source:
            for d in read_jsonl(path):
                doc_id = str(d["document_id"])
                for sp in d.get("sampled_pairs", []):
                    key = (doc_id, f"{sp['pers_entity_id']}{sp['loc_entity_id']}")
                    self.lookup[key] = {
                        "at": cfg.norm_label(sp.get("at"), "at"),
                        "isAt": cfg.norm_label(sp.get("isAt"), "isAt"),
                    }

    def predict(self, pairs):
        out = []
        for p in pairs:
            v = self.lookup.get((p.doc_id, pair_key(p)),
                                {"at": "FALSE", "isAt": "FALSE"})
            out.append({"at": v["at"], "isAt": v["isAt"],
                        "at_proba": None, "isAt_proba": None})
        return out
