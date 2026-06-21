# hipe/models/baselines.py
import random
from collections import Counter
from hipe import config
from hipe.models.base import RelationModel
from hipe.models import registry


@registry.register("majority")
class MajorityModel(RelationModel):
    name = "majority"

    def __init__(self):
        self._at = "FALSE"
        self._isat = "FALSE"

    def fit(self, train, dev=None):
        if train:
            self._at = Counter(p.gold_at for p in train).most_common(1)[0][0]
            self._isat = Counter(p.gold_isat for p in train).most_common(1)[0][0]

    def predict(self, pairs):
        return [{"at": self._at, "isAt": self._isat} for _ in pairs]


@registry.register("random")
class RandomModel(RelationModel):
    name = "random"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, train, dev=None):
        pass

    def predict(self, pairs):
        rng = random.Random(self.seed)
        return [{"at": rng.choice(config.AT_LABELS),
                 "isAt": rng.choice(config.ISAT_LABELS)} for _ in pairs]
