from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    entity_id: str                   # pers_entity_id / loc_entity_id
    etype: str                       # "person" | "place"
    mentions: list[str]              # surface forms (mention list)
    qid: Optional[str] = None        # pre-supplied or resolved by linking
    link_score: Optional[float] = None

    @property
    def surface(self) -> str:        # primary mention
        return self.mentions[0] if self.mentions else ""


@dataclass
class Document:
    doc_id: str
    text: str
    language: str
    pub_date: Optional[str]          # ISO date / "YYYY" / None
    media: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class Pair:
    doc_id: str
    person: Entity
    place: Entity
    context: str
    language: str
    pub_date: Optional[str]
    gold_at: str = "FALSE"           # TRUE | PROBABLE | FALSE
    gold_isat: str = "FALSE"         # TRUE | FALSE
    is_gold: bool = False            # True = human gold (newspapers); False = silver (sandbox)
    features: dict = field(default_factory=dict)
