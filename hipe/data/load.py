import json
from pathlib import Path
from hipe.data.schema import Document


def read_jsonl(path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_documents(path) -> list[Document]:
    docs = []
    for raw in read_jsonl(path):
        docs.append(Document(
            doc_id=str(raw["document_id"]),
            text=raw.get("text", ""),
            language=raw.get("language", ""),
            pub_date=raw.get("date"),
            media=raw.get("media", {}) or {},
            source=raw.get("source", ""),
        ))
    return docs
