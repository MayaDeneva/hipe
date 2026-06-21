from hipe.data.schema import Entity, Pair
from hipe.data.split import split_by_document


def _pair(doc_id):
    return Pair(doc_id=doc_id, person=Entity("p", "person", ["x"]),
                place=Entity("l", "place", ["y"]), context="", language="en",
                pub_date=None)


def test_no_document_leakage():
    pairs = [_pair(f"d{i}") for i in range(10) for _ in range(3)]
    train, dev = split_by_document(pairs, dev_frac=0.2, seed=0)
    train_docs = {p.doc_id for p in train}
    dev_docs = {p.doc_id for p in dev}
    assert train_docs.isdisjoint(dev_docs)
    assert len(dev_docs) == 2          # 20% of 10 docs


def test_split_is_deterministic():
    pairs = [_pair(f"d{i}") for i in range(10)]
    a = split_by_document(pairs, seed=0)
    b = split_by_document(pairs, seed=0)
    assert [p.doc_id for p in a[1]] == [p.doc_id for p in b[1]]
