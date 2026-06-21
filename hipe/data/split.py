import random


def split_by_document(pairs, dev_frac=0.2, seed=0):
    docs = sorted({p.doc_id for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_dev = max(1, int(len(docs) * dev_frac))
    dev_docs = set(docs[:n_dev])
    train = [p for p in pairs if p.doc_id not in dev_docs]
    dev = [p for p in pairs if p.doc_id in dev_docs]
    return train, dev
