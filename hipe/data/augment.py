# hipe/data/augment.py
"""Label-preserving OCR-noise augmentation. Corrupts the context (simulating
different OCR / sources) so the encoder learns meaning over impresso surface
cues, but PROTECTS the entity mention spans so fuzzy entity-marking still works
and the (person, place) relation labels are unchanged. Experiment #2 (robustness)."""
import random
from dataclasses import replace
from hipe.features.markers import _locate

# common OCR confusions (historical newspaper digitization)
_SUB = {"e": "c", "c": "e", "o": "0", "0": "o", "l": "i", "i": "l", "n": "m",
        "m": "n", "u": "v", "v": "u", "s": "f", "rn": "m", "a": "o", "t": "f"}


def _ocr_noise(text, protected, prob, rng):
    out, i, n = [], 0, len(text)
    while i < n:
        if i in protected or text[i].isspace():
            out.append(text[i]); i += 1; continue
        if rng.random() < prob:
            r = rng.random()
            if r < 0.55:                                   # substitute
                out.append(_SUB.get(text[i].lower(), text[i]))
            elif r < 0.75:                                 # delete
                pass
            else:                                          # duplicate
                out.append(text[i]); out.append(text[i])
        else:
            out.append(text[i])
        i += 1
    return "".join(out)


def _protected_spans(pair):
    spans = set()
    for ms in (pair.person.mentions, pair.place.mentions):
        sp = _locate(pair.context, ms)                     # single (start, end) or None
        if sp is not None:
            s, e = sp
            spans.update(range(max(0, s - 2), e + 2))      # +slack so fuzzy_find still hits
    return spans


def augment_pairs(pairs, n_aug=1, prob=0.06, seed=0):
    """Return n_aug OCR-noised copies per pair (entities protected, labels kept)."""
    rng = random.Random(seed)
    out = []
    for p in pairs:
        prot = _protected_spans(p)
        for _ in range(n_aug):
            out.append(replace(p, context=_ocr_noise(p.context, prot, prob, rng)))
    return out
