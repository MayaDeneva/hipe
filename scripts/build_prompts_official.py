"""Build hipe_prompts_official.json for the OFFICIAL impresso-test pairs, so the
same Kaggle notebook can produce frontier predictions on the real test set.

Copies the labeled reference test files from the hipe-eval clone into the repo
(so the transformer official configs can also run on them), then bakes prompts
(gold few-shot + KB known-places + date + context)."""
import os, json, shutil, glob
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REF = "/tmp/hipe-2026-eval/data/reference"
DEST = "data/raw/HIPE-2026-data/data/newspapers/v1.0"
TEST = []
for lang in ("en", "de", "fr"):
    src = f"{REF}/HIPE-2026-v1.0-impresso-test-{lang}.jsonl"
    dst = f"{DEST}/HIPE-2026-v1.0-impresso-test-{lang}.jsonl"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)          # python copy (not shell cp)
    TEST.append(dst)

from hipe.harness import _load_pairs_spec
from hipe.data.pairs import pair_key
from hipe.models.llm import LLMModel

NW = [f"{DEST}/HIPE-2026-v1.0-impresso-train-{l}.jsonl" for l in ("en", "de", "fr")]
train = _load_pairs_spec(NW)           # gold newspapers train -> few-shot pool
for p in train:
    p.is_gold = True
test = _load_pairs_spec(TEST)          # official test pairs
m = LLMModel(backend="kbench", model="frontier", kbench_model=None,
             prompt_version="claude-lean", use_known_places=True, resolve_nil=False, n_shots=3)
m.fit(train)

rows = []
for p in test:
    msgs = m._messages(p)
    text = "\n\n".join((("ANSWER: " + x["content"]) if x["role"] == "assistant"
                        else x["content"]) for x in msgs)
    rows.append({"key": f"{p.doc_id}|{pair_key(p)}", "prompt": text,
                 "gold_at": p.gold_at, "gold_isAt": p.gold_isat})
json.dump(rows, open("hipe_prompts_official.json", "w"), ensure_ascii=False)
print(f"wrote hipe_prompts_official.json: {len(rows)} official-test prompts")
