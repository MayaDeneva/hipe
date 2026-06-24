"""Build hipe_prompts_surprise.json for the SURPRISE-test-fr bonus track (480 fr
pairs). Copies the LABELED eval reference into the repo (data/inject/, which the
Kaggle kernel injects into the data dir so the transformer can predict on it),
then bakes prompts (gold few-shot + KB known-places + date + context)."""
import os, json, shutil
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REF = "/tmp/hipe-2026-eval/data/reference/HIPE-2026-v1.0-surprise-test-fr.jsonl"
INJECT = "data/inject"
os.makedirs(INJECT, exist_ok=True)
dst = f"{INJECT}/HIPE-2026-v1.0-surprise-test-fr.jsonl"
shutil.copyfile(REF, dst)                                # labeled, for the transformer kernel

from hipe.harness import _load_pairs_spec
from hipe.data.pairs import pair_key
from hipe.models.llm import LLMModel

NW = [f"data/raw/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-{l}.jsonl"
      for l in ("en", "de", "fr")]
train = _load_pairs_spec(NW)
for p in train:
    p.is_gold = True
test = _load_pairs_spec([dst])
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
json.dump(rows, open("hipe_prompts_surprise.json", "w"), ensure_ascii=False)
print(f"wrote hipe_prompts_surprise.json: {len(rows)} surprise-fr prompts; injected ref -> {dst}")
