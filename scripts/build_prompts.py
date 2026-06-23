"""Bake the full per-pair prompts (system + gold few-shot + KB known-places + date
+ context) for the 401 in-domain test pairs into hipe_prompts.json, so a Kaggle
notebook can run any frontier model over them with no HIPE code/data on Kaggle."""
import os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from hipe.harness import _load_pairs_spec
from hipe.data.split import split_by_document
from hipe.data.pairs import pair_key
from hipe.models.llm import LLMModel

NW = [f"data/raw/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-{l}.jsonl"
      for l in ("en", "de", "fr")]
dev = _load_pairs_spec(NW)
din, test = split_by_document(dev, dev_frac=0.3, seed=0)
for p in din:
    p.is_gold = True
m = LLMModel(backend="kbench", model="frontier", kbench_model=None,
             prompt_version="claude-lean", use_known_places=True, resolve_nil=False, n_shots=3)
m.fit(din)

rows = []
for p in test:
    msgs = m._messages(p)
    text = "\n\n".join((("ANSWER: " + x["content"]) if x["role"] == "assistant"
                        else x["content"]) for x in msgs)
    rows.append({"key": f"{p.doc_id}|{pair_key(p)}", "prompt": text,
                 "gold_at": p.gold_at, "gold_isAt": p.gold_isat})
json.dump(rows, open("hipe_prompts.json", "w"), ensure_ascii=False)
print(f"wrote hipe_prompts.json: {len(rows)} prompts; avg chars={sum(len(r['prompt']) for r in rows)//len(rows)}")
