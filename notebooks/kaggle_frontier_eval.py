# HIPE-2026 frontier-LLM eval via Kaggle Community Benchmarks
# ----------------------------------------------------------------------
# Runs a frontier model over the 401 in-domain test prompts (baked locally
# with few-shot + KB known-places + date) and writes per-pair at/isAt.
#
# SETUP (once):
#   1. Kaggle -> Datasets -> New Dataset -> upload `hipe_prompts.json`
#      (name it so it mounts at /kaggle/input/hipe-prompts/).
#   2. New Notebook -> Add Data -> your hipe-prompts dataset.
#   3. Enable the Community Benchmarks / Model Proxy add-on for the notebook
#      (so kaggle_benchmarks can reach the hosted models).
#   4. Paste this file into a cell and Run All.
#   5. Download /kaggle/working/hipe_preds.json and send it back.
#
# Resumable: re-running continues from whatever is already in hipe_preds.json.
import json, os, time
import kaggle_benchmarks as kbench
from dataclasses import make_dataclass

# pick a model that is reachable (gemini-3-flash was 503 at our last check):
#   anthropic/claude-haiku-4-5@20251001   deepseek-ai/deepseek-v3.2
#   openai/gpt-oss-120b                   qwen/qwen3-next-80b-a3b-instruct   zai/glm-5
MODEL = "anthropic/claude-haiku-4-5@20251001"
INPUT = "/kaggle/input/hipe-prompts/hipe_prompts.json"
OUT = "/kaggle/working/hipe_preds.json"

try:
    kbench.config.disable_console_mode()
    kbench.config.disable_tqdm()
except Exception:
    pass

rows = json.load(open(INPUT))
P = make_dataclass("P", [("at", str), ("isAt", str)])
llm = kbench.llms[MODEL]
out = json.load(open(OUT)) if os.path.exists(OUT) else {}

for i, r in enumerate(rows):
    if r["key"] in out:
        continue
    for attempt in range(4):
        try:
            p = llm.prompt(r["prompt"], schema=P)
            out[r["key"]] = {"at": str(p.at), "isAt": str(p.isAt)}
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))   # backoff for transient 429/503
    else:
        out[r["key"]] = {"at": "FALSE", "isAt": "FALSE"}
    if i % 25 == 0:
        json.dump(out, open(OUT, "w"))
        print(f"{i}/{len(rows)}", flush=True)

json.dump(out, open(OUT, "w"))
print("DONE", len(out))

# quick sanity: raw at-accuracy vs gold (the macro-recall ensemble is computed locally)
def nrm(s):
    s = str(s).upper()
    return s if s in ("TRUE", "PROBABLE", "FALSE") else ("TRUE" if "TRUE" in s else
           "PROBABLE" if "PROB" in s else "FALSE")
acc = sum(1 for r in rows if nrm(out.get(r["key"], {}).get("at", "")) == r["gold_at"])
print(f"raw at-accuracy: {acc}/{len(rows)} = {acc/len(rows):.3f}")
