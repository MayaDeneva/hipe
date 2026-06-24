# HIPE-2026 frontier eval — runs inside a Kaggle "Community Benchmarks" notebook
# (created from https://www.kaggle.com/benchmarks/tasks/new, which REQUIRES a
# @kbench.task). It defines a task, evaluates a model over the 401 in-domain test
# prompts, and EXTRACTS the raw at/isAt per pair into hipe_preds.json for our local
# transformer ensemble (scripts/router_from_json.py).
#
# SETUP:
#   1. Start a notebook from https://www.kaggle.com/benchmarks/tasks/new
#      (this pre-installs kaggle_benchmarks + wires the Model Proxy).
#   2. Add Data -> your `hipe-prompts` dataset (mounts at /kaggle/input/hipe-prompts/).
#   3. Paste this file, Run All.
#   4. Download /kaggle/working/hipe_preds.json and send it back.
import json, glob, os, time
import pandas as pd
import kaggle_benchmarks as kbench
from dataclasses import dataclass

THROTTLE = 0.8   # proactive pause/call: spaces requests so the proxy 429s less -> fewer slow retries

# reachable models (gemini-3-flash was 503 last we checked):
#   anthropic/claude-haiku-4-5@20251001  openai/gpt-oss-120b
#   deepseek-ai/deepseek-v3.2            qwen/qwen3-next-80b-a3b-instruct   zai/glm-5
MODEL = "anthropic/claude-haiku-4-5@20251001"
OUT = "/kaggle/working/hipe_preds.json"


@dataclass
class Pred:
    at: str
    isAt: str


# which test set: "official" (638 impresso), "surprise" (480 surprise-fr), "indomain" (401)
WHICH = "surprise"
_EXPECT = {"official": 638, "surprise": 480, "indomain": 401}
_FILE = {"official": "hipe_prompts_official.json", "surprise": "hipe_prompts_surprise.json",
         "indomain": "hipe_prompts.json"}


def find_prompts():
    """Locate the prompts file for the chosen set, then any prompts json."""
    pats = [f"/kaggle/input/**/{_FILE[WHICH]}", f"./{_FILE[WHICH]}",
            "/kaggle/input/**/hipe_prompts*.json", "/kaggle/input/**/*.json"]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "prompts json not found. Mounts under /kaggle/input/: "
        + str(os.listdir("/kaggle/input")) + ". Attach the dataset that contains it.")


INPUT = find_prompts()
rows = json.load(open(INPUT))
print(f"using prompts file: {INPUT}  ({len(rows)} pairs, WHICH={WHICH})", flush=True)
assert len(rows) == _EXPECT[WHICH], \
    f"expected {_EXPECT[WHICH]} {WHICH} pairs, got {len(rows)} — wrong dataset attached?"
DATA = pd.DataFrame(rows)            # columns: key, prompt, gold_at, gold_isAt
_done = [0]


@kbench.task(name="hipe-person-place-at")
def hipe_at(llm, prompt: str, gold_at: str, gold_isAt: str, key: str) -> dict:
    """Predict at (ever there: TRUE/PROBABLE/FALSE) and isAt (present now: TRUE/FALSE)
    for a historical person-place pair. Returns the labels; we score macro-recall +
    ensemble locally, so no assertion is needed here."""
    time.sleep(THROTTLE)
    p = kbench.llm.prompt(prompt, schema=Pred)
    _done[0] += 1
    if _done[0] % 10 == 0:
        print(f"  ...{_done[0]} pairs done", flush=True)
    return {"at": str(getattr(p, "at", "FALSE")).upper(),
            "isAt": str(getattr(p, "isAt", "FALSE")).upper()}


print(f"=== {MODEL} | {len(DATA)} pairs ===", flush=True)
runs = hipe_at.evaluate(
    llm=[kbench.llms[MODEL]],
    evaluation_data=DATA,
    on_failure="continue",
    max_attempts=3,
)

preds = {}
for r in runs.completed_runs:
    preds[r.params["key"]] = r.result or {"at": "FALSE", "isAt": "FALSE"}
json.dump(preds, open(OUT, "w"))
print(f"DONE: {len(preds)} predictions, {len(runs.errored_runs)} errored -> {OUT}", flush=True)
