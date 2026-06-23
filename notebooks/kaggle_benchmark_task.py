# HIPE-2026 person-place relation as a Kaggle Community Benchmark TASK.
# ----------------------------------------------------------------------
# This is the *scored* benchmark version (vs. kaggle_frontier_eval.py, which just
# collects raw predictions for our local macro-recall ensemble). A Task is graded
# by assertions (pass/fail per pair) -> Kaggle builds a per-model leaderboard.
#
#   # one-time: push the task definition to Kaggle
#   kaggle b t push hipe-person-place-at -f notebooks/kaggle_benchmark_task.py
#
#   # run it across models (or use the task page's "Evaluate More Models" button)
#   kaggle b t run hipe-person-place-at -m anthropic/claude-haiku-4-5 \
#                                       -m openai/gpt-oss-120b -m deepseek-ai/deepseek-v3.2
#
#   kaggle b t status hipe-person-place-at         # leaderboard / per-model status
#   kaggle b t publish hipe-person-place-at        # make it public/shareable
#
# A *Benchmark* (collection of tasks) is created/managed on the Kaggle web UI;
# add this task to it there.
#
# Eval data = the baked prompts; attach the `hipe-prompts` dataset to the notebook
# so it mounts at /kaggle/input/hipe-prompts/.
import json
import pandas as pd
import kaggle_benchmarks as kbench
from dataclasses import dataclass


@dataclass
class Pred:
    at: str
    isAt: str


rows = json.load(open("/kaggle/input/hipe-prompts/hipe_prompts.json"))
DATA = pd.DataFrame(rows)   # columns: key, prompt, gold_at, gold_isAt


@kbench.task(name="hipe-person-place-at")
def hipe_at(llm, prompt: str, gold_at: str, gold_isAt: str, key: str) -> dict:
    """Given a historical PERSON, PLACE and TEXT, decide `at` (was the person ever
    at the place: TRUE/PROBABLE/FALSE) and `isAt` (present there in this context:
    TRUE/FALSE). Graded on the `at` label — the reasoning/world-knowledge target."""
    p = kbench.llm.prompt(prompt, schema=Pred)
    at = str(getattr(p, "at", "FALSE")).upper()
    isAt = str(getattr(p, "isAt", "FALSE")).upper()
    kbench.assertions.assert_equal(gold_at, at, expectation="`at` matches the gold label")
    return {"at": at, "isAt": isAt}


if __name__ == "__main__":   # local / in-notebook smoke run on one model
    runs = hipe_at.evaluate(
        llm=[kbench.llms["anthropic/claude-haiku-4-5@20251001"]],
        evaluation_data=DATA, on_failure="continue", max_attempts=3)
    df = runs.completed_runs.as_dataframe()
    print(f"completed {len(df)} / {len(DATA)}  (errored {len(runs.errored_runs)})")
