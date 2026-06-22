# hipe/cli.py
import argparse
import os
from datetime import datetime
from pathlib import Path
import yaml
from hipe import config as cfg
from hipe.harness import run_experiment
from hipe.eval.scorer import score_files


def _runs_root():
    return Path(os.environ.get("HIPE_RUNS_DIR", cfg.RUNS_DIR))


def _cmd_run(args) -> int:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result = run_experiment(config, now=now, runs_root=_runs_root())
    print(f"run_dir: {result['run_dir']}")
    print(f"at_recall: {result['at_recall']:.4f}  "
          f"isAt_recall: {result['isAt_recall']:.4f}  "
          f"global: {result['global']:.4f}  (n_dev={result['n_dev']})")
    return 0


def _cmd_score(args) -> int:
    m = score_files(args.gold, args.pred)
    print(f"at macro_recall:    {m['at']['macro_recall']:.4f}")
    print(f"isAt macro_recall:  {m['isAt']['macro_recall']:.4f}")
    print(f"global macro_recall: {m['global']['macro_recall']:.4f}")
    return 0


def _cmd_kaggle_run(args) -> int:
    import tempfile
    from hipe.kaggle.bridge import run_kaggle
    staging = tempfile.mkdtemp(prefix="hipe_kaggle_")
    result = run_kaggle(args.config, repo_root=cfg.ROOT, runs_root=_runs_root(),
                        staging_dir=staging)
    print(f"kernel: {result['kernel_id']}")
    print(f"ingested runs: {result['runs']}")
    return 0


def _cmd_leaderboard(args) -> int:
    path = _runs_root() / "leaderboard.csv"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
    else:
        print("(no runs yet)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hipe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run an experiment from a config")
    p_run.add_argument("config")
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser("score", help="score a prediction file vs gold")
    p_score.add_argument("gold")
    p_score.add_argument("pred")
    p_score.set_defaults(func=_cmd_score)

    p_lb = sub.add_parser("leaderboard", help="print the leaderboard")
    p_lb.set_defaults(func=_cmd_leaderboard)

    p_kaggle = sub.add_parser("kaggle-run",
                              help="run a config on Kaggle GPU and ingest results")
    p_kaggle.add_argument("config")
    p_kaggle.set_defaults(func=_cmd_kaggle_run)

    args = parser.parse_args(argv)
    return args.func(args)
