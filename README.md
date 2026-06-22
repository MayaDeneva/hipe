# HIPE-2026 Relation-Extraction Harness

Person–place relation extraction for CLEF HIPE-2026. See the design spec in
`docs/superpowers/specs/` and the plans in `docs/superpowers/plans/`.

## Setup

```bash
pip install -e ".[dev]"
python scripts/fetch_data.py        # clone pinned HIPE-2026-data into data/raw/
```

## Run

```bash
hipe run configs/baseline_majority.yaml   # train -> predict -> score -> leaderboard row
hipe leaderboard                          # show all runs
hipe score <gold.jsonl> <pred.jsonl>      # official-parity score of a file
```

Each run writes a folder under `runs/` (config, predictions, metrics) and appends
one row to the committed `runs/leaderboard.csv` — nothing is lost.

## Run on Kaggle GPU (automated)

Requires a phone-verified Kaggle account and `~/.kaggle/kaggle.json`, plus `pip install -e ".[kaggle]"`.

```bash
hipe kaggle-run configs/xlmr.yaml   # packages code -> Kaggle dataset -> GPU kernel -> ingests results
hipe leaderboard                    # the xlmr row, trained on Kaggle, appears here
```

It pushes the code as a private dataset, runs a GPU+internet kernel that clones the
HIPE data and runs `hipe run`, polls to completion, and merges the run folder +
leaderboard row into local `runs/`.
