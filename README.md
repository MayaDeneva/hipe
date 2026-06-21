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
