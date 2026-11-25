#!/usr/bin/env sh
# Reproduce every number, table and figure in the letter.
# Needs data/sessions_claude-45sonnet-10132025.json.gz (see README) and
# numpy + matplotlib. Takes roughly 20 minutes on a laptop.
set -e
T=data/sessions_claude-45sonnet-10132025.json.gz
OUT=${1:-out}
python scripts/run_all.py          --traj "$T" --out "$OUT"   # Table 1, Figs 1-3, time-averaged TTL sweep
python scripts/review_analyses.py  --traj "$T" --out "$OUT"   # Table 2, Table 3a, bootstrap CIs
python scripts/review_extra.py     --traj "$T" --out "$OUT"   # per-seed thresholds, Table 3b, sensitivities
python scripts/review_cancel_pool.py --traj "$T" --prev "$OUT/results_review.json" --out "$OUT"
echo "Results in $OUT/"
