"""Shrink SWE-agent .traj files to per-step (ctx, out, new, pause) records.

Run on your machine, then attach the output (a few MB) back to the chat:

  pip install tiktoken numpy
  aws s3 cp s3://scaleapi-results/swe-bench-pro/claude-45sonnet-10132025/ data/claude45/ \
      --recursive --no-sign-request --exclude "*" --include "*.traj"
  python scripts/reduce_traj.py data/claude45 sessions_claude45.json.gz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import gzip
import json
import sys

from kvpark.workload import load_traj_dir

src, dst = sys.argv[1], sys.argv[2]
sess = load_traj_dir(src)
out = [{k: [round(float(x), 3) for x in v] for k, v in s.items()} for s in sess]
with gzip.open(dst, "wt") as f:
    json.dump(out, f)
print(f"{len(out)} sessions, {sum(len(s['ctx']) for s in out)} steps -> {dst}")
