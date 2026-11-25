"""Offload-cancellation and 352 GB/s shared-pool checks (Llama-3.1-70B).

  python scripts/review_cancel_pool.py --traj data/sessions_claude-45sonnet-10132025.json.gz \
      --prev out/results_review.json --out out

Compares aborting an in-flight offload when its tool returns (default) with
letting the copy finish and discarding it, at TTL 0.5 s and 2 s and at 16 and
51.2 GB/s; and evaluates a 352 GB/s shared pool (in-order, reads-only and
fluid) plus the fluid pool across all five seeds at their p99-provisioned C*.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import argparse
import json
import os

from kvpark.hw import MODELS, EFF
from kvpark.workload import load_sessions_json
from kvpark.replay import replay, _prep

K = ("stall_p99_ms", "wait_p99_ms", "frac_stall_gt100", "frac_returns_gt100", "hbm_p99_gb",
     "cancelled", "wasted_copies", "returns", "idle_freed_actual")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True); ap.add_argument("--prev", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    S = load_sessions_json(a.traj)
    m = MODELS["Llama-3.1-70B"]; pre = _prep(m, S, 8, 0.03)
    out = {}
    for ttl, c in [(0.5, 33), (2.0, 32)]:
        for bwe in [16, 51.2]:
            for pre_ in (True, False):
                r = replay(m, S, c, ttl, 8, bwe / EFF, prep=pre, preempt_inflight=pre_)
                out[f"{ttl}|{bwe}|{pre_}"] = {k: r[k] for k in K}
                print(ttl, bwe, pre_, {k: round(float(r[k]), 3) for k in K}, flush=True)
    row = {}
    for mode, ro in (("fifo", False), ("fifo", True), ("fluid", False)):
        r = replay(m, S, 32, 2.0, 8, 64, prep=pre, pool_gbs=352, pool_reads_only=ro, pool_mode=mode)
        row[mode + ("_r" if ro else "")] = {k: r[k] for k in ("stall_p99_ms", "wait_p99_ms", "frac_stall_gt100")}
    out["pool352"] = row
    R = json.load(open(a.prev))
    runs = [p for p in R["provision"] if p["model"] == "Llama-3.1-70B"][0]["runs"]
    out["pool352_fluid_seeds"] = [replay(m, S, rr["c_ttl"], 2.0, 8, 64, prep=pre, pool_gbs=352,
                                         pool_mode="fluid", seed=sd)["stall_p99_ms"] for sd, rr in enumerate(runs)]
    print("352", row, out["pool352_fluid_seeds"], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "results_cancel_pool.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
