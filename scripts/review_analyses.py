"""Contention-aware provisioning analyses used in the letter.

  python scripts/review_analyses.py --traj data/sessions_claude-45sonnet-10132025.json.gz --out out

Writes results_review.json and the LaTeX rows for the provisioning, bandwidth,
shared-pool and TTL-by-bandwidth tables.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import argparse
import json
import os
import numpy as np

from kvpark.hw import MODELS, TIERS, H100, EFF, breakeven_bw_gbs
from kvpark.workload import load_sessions_json, load_traj_dir
from kvpark.replay import replay, safe_concurrency, _prep
from kvpark.sim import flatten

# Deployment: GPUs, weight precision, total weight GB, and weights replicated on
# every GPU (MLA attention under data-parallel attention).
DEPLOY = {
    "Llama-3.1-8B":    dict(gpus=1,  prec="BF16",  w_gb=16.1,  rep_gb=0.0),
    "Gemma-3-27B":     dict(gpus=1,  prec="BF16",  w_gb=54.9,  rep_gb=0.0),
    "gpt-oss-120b":    dict(gpus=1,  prec="MXFP4", w_gb=65.0,  rep_gb=0.0),
    "Llama-3.1-70B":   dict(gpus=8,  prec="BF16",  w_gb=141.1, rep_gb=0.0),
    "Qwen3-235B-A22B": dict(gpus=8,  prec="BF16",  w_gb=470.0, rep_gb=0.0),
    "DeepSeek-V3":     dict(gpus=16, prec="FP8",   w_gb=685.0, rep_gb=11.4),
    "Llama-3.1-405B":  dict(gpus=16, prec="BF16",  w_gb=811.0, rep_gb=0.0),
}
ORDER = list(DEPLOY)
HOST, SSD = TIERS["Host DRAM (PCIe 5.0 x16)"], TIERS["NVMe Gen5 SSD"]
TTL, KV_FRAC, SLO = 2.0, 0.9, 0.100
SEEDS = [0, 1, 2, 3, 4]
BW_GRID = [16, 24, 32, 40, 48, 51.2, 64, 80, 96, 128, 160, 192, 256, 320]   # effective GB/s per link


def per_gpu_weights(d):
    return d["rep_gb"] + (d["w_gb"] - d["rep_gb"]) / d["gpus"]


def rng_str(v, fmt="{:.0f}"):
    lo, hi = min(v), max(v)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}--{fmt.format(hi)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True); ap.add_argument("--out", default="out")
    ap.add_argument("--horizon", type=float, default=2 * 3600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sess = load_sessions_json(a.traj) if a.traj.endswith(".json.gz") else load_traj_dir(a.traj)
    R = {}
    P = lambda *x: print(*x, flush=True)

    # ---- per-model provisioning, 5 replay seeds ------------------------------
    prov = []
    for name in ORDER:
        m, d = MODELS[name], DEPLOY[name]
        g = d["gpus"]; wg = per_gpu_weights(d)
        pre = _prep(m, sess, g, 0.03)
        budget = H100.hbm_gb * KV_FRAC - wg
        kw = dict(horizon=a.horizon, prep=pre)
        runs = []
        for sd in SEEDS:
            c_inf, _ = safe_concurrency(m, sess, 1e12, g, wg * g, HOST.bw_gbs, seed=sd, **kw)
            c_ttl, _ = safe_concurrency(m, sess, TTL, g, wg * g, HOST.bw_gbs, seed=sd, **kw)
            rh = replay(m, sess, c_ttl, TTL, g, HOST.bw_gbs, seed=sd, budget_gb=budget, **kw)
            rs = replay(m, sess, c_ttl, TTL, g, SSD.bw_gbs, seed=sd, **kw)
            runs.append(dict(c_inf=c_inf, c_ttl=c_ttl, **{k: rh[k] for k in (
                "hbm_p99_gb", "hbm_max_gb", "frac_time_over_budget", "off_p99_gb", "util_mean",
                "util_p99", "wait_p99_ms", "stall_p99_ms", "frac_stall_gt100")},
                stall_p99_ssd_ms=rs["stall_p99_ms"]))
        r0 = runs[0]
        # smallest effective per-link bandwidth meeting a 100 ms p99 resume SLO
        # (seed 0, at seed-0 concurrency), and the smallest with p99 queueing <= 10 ms
        c0 = r0["c_ttl"]
        bw_slo = bw_q = None
        for bwe in BW_GRID:
            r = replay(m, sess, c0, TTL, g, bwe / EFF, prep=pre, horizon=a.horizon)
            if bw_q is None and r["wait_p99_ms"] <= 10:
                bw_q = bwe
            if r["stall_p99_ms"] <= SLO * 1e3:
                bw_slo = bwe; break
        row = dict(model=name, prec=d["prec"], gpus=g, w_gpu_gb=wg, budget_gb=budget,
                   runs=runs, bw_slo=bw_slo, bw_queue=bw_q,
                   assignment="round-robin slot-to-GPU" if m.mla_dim else "all GPUs (tensor parallel)")
        prov.append(row)
        P(name, "C*", [r["c_inf"] for r in runs], [r["c_ttl"] for r in runs], "bw_slo", bw_slo, "bw_q", bw_q,
          "stall", [round(r["stall_p99_ms"]) for r in runs], "over", [round(r["frac_time_over_budget"], 4) for r in runs])
    R["provision"] = prov
    with open(os.path.join(a.out, "table_provision.tex"), "w") as f:
        for p in prov:
            rs_ = p["runs"]; g = lambda k: [r[k] for r in rs_]
            ci, ct = g("c_inf"), g("c_ttl")
            f.write(f"{p['model']} & {p['prec']} & {p['w_gpu_gb']:.0f} & {p['budget_gb']:.0f} & "
                    f"{rng_str(ci)}$\\to${rng_str(ct)} & {rng_str(g('off_p99_gb'))} & "
                    f"{rng_str([100*x for x in g('util_p99')])} & "
                    f"{rng_str(g('stall_p99_ms'))} & {rng_str(g('stall_p99_ssd_ms'))} & "
                    f"{p['bw_slo'] if p['bw_slo'] else '$>$' + str(BW_GRID[-1])} \\\\\n")

    # ---- Llama-3.1-70B: bandwidth, shared pool, TTL x bandwidth ------------
    name = "Llama-3.1-70B"; m, d = MODELS[name], DEPLOY[name]
    g = d["gpus"]; wg = per_gpu_weights(d); budget = H100.hbm_gb * KV_FRAC - wg
    pre = _prep(m, sess, g, 0.03)
    c = prov[ORDER.index(name)]["runs"][0]["c_ttl"]
    sens = []
    for bwe in [4, 8, 16, 32, 51.2]:
        r = replay(m, sess, c, TTL, g, bwe / EFF, horizon=a.horizon, prep=pre)
        sens.append(dict(bw=bwe, **{k: r[k] for k in ("stall_p50_ms", "stall_p99_ms", "wait_p99_ms",
                                                         "frac_stall_gt100", "util_p99")}))
    R["bw_sensitivity"] = sens; P("bw", [(s["bw"], round(s["stall_p99_ms"]), round(s["wait_p99_ms"])) for s in sens])
    with open(os.path.join(a.out, "table_bw.tex"), "w") as f:
        for s in sens:
            f.write(f"{s['bw']:g} & {s['stall_p50_ms']:.0f} & {s['stall_p99_ms']:.0f} & {s['wait_p99_ms']:.0f} & "
                    f"{100*s['frac_stall_gt100']:.0f}\\% & {100*min(s['util_p99'],1):.0f}\\% \\\\\n")

    pool = []
    for pg in [64, 128, 256, 512, None]:
        r = replay(m, sess, c, TTL, g, HOST.bw_gbs, horizon=a.horizon, prep=pre, pool_gbs=pg)
        pool.append(dict(pool=pg, **{k: r[k] for k in ("stall_p99_ms", "wait_p99_ms", "frac_stall_gt100")}))
    R["pool"] = pool; P("pool", [(p["pool"], round(p["stall_p99_ms"]), round(100 * p["frac_stall_gt100"])) for p in pool])

    joint = []
    for ttl in [0.5, 1, 2, 5, 10]:
        cs, _ = safe_concurrency(m, sess, ttl, g, wg * g, HOST.bw_gbs, horizon=a.horizon, prep=pre)
        row = dict(ttl=ttl, c=cs)
        for bwe in [8, 16, 32, 51.2]:
            r = replay(m, sess, cs, ttl, g, bwe / EFF, horizon=a.horizon, prep=pre)
            row[str(bwe)] = r["frac_stall_gt100"]
        joint.append(row)
    R["ttl_bw"] = joint; P("joint", joint)
    with open(os.path.join(a.out, "table_ttl_bw.tex"), "w") as f:
        for r in joint:
            f.write(f"{r['ttl']:g} & {r['c']} & " + " & ".join(f"{100*r[str(b)]:.0f}" for b in [8, 16, 32, 51.2]) + " \\\\\n")
    with open(os.path.join(a.out, "table_pool.tex"), "w") as f:
        for p in pool:
            lab = "unlimited" if p["pool"] is None else f"{p['pool']}"
            f.write(f"{lab} & {p['stall_p99_ms']:.0f} & {p['wait_p99_ms']:.0f} & {100*p['frac_stall_gt100']:.0f}\\% \\\\\n")

    # ---- MLA striping with NVLink gather (analytic) --------------------------
    ds = MODELS["DeepSeek-V3"]; B = float(ds.kv_bytes(32768)); link = HOST.bw_gbs * 1e9 * EFF
    nvl = 450e9 * EFF  # H100 NVLink, per direction
    R["mla_stripe"] = {str(k): 1e3 * (B / (k * link) + (B * (k - 1) / k) / nvl) for k in (1, 2, 4, 8)}
    P("stripe", R["mla_stripe"])

    # ---- prefill-utilization sensitivity of the break-even bandwidth --------
    us = [0.2, 0.35, 0.5, 0.7, 0.9]
    R["u_sensitivity"] = {n: {str(u): float(breakeven_bw_gbs(MODELS[n], 32768, gpus=DEPLOY[n]["gpus"])) * u / H100.mfu_prefill
                              for u in us} for n in ORDER}
    R["u_max_be"] = {str(u): max(v[str(u)] for v in R["u_sensitivity"].values()) for u in us}

    # ---- bootstrap CIs (session resampling) ---------------------------------
    rng = np.random.default_rng(7)
    B_, P_, D_, sid, *_ = flatten(m, sess, g, 0.03)
    starts = np.searchsorted(sid, np.arange(len(sess))); ends = np.append(starts[1:], len(sid))
    def stats(idx):
        sl = np.concatenate([np.arange(starts[i], ends[i]) for i in idx])
        b, p, dd = B_[sl], P_[sl], D_[sl]
        keep = p > 0; bs = (b * p)[keep]; o = np.argsort(-p[keep])
        k = max(1, int(0.02 * keep.sum()))
        return bs[o][:k].sum() / bs.sum(), (b * p).sum() / (b * (p + dd)).sum()
    boots = np.array([stats(rng.integers(len(sess), size=len(sess))) for _ in range(300)])
    R["bootstrap"] = dict(top2_share=[float(np.percentile(boots[:, 0], q)) for q in (2.5, 50, 97.5)],
                          idle_share=[float(np.percentile(boots[:, 1], q)) for q in (2.5, 50, 97.5)])

    ctx = np.concatenate([s["ctx"] for s in sess])
    R["ctx"] = dict(p99=float(np.percentile(ctx, 99)), max=float(ctx.max()),
                    frac_gt_128k=float((ctx > 131072).mean()))
    with open(os.path.join(a.out, "results_review.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    P("done")


if __name__ == "__main__":
    main()
