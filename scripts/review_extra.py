"""Further sensitivity analyses for the letter (run after review_analyses.py).

  python scripts/review_extra.py --traj data/sessions_claude-45sonnet-10132025.json.gz --out out

Adds: per-link bandwidth for a 100 ms p99 resume across five seeds; shared-pool
sweep with and without offload writes on the pool; idle-KV concentration for
every model; provisioning at mean/p95/p99/p99.9/max occupancy; usable-HBM
sensitivity; and the share of all tool returns that stall over 100 ms.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import argparse
import importlib.util
import json
import os
import numpy as np

from kvpark.hw import MODELS, TIERS, H100, EFF
from kvpark.workload import load_sessions_json
from kvpark.replay import replay, safe_concurrency, _prep
from kvpark.sim import flatten, ttl_metrics

_spec = importlib.util.spec_from_file_location("ra", os.path.join(os.path.dirname(__file__), "review_analyses.py"))
ra = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ra)
DEPLOY, ORDER, per_gpu_weights = ra.DEPLOY, ra.ORDER, ra.per_gpu_weights
HOST = TIERS["Host DRAM (PCIe 5.0 x16)"]
TTL, SLO = 2.0, 100.0
GRID = [16, 24, 32, 40, 48, 51.2, 56, 64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True); ap.add_argument("--out", default="out")
    ap.add_argument("--prev", default=None, help="results_review.json with per-seed C*")
    a = ap.parse_args()
    sess = load_sessions_json(a.traj)
    prev = json.load(open(a.prev or os.path.join(a.out, "results_review.json")))
    cstar = {p["model"]: [r["c_ttl"] for r in p["runs"]] for p in prev["provision"]}
    R = {}; P = lambda *x: print(*x, flush=True)

    # 1) smallest tested per-link bandwidth meeting a 100 ms p99 resume, per seed
    bw = {}
    for name in ORDER:
        m, d = MODELS[name], DEPLOY[name]; g = d["gpus"]; pre = _prep(m, sess, g, 0.03)
        per_seed = []
        for sd, c in enumerate(cstar[name]):
            hit = None
            for b in GRID:
                if replay(m, sess, c, TTL, g, b / EFF, prep=pre, seed=sd)["stall_p99_ms"] <= SLO:
                    hit = b; break
            per_seed.append(hit)
        bw[name] = per_seed; P("bw", name, per_seed)
    R["bw_slo_seeds"] = bw

    # 2) shared pool: reads+writes vs reads only (Llama-3.1-70B, C=32)
    name = "Llama-3.1-70B"; m, d = MODELS[name], DEPLOY[name]; g = d["gpus"]
    pre = _prep(m, sess, g, 0.03); c = cstar[name][0]
    pool = []
    for A in [128, 256, 320, 384, 409.6, 448, 512, None]:
        row = dict(pool=A)
        for tag, ro, mode in (("rw", False, "fifo"), ("r", True, "fifo"), ("fluid", False, "fluid")):
            r = replay(m, sess, c, TTL, g, HOST.bw_gbs, prep=pre, pool_gbs=A, pool_reads_only=ro, pool_mode=mode)
            row[tag] = {k: r[k] for k in ("stall_p99_ms", "wait_p99_ms", "frac_stall_gt100")}
        pool.append(row); P("pool", A, round(row["rw"]["stall_p99_ms"]), round(row["r"]["stall_p99_ms"]),
                            round(100 * row["rw"]["frac_stall_gt100"]), round(100 * row["r"]["frac_stall_gt100"]))
    R["pool2"] = pool
    # smallest tested pool meeting the objective, per seed (FIFO and fluid)
    pseeds = {}
    for mode in ("fifo", "fluid"):
        out = []
        for sd, cs in enumerate(cstar[name]):
            hit = None
            for A in [256, 288, 320, 352, 384, 409.6, 448, 512]:
                if replay(m, sess, cs, TTL, g, HOST.bw_gbs, prep=pre, pool_gbs=A, seed=sd,
                          pool_mode=mode)["stall_p99_ms"] <= SLO:
                    hit = A; break
            out.append(hit)
        pseeds[mode] = out; P("pool seeds", mode, out)
    R["pool_seeds"] = pseeds

    # 3) idle-KV concentration and TTL effect per model
    conc = {}
    for nm in ORDER:
        mm, gg = MODELS[nm], DEPLOY[nm]["gpus"]
        B, Pp, D, *_ = flatten(mm, sess, gg, 0.03)
        keep = Pp > 0; bs = (B * Pp)[keep]; o = np.argsort(-Pp[keep]); k = int(0.02 * keep.sum())
        ra_ = replay(mm, sess, cstar[nm][0], TTL, gg, HOST.bw_gbs, prep=_prep(mm, sess, gg, 0.03))
        conc[nm] = dict(top2=float(bs[o][:k].sum() / bs.sum()),
                        freed_ideal=ra_["idle_freed_ideal"], freed_actual=ra_["idle_freed_actual"],
                        reload_frac=ra_["reload_frac"],
                        idle=float((B * Pp).sum() / (B * (Pp + D)).sum()))
        P("conc", nm, {k2: round(v, 3) for k2, v in conc[nm].items()})
    R["concentration_models"] = conc

    # 4) provisioning quantile and usable-HBM sensitivity
    quant = {}
    for nm in ["Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.1-405B"]:
        mm, dd = MODELS[nm], DEPLOY[nm]; gg = dd["gpus"]; wg = per_gpu_weights(dd); pp = _prep(mm, sess, gg, 0.03)
        row = {}
        for lab, kw in (("mean", dict(mean=True)), ("p95", dict(q=95)), ("p99", dict(q=99)),
                        ("p99.9", dict(q=99.9)), ("max", dict(q=100))):
            ci, _ = safe_concurrency(mm, sess, 1e12, gg, wg * gg, HOST.bw_gbs, prep=pp, **kw)
            ct, _ = safe_concurrency(mm, sess, TTL, gg, wg * gg, HOST.bw_gbs, prep=pp, **kw)
            row[lab] = (ci, ct)
        quant[nm] = row; P("quant", nm, row)
    R["quantile_provisioning"] = quant
    frac = {}
    mm, dd = MODELS["Llama-3.1-70B"], DEPLOY["Llama-3.1-70B"]; gg = dd["gpus"]; wg = per_gpu_weights(dd)
    pp = _prep(mm, sess, gg, 0.03)
    for f in (0.80, 0.85, 0.90):
        ci, _ = safe_concurrency(mm, sess, 1e12, gg, wg * gg, HOST.bw_gbs, prep=pp, kv_frac=f)
        ct, _ = safe_concurrency(mm, sess, TTL, gg, wg * gg, HOST.bw_gbs, prep=pp, kv_frac=f)
        frac[str(f)] = (ci, ct); P("hbmfrac", f, ci, ct)
    R["hbm_frac"] = frac

    # 5) share of all tool returns stalling >100 ms, TTL x bandwidth (Llama-3.1-70B)
    joint = []
    for ttl in [0.5, 1, 2, 5, 10]:
        cs, _ = safe_concurrency(mm, sess, ttl, gg, wg * gg, HOST.bw_gbs, prep=pp)
        row = dict(ttl=ttl, c=cs)
        for b in [8, 16, 32, 51.2]:
            r = replay(mm, sess, cs, ttl, gg, b / EFF, prep=pp)
            row[str(b)] = dict(of_resumes=r["frac_stall_gt100"], of_returns=r["frac_returns_gt100"],
                               reload_frac=r["reload_frac"], resumes=r["resumes"], returns=r["returns"],
                               cancelled=r["cancelled"])
        joint.append(row); P("joint", ttl, cs, {b: round(100 * row[str(b)]["of_returns"], 1) for b in [8, 16, 32, 51.2]})
    R["ttl_bw_returns"] = joint

    # 6) TPOT sensitivity (Llama-3.1-70B)
    tp = {}
    for tpot in (0.010, 0.030, 0.060):
        p6 = _prep(mm, sess, gg, tpot)
        ci, _ = safe_concurrency(mm, sess, 1e12, gg, wg * gg, HOST.bw_gbs, prep=p6, tpot=tpot)
        ct, _ = safe_concurrency(mm, sess, TTL, gg, wg * gg, HOST.bw_gbs, prep=p6, tpot=tpot)
        r = replay(mm, sess, ct, TTL, gg, HOST.bw_gbs, prep=p6, tpot=tpot)
        tp[str(tpot)] = dict(c_inf=ci, c_ttl=ct, stall_p99_ms=r["stall_p99_ms"], off_p99_gb=r["off_p99_gb"],
                             util_p99=r["util_p99"])
        P("tpot", tpot, tp[str(tpot)])
    R["tpot"] = tp

    # 7) token-count sensitivity: scale contexts by +-15% (Llama-3.1-70B)
    tk = {}
    for f_ in (0.85, 1.0, 1.15):
        s2 = [dict(s, ctx=s["ctx"] * f_, new=s["new"] * f_) for s in sess]
        p7 = _prep(mm, s2, gg, 0.03)
        ci, _ = safe_concurrency(mm, s2, 1e12, gg, wg * gg, HOST.bw_gbs, prep=p7)
        ct, _ = safe_concurrency(mm, s2, TTL, gg, wg * gg, HOST.bw_gbs, prep=p7)
        r = replay(mm, s2, ct, TTL, gg, HOST.bw_gbs, prep=p7)
        tk[str(f_)] = dict(c_inf=ci, c_ttl=ct, stall_p99_ms=r["stall_p99_ms"], off_p99_gb=r["off_p99_gb"])
        P("tokens", f_, tk[str(f_)])
    R["token_scale"] = tk

    # 8) context feasibility: one 128k-token session per GPU vs KV budget
    from kvpark.hw import H100 as _H
    feas = {}
    for nm in ORDER:
        mm2, dd = MODELS[nm], DEPLOY[nm]; gg2 = dd["gpus"]; wg2 = per_gpu_weights(dd)
        B = float(mm2.kv_bytes(131072))
        per_gpu = B if mm2.mla_dim else B * mm2.kv_replication(gg2) / gg2
        feas[nm] = dict(kv_128k_gb_gpu=per_gpu / 1e9,
                        budget={str(f_): _H.hbm_gb * f_ - wg2 for f_ in (0.8, 0.85, 0.9)})
        P("feas", nm, round(per_gpu / 1e9, 2), {k2: round(v, 1) for k2, v in feas[nm]["budget"].items()})
    R["context_feasibility"] = feas

    with open(os.path.join(a.out, "results_extra.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    P("done")


if __name__ == "__main__":
    main()
