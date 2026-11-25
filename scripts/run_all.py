"""Reproduce every figure, table and number in the letter.

  python scripts/run_all.py                       # calibrated stand-in workload (draft)
  python scripts/run_all.py --traj DIR [--limit N]  # real SWE-agent .traj files
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kvpark.hw import MODELS, TIERS, H100, reload_seconds, recompute_seconds, breakeven_bw_gbs
from kvpark.workload import calibrated_sessions, load_traj_dir, load_sessions_json, summarize, fit_pause, TARGET
from kvpark.sim import ttl_metrics, timeline, flatten

# fixed categorical order (validated palette) + line styles as secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W1 = 3.45  # IEEE single column (in)

# Deployment per model: GPUs per replica (H100-80GB) and weight bytes (GB).
DEPLOY = {
    "Llama-3.1-8B": (1, 16.1), "Gemma-3-27B": (1, 54.9), "gpt-oss-120b": (1, 65.0),
    "Llama-3.1-70B": (8, 141.1), "Qwen3-235B-A22B": (8, 470.0),
    "DeepSeek-V3": (16, 685.0), "Llama-3.1-405B": (16, 811.0),
}
ORDER = ["Llama-3.1-8B", "Gemma-3-27B", "gpt-oss-120b", "Llama-3.1-70B",
         "Qwen3-235B-A22B", "DeepSeek-V3", "Llama-3.1-405B"]
FOCUS = "Llama-3.1-70B"
HOST = TIERS["Host DRAM (PCIe 5.0 x16)"]
TTL_STAR = 2.0
TPOT = 0.03
KV_FRAC = 0.9  # usable HBM fraction after activations/fragmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj"); ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    figdir = os.path.join(a.out, "figs"); os.makedirs(figdir, exist_ok=True)
    R = {}

    if a.traj and a.traj.endswith(".json.gz"):
        sess = load_sessions_json(a.traj); R["workload"] = "trace:" + a.traj
    elif a.traj:
        sess = load_traj_dir(a.traj, a.limit); R["workload"] = "trace:" + a.traj
    else:
        err, mu, sig, *_ = fit_pause()
        sess = calibrated_sessions(2000, mu=mu, sig=sig)
        R["workload"] = "calibrated"; R["pause_fit"] = dict(mu=mu, sigma=sig, err=err)
    R["summary"] = {k: float(v) for k, v in summarize(sess).items()}
    R["target"] = TARGET

    # ---------------- Table I: geometry + breakeven -----------------------
    rows = []
    for name in ORDER:
        m = MODELS[name]; g, _ = DEPLOY[name]
        rows.append(dict(
            model=name, attn=m.attn, gpus=g,
            kib_tok=m.bytes_per_token_at(32768) / 1024,
            kib_tok_link=m.bytes_per_token_at(32768) / 1024 / m.kv_shards(g),
            gib32=float(m.kv_bytes(32768)) / 2**30, gib128=float(m.kv_bytes(131072)) / 2**30,
            be_gbs=float(breakeven_bw_gbs(m, 32768, gpus=g)), shards=m.kv_shards(g),
            recompute_s=float(recompute_seconds(m, 32768, gpus=g)),
            reload_ms=1e3 * float(reload_seconds(m, 32768, HOST, gpus=g)),
            reload_ssd_ms=1e3 * float(reload_seconds(m, 32768, TIERS["NVMe Gen5 SSD"], gpus=g)),
        ))
    R["table"] = rows
    with open(os.path.join(a.out, "table_models.tex"), "w") as f:
        for r in rows:
            f.write(f"{r['model']} & {r['attn']} & {r['gpus']}/{r['shards']} & {r['kib_tok']:.0f} & "
                    f"{r['kib_tok_link']:.0f} & {r['be_gbs']:.2f} & {r['recompute_s']:.2f} & "
                    f"{r['reload_ms']:.0f} \\\\\n")

    # ---------------- Fig 1: pauses and byte-second concentration ----------
    m = MODELS[FOCUS]; g = DEPLOY[FOCUS][0]
    B, P, D, _, _, _, _ = flatten(m, sess, g, TPOT)
    keep = P > 0
    Bp, Pp = B[keep], P[keep]
    o = np.argsort(-Pp); bs = (Bp * Pp)[o]; share = np.cumsum(bs) / bs.sum()
    ev = np.arange(1, len(o) + 1) / len(o)
    conc = {}
    for q in [0.01, 0.02, 0.05, 0.10, 0.20]:
        k = int(q * len(o)) - 1
        conc[str(q)] = dict(pause_gt_s=float(Pp[o][k]), byte_sec_share=float(share[k]))
    R["concentration"] = conc
    R["idle_share_inf"] = float((B * P).sum() / (B * (P + D)).sum())
    R["idle_share_tpot"] = {str(t): float((lambda b, p, d: (b * p).sum() / (b * (p + d)).sum())(*flatten(m, sess, g, t)[:3]))
                            for t in [0.01, 0.02, 0.03, 0.05]}
    R["pause_pcts"] = {str(q): float(np.percentile(Pp, q)) for q in [50, 90, 99, 99.9]}

    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.45))
    xs = np.sort(Pp); ccdf = 1 - np.arange(len(xs)) / len(xs)
    ax[0].loglog(xs, ccdf, color=C[0]); ax[0].set_xlim(1e-2, None)
    ax[0].set_xlabel("Tool-call pause (s)"); ax[0].set_ylabel("P(pause > x)")
    ax[0].set_title("(a) Pause CCDF", fontsize=7, color=INK)
    ax[1].plot(100 * ev, 100 * share, color=C[0])
    ax[1].plot([0, 100], [0, 100], color=MUTED, lw=0.6, ls=":")
    ax[1].set_xscale("symlog", linthresh=1); ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("Longest pauses (% of events)"); ax[1].set_ylabel("Idle KV byte-s (%)")
    ax[1].set_title("(b) Idle-KV concentration", fontsize=7, color=INK)
    k = int(0.02 * len(o)) - 1
    ax[1].plot([100 * ev[k]], [100 * share[k]], "o", ms=3, color=C[1])
    ax[1].annotate(f"2% of pauses\n{100*share[k]:.0f}% of byte-s", (100 * ev[k], 100 * share[k]),
                   xytext=(4, 30), textcoords="data", fontsize=6, color=INK,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(figdir, "fig1_pauses.pdf")); plt.close(fig)

    # ---------------- Fig 2: resume stall vs context ----------------------
    ctx = np.logspace(np.log10(2e3), np.log10(131072), 60)
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    curves = [("NVMe Gen5 SSD", TIERS["NVMe Gen5 SSD"]),
              ("Host DRAM or CXL (PCIe 5.0 x16)", HOST),
              ("Grace LPDDR5X (NVLink-C2C)", TIERS["Grace LPDDR5X (NVLink-C2C)"])]
    for i, (lab, t) in enumerate(curves):
        y = 1e3 * reload_seconds(m, ctx, t, gpus=g)
        ax.loglog(ctx / 1e3, y, color=C[i], ls=LS[i], label=lab)
    y = 1e3 * recompute_seconds(m, ctx, gpus=g)
    ax.loglog(ctx / 1e3, y, color=INK, lw=1.4, label="Recompute (prefill)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, fontsize=5.8, handlelength=2.2)
    ax.axhline(100, color=MUTED, lw=0.6, ls=":"); ax.text(60, 125, "100 ms", fontsize=5.8, color=MUTED)
    ax.set_ylim(0.1, 3e4)
    ax.set_xlabel("Parked context (K tokens)"); ax.set_ylabel("Resume stall (ms)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig2_stall.pdf")); plt.close(fig)
    R["fig2"] = {tn: dict(ms_32k=float(1e3 * reload_seconds(m, 32768, t, gpus=g)),
                          ms_128k=float(1e3 * reload_seconds(m, 131072, t, gpus=g)))
                 for tn, t in TIERS.items()}
    R["fig2"]["recompute"] = dict(ms_32k=float(1e3 * recompute_seconds(m, 32768, gpus=g)),
                                  ms_128k=float(1e3 * recompute_seconds(m, 131072, gpus=g)))

    # ---------------- Fig 3: TTL sweep (per session, per GPU) --------------
    ttls = np.array([0.0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300])
    sw = [ttl_metrics(m, sess, HOST, t, g, TPOT) for t in ttls]
    sw_inf = ttl_metrics(m, sess, HOST, 1e12, g, TPOT)
    R["ttl_sweep"] = {str(t): {k: float(v) for k, v in r.items()} for t, r in zip(ttls, sw)}
    R["ttl_inf"] = {k: float(v) for k, v in sw_inf.items()}
    x = np.where(ttls == 0, 0.05, ttls)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.5))
    ax[0].semilogx(x, [r["parked_hbm_gb"] * 1e3 for r in sw], color=C[0], marker="o", ms=2.5, label="Idle in HBM")
    ax[0].semilogx(x, [r["offloaded_gb"] * 1e3 for r in sw], color=C[1], ls="--", marker="s", ms=2.5, label="Off-HBM tier")
    ax[0].axvline(TTL_STAR, color=MUTED, lw=0.6, ls=":")
    ax[0].set_xlabel("TTL (s)"); ax[0].set_ylabel("Idle KV (MB/GPU/session)")
    ymax = 1e3 * max(max(r["offloaded_gb"], r["parked_hbm_gb"]) for r in sw) * 1.18
    ax[0].set_ylim(0, ymax)
    ax[0].text(0.06, ymax * 0.92, "off-HBM tier", fontsize=5.8, color=INK)
    ax[0].text(0.06, ymax * 0.30, "idle in HBM", fontsize=5.8, color=INK)
    ax[0].set_title("(a) Where idle KV lives", fontsize=7, color=INK)
    ax[1].loglog(x, [r["link_gbs"] * 1e3 for r in sw], color=C[0], marker="o", ms=2.5)
    ax[1].axvline(TTL_STAR, color=MUTED, lw=0.6, ls=":")
    ax[1].set_xlabel("TTL (s)"); ax[1].set_ylabel("Link traffic (MB/s/GPU/session)")
    ax[1].set_title("(b) Offload+reload traffic", fontsize=7, color=INK)
    fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(figdir, "fig3_ttl.pdf")); plt.close(fig)

    # ---------------- Fig 4: per-model provisioning at TTL* -----------------
    prov = []
    for name in ORDER:
        mm = MODELS[name]; gg, wgb = DEPLOY[name]
        budget = gg * 80.0 * KV_FRAC - wgb  # GB of HBM for KV per replica
        r_inf = ttl_metrics(mm, sess, HOST, 1e12, gg, TPOT)
        r_ttl = ttl_metrics(mm, sess, HOST, TTL_STAR, gg, TPOT)
        per_inf = (r_inf["active_gb"] + r_inf["parked_hbm_gb"]) * gg  # GB per session per replica
        per_ttl = (r_ttl["active_gb"] + r_ttl["parked_hbm_gb"]) * gg
        c_inf = int(budget // per_inf); c_ttl = int(budget // per_ttl)
        _, _, off = timeline(mm, sess, c_ttl, TTL_STAR, gg, tier=HOST, tpot=TPOT, seed=1)
        warm = off[600:]
        prov.append(dict(model=name, budget_gb=budget, sess_inf=c_inf, sess_ttl=c_ttl,
                         gain=c_ttl / max(c_inf, 1),
                         off_mean_gb_gpu=float(warm.mean()), off_p99_gb_gpu=float(np.percentile(warm, 99)),
                         link_gbs_gpu=r_ttl["link_gbs"] * c_ttl,
                         stall_p99_ms_host=r_ttl["stall_p99_ms"],
                         stall_p99_ms_ssd=ttl_metrics(mm, sess, TIERS["NVMe Gen5 SSD"], TTL_STAR, gg, TPOT)["stall_p99_ms"],
                         idle_share=r_inf["parked_hbm_gb"] / (r_inf["active_gb"] + r_inf["parked_hbm_gb"])))
    R["provision"] = prov
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.95), gridspec_kw=dict(width_ratios=[1, 1]))
    yy = np.arange(len(prov))[::-1]
    labels = [p["model"].replace("-Instruct", "") for p in prov]
    ax[0].barh(yy + 0.18, [p["stall_p99_ms_host"] for p in prov], height=0.34, color=C[0], label="DRAM/CXL")
    ax[0].barh(yy - 0.18, [p["stall_p99_ms_ssd"] for p in prov], height=0.34, color=C[1], label="NVMe SSD")
    ax[0].set_xscale("log"); ax[0].axvline(100, color=MUTED, lw=0.6, ls=":")
    ax[0].set_yticks(yy); ax[0].set_yticklabels(labels); ax[0].set_xlabel("p99 stall (ms)")
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=5.5, handlelength=1); ax[0].grid(axis="y", visible=False)
    ax[0].set_title("(a) Resume stall", fontsize=7, color=INK)
    ax[1].barh(yy, [p["off_p99_gb_gpu"] for p in prov], height=0.5, color=C[0])
    for yv, p in zip(yy, prov):
        ax[1].text(p["off_p99_gb_gpu"] + 0.3, yv, f"{p['off_p99_gb_gpu']:.0f}", va="center", fontsize=5.5, color=INK)
    ax[1].set_yticks(yy); ax[1].set_yticklabels([]); ax[1].set_xlabel("p99 off-HBM (GB/GPU)")
    ax[1].grid(axis="y", visible=False); ax[1].set_title("(b) Tier capacity", fontsize=7, color=INK)
    fig.tight_layout(w_pad=1.2); fig.savefig(os.path.join(figdir, "fig4_provision.pdf")); plt.close(fig)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(json.dumps({k: R[k] for k in ["workload", "summary", "concentration", "idle_share_inf", "idle_share_tpot", "pause_pcts"]}, indent=1, default=float))
    print(json.dumps(R["ttl_sweep"]["2.0"], indent=1)); print(json.dumps(R["fig2"], indent=1))
    for p in prov: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
