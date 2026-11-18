"""Replay agent sessions against a TTL park-or-offload policy.

During each tool pause the session's KV is either kept in HBM ("parked")
for up to TTL seconds, after which it is moved to an off-HBM tier, and
reloaded when the tool returns (adding a resume stall).
"""
from __future__ import annotations
import numpy as np
from .hw import Model, Tier, GPU, H100, EFF


def step_seconds(model: Model, s: dict, gpus: int, tpot: float, gpu: GPU = H100):
    """LLM step duration: prefill of appended tokens + decode of output tokens."""
    d = model.q_heads * model.head_dim
    full = model.layers if model.global_layers < 0 else model.global_layers
    flops = 2 * model.active_params * s["new"] + 4 * d * full * s["new"] * s["ctx"]
    return flops / (gpu.bf16_tflops * 1e12 * gpu.mfu_prefill * gpus) + s["out"] * tpot


def flatten(model: Model, sess: list[dict], gpus: int, tpot: float):
    B, P, D, sid = [], [], [], []
    for i, s in enumerate(sess):
        B.append(model.kv_bytes(s["ctx"]) / gpus)  # per GPU
        P.append(s["pause"]); D.append(step_seconds(model, s, gpus, tpot)); sid.append(np.full(len(s["ctx"]), i))
    return np.concatenate(B), np.concatenate(P), np.concatenate(D), np.concatenate(sid)


def ttl_metrics(model: Model, sess, tier: Tier, ttl: float, gpus: int,
                tpot: float = 0.03, eff: float = EFF):
    """Per-session time averages (per GPU). Multiply by concurrency for totals."""
    B, P, D, _ = flatten(model, sess, gpus, tpot)
    off = P > ttl
    stall = np.where(off, B / (tier.bw_gbs * 1e9 * eff), 0.0)
    T = (D + P + stall).sum()
    return dict(
        active_gb=(B * D).sum() / T / 1e9,
        parked_hbm_gb=(B * np.minimum(P, ttl)).sum() / T / 1e9,
        offloaded_gb=(B * np.maximum(P - ttl, 0)).sum() / T / 1e9,
        link_gbs=(2 * B * off).sum() / T / 1e9,
        stall_mean_ms=1e3 * stall[off].mean() if off.any() else 0.0,
        stall_p99_ms=1e3 * np.percentile(stall[off], 99) if off.any() else 0.0,
        offload_frac=off.mean(),
        stall_time_frac=stall.sum() / T,
    )


def timeline(model: Model, sess, conc: int, ttl: float, gpus: int,
             horizon: float = 4 * 3600, tpot: float = 0.03, seed: int = 0,
             tier: Tier | None = None, eff: float = EFF, dt: float = 1.0):
    """Closed-loop: `conc` session slots per replica run sessions back to back.
    Returns per-GPU byte time series (GB) sampled every dt seconds."""
    rng = np.random.default_rng(seed)
    nbin = int(horizon / dt) + 1
    act = np.zeros(nbin + 1); hbm = np.zeros(nbin + 1); offl = np.zeros(nbin + 1)
    pre = [(model.kv_bytes(s["ctx"]) / gpus, s["pause"], step_seconds(model, s, gpus, tpot)) for s in sess]

    def add(arr, t0, t1, v):
        i0, i1 = int(t0 / dt), int(t1 / dt)
        if i0 >= nbin:
            return
        arr[i0] += v; arr[min(i1, nbin)] -= v

    bw = (tier.bw_gbs * 1e9 * eff) if tier else np.inf
    for _ in range(conc):
        t = -rng.uniform(0, 3600)  # random phase, warm start
        while t < horizon:
            B, P, D = pre[rng.integers(len(pre))]
            for b, p, d in zip(B, P, D):
                if t + d > 0: add(act, max(t, 0), t + d, b)
                t += d
                h = min(p, ttl)
                if t + h > 0 and h > 0: add(hbm, max(t, 0), t + h, b)
                if p > ttl:
                    if t + p > 0: add(offl, max(t + h, 0), t + p, b)
                    t += p + b / bw
                else:
                    t += p
                if t >= horizon:
                    break
    f = lambda a: np.cumsum(a)[:nbin] / 1e9
    return f(act), f(hbm), f(offl)
