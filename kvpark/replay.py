"""Event-driven, contention-aware replay of agent sessions on one serving replica.

Compared with the time-averaged model in sim.py, this replay
  * queues transfers on each GPU link, with host-to-device (reload) and
    device-to-host (offload) as separate FIFO queues, so simultaneous resumes
    wait for one another;
  * frees HBM only when an offload completes, and cancels an offload whose
    tool returns first (the cache then never left HBM);
  * reloads every replica of a replicated GQA shard over its own GPU's link;
  * places MLA sessions on one GPU each (data-parallel attention) and tracks
    every GPU's HBM separately;
  * records HBM and off-HBM occupancy exactly, as step functions changed at
    every allocation and release, so maxima and time-weighted percentiles have
    no sampling error.

Two engines share this interface:
  replay(...)        transfer completion is computed when a transfer starts;
                     an optional shared pool is served in order (pool="fifo").
  replay(..., pool_mode="fluid")
                     transfers are active concurrently and share the pool's
                     bandwidth max-min fairly (work-conserving), each capped at
                     its link rate; completions are rescheduled whenever the set
                     of active transfers changes.
With no pool, both engines give the same result.
"""
from __future__ import annotations
import heapq
from collections import deque
import numpy as np
from .hw import Model, H100, GPU, EFF
from .sim import step_seconds


def _prep(model: Model, sess, gpus, tpot):
    return [(model.kv_bytes(s["ctx"]), s["pause"].copy(), step_seconds(model, s, gpus, tpot)) for s in sess]


# ---------------------------------------------------------------------------
# exact occupancy from (time, delta) events
# ---------------------------------------------------------------------------
def _step_series(events, horizon):
    """events: list of (t, delta). Returns (times, values) of the step function
    on [0, horizon): value[i] holds on [times[i], times[i+1])."""
    if not events:
        return np.array([0.0]), np.array([0.0])
    a = np.asarray(events, dtype=float)
    order = np.lexsort((a[:, 1], a[:, 0]))       # time, then releases before allocations
    t, d = a[order, 0], a[order, 1]
    v = np.cumsum(d)
    keep = np.append(t[1:] != t[:-1], True)      # last value at each distinct time
    t, v = t[keep], v[keep]
    if t[0] > 0:
        t = np.concatenate([[0.0], t]); v = np.concatenate([[0.0], v])
    return t, np.maximum(v, 0.0)


def _occupancy(ev, horizon):
    """ev: list per GPU of event lists. Returns union times, per-time maximum
    over GPUs, durations, and the per-GPU time-weighted mean."""
    series = [_step_series(e, horizon) for e in ev]
    T = np.unique(np.concatenate([s[0] for s in series]))
    T = T[T < horizon]
    vals = np.empty((len(series), len(T)))
    for i, (t, v) in enumerate(series):
        idx = np.searchsorted(t, T, side="right") - 1
        vals[i] = np.where(idx >= 0, v[np.maximum(idx, 0)], 0.0)
    dur = np.diff(np.append(T, horizon))
    mx = vals.max(axis=0)
    mean_per_gpu = float((vals * dur).sum() / (horizon * len(series)))
    return mx, dur, mean_per_gpu


def _wquantile(x, w, q):
    if q >= 100:
        return float(x.max())
    o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(cw, q / 100.0), len(x) - 1)])


# ---------------------------------------------------------------------------
def replay(model: Model, sess, conc: int, ttl: float, gpus: int, bw_gbs: float,
           eff: float = EFF, tpot: float = 0.03, horizon: float = 2 * 3600,
           warm: float = 1800.0, seed: int = 0, dt: float = 0.1, prep=None,
           pool_gbs: float | None = None, budget_gb: float | None = None,
           pool_reads_only: bool = False, occ_q: float = 99.0, pool_mode: str = "fifo",
           preempt_inflight: bool = True):
    """pool_gbs: effective bandwidth of the memory pool shared by all links of
    the replica (host DRAM or a CXL expander); None means unlimited.
    pool_mode "fifo": a transfer starts when its link is free and the pool has
    served all earlier bytes, and lasts max(link time, pool time).
    pool_mode "fluid": active transfers share the pool max-min fairly.
    pool_reads_only: only reloads (host reads) use the pool.
    preempt_inflight: if the tool returns while an offload is copying, abort
    the copy and free the link at once (True), or let the copy run to
    completion and discard it, keeping the link and pool busy (False). Either
    way the cache never left HBM and the session resumes immediately. Queued
    offloads that have not started are always dropped. (In-order engine only.)
    occ_q: occupancy percentile reported as hbm_q_gb (100 = exact maximum).
    budget_gb: per-GPU KV budget, used only to report overflow statistics;
    the replay itself does not block or evict (memory may be overcommitted)."""
    rng = np.random.default_rng(seed)
    pre = prep if prep is not None else _prep(model, sess, gpus, tpot)
    S = model.kv_shards(gpus)
    R = model.kv_replication(gpus)
    dp = bool(model.mla_dim)            # data-parallel attention: one GPU per session
    nq = gpus if dp else 1              # GQA: all links move equal shards in lockstep
    bw = bw_gbs * 1e9 * eff
    nb = int(horizon / dt) + 2
    ev = [[] for _ in range(nq)]        # per-GPU HBM (time, delta) events
    off_ev = []                         # off-HBM bytes, one copy, whole replica
    busy_h2d = np.zeros((nq, nb))
    pbw = pool_gbs * 1e9 if pool_gbs else None
    # pool bytes per link byte: every replica re-reads its shard on reload, but
    # only one copy of each unique shard is written on offload and stored
    pool_mult = 1.0 if dp else gpus
    write_mult = 1.0 if dp else S
    fluid = pool_mode == "fluid"

    def per_gpu(B):
        return B if dp else B * R / gpus

    def hold(lst, t0, t1, v):
        if t1 <= 0 or t1 <= t0 or t0 >= horizon:
            return
        lst.append((max(t0, 0.0), v)); lst.append((min(t1, horizon), -v))

    def busy(q, t0, t1):
        t0 = max(t0, 0.0); t1 = min(t1, horizon)
        i = int(t0 / dt)
        while t0 < t1 and i < nb:
            e = min((i + 1) * dt, t1)
            if e > t0:
                busy_h2d[q, i] += e - t0; t0 = e
            i += 1

    st = dict(stalls=[], waits=[], resume=0, offload=0, cancel=0, returns=0, wasted=0,
              idle_tot=0.0, idle_held=0.0, idle_ideal=0.0)

    def idle(t_end, p, held, v):
        if t_end >= 0:
            st["idle_tot"] += v * p; st["idle_held"] += v * held
            st["idle_ideal"] += v * min(p, ttl)

    heap, seq = [], [0]
    def push(t, c, kind, info=None):
        heapq.heappush(heap, (t, seq[0], c, kind, info)); seq[0] += 1

    slot = []
    for c in range(conc):
        slot.append([int(rng.integers(len(pre))), 0, c % nq])
        push(-rng.uniform(0, warm), c, "start")

    # ---------------- analytic engine (link FIFO, optional FIFO pool) --------
    d2h_free = np.full(nq, -warm - 1.0); h2d_free = np.full(nq, -warm - 1.0)
    poolf = [-warm - 1.0]

    def transfer(free_arr, q, t, L, read=True):
        start = max(t, free_arr[q]); dur = L / bw
        if pbw and (read or not pool_reads_only):
            start = max(start, poolf[0]); pdur = L * (pool_mult if read else write_mult) / pbw
            poolf[0] = start + pdur; dur = max(dur, pdur)
        return start, start + dur

    # ---------------- fluid engine state ------------------------------------
    links = {}                          # (q, dir) -> {"active": tid|None, "wait": deque}
    xfer = {}                           # tid -> transfer dict
    active = set()
    clock = [None]; ver = [0]; tid_n = [0]

    def lk(q, dr):
        return links.setdefault((q, dr), {"active": None, "wait": deque()})

    def rates():
        r = {}
        on_pool = []
        for i in active:
            x = xfer[i]
            if pbw and x["mult"] > 0:
                on_pool.append(i)
            else:
                r[i] = bw
        if on_pool:                     # max-min fair share of pool bytes/s
            cap = {i: bw * xfer[i]["mult"] for i in on_pool}
            left, rem = pbw, sorted(on_pool, key=lambda i: cap[i])
            while rem:
                share = left / len(rem)
                i = rem[0]
                if cap[i] <= share:
                    r[i] = bw; left -= cap[i]; rem.pop(0)
                else:
                    for j in rem:
                        r[j] = share / xfer[j]["mult"]
                    rem = []
        return r

    def advance(t):
        if clock[0] is not None and t > clock[0]:
            r = rates(); dtt = t - clock[0]
            for i in active:
                xfer[i]["L"] -= r[i] * dtt
        clock[0] = t

    def reschedule(t):
        ver[0] += 1
        if active:
            r = rates()
            tn = min(t + max(xfer[i]["L"] / r[i], 1e-6) for i in active)
            push(tn, -1, "xdone", ver[0])

    def activate(i, t):
        x = xfer[i]; x["start"] = t; active.add(i); lk(x["q"], x["dir"])["active"] = i

    def request(t, c, q, dr, L, mult, info):
        advance(t)
        i = tid_n[0]; tid_n[0] += 1
        xfer[i] = dict(c=c, q=q, dir=dr, L=L, mult=mult, req=t, start=None, info=info)
        l = lk(q, dr)
        if l["active"] is None:
            activate(i, t)
        else:
            l["wait"].append(i)
        reschedule(t)
        return i

    def release_link(q, dr, t):
        l = lk(q, dr); l["active"] = None
        if l["wait"]:
            activate(l["wait"].popleft(), t)

    def cancel(i, t):
        advance(t)
        x = xfer[i]; l = lk(x["q"], x["dir"])
        if i in active:
            active.discard(i); release_link(x["q"], x["dir"], t)
        elif i in l["wait"]:
            l["wait"].remove(i)
        reschedule(t)

    pending = {}                        # slot -> dict for an in-progress pause

    # ---------------- main loop ---------------------------------------------
    while heap:
        t, _, c, kind, info = heapq.heappop(heap)
        if t >= horizon:
            continue
        if kind == "xdone":
            if info != ver[0]:
                continue
            advance(t)
            rr = rates()
            done = [i for i in active if xfer[i]["L"] <= rr[i] * 1e-6]   # within 1 us
            for i in done:
                x = xfer[i]; active.discard(i); release_link(x["q"], x["dir"], t)
                cc = x["c"]; si, k, q = slot[cc]; B, P, D = pre[si]
                pz = pending[cc]
                if x["dir"] == "d2h":
                    pz["fin"] = t; pz["hstart"] = x["start"]; st["offload"] += 1
                else:                       # reload complete
                    tr = pz["tr"]
                    busy(q, x["start"], t)
                    hold(off_ev, pz["hstart"], t, B[k])     # host buffer reserved at copy start
                    hold(ev[q], x["start"], t, per_gpu(B[k]))
                    if tr >= 0:
                        st["resume"] += 1; st["stalls"].append(t - tr); st["waits"].append(x["start"] - tr)
                    del pending[cc]; slot[cc][1] = k + 1
                    push(t, cc, "start")
            reschedule(t)
            continue

        si, k, q = slot[c]
        B, P, D = pre[si]
        if kind == "start":
            d = D[k]
            hold(ev[q], t, t + d, per_gpu(B[k]))
            push(t + d, c, "end")
        elif kind == "end":
            p = P[k]
            last = k == len(B) - 1
            if not last and p > 0 and t + p >= 0:
                st["returns"] += 1
            if last or p <= 0:
                if last:
                    slot[c][0] = int(rng.integers(len(pre))); slot[c][1] = 0
                else:
                    slot[c][1] = k + 1
                push(t + (0 if last else p), c, "start")
            elif p <= ttl:
                hold(ev[q], t, t + p, per_gpu(B[k])); idle(t, p, p, per_gpu(B[k]))
                slot[c][1] = k + 1
                push(t + p, c, "start")
            else:
                push(t + ttl, c, "offload", (t, t + p))
                if fluid:
                    push(t + p, c, "toolret", (t, t + p))
        elif kind == "offload":
            t_end, tr = info
            L = B[k] / S if not dp else B[k]
            v = per_gpu(B[k])
            if fluid:
                mult = write_mult if (pbw and not pool_reads_only) else 0.0
                i = request(t, c, q, "d2h", L, mult, None)
                pending[c] = dict(t_end=t_end, tr=tr, off=i, fin=None)
                continue
            start = max(t, d2h_free[q], poolf[0] if (pbw and not pool_reads_only) else -1e18)
            if start >= tr:                       # never started: cancel
                st["cancel"] += 1
                hold(ev[q], t_end, tr, v); idle(t_end, tr - t_end, tr - t_end, v)
                slot[c][1] = k + 1; push(tr, c, "start"); continue
            start, fin = transfer(d2h_free, q, t, L, read=False)
            if fin > tr:                          # tool returned first: abort
                st["cancel"] += 1
                if preempt_inflight:
                    d2h_free[q] = tr
                    if pbw and not pool_reads_only:
                        poolf[0] = min(poolf[0], tr)
                else:                             # copy completes and is discarded
                    d2h_free[q] = fin; st["wasted"] += 1
                hold(off_ev, start, tr if preempt_inflight else fin, B[k])
                hold(ev[q], t_end, tr, v); idle(t_end, tr - t_end, tr - t_end, v)
                slot[c][1] = k + 1; push(tr, c, "start"); continue
            st["offload"] += 1
            d2h_free[q] = fin
            hold(ev[q], t_end, fin, v); idle(t_end, tr - t_end, fin - t_end, v)
            push(tr, c, "reload", start)
        elif kind == "toolret":                   # fluid engine only
            t_end, tr = info
            pz = pending[c]; v = per_gpu(B[k])
            if pz["fin"] is None:                 # offload not finished: cancel
                xs = xfer[pz["off"]]["start"]
                if xs is not None:                # host buffer held while it copied
                    hold(off_ev, xs, t, B[k])
                cancel(pz["off"], t); st["cancel"] += 1
                hold(ev[q], t_end, tr, v); idle(t_end, tr - t_end, tr - t_end, v)
                del pending[c]; slot[c][1] = k + 1; push(tr, c, "start")
            else:
                hold(ev[q], t_end, pz["fin"], v); idle(t_end, tr - t_end, pz["fin"] - t_end, v)
                L = B[k] / S if not dp else B[k]
                request(t, c, q, "h2d", L, pool_mult if pbw else 0.0, None)
        elif kind == "reload":                    # analytic engine only
            host_start = info                     # host buffer reserved when the offload began
            L = B[k] / S if not dp else B[k]
            rs, rf = transfer(h2d_free, q, t, L)
            h2d_free[q] = rf
            busy(q, rs, rf)
            hold(off_ev, host_start, rf, B[k])
            hold(ev[q], rs, rf, per_gpu(B[k]))
            if t >= 0:
                st["resume"] += 1; st["stalls"].append(rf - t); st["waits"].append(rs - t)
            slot[c][1] = k + 1
            push(rf, c, "start")

    mx, dur, mean_gpu = _occupancy(ev, horizon)
    mx = mx / 1e9
    ot, ov = _step_series(off_ev, horizon)
    od = np.diff(np.append(ot[ot < horizon], horizon)); ov = ov[ot < horizon] / 1e9 / gpus
    util = busy_h2d[:, :-1] / dt
    s_ = np.array(st["stalls"]) if st["stalls"] else np.zeros(1)
    w_ = np.array(st["waits"]) if st["waits"] else np.zeros(1)
    nret = max(st["returns"], 1)
    over = float(dur[mx > budget_gb].sum() / horizon) if budget_gb is not None else None
    return dict(
        hbm_p99_gb=_wquantile(mx, dur, 99), hbm_mean_gb=mean_gpu / 1e9,
        hbm_max_gb=float(mx.max()), frac_time_over_budget=over,
        hbm_q_gb=_wquantile(mx, dur, occ_q),
        off_p99_gb=_wquantile(ov, od, 99), off_mean_gb=float((ov * od).sum() / horizon),
        stall_p50_ms=1e3 * float(np.percentile(s_, 50)), stall_p99_ms=1e3 * float(np.percentile(s_, 99)),
        wait_p99_ms=1e3 * float(np.percentile(w_, 99)),
        frac_stall_gt100=float((s_ > 0.1).mean()) if st["stalls"] else 0.0,
        frac_returns_gt100=float((s_ > 0.1).sum()) / nret if st["stalls"] else 0.0,
        reload_frac=st["resume"] / nret,
        util_mean=float(util.mean()), util_p99=float(np.percentile(util.max(axis=0), 99)),
        util_p999=float(np.percentile(util.max(axis=0), 99.9)),
        idle_freed_actual=1 - st["idle_held"] / st["idle_tot"] if st["idle_tot"] else 0.0,
        idle_freed_ideal=1 - st["idle_ideal"] / st["idle_tot"] if st["idle_tot"] else 0.0,
        resumes=st["resume"], offloads=st["offload"], cancelled=st["cancel"], returns=st["returns"],
        wasted_copies=st["wasted"],
    )


def safe_concurrency(model: Model, sess, ttl, gpus, weight_gb, bw_gbs, gpu: GPU = H100,
                     kv_frac=0.9, prep=None, q=99.0, mean=False, **kw):
    """Largest C whose per-GPU HBM occupancy at time-weighted percentile q (or
    its mean, or its exact maximum when q=100) fits the per-GPU KV budget."""
    budget = gpu.hbm_gb * kv_frac - weight_gb / gpus
    if budget <= 0:
        return 0, budget
    pre = prep if prep is not None else _prep(model, sess, gpus, kw.get("tpot", 0.03))
    def fits(c):
        r = replay(model, sess, c, ttl, gpus, bw_gbs, prep=pre, occ_q=q, **kw)
        return (r["hbm_mean_gb"] if mean else r["hbm_q_gb"]) <= budget
    lo, hi = 0, 1
    while fits(hi):
        lo, hi = hi, hi * 2
        if hi > 4096:
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo, budget
