"""Agent-session workloads: real-trace ingest and a calibrated stand-in.

A session is a dict of equal-length numpy arrays, one entry per LLM step:
  ctx     tokens resident in KV at the end of the step (parked during the pause)
  pause   seconds the following tool call takes (KV is idle meanwhile)
  out     output tokens generated in the step
  new     tokens appended to the prompt before the step (prefill work)
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# 1) Real traces: SWE-agent .traj files (e.g. Scale AI SWE-Bench Pro runs,
#    s3://scaleapi-results/swe-bench-pro/<run>/, public, --no-sign-request)
# ---------------------------------------------------------------------------

try:  # optional, more accurate token counts
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def ntok(s: str) -> int:
        return len(_ENC.encode(s, disallowed_special=()))
except Exception:  # pragma: no cover
    def ntok(s: str) -> int:
        return max(1, len(s) // 4)


def _msg_text(m) -> str:
    if isinstance(m, str):
        return m
    c = m.get("content", "")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    extra = ""
    for tc in m.get("tool_calls") or []:
        extra += json.dumps(tc.get("function", tc))
    return (c or "") + extra


def load_traj(path: str) -> dict | None:
    with open(path) as f:
        d = json.load(f)
    steps = d.get("trajectory") or []
    if not steps:
        return None
    ctx, pause, out, new = [], [], [], []
    prev_prompt = 0
    for st in steps:
        q = st.get("query") or []
        prompt = sum(ntok(_msg_text(m)) for m in q) if isinstance(q, list) else ntok(str(q))
        resp = ntok((st.get("response") or "") + (st.get("action") or ""))
        ctx.append(prompt + resp)
        out.append(resp)
        new.append(max(prompt - prev_prompt, 0))
        prev_prompt = prompt + resp
        pause.append(float(st.get("execution_time") or 0.0))
    return {k: np.asarray(v, float) for k, v in
            dict(ctx=ctx, pause=pause, out=out, new=new).items()}


def load_sessions_json(path: str) -> list[dict]:
    """Load the gzipped output of reduce_traj.py."""
    import gzip
    with gzip.open(path, "rt") as f:
        return [{k: np.asarray(v, float) for k, v in s.items()} for s in json.load(f)]


def load_traj_dir(root: str, limit: int | None = None) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(root, "**", "*.traj"), recursive=True))
    if limit:
        paths = paths[:limit]
    sess = [s for s in (load_traj(p) for p in paths) if s is not None]
    return sess


# ---------------------------------------------------------------------------
# 2) Calibrated stand-in (DRAFT ONLY). Matches published aggregates of the
#    Scale SWE-Bench Pro Claude Sonnet 4.5 runs (nilenso analysis, 616 tasks):
#    mean 78 calls/task, mean 34,705 prompt tokens/call, mean tool time
#    265.7 s/task, 1,757 tool steps >10 s, mean per-task max step 123.5 s,
#    mean 17,705 output tokens/task. First-call context ~12.3k tokens
#    (Inferact Codex SWE-Bench Pro traces).
# ---------------------------------------------------------------------------
TARGET = dict(calls=78.0, tok_per_call=34705.0, tool_s_per_task=265.7,
              frac_gt10=1757 / (616 * 78.0), max_step=123.5,
              out_per_task=17705.0, ctx0=12300.0)
PAUSE_CAP = 1800.0  # SWE-agent command timeout scale


def _pause_stats(mu, sig, n_steps, rng, n_tasks=616):
    tot, gt10, mx = [], 0, []
    for n in n_steps[:n_tasks]:
        p = np.minimum(rng.lognormal(mu, sig, int(n)), PAUSE_CAP)
        tot.append(p.sum()); gt10 += (p > 10).sum(); mx.append(p.max())
    return np.mean(tot) / np.mean(n_steps[:n_tasks]), gt10 / n_steps[:n_tasks].sum(), np.mean(mx)


def fit_pause(seed=0):
    rng = np.random.default_rng(seed)
    n_steps = _steps(rng, 616)
    best = None
    t_mean = TARGET["tool_s_per_task"] / TARGET["calls"]
    for sig in np.linspace(1.0, 3.2, 45):
        for mu in np.linspace(-3.5, 1.5, 51):
            m, f, x = _pause_stats(mu, sig, n_steps, np.random.default_rng(1))
            err = (np.log(m / t_mean)) ** 2 + (np.log(f / TARGET["frac_gt10"])) ** 2 \
                + (np.log(x / TARGET["max_step"])) ** 2
            if best is None or err < best[0]:
                best = (err, mu, sig, m, f, x)
    return best


def _steps(rng, n):
    s = rng.lognormal(np.log(TARGET["calls"]) - 0.6 ** 2 / 2, 0.6, n)
    return np.clip(np.round(s), 5, 250)


def calibrated_sessions(n=2000, seed=0, mu=None, sig=None) -> list[dict]:
    rng = np.random.default_rng(seed)
    if mu is None:
        _, mu, sig, *_ = fit_pause(seed)
    n_steps = _steps(rng, n)
    # growth g so that call-weighted mean ctx matches tok_per_call
    c0 = TARGET["ctx0"]
    w = n_steps * (n_steps - 1) / 2
    g_mean = (TARGET["tok_per_call"] - c0) * n_steps.sum() / w.sum()
    out_mean = TARGET["out_per_task"] / TARGET["calls"]
    sess = []
    for n_i in n_steps.astype(int):
        g = rng.lognormal(np.log(g_mean) - 0.5 ** 2 / 2, 0.5)
        k = np.arange(n_i)
        ctx = c0 + g * k
        out = rng.lognormal(np.log(out_mean) - 1.0 / 2, 1.0, n_i)
        new = np.diff(np.concatenate([[0.0], ctx]))
        pause = np.minimum(rng.lognormal(mu, sig, n_i), PAUSE_CAP)
        pause[-1] = 0.0  # session ends after final step
        sess.append(dict(ctx=ctx, pause=pause, out=out, new=new))
    return sess


def summarize(sess: list[dict]) -> dict:
    calls = np.array([len(s["ctx"]) for s in sess])
    allp = np.concatenate([s["pause"][:-1] for s in sess])
    return dict(
        sessions=len(sess), mean_calls=calls.mean(),
        tok_per_call=np.concatenate([s["ctx"] for s in sess]).mean(),
        tool_s_per_task=np.mean([s["pause"].sum() for s in sess]),
        frac_gt10=(allp > 10).mean(), median_pause=np.median(allp),
        p99_pause=np.percentile(allp, 99),
        max_step=np.mean([s["pause"].max() for s in sess]),
    )
