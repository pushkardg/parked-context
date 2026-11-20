# parked-context

Memory-tier capacity and bandwidth analysis for **paused KV caches in agentic LLM serving**.

When an LLM agent calls a tool, its KV cache sits idle until the tool returns. This
repository models where that idle KV cache should live (HBM, host DRAM, CXL memory,
Grace LPDDR5X over NVLink-C2C, or NVMe), and what each choice costs in capacity, link
bandwidth and resume latency. The models covered are seven open LLMs with different
attention designs (GQA, MLA, sliding-window, MoE).

## What it computes

| Output | Description |
|---|---|
| KV geometry | Bytes per token and per context for each model, including MLA latents and sliding-window layers |
| Reload vs. recompute | Time to reload from each tier vs. re-running prefill, and the break-even link bandwidth (shard-aware) |
| Idle-KV concentration | How idle KV byte-seconds are spread across tool pauses |
| TTL sweep | For a time-to-live policy: idle KV left in HBM vs. moved off it, link traffic, and resume stalls |
| Provisioning | Sessions per replica, p99 off-HBM capacity per GPU, and p99 resume stall per model and tier |

## Layout

```
kvpark/
  hw.py          model configs, memory tiers, reload/recompute and break-even models
  workload.py    SWE-agent .traj parser, reduced-trace loader, calibrated synthetic workload
  sim.py         TTL park-or-offload replay and closed-loop occupancy timeline
scripts/
  run_all.py     produces all figures (PDF), table_models.tex and results.json
  reduce_traj.py shrinks raw .traj files to compact per-step records (.json.gz)
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_all.py --out out          # calibrated synthetic workload
```

The synthetic workload is only for development. Its tool-pause distribution is fitted
to published aggregate statistics, not replayed from per-step traces.

## Running on real agent traces

The SWE-Bench Pro trajectories released by Scale AI record the tool execution time of
every step:

```bash
aws s3 cp s3://scaleapi-results/swe-bench-pro/claude-45sonnet-10132025/ data/claude45/ \
    --recursive --no-sign-request --exclude "*" --include "*.traj"
python scripts/reduce_traj.py data/claude45 data/sessions_claude45.json.gz
python scripts/run_all.py --traj data/sessions_claude45.json.gz --out out
```

`run_all.py` also accepts a directory of `.traj` files directly (`--traj data/claude45`).

## Assumptions to check

- **Model configs** in `kvpark/hw.py` (layers, KV heads, head dim, MLA rank, window,
  global layers) follow each model's public `config.json`.
- **Deployments** (`DEPLOY` in `scripts/run_all.py`): H100-80GB GPUs per replica and weight sizes.
- **KV placement**: GQA caches split across `min(GPUs, KV heads)` shards under tensor
  parallelism (replicated beyond that); MLA caches use data-parallel attention, so a
  session's KV lives on one GPU and reloads over one link.
- **Tier bandwidths** are nominal per-direction figures, scaled by `EFF = 0.8`.
- **Recompute** assumes 989 TFLOP/s BF16 dense at 50% utilization; decode uses a fixed TPOT of 30 ms.

## License

MIT
