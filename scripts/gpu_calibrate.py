"""Calibrate the transfer model on a real GPU.

Measures host<->device copies of KV-sized buffers with pinned and pageable host
memory, alone and with several copies in flight, and compares them with the
model's prediction T = B / (eta * BW). Optionally times a prefill to check the
recompute model.

  pip install torch
  python scripts/gpu_calibrate.py                    # transfer benchmarks
  python scripts/gpu_calibrate.py --prefill Qwen/Qwen2.5-7B   # also time prefill (needs transformers)

Writes gpu_calibration.json. If the GPU is not an H100, treat the result as a
check of the model's structure (linear in bytes, queueing under concurrency),
not of H100 numbers.
"""
import argparse
import json
import platform
import time

import torch

SIZES_MB = [128, 512, 1024, 2048, 4096]


def timed(fn, reps=5):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal-gbs", type=float, default=64.0, help="nominal link GB/s (PCIe 5.0 x16 = 64)")
    ap.add_argument("--eta", type=float, default=0.8)
    ap.add_argument("--prefill", default=None, help="HF model id to time prefill at several context lengths")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = torch.device("cuda")
    out = dict(gpu=torch.cuda.get_device_name(0), host=platform.platform(), torch=torch.__version__,
               nominal_gbs=a.nominal_gbs, eta=a.eta, single=[], concurrent=[])

    for mb in SIZES_MB:
        n = mb * 2**20
        d = torch.empty(n, dtype=torch.uint8, device=dev)
        for pinned in (True, False):
            h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            h2d = timed(lambda: d.copy_(h, non_blocking=pinned))
            d2h = timed(lambda: h.copy_(d, non_blocking=pinned))
            pred = n / (a.eta * a.nominal_gbs * 1e9)
            out["single"].append(dict(mb=mb, pinned=pinned, h2d_ms=1e3 * h2d, d2h_ms=1e3 * d2h,
                                      h2d_gbs=n / h2d / 1e9, d2h_gbs=n / d2h / 1e9, model_ms=1e3 * pred))
            print(out["single"][-1], flush=True)
            del h
        del d

    # k simultaneous 1 GiB reloads on separate streams (one link): queueing check
    n = 2**30
    for k in (1, 2, 4, 8):
        hs = [torch.empty(n, dtype=torch.uint8, pin_memory=True) for _ in range(k)]
        ds = [torch.empty(n, dtype=torch.uint8, device=dev) for _ in range(k)]
        ss = [torch.cuda.Stream() for _ in range(k)]
        def go():
            for h, d, s in zip(hs, ds, ss):
                with torch.cuda.stream(s):
                    d.copy_(h, non_blocking=True)
        t = timed(go, reps=3)
        out["concurrent"].append(dict(k=k, total_ms=1e3 * t, per_copy_model_ms=1e3 * n / (a.eta * a.nominal_gbs * 1e9),
                                      model_serial_ms=1e3 * k * n / (a.eta * a.nominal_gbs * 1e9)))
        print(out["concurrent"][-1], flush=True)
        del hs, ds

    if a.prefill:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.prefill, torch_dtype=torch.bfloat16).to(dev).eval()
        out["prefill"] = []
        for c in (2048, 8192, 16384, 32768):
            ids = torch.randint(100, 1000, (1, c), device=dev)
            with torch.no_grad():
                t = timed(lambda: model(ids), reps=3)
            out["prefill"].append(dict(ctx=c, ms=1e3 * t))
            print(out["prefill"][-1], flush=True)

    with open("gpu_calibration.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote gpu_calibration.json")


if __name__ == "__main__":
    main()
