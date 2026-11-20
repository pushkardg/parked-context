"""Download the public SWE-Bench Pro trajectories and reduce them on the fly.

No AWS CLI needed. Each .traj file (~17 MB on average) is downloaded, reduced to
per-step (ctx, out, new, pause) records, and discarded, so the full 12 GB never
sits on disk. Re-running resumes where it stopped.

  python scripts/fetch_and_reduce.py                      # Claude Sonnet 4.5 run
  python scripts/fetch_and_reduce.py --run gpt-5-250-turns-10132025
  python scripts/fetch_and_reduce.py --limit 20           # quick test
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import argparse
import gzip
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from kvpark.workload import load_traj

BUCKET = "https://scaleapi-results.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_traj(base, prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{base}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        keys += [c.find(NS + "Key").text for c in root.iter(NS + "Contents")
                 if c.find(NS + "Key").text.endswith(".traj")]
        t = root.find(NS + "NextContinuationToken")
        if t is None:
            return keys
        token = t.text


def fetch(url, dst, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dst, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="claude-45sonnet-10132025")
    ap.add_argument("--out", default=None, help="default: data/sessions_<run>.json.gz")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base", default=BUCKET)
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    out = a.out or f"data/sessions_{a.run}.json.gz"
    part = out + ".partial.jsonl"

    keys = list_traj(a.base, f"swe-bench-pro/{a.run}/")
    if a.limit:
        keys = keys[: a.limit]
    done = set()
    if os.path.exists(part):
        with open(part) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    print(f"{len(keys)} trajectories, {len(done)} already reduced")

    tmp = os.path.join(tempfile.gettempdir(), "kvpark_tmp.traj")
    with open(part, "a") as pf:
        for i, k in enumerate(keys, 1):
            if k in done:
                continue
            fetch(f"{a.base}/{urllib.parse.quote(k)}", tmp)
            s = load_traj(tmp)
            if s is not None:
                rec = {"key": k, **{n: [round(float(x), 3) for x in v] for n, v in s.items()}}
                pf.write(json.dumps(rec) + "\n"); pf.flush()
            if i % 25 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}")
    if os.path.exists(tmp):
        os.remove(tmp)

    with open(part) as f:
        sess = [{n: v for n, v in json.loads(l).items() if n != "key"} for l in f if l.strip()]
    with gzip.open(out, "wt") as f:
        json.dump(sess, f)
    print(f"{len(sess)} sessions, {sum(len(s['ctx']) for s in sess)} steps -> {out}")


if __name__ == "__main__":
    main()
