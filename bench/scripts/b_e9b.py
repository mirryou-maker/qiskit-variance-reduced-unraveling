"""E9b (docs/plan_B_characterization_2026-11-06.md): in the regime where truncated Pauli propagation does NOT converge
(E9a TN groups, n = 20), does a usable adaptive unraveling policy beat the best fixed unraveling in wall-clock?

Policy: greedy per-trajectory argmin of the increment variance (traj_kraus.run greedy) with O~ from SPARSE truncated
Pauli propagation at M terms (sparse_policy.record_after; truncation after every op), M in {10, 100}.
Channels: E9a deph -> deph_sweep candidates, ad -> ad_phase candidates (same physical channels).

Groups: bench/results/b_e9b_groups.csv (made by e9b_groups.py from b_e9a_summary.csv). Work unit = config
(channel, family, variant, L, gt); its TN observables are the policy targets.
Rows (ledger CSV, resumable):
  kind=fixed   scheme = candidate label, obs, value/var/shots, wall_s (whole run, all 4 observables)
  kind=greedy  M, obs = policy observable (statistics of that observable), wall_s (run), bp_s (record_after),
               nonstd = fraction of events x trajectories not using candidate 0
Raw per-trajectory arrays (float32) in bench/results/b_e9b_raw/<cfg>.npz for bootstrap CIs.
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402

OBS = ("x", "zz", "xc", "zc")
CHMAP = {"deph": "deph_sweep", "ad": "ad_phase"}
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "scheme", "M", "value", "var", "shots",
          "wall_s", "bp_s", "nonstd", "seed", "host", "gpu_other", "load1", "timestamp"]


def load1():
    return round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else ""


def seed_of(*k):
    return int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)


def gpu_other():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        return f"err:{e}"
    me = str(os.getpid())
    others = [l.strip() for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != me]
    return ";".join(others) if others else "none"


def configs(groups_csv):
    cfg = {}
    for r in csv.DictReader(open(groups_csv)):
        k = (r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]))
        cfg.setdefault(k, []).append(r["obs"])
    return cfg


def cost(k, obs, N, Ms):
    ncand = {"deph": 7, "ad": 5}[k[0]]
    return k[4] * N * (ncand + len(obs) * len(Ms) * 3.6)       # ~ L x (fixed runs + greedy at ~2-5x)


def partition(cfg, part, nparts, N, Ms):
    """Deterministic greedy bin packing by estimated cost (part 0 gets the larger share: weights)."""
    wts = [1.4, 1.0] if nparts == 2 else [1.0] * nparts
    load = [0.0] * nparts
    out = [[] for _ in range(nparts)]
    for k in sorted(cfg, key=lambda k: -cost(k, cfg[k], N, Ms)):
        j = min(range(nparts), key=lambda j: (load[j] + cost(k, cfg[k], N, Ms)) / wts[j])
        load[j] += cost(k, cfg[k], N, Ms)
        out[j].append(k)
    return sorted(out[part], key=lambda k: (k[0], k[1], k[2], k[4], k[5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e9b_groups.csv"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--M", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--mem_gb", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None, help="first k configs only (smoke)")
    ap.add_argument("--reverse", action="store_true",
                    help="process the part from the end (a second host taking over the tail of a slow host's part)")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e9b.csv"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_e9b_raw"))
    a = ap.parse_args()
    cfg = configs(a.groups)
    todo = partition(cfg, a.part, a.nparts, a.N, a.M)
    todo = (todo[::-1] if a.reverse else todo)[:a.limit]
    os.makedirs(a.raw, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]),
                      r["kind"], r["scheme"], r["M"], r["obs"]))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    host = socket.gethostname()
    print(f"part {a.part}/{a.nparts}: {len(todo)} configs", flush=True)

    for k in todo:
        ch, fam, var, n, L, gt = k
        tn_obs = cfg[k]
        chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (8 * 16 * 2 ** n))))
        key = f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"
        rawf = os.path.join(a.raw, key + ".npz")
        raw = dict(np.load(rawf)) if os.path.exists(rawf) else {}
        ops = K.build(fam, var, n, L, gt, CHMAP[ch])
        labels = next(op[2].labels for op in ops if op[0] == "ch")
        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host)
        t0 = time.time()

        def emit(rows):
            ts = dt.datetime.now().isoformat(timespec="seconds")
            for rr in rows:
                rr["timestamp"] = ts
                w.writerow(rr)
            f.flush()
            np.savez(rawf, **raw)

        for c, lab in enumerate(labels):
            if all(k + ("fixed", lab, "", o) in done for o in tn_obs):
                continue
            b = dict(base, gpu_other=gpu_other(), load1=load1())
            sd = seed_of("e9b-fixed", *k, c)
            P.sync()
            t1 = time.perf_counter()
            r = K.run(ops, n, list(OBS), a.N, ("fixed", c), seed=sd, chunk=chunk, fast=True)
            P.sync()
            wt = time.perf_counter() - t1
            rows = []
            for o in tn_obs:
                x = r[o]
                raw[f"fixed_{lab}_{o}"] = x.astype(np.float32)
                rows.append(dict(b, obs=o, kind="fixed", scheme=lab, M="", value=float(x.mean()),
                                 var=float(x.var(ddof=1)), shots=len(x), seed=sd, wall_s=round(wt, 3)))
            emit(rows)
        for o in tn_obs:
            for m in a.M:
                if k + ("greedy", "greedy", str(m), o) in done:
                    continue
                b = dict(base, gpu_other=gpu_other(), load1=load1())
                sd = seed_of("e9b-greedy", *k, o, m)
                P.sync()
                t1 = time.perf_counter()
                rec, _, _ = S.record_after(ops, n, o, m)
                P.sync()
                bp = time.perf_counter() - t1
                t1 = time.perf_counter()
                r = K.run(ops, n, list(OBS), a.N, ("greedy", None), seed=sd, chunk=chunk, after=rec, fast=True)
                P.sync()
                wt = time.perf_counter() - t1
                x = r[o]
                raw[f"greedy_M{m}_{o}"] = x.astype(np.float32)
                emit([dict(b, obs=o, kind="greedy", scheme="greedy", M=m, value=float(x.mean()),
                           var=float(x.var(ddof=1)), shots=len(x), seed=sd, wall_s=round(wt, 3),
                           bp_s=round(bp, 3), nonstd=round(float(1 - r["_choice"][:, 0].mean()), 5))])
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} obs={','.join(tn_obs)} chunk={chunk} "
              f"({time.time() - t0:.0f}s) other={gpu_other()}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
