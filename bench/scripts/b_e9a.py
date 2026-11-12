"""E9a (docs/plan_B_characterization_2026-11-06.md): does truncated Pauli propagation converge at affordable M
on larger / deeper circuits, or is there a regime that needs trajectories?

Rows (ledger CSV, resumable):
  kind=pp     value = <0|O~_0^M|0> from sparse truncated PP, peak = max term count, wall_s = PP time (GPU synced)
  kind=traj   value = mean, var = per-trajectory variance of the std unraveling, shots = N_ref, wall_s = time of the
              whole run (all 4 observables at once)
  kind=exact  dense answer (n <= 12 only)
gpu_other = compute processes on the GPU other than this one at the start of the config (idle check).
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
from src.noise import traj_kraus as K  # noqa: E402

VARIANTS = [("brickwork", "s7"), ("brickwork", "s8"), ("ising", "h0.7"), ("ising", "h1.4")]
OBS = ("x", "zz", "xc", "zc")
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "M", "value", "var", "shots", "peak",
          "wall_s", "seed", "host", "gpu_other", "load1", "timestamp"]
NREF = {12: 2000, 20: 1000, 24: 64}   # n = 24 reduced from 256 after the smoke (plan doc)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", nargs="+", default=["deph", "ad"])
    ap.add_argument("--n", type=int, nargs="+", default=[12, 20, 24])
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--gt", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--M", type=int, nargs="+", default=[100, 1000, 10000, 100000, 1000000])
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--nref", type=int, default=None, help="override N_ref (smoke only)")
    ap.add_argument("--mem_gb", type=float, default=6.0, help="GPU memory budget for trajectory chunks")
    ap.add_argument("--skip_traj", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e9a.csv"))
    a = ap.parse_args()
    variants = [v for v in VARIANTS if a.families is None or v[0] in a.families]
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]), r["obs"],
                      r["kind"], r["M"]))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    host = socket.gethostname()

    for n in a.n:
        nref = a.nref or NREF.get(n, 1000)
        chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (6 * 16 * 2 ** n))))
        for ch in a.channel:
            for fam, var in variants:
                for L in a.L:
                    for gt in a.gt:
                        cfg = (ch, fam, var, n, L, gt)
                        want = [(o, "pp", str(m)) for o in OBS for m in a.M]
                        if not a.skip_traj:
                            want += [(o, "traj", "") for o in OBS]
                        if n <= 12:
                            want += [(o, "exact", "") for o in OBS]
                        if all(cfg + x in done for x in want):
                            continue
                        key = f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"
                        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host,
                                    gpu_other=gpu_other(), load1=round(os.getloadavg()[0], 2))
                        ops = K.build(fam, var, n, L, gt, ch)
                        t0 = time.time()
                        rows = []
                        if n <= 12 and not all(cfg + (o, "exact", "") in done for o in OBS):
                            ex = K.dm_means_fast(ops, n, list(OBS))     # == exact_mean (validation/test_pauli_prop.py)
                            for o in OBS:
                                rows.append(dict(base, obs=o, kind="exact", M="", value=ex[o]))
                        tables = {}
                        msg = []
                        for o in OBS:
                            for m in a.M:
                                if cfg + (o, "pp", str(m)) in done:
                                    continue
                                P.sync()
                                t1 = time.perf_counter()
                                v, peak = P.propagate(ops, n, o, m, tables)
                                P.sync()
                                rows.append(dict(base, obs=o, kind="pp", M=m, value=v, peak=peak,
                                                 wall_s=round(time.perf_counter() - t1, 4)))
                            msg.append(o)
                        if not a.skip_traj and not all(cfg + (o, "traj", "") in done for o in OBS):
                            sd = seed_of("e9a", *cfg)
                            P.sync()
                            t1 = time.perf_counter()
                            r = K.run(ops, n, list(OBS), nref, ("fixed", 0), seed=sd, chunk=chunk, fast=True)
                            P.sync()
                            wt = time.perf_counter() - t1
                            for o in OBS:
                                x = r[o]
                                rows.append(dict(base, obs=o, kind="traj", M="", value=float(x.mean()),
                                                 var=float(x.var(ddof=1)), shots=len(x), seed=sd,
                                                 wall_s=round(wt, 3)))
                        ts = dt.datetime.now().isoformat(timespec="seconds")
                        for rr in rows:
                            rr["timestamp"] = ts
                            w.writerow(rr)
                        f.flush()
                        print(f"{ts} {key:40} rows={len(rows)} chunk={chunk} ({time.time() - t0:.0f}s) "
                              f"other={base['gpu_other']}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
