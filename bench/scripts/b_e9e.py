"""E9e (docs/plan_B_characterization_2026-11-06.md, "E9e pre-registration"): re-time the E9b runs with the optimized
policy implementation (sparse_policy.IMPL = "v2", traj_kraus.run(opt2=True)). Same groups, seeds and chunk as E9b, so the
trajectories must equal the E9b ones; this is checked run by run against the E9b raw arrays, and variances are then taken
from E9b (analyze_e9e.py).

Rows (ledger CSV, resumable):
  kind=fixed   scheme = candidate label, obs = each TN observable, wall_s (run of N_t trajectories), match / maxdiff vs
               the first N_t E9b raw values of that observable
  kind=greedy  M, obs = policy observable, wall_s, bp_s (record_after + group precompute), nonstd, match / maxdiff
"""
import argparse
import csv
import datetime as dt
import importlib.util
import os
import socket
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402

_spec = importlib.util.spec_from_file_location("b_e9b", os.path.join(os.path.dirname(__file__), "b_e9b.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "scheme", "M", "shots", "wall_s", "bp_s",
          "nonstd", "match", "maxdiff", "seed", "chunk", "host", "gpu_other", "load1", "impl", "timestamp"]


def compare(x, ref):
    x = np.asarray(x, np.float64)[:len(ref)]
    ref = np.asarray(ref, np.float64)[:len(x)]
    d = np.abs(x - ref)
    return float(np.mean(d <= 1e-5 * np.maximum(1.0, np.abs(ref)))), float(d.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e9b_groups.csv"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_e9b_raw_all"), help="E9b raw arrays (all 37)")
    ap.add_argument("--Nt", type=int, default=296)
    ap.add_argument("--M", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--mem_gb", type=float, default=5.0, help="must equal E9b (sets the chunk)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e9e.csv"))
    a = ap.parse_args()
    S.IMPL = "v2"
    cfg = E.configs(a.groups)
    todo = E.partition(cfg, a.part, a.nparts, 2000, a.M)[:a.limit]
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
    print(f"E9e part {a.part}/{a.nparts}: {len(todo)} configs, N_t {a.Nt}", flush=True)

    # warm-up (kernel compilation outside every timed region)
    ops0 = K.build("brickwork", "s7", 20, 4, 0.01, "ad_phase")
    rec0, _, _ = S.record_after(ops0, 20, "zz", 100)
    K.run(ops0, 20, list(E.OBS), 4, ("greedy", None), seed=0, chunk=4, after=rec0, fast=True, opt2=True)
    K.run(ops0, 20, list(E.OBS), 4, ("fixed", 0), seed=0, chunk=4, fast=True, opt2=True)

    for k in todo:
        ch, fam, var, n, L, gt = k
        tn_obs = cfg[k]
        chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (8 * 16 * 2 ** n))))
        key = f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"
        raw = np.load(os.path.join(a.raw, key + ".npz"))
        ops = K.build(fam, var, n, L, gt, E.CHMAP[ch])
        labels = next(op[2].labels for op in ops if op[0] == "ch")
        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, chunk=chunk, impl="v2+opt2")
        t0 = time.time()

        def emit(rows):
            ts = dt.datetime.now().isoformat(timespec="seconds")
            for rr in rows:
                rr["timestamp"] = ts
                w.writerow(rr)
            f.flush()

        for c, lab in enumerate(labels):
            if all(k + ("fixed", lab, "", o) in done for o in tn_obs):
                continue
            b = dict(base, gpu_other=E.gpu_other(), load1=E.load1())
            sd = E.seed_of("e9b-fixed", *k, c)
            P.sync()
            t1 = time.perf_counter()
            r = K.run(ops, n, list(E.OBS), a.Nt, ("fixed", c), seed=sd, chunk=chunk, fast=True, opt2=True)
            P.sync()
            wt = time.perf_counter() - t1
            rows = []
            for o in tn_obs:
                mt, md = compare(r[o], raw[f"fixed_{lab}_{o}"])
                rows.append(dict(b, obs=o, kind="fixed", scheme=lab, M="", shots=a.Nt, seed=sd, wall_s=round(wt, 4),
                                 match=round(mt, 5), maxdiff=f"{md:.3g}"))
            emit(rows)
        for o in tn_obs:
            for m in a.M:
                if k + ("greedy", "greedy", str(m), o) in done:
                    continue
                b = dict(base, gpu_other=E.gpu_other(), load1=E.load1())
                sd = E.seed_of("e9b-greedy", *k, o, m)
                P.sync()
                t1 = time.perf_counter()
                rec, _, _ = S.record_after(ops, n, o, m)
                for ev in rec:
                    S._groups(ev, S.GMAX)
                P.sync()
                bp = time.perf_counter() - t1
                t1 = time.perf_counter()
                r = K.run(ops, n, list(E.OBS), a.Nt, ("greedy", None), seed=sd, chunk=chunk, after=rec, fast=True,
                          opt2=True)
                P.sync()
                wt = time.perf_counter() - t1
                mt, md = compare(r[o], raw[f"greedy_M{m}_{o}"])
                emit([dict(b, obs=o, kind="greedy", scheme="greedy", M=m, shots=a.Nt, seed=sd, wall_s=round(wt, 4),
                           bp_s=round(bp, 3), nonstd=round(float(1 - r["_choice"][:, 0].mean()), 5),
                           match=round(mt, 5), maxdiff=f"{md:.3g}")])
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} obs={','.join(tn_obs)} "
              f"({time.time() - t0:.0f}s) other={E.gpu_other()}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
