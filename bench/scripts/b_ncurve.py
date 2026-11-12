"""E1 (docs/plan_B_characterization_2026-11-06.md): adaptive-vs-best-fixed gain as a function of n.

Pre-registered ensemble: brickwork random (instances s7, s8, s9), Ising h0.7 / h1.4;
L in {4, 8}; gamma*t per layer in {0.01, 0.03, 0.1, 0.3}; observables x, zz, xc, zc; n in {6, 8, 10, 12}.
Schemes per group: std (lam=0), proj (lam=1) -- shared across observables --, greedy (per observable,
local rule with the dense back-propagated observable, local_bp), PI (n = 6 only).

Variance only (no timing claims). Ledger CSV: rows already present are skipped; raw per-trajectory
estimates (float32) go to bench/results/b_ncurve_raw/<circuit>.npz for bootstrap CIs in the analysis.
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import socket
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import traj_ref as T  # noqa: E402

VARIANTS = [("brickwork", "s7"), ("brickwork", "s8"), ("brickwork", "s9"),
            ("ising", "h0.7"), ("ising", "h1.4")]
OBS = ("x", "zz", "xc", "zc")
LS = (4, 8)
GTS = (0.01, 0.03, 0.1, 0.3)
FIELDS = ["family", "variant", "n", "L", "gt", "obs", "scheme", "shots", "exact", "mean", "z",
          "var", "frac_proj", "seed", "wall_s", "host", "timestamp"]


def build(fam, var, n, L, gt):
    if fam == "ising":
        return T.ising_ops(n, L, gt, h=float(var[1:]))
    return T.brickwork_ops(n, L, gt, seed=int(var[1:]))


def seed_of(*k):
    return int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--L", type=int, nargs="+", default=list(LS))
    ap.add_argument("--gt", type=float, nargs="+", default=list(GTS))
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_ncurve.csv"))
    a = ap.parse_args()
    rawdir = os.path.join(ROOT, "bench/results/b_ncurve_raw")
    os.makedirs(rawdir, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]),
                      r["obs"], r["scheme"]))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    host = socket.gethostname()

    for n in a.n:
        for fam, var in VARIANTS:
            for L in a.L:
                for gt in a.gt:
                    schemes = ["std", "proj", "greedy"] + (["pi"] if n <= 6 else [])
                    if all((fam, var, n, L, gt, o, s) in done for o in OBS for s in schemes):
                        continue
                    key = f"{fam}_{var}_n{n}_L{L}_gt{gt}"
                    ops = build(fam, var, n, L, gt)
                    sd = seed_of(fam, var, n, L, gt)
                    t0 = time.time()
                    exact = {o: T.exact_mean(ops, n, o, local_bp=True) for o in OBS}
                    base = T.run(ops, n, list(OBS), a.shots, ("lam", 0.0), seed=sd, chunk=a.chunk)
                    t_std = time.time() - t0
                    t1 = time.time()
                    prj = T.run(ops, n, list(OBS), a.shots, ("lam", 1.0), seed=sd + 1, chunk=a.chunk)
                    t_prj = time.time() - t1
                    raw = {}
                    rows = []

                    def row(o, s, x, fr, seed, wall):
                        v = float(x.var(ddof=1))
                        raw[f"{o}_{s}"] = x.astype(np.float32)
                        return dict(family=fam, variant=var, n=n, L=L, gt=gt, obs=o, scheme=s,
                                    shots=len(x), exact=exact[o], mean=float(x.mean()),
                                    z=(float(x.mean()) - exact[o]) / np.sqrt(max(v, 1e-300) / len(x)),
                                    var=v, frac_proj=fr, seed=seed, wall_s=round(wall, 2), host=host)

                    for o in OBS:
                        rows.append(row(o, "std", base[o], 0.0, sd, t_std))
                        rows.append(row(o, "proj", prj[o], 1.0, sd + 1, t_prj))
                        t2 = time.time()
                        gr = T.run(ops, n, [o], a.shots, ("greedy", None), seed=sd + 2,
                                   obs_for_policy=o, local_bp=True, chunk=a.chunk)
                        rows.append(row(o, "greedy", gr[o], float(np.mean(gr["_frac_proj"])),
                                        sd + 2, time.time() - t2))
                        if "pi" in schemes:
                            t3 = time.time()
                            # host-stored two-copy operators: L=8 at n=6 is 48 x 128 MB
                            px, pf = T.run_pi(ops, n, o, a.shots, seed=sd + 3, chunk=5000,
                                              host=(L > 4))
                            rows.append(row(o, "pi", px, float(np.mean(pf)), sd + 3, time.time() - t3))
                    np.savez_compressed(os.path.join(rawdir, key + ".npz"), **raw)
                    ts = dt.datetime.now().isoformat(timespec="seconds")
                    for r in rows:
                        r["timestamp"] = ts
                        w.writerow(r)
                    f.flush()
                    vs = {(r["obs"], r["scheme"]): r["var"] for r in rows}
                    msg = " ".join(
                        f"{o}:G={min(vs[(o, 'std')], vs[(o, 'proj')]) / max(vs[(o, 'greedy')], 1e-300):.2f}"
                        for o in OBS)
                    print(f"{ts} {key:32} {msg} ({time.time() - t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
