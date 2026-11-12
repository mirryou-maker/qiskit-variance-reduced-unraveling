"""Plan A: tune the continuation-table threshold abs_frac on the n=8 TUNING set (fastsim).

The value chosen here is frozen before the held-out n=20/24 timing runs (a_timing.py).
Ratio = Var_std / Var_table (same fastsim, independent seeds), bootstrap 95% CI.
Ledger-style CSV bench/results/a_tune.csv.
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import fastsim as F  # noqa: E402

FIELDS = ["family", "variant", "n", "L", "gt", "obs", "abs_frac", "reps", "shots", "var_std", "var",
          "ratio", "ratio_lo", "ratio_hi", "frac_table", "table_s", "host", "timestamp"]


def boot(xs, xq, B=500, seed=3):
    rng = np.random.default_rng(seed)
    r = [xs[rng.integers(0, len(xs), len(xs))].var(ddof=1)
         / max(xq[rng.integers(0, len(xq), len(xq))].var(ddof=1), 1e-300) for _ in range(B)]
    return np.percentile(r, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/a_tune.csv"))
    a = ap.parse_args()
    host = os.uname().nodename
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["family"], r["variant"], r["L"], r["gt"], r["obs"]))
    new = not os.path.exists(a.out)
    fo = open(a.out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    if new:
        w.writeheader()
    n = a.n
    sim = F.FastSim(n)
    for fam, var in (("ising", "h0.7"), ("ising", "h1.4"), ("brickwork", "random")):
        for L in (4, 8):
            for gt in (0.01, 0.03, 0.1, 0.3):
                st = (F.ising_circuit(n, L, gt, h=float(var[1:])) if fam == "ising"
                      else F.brickwork_circuit(n, L, gt))
                for o in ("x", "zz", "xc", "zc"):
                    if (fam, var, str(L), str(gt), o) in done:
                        continue
                    sd = int(hashlib.md5(repr((fam, var, L, gt, o)).encode()).hexdigest()[:8], 16)
                    xs = sim.run(st, o, a.shots, seed=sd)
                    vs = xs.var(ddof=1)
                    t0 = time.time()
                    _, s_ev, q_ev = sim.continuation_table(st, o, reps=a.reps, seed=sd + 1)
                    t_tab = time.time() - t0
                    ts = dt.datetime.now().isoformat(timespec="seconds")
                    line = []
                    for fr in a.fracs:
                        thr = fr * (s_ev.max() if len(s_ev) else 0.0)
                        tab = ((q_ev < (1 - 1e-3) * s_ev) & (s_ev - q_ev > thr)).astype(float)
                        x = sim.run(st, o, a.shots, table=tab, seed=sd + 2)
                        v = x.var(ddof=1)
                        ok = v > 1e-14 and vs > 1e-14
                        lo, hi = boot(xs, x) if ok else (np.nan, np.nan)
                        rat = vs / v if v > 1e-14 else (np.inf if vs > 1e-14 else np.nan)
                        w.writerow(dict(family=fam, variant=var, n=n, L=L, gt=gt, obs=o, abs_frac=fr,
                                        reps=a.reps, shots=a.shots, var_std=vs, var=v, ratio=rat,
                                        ratio_lo=lo, ratio_hi=hi, frac_table=tab.mean(),
                                        table_s=round(t_tab, 2), host=host, timestamp=ts))
                        line.append(f"{fr:g}:{rat:.2f}")
                    fo.flush()
                    print(f"{fam:9} {var:6} L={L} gt={gt:<5} {o:3} " + " ".join(line), flush=True)
    fo.close()


if __name__ == "__main__":
    main()
