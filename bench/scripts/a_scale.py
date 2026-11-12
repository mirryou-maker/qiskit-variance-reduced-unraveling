"""Plan A: variance gain of the continuation table vs n (fastsim, variance only, no timing claims).

Ising h0.7 / brickwork random, L=8, several gt and observables, n = 8..20, abs_frac grid.
Ledger-style CSV bench/results/a_scale.csv.
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import socket
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import fastsim as F  # noqa: E402

FIELDS = ["family", "variant", "n", "L", "gt", "obs", "abs_frac", "reps", "shots", "var_std", "var",
          "ratio", "ratio_lo", "ratio_hi", "frac_table", "host", "timestamp"]


def boot(xs, xq, B=500, seed=3):
    rng = np.random.default_rng(seed)
    r = [xs[rng.integers(0, len(xs), len(xs))].var(ddof=1)
         / max(xq[rng.integers(0, len(xq), len(xq))].var(ddof=1), 1e-300) for _ in range(B)]
    return np.percentile(r, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--families", nargs="+", default=["ising", "brickwork"])
    ap.add_argument("--gt", type=float, nargs="+", default=[0.03, 0.1])
    ap.add_argument("--obs", nargs="+", default=["x", "zz", "xc", "zc"])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/a_scale.csv"))
    a = ap.parse_args()
    host = socket.gethostname()
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["family"], r["n"], r["gt"], r["obs"]))
    new = not os.path.exists(a.out)
    fo = open(a.out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for n in a.n:
        sim = F.FastSim(n, B=a.batch)
        for fam in a.families:
            var = "h0.7" if fam == "ising" else "random"
            for gt in a.gt:
                st = F.ising_circuit(n, a.L, gt) if fam == "ising" else F.brickwork_circuit(n, a.L, gt)
                for o in a.obs:
                    if (fam, str(n), str(gt), o) in done:
                        continue
                    sd = int(hashlib.md5(repr((fam, n, gt, o, "scale")).encode()).hexdigest()[:8], 16)
                    xs = sim.run(st, o, a.shots, seed=sd)
                    vs = xs.var(ddof=1)
                    _, s_ev, q_ev = sim.continuation_table(st, o, reps=a.reps, seed=sd + 1,
                                                           max_rows=a.max_rows)
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
                        w.writerow(dict(family=fam, variant=var, n=n, L=a.L, gt=gt, obs=o, abs_frac=fr,
                                        reps=a.reps, shots=a.shots, var_std=vs, var=v, ratio=rat,
                                        ratio_lo=lo, ratio_hi=hi, frac_table=tab.mean(), host=host,
                                        timestamp=ts))
                        line.append(f"{fr:g}:{rat:.2f}")
                    fo.flush()
                    print(f"n={n} {fam:9} gt={gt:<5} {o:3} " + " ".join(line), flush=True)
    fo.close()


if __name__ == "__main__":
    main()
