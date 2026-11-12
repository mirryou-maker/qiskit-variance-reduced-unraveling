"""Plan A step 1: STATIC per-event std/projector tables vs std / proj (reference sampler).

Static tables cost nothing per trajectory (decided once, before production), so they are the
cheapest scalable candidate. Variants (pilot shots counted in the CSV):
  s1     one std pilot round                    (pilot_shots)
  pi3    3 rounds of re-piloting under the table (3 x pilot_shots)
  ideal  table from the noiseless trajectory    (0 pilot shots, 1 deterministic pass)
Also runs greedy (per-trajectory, dense) at n=8 for reference; PI/greedy at n=5,6 come from
bench/results/p3_adapt.csv (same groups). Ratio = Var_std / Var_scheme, bootstrap 95% CI.
Ledger-style CSV. Usage: a_static.py --n 5|6|8 [--shots 20000] [--pilot 2000]
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
from src.noise import traj_ref as T  # noqa: E402


def build(fam, var, n, L, gt):
    if fam == "ising":
        return T.ising_ops(n, L, gt, h=float(var[1:]))
    return T.brickwork_ops(n, L, gt, clifford=(var == "clifford"))


def boot(xs, xq, B=500, seed=3):
    rng = np.random.default_rng(seed)
    r = [xs[rng.integers(0, len(xs), len(xs))].var(ddof=1)
         / max(xq[rng.integers(0, len(xq), len(xq))].var(ddof=1), 1e-300) for _ in range(B)]
    return np.percentile(r, [2.5, 97.5])


def seed_of(*k):
    return int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)


FIELDS = ["family", "variant", "n", "L", "gt", "obs", "scheme", "shots", "pilot_shots",
          "exact", "mean", "z", "var", "var_std", "ratio", "ratio_lo", "ratio_hi",
          "frac_proj", "seed", "wall_s", "host", "timestamp"]


def groups(n):
    if n <= 6:
        variants = [("brickwork", "random"), ("brickwork", "clifford"), ("ising", "h0"),
                    ("ising", "h0.7"), ("ising", "h1.4"), ("ising", "h3")]
        Ls = (2, 4, 8) if n <= 5 else (2, 4)
    else:
        variants = [("brickwork", "random"), ("ising", "h0.7"), ("ising", "h1.4")]
        Ls = (4, 8)
    for fam, var in variants:
        for L in Ls:
            for gt in (0.01, 0.03, 0.1, 0.3):
                yield fam, var, n, L, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--pilot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/a_static.csv"))
    ap.add_argument("--static-pi", action="store_true", help="also static PI tables (dense, n<=6)")
    ap.add_argument("--only-new", action="store_true", help="skip s1/pi3/ideal (static-PI study)")
    a = ap.parse_args()
    host = os.uname().nodename
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]), r["obs"]))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    obs_list = ("x", "zz", "xc", "zc")
    for fam, var, n, L, gt in groups(a.n):
        ops = build(fam, var, n, L, gt)
        sd = seed_of(fam, var, n, L, gt, "static")
        t0 = time.time()
        base = T.run(ops, n, list(obs_list), a.shots, ("lam", 0.0), seed=sd)
        prj = T.run(ops, n, list(obs_list), a.shots, ("lam", 1.0), seed=sd + 1)
        for o in obs_list:
            if (fam, var, n, L, gt, o) in done:
                continue
            ex = T.exact_mean(ops, n, o)
            xs = base[o]
            vs = xs.var(ddof=1)
            cands = [("std", xs, 0.0, 0), ("proj", prj[o], 1.0, 0)]
            if not a.only_new:
                tab1, _ = T.static_table(ops, n, o, a.pilot, rounds=1, seed=sd + 10)
                x1 = T.run(ops, n, [o], a.shots, ("table", tab1), seed=sd + 11)[o]
                cands.append(("s1", x1, float(tab1.mean()), a.pilot))
                tab3, _ = T.static_table(ops, n, o, a.pilot, rounds=3, seed=sd + 20)
                x3 = T.run(ops, n, [o], a.shots, ("table", tab3), seed=sd + 21)[o]
                cands.append(("pi3", x3, float(tab3.mean()), 3 * a.pilot))
                s_ev, q_ev = T.ideal_events(ops, n, o)
                tabi = (q_ev < (1 - 1e-3) * s_ev).astype(float)
                xi = T.run(ops, n, [o], a.shots, ("table", tabi), seed=sd + 31)[o]
                cands.append(("ideal", xi, float(tabi.mean()), 0))
            if a.static_pi and n <= 6:
                for nm, ps_ in (("spi_ideal", 0), ("spi_pilot", a.pilot)):
                    tb, _ = T.static_pi_table(ops, n, o, pilot_shots=ps_, seed=sd + 50)
                    xx = T.run(ops, n, [o], a.shots, ("table", tb), seed=sd + 51)[o]
                    cands.append((nm, xx, float(tb.mean()), ps_))
            if n >= 7:
                gr = T.run(ops, n, [o], a.shots, ("greedy", None), seed=sd + 41, obs_for_policy=o)
                cands.append(("greedy", gr[o], float(np.mean(gr["_frac_proj"])), 0))
            rows = []
            ts = dt.datetime.now().isoformat(timespec="seconds")
            for sch, x, fr, ps in cands:
                v = x.var(ddof=1)
                ok = v > 1e-14 and vs > 1e-14
                lo, hi = boot(xs, x) if ok and sch != "std" else (np.nan, np.nan)
                rows.append(dict(family=fam, variant=var, n=n, L=L, gt=gt, obs=o, scheme=sch,
                                 shots=a.shots, pilot_shots=ps, exact=ex, mean=x.mean(),
                                 z=(x.mean() - ex) / np.sqrt(max(v, 1e-300) / len(x)),
                                 var=v, var_std=vs,
                                 ratio=(vs / v if v > 1e-14 else (np.inf if vs > 1e-14 else np.nan)),
                                 ratio_lo=lo, ratio_hi=hi, frac_proj=fr, seed=sd,
                                 wall_s=round(time.time() - t0, 2), host=host, timestamp=ts))
            for r in rows:
                w.writerow(r)
            f.flush()
            print(f"n={n} {fam:9} {var:8} L={L} gt={gt:<5} {o:3} "
                  + " ".join(f"{r['scheme']}={r['ratio']:.2f}" for r in rows[1:]), flush=True)
    f.close()


if __name__ == "__main__":
    main()
