"""POST HOC summary of the small-n static per-location tables (Plan A step 1) for the PRA revision (2026-11-12).

Referee issue 2 asks how much of the per-trajectory gain a static (trajectory-independent) per-location table recovers.
Inputs (read only): bench/results/a_static.csv (std, proj, s1, pi3, ideal; greedy at n = 8) and
bench/results/p3_adapt.csv (std, proj, greedy, pi at n = 5, 6; same groups, separate runs).
Metric per group: Var_best(std, proj) / Var_scheme, with std and proj taken from the SAME file as the scheme
(a_static for s1 / pi3 / ideal / greedy(n=8); p3_adapt for greedy / pi at n = 5, 6).
Groups: non-degenerate (a_static Var_std >= 1e-10, the analyze_a_static.py rule), all gt and gt <= 0.1.
Prints medians, 95 % group-bootstrap CIs of the median (B = 2000, seed 5), and counts of per-group ratio > 1.
Pilot cost (from the CSV): s1 2000 std trajectories, pi3 3 x 2000, ideal 0 (one noiseless pass); production 20000.
Usage: python bench/scripts/pra_static_summary.py > bench/results/pra_static_summary.txt
"""
import collections
import csv
import math

import numpy as np

VMIN = 1e-10
f = lambda x: float(x) if x not in ("", None) else math.nan  # noqa: E731
key = lambda r: (r["family"], r["variant"], r["n"], r["L"], r["gt"], r["obs"])  # noqa: E731

st = collections.defaultdict(dict)
for r in csv.DictReader(open("bench/results/a_static.csv")):
    st[key(r)][r["scheme"]] = r
p3 = collections.defaultdict(dict)
for r in csv.DictReader(open("bench/results/p3_adapt.csv")):
    p3[key(r)][r["scheme"]] = r

nondeg = [k for k, d in st.items() if f(d["std"]["var"]) >= VMIN]
rng = np.random.default_rng(5)


def boot_med(x, B=2000):
    x = np.asarray(x)
    m = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.percentile(m, [2.5, 97.5])


def gain(d, s):
    vb = min(f(d["std"]["var"]), f(d["proj"]["var"]))
    vs = f(d[s]["var"])
    return vb / vs if vs > 0 else math.nan


print(f"non-degenerate groups: {len(nondeg)} (of {len(st)})")
for nn in sorted({k[2] for k in nondeg}, key=int):
    ks = [k for k in nondeg if k[2] == nn]
    print(f"\n n={nn}: {len(ks)} non-degenerate groups")
    rows = [("s1", st), ("pi3", st), ("ideal", st), ("greedy", st), ("greedy", p3), ("pi", p3)]
    for s, src in rows:
        tag = ("p3_" if src is p3 else "") + s
        for lab, sel in (("all gt", lambda k: True), ("gt<=0.1", lambda k: float(k[4]) <= 0.1)):
            g = [gain(src[k], s) for k in ks if sel(k) and k in src and s in src[k]]
            g = np.array([x for x in g if math.isfinite(x)])
            if len(g) == 0:
                continue
            lo, hi = boot_med(g)
            print(f"  {tag:10} {lab:8} groups={len(g):3}  median Var_best(std,proj)/Var = {np.median(g):.3f} "
                  f"[{lo:.3f}, {hi:.3f}]  frac>1 = {np.mean(g > 1):.2f}")
