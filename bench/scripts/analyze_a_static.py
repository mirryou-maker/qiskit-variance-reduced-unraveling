"""Summarise Plan A step 1 (bench/results/a_static.csv), joined with PI/greedy from p3_adapt.csv.
F1 check: static policies' median ratio in gt<=0.1 groups and vs best-of(std, proj)."""
import collections
import csv
import math
import sys

import numpy as np

VMIN = 1e-10
f = lambda x: float(x) if x not in ("", None) else math.nan  # noqa: E731
st = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "bench/results/a_static.csv")))
p3 = list(csv.DictReader(open("bench/results/p3_adapt.csv")))
key = lambda r: (r["family"], r["variant"], r["n"], r["L"], r["gt"], r["obs"])  # noqa: E731
g = collections.defaultdict(dict)
for r in st:
    g[key(r)][r["scheme"]] = r
for r in p3:
    if key(r) in g and r["scheme"] in ("pi", "greedy"):
        g[key(r)]["p3_" + r["scheme"]] = r
nondeg = {k: d for k, d in g.items() if f(d["std"]["var"]) >= VMIN}
deg = {k: d for k, d in g.items() if f(d["std"]["var"]) < VMIN}
print(f"groups {len(g)}, non-degenerate {len(nondeg)}, degenerate (Var_std~0) {len(deg)}")
for s in ("proj", "s1", "pi3", "ideal", "greedy", "p3_greedy", "p3_pi"):
    print(f"  degenerate: max var {s:9} = "
          f"{max((f(d[s]['var']) for d in deg.values() if s in d), default=math.nan):.3g}")
zs = [abs(f(d[s]["z"])) for d in nondeg.values() for s in ("s1", "pi3", "ideal") if s in d and f(d[s]["var"]) >= VMIN]
print(f"bias static: {len(zs)} tests max|z|={max(zs):.2f} n>3={sum(z > 3 for z in zs)}")
for nn in sorted({k[2] for k in nondeg}):
    sub = {k: d for k, d in nondeg.items() if k[2] == nn}
    print(f"\n n={nn}: {len(sub)} non-degenerate groups")
    for s in ("proj", "s1", "pi3", "ideal", "greedy", "p3_greedy", "p3_pi"):
        rr = [d[s] for d in sub.values() if s in d and math.isfinite(f(d[s]["ratio"])) and f(d[s]["ratio"]) > 0]
        if not rr:
            continue
        rat = np.array([f(r["ratio"]) for r in rr])
        real = np.array([f(r["ratio"]) for r in rr if f(r["gt"]) <= 0.1])
        b = sum(f(r["ratio_lo"]) > 1 for r in rr)
        w = sum(f(r["ratio_hi"]) < 1 for r in rr)
        print(f"  {s:9} n={len(rr):3} better={b:3} worse={w:3} median={np.median(rat):.3f} "
              f"[gt<=0.1 median={np.median(real):.3f}, frac>=1.3={np.mean(real >= 1.3):.2f}]")
    for s in ("s1", "pi3", "ideal"):
        rel = []
        for d in sub.values():
            if s in d and f(d["gt"] if "gt" in d else d[s]["gt"]) <= 0.1:
                rs, rp = f(d[s]["ratio"]), f(d["proj"]["ratio"])
                if math.isfinite(rs) and math.isfinite(rp):
                    rel.append(rs / max(1.0, rp))
        rel = np.array(rel)
        print(f"  {s:9} / best-of(std,proj) [gt<=0.1]: median={np.median(rel):.3f} frac>1.05={np.mean(rel > 1.05):.2f} "
              f"frac<0.95={np.mean(rel < 0.95):.2f}")
    print("  per family/variant/obs medians (s1 | pi3 | ideal | PI):")
    gg = collections.defaultdict(list)
    for k, d in sub.items():
        gg[(k[0], k[1], k[5])].append(d)
    for k in sorted(gg):
        def med(s):
            v = [f(d[s]["ratio"]) for d in gg[k] if s in d and math.isfinite(f(d[s]["ratio"]))]
            return np.median(v) if v else math.nan
        print(f"    {k[0]:9} {k[1]:8} {k[2]:3}: {med('s1'):6.3f} | {med('pi3'):6.3f} | {med('ideal'):6.3f} | "
              f"{med('p3_pi'):6.3f}")
