"""Review 4 (2026-11-15) supporting numbers for the E13 revision. Reads summary CSVs only (no simulation, no raw arrays).

(A) POST HOC instance-level summary of the E10 pilot table and E11 Q1 (review 4, M4d): per-instance medians of the
    amplitude-damping W_bf from pra_family_split.csv (by = variant), count of instances with median > 1 and the
    two-sided exact sign test (same definition as analyze_e13.sign_p); also how many instances have a 98.75% lower
    bound above 1.
(B) POST HOC cost of the cap-1e7 classification on the E13 groups (b_e13_select_summary.csv, TN at 1e7):
    (t_PP(1e7) + t_PP(1e8)) / t_traj20k, as pra_rev3b_misc (C) for the E10 groups. The selection was not a timing
    experiment: P100 and V100 workers, other compute processes present on some nodes; indicative only.
Usage: python bench/scripts/pra_rev4_misc.py > bench/results/pra_rev4_misc.txt
"""
import os
from math import comb

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench", "results")


def sign_p(k, n):
    """Two-sided exact binomial (sign) test p-value for k successes of n at p = 1/2 (as analyze_e13.sign_p)."""
    tail = sum(comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


fam = pd.read_csv(os.path.join(R, "pra_family_split.csv"))
print("(A) POST HOC instance-level summary, amplitude damping (pra_family_split.csv, by = variant)")
tot_k = tot_n = 0
for exp, metric in [("E10", "pilot_W_bf"), ("E11Q1", "single_W_bf")]:
    d = fam[(fam.exp == exp) & (fam.metric == metric) & (fam.channel == "ad") & (fam.by == "variant") & (fam.level != "all")]
    k, n = int((d["median"] > 1).sum()), len(d)
    tot_k, tot_n = tot_k + k, tot_n + n
    print(f"  {exp:6s} {metric}: " + ", ".join(f"{r.level} {r['median']:.3f} [{r.lo:.3f}, {r.hi:.3f}]" for _, r in d.iterrows()))
    print(f"         instances with median > 1: {k}/{n}; lower bound > 1: {int((d.lo > 1).sum())}/{n}; "
          f"Ising lower bound > 1: {int(((d.lo > 1) & d.level.str.startswith('h')).sum())}")
print(f"  E10 + E11 Q1 pooled: instances with median > 1: {tot_k}/{tot_n}; two-sided sign test p = {sign_p(tot_k, tot_n):.4f}")

s = pd.read_csv(os.path.join(R, "b_e13_select_summary.csv"))
t = s[s.TN10000000 == 1]
r = (t.t10000000 + t.t100000000) / t.t_traj20k
print("\n(B) POST HOC cap-1e7 classification cost on the E13 groups (b_e13_select_summary.csv; indicative, mixed hosts)")
print(f"  groups TN at 1e7: {len(t)}; (t_PP(1e7) + t_PP(1e8)) / t_traj20k: median {r.median():.2f}, "
      f"min {r.min():.2f}, max {r.max():.2f}; groups with ratio < 1: {int((r < 1).sum())}")
print(f"  t_PP(1e8) median {t.t100000000.median():.0f} s; t_traj20k median {t.t_traj20k.median():.0f} s")
