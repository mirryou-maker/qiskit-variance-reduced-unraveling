"""POST HOC extras for the second PRA revision (2026-11-12), referee issue 17.

Sign test and paired median CI for Var_greedy / Var_PI on the E1 n = 6 primary groups (the M1 set of pra_revision.py:
b_ncurve_summary.csv, Var_bf = min(Var_std, Var_proj) >= 1e-12, gt <= 0.1, n = 6, G_pi present).
Point ratios use the summary-CSV variances (same as the M1 point estimates). One-sided sign test of
H0: P(Var_greedy > Var_PI) = 1/2 against "PI better more often". Configuration-cluster bootstrap (B = 4000, seed 11,
cluster = family, variant, n, L, gt) of the median ratio.
Usage: python bench/scripts/pra_rev2_extra.py > bench/results/pra_rev2_extra.txt
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

s = pd.read_csv("bench/results/b_ncurve_summary.csv")
s = s[np.minimum(s.var_std, s.var_proj) >= 1e-12]
pi = s[(s["gt"] <= 0.1) & (s.n == 6) & s.G_pi.notna()].copy()
pi["r"] = pi.var_greedy / pi.var_pi
k = int((pi.r > 1).sum())
m = len(pi)
p = binomtest(k, m, 0.5, alternative="greater").pvalue
print(f"M1 sign test: groups {m}; Var_greedy > Var_PI in {k}; ties {(pi.r == 1).sum()}; one-sided p = {p:.3g}")
rng = np.random.default_rng(11)
cfg = ["family", "variant", "n", "L", "gt"]
groups = [g.r.values for _, g in pi.groupby(cfg)]
meds = []
for _ in range(4000):
    idx = rng.integers(0, len(groups), len(groups))
    meds.append(np.median(np.concatenate([groups[i] for i in idx])))
lo, hi = np.percentile(meds, [2.5, 97.5])
print(f"M1 median Var_greedy/Var_PI {np.median(pi.r):.3f}, cluster95 [{lo:.3f}, {hi:.3f}] ({len(groups)} configurations)")
