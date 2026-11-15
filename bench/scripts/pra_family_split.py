"""POST HOC (review 3, 2026-11-14): static-table gain split by circuit family and by circuit instance.

Reads only summary CSVs written by analyze_e10.py / analyze_e11.py (no simulation, no raw arrays):
  b_e10_summary.csv   per group (E10, RTX 2060): pilot_W_bf, ideal_W_bf, family, variant, L, gt
  b_e11_summary.csv   per group (E11 Q1): single_W_bf
  b_e11_multi.csv     per configuration (E11 Q2): W_multi (edef A20k, subset all)
Intervals: 98.75% percentile bootstrap (as the pre-specified E10/E11 rules); groups resampled (B = 2000) and
configurations resampled (cluster, B = 2000) for per-group metrics; configurations resampled (B = 4000) for W_multi.
One generator default_rng(20261114) is used for every interval in the order printed (draws differ from analyze_e1x).
Output: bench/results/pra_family_split.txt (stdout) and pra_family_split.csv.
Usage (ubuntu94, env sci): python bench/scripts/pra_family_split.py > bench/results/pra_family_split.txt
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench", "results")
CFG = ["channel", "family", "variant", "n", "L", "gt"]
LO, HI = 0.625, 99.375
rng = np.random.default_rng(20261114)


def med_ci(x, B=2000):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [LO, HI])


def cluster_ci(d, col, B=2000):
    x = d[col].to_numpy(float)
    ids = d[CFG].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    if len(u) < 2:
        return np.nan, np.nan
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return tuple(np.percentile(cb, [LO, HI]))


rows = []


def report(exp, metric, d, col, unit, by):
    for ch, D0 in d.groupby("channel"):
        for key, D in [("all", D0)] + list(D0.groupby(by)):
            m, lo, hi = med_ci(D[col], 4000 if unit == "config" else 2000)
            clo, chi = cluster_ci(D, col) if unit == "group" else (np.nan, np.nan)
            ncfg = D[CFG].drop_duplicates().shape[0]
            rows.append(dict(exp=exp, metric=metric, channel=ch, by=by, level=key, units=len(D), configs=ncfg,
                             median=m, lo=lo, hi=hi, clo=clo, chi=chi, n_gt1=int((D[col] > 1).sum())))
            cl = f" {{cluster {clo:.3f}, {chi:.3f}}}" if unit == "group" else ""
            print(f"  {exp:7s} {metric:12s} {ch:4s} {by:7s}={str(key):9s} {unit}s {len(D):3d} (configs {ncfg:2d}) "
                  f"median {m:.3f} [{lo:.3f}, {hi:.3f}]{cl}  >1 {int((D[col] > 1).sum())}/{len(D)}")


e10 = pd.read_csv(os.path.join(R, "b_e10_summary.csv"))
e11 = pd.read_csv(os.path.join(R, "b_e11_summary.csv"))
mu = pd.read_csv(os.path.join(R, "b_e11_multi.csv"))
mu = mu[(mu.edef == "A20k") & (mu.subset == "all")]
print("POST HOC family / instance split of the static-table gain (98.75% percentile bootstrap; [groups or configs]"
      " {config cluster})")
for by in ("family", "variant"):
    print(f"\n== by {by} ==")
    report("E10", "pilot_W_bf", e10, "pilot_W_bf", "group", by)
    report("E10", "ideal_W_bf", e10, "ideal_W_bf", "group", by)
    report("E11Q1", "single_W_bf", e11, "single_W_bf", "group", by)
    report("E11Q2", "W_multi", mu, "W_multi", "config", by)
print("\ncircuit instances per experiment (variant values):")
for exp, d in (("E10", e10), ("E11", e11)):
    print(f"  {exp}: {sorted(set(d.variant))} ({d.variant.nunique()} instances; families "
          f"{d.groupby('family').variant.nunique().to_dict()})")
pd.DataFrame(rows).to_csv(os.path.join(R, "pra_family_split.csv"), index=False)
