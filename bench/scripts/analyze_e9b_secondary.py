"""E9c pre-registered secondary: E9b R and W restricted to groups still TN at cap 1e6 (and 3e6), config-cluster CIs.

Inputs: bench/results/b_e9b_summary.csv, bench/results/b_e9c_summary.csv.  Output: printed table.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = KEYS[:-1]


def cluster_med(d, col, rng, lo, hi, B=4000):
    ids = d[CFG].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    x = d[col].to_numpy(float)
    b = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return np.median(x), *np.percentile(b, [lo, hi]), len(x), len(u)


def main():
    rng = np.random.default_rng(11)
    b = pd.read_csv(os.path.join(R, "b_e9b_summary.csv"))
    c = pd.read_csv(os.path.join(R, "b_e9c_summary.csv"))[KEYS + ["TN1000000", "TN3000000"]]
    d = b.merge(c, on=KEYS, how="left")
    print(f"E9b groups {len(b)}; matched {int(d.TN1000000.notna().sum())}; "
          f"still TN at 1e6: {int(d.TN1000000.sum())}, at 3e6: {int(d.TN3000000.sum())}")
    for cap in ("TN1000000", "TN3000000"):
        s = d[d[cap] == 1]
        print(f"\n== subset {cap} ({len(s)} groups) ==")
        for m in (10, 100):
            for ch in ("ad", "deph"):
                g = s[s.channel == ch]
                if len(g) < 3:
                    print(f"  M={m} {ch}: {len(g)} groups (too few)")
                    continue
                r = cluster_med(g, f"R{m}", rng, 2.5, 97.5)
                w = cluster_med(g, f"W{m}", rng, 0.625, 99.375)
                print(f"  M={m:3d} {ch:4s} n={r[3]:2d} ({r[4]} cfg): R med {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]  "
                      f"W med {w[0]:.3f} [{w[1]:.3f}, {w[2]:.3f}] (98.75%)  W>1 groups {int((g[f'W{m}'] > 1).sum())}")


if __name__ == "__main__":
    main()
