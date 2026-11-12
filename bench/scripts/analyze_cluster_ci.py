"""Config-level cluster bootstrap for every median-of-groups CI in E1-E9b (plan doc, "Cluster-bootstrap re-analysis
pre-registration"). The 4 observables of a configuration share runs (same circuit, same seed), so groups are not
independent; here configurations (all group keys except obs) are resampled with replacement.

Output: bench/results/cluster_ci.csv and a printed table: naive CI (groups resampled) vs cluster CI.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
B = 4000


def rd(name):
    return pd.read_csv(os.path.join(R, name))


def nondet(d):
    return d[~d.deterministic.astype(str).eq("True")] if "deterministic" in d else d


def boot(d, col, stat, cluster, lo, hi, rng):
    x = d[col].to_numpy(float)
    pt = stat(x)
    nb = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    ids = d[cluster].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    members = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        cb.append(stat(x[np.concatenate([members[j] for j in pick])]))
    return pt, *np.percentile(nb, [lo, hi]), *np.percentile(cb, [lo, hi]), len(x), len(u)


def main():
    rng = np.random.default_rng(20261110)
    med = np.median
    mean = np.mean
    K5 = ["family", "variant", "n", "L", "gt"]
    K6 = ["channel"] + K5
    specs = []                       # (experiment, label, df, col, stat, cluster keys, lo, hi, decision threshold on lower CI)
    e1 = nondet(rd("b_ncurve_summary.csv"))
    e1 = e1[e1["gt"] <= 0.1]
    for n in (6, 8, 10, 12):
        specs.append(("E1", f"G n={n}", e1[e1.n == n], "G", med, K5, 2.5, 97.5, 1.1 if n == 12 else None))
    specs.append(("E1", "G_pi n=6", e1[(e1.n == 6) & e1.G_pi.notna()], "G_pi", med, K5, 2.5, 97.5, None))
    e2 = nondet(rd("b_e2_summary.csv"))
    e2 = e2[(e2["gt"] <= 0.1) & (e2.n == 10)]
    for ch in ("depol", "ad"):
        specs.append(("E2", f"G {ch} n=10", e2[e2.channel == ch], "G", med, K6, 2.5, 97.5, 1.1))
    e3 = nondet(rd("b_e3_summary.csv"))
    e3 = e3[(e3["gt"] <= 0.1) & (e3.n == 10)]
    for c, thr in (("G", 1.1), ("M", 1.0), ("T", None), ("A", 1.1)):
        specs.append(("E3", f"{c} n=10", e3, c, med, K6, 2.5, 97.5, thr))
    e7 = nondet(rd("b_e7_summary.csv"))
    e7 = e7[e7["gt"] <= 0.1]
    for ch in ("deph_sweep", "ad_phase"):
        specs.append(("E7", f"G {ch}", e7[e7.channel == ch], "G", med, K6, 2.5, 97.5, None))
    e7b = nondet(rd("b_e7b_summary.csv"))
    for c in ("r1", "r2", "Gp"):
        specs.append(("E7b", c, e7b, c, med, K5, 2.5, 97.5, 1.0 if c == "Gp" else None))
    e8 = nondet(rd("b_e8_summary.csv"))
    e8 = e8[(e8.n == 10) & e8.M.notna()]
    for ch in ("deph_sweep", "ad_phase"):
        for m in (1, 10, 100, 1000, 10000):
            specs.append(("E8", f"R {ch} M={m}", e8[(e8.channel == ch) & (e8.M == m)], "R", med, K6, 2.5, 97.5, None))
    e9a = nondet(rd("b_e9a_summary.csv"))
    e9a = e9a[e9a.n.isin([20, 24])].assign(TNf=lambda d: d.TN.astype(str).eq("True").astype(float))
    specs.append(("E9a", "TN fraction all", e9a, "TNf", mean, K6, 2.5, 97.5, 0.05))
    for ch in ("ad", "deph"):
        specs.append(("E9a", f"TN fraction {ch}", e9a[e9a.channel == ch], "TNf", mean, K6, 2.5, 97.5, None))
    e9b = rd("b_e9b_summary.csv")
    for m in (10, 100):
        for ch in ("ad", "deph"):
            g = e9b[e9b.channel == ch]
            specs.append(("E9b", f"R{m} {ch}", g, f"R{m}", med, K6, 2.5, 97.5, None))
            specs.append(("E9b", f"W{m} {ch}", g, f"W{m}", med, K6, 0.625, 99.375, 1.0))
    rows = []
    for ex, lab, d, col, stat, cl, lo, hi, thr in specs:
        pt, nlo, nhi, clo, chi, ng, nc = boot(d, col, stat, cl, lo, hi, rng)
        rows.append(dict(exp=ex, quantity=lab, level=f"{hi - lo:.2f}%", groups=ng, configs=nc, point=pt,
                         naive_lo=nlo, naive_hi=nhi, cluster_lo=clo, cluster_hi=chi,
                         width_ratio=(chi - clo) / max(nhi - nlo, 1e-300), threshold=thr,
                         naive_pass=None if thr is None else bool(nlo > thr or (thr == 1.1 and nlo >= thr)),
                         cluster_pass=None if thr is None else bool(clo > thr or (thr == 1.1 and clo >= thr))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, "cluster_ci.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    ch = out[out.threshold.notna() & (out.naive_pass != out.cluster_pass)]
    print("\nlower-bound threshold checks that change under the cluster bootstrap:",
          "none" if ch.empty else ch[["exp", "quantity", "threshold", "naive_lo", "cluster_lo"]].to_string(index=False))
    print("note: thresholds here are lower-CI checks only; E9b W uses the pre-registered rule median > 1 AND lower > 1.")


if __name__ == "__main__":
    main()
