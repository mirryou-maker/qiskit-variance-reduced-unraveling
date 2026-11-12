"""Analysis of E7b (pre-registered in docs/plan_B_characterization_2026-11-06.md): Ising dephasing, theta -> 0.

r1 = Var(analog 0.05) / Var(analog 0.025),  r2 = Var(analog 0.025) / Var(analog 0.0125)   (E7b runs)
G' = min(all E7 fixed, all E7b fixed) / Var(E7 greedy)  (extended baseline; E7 G for comparison)
Secondary: G'' = same baseline / Var(E7b greedy over {std, proj, 0.0125, 0.025, 0.05}).
Decision: converged in theta iff median r2 <= 1.02. Exclusion: best fixed variance < 1e-12.
Per-group CIs: bootstrap of raw trajectory arrays (B = 500); median CIs: bootstrap over groups (2000).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
EPS, NB = 1e-12, 500
KEYS = ["family", "variant", "n", "L", "gt", "obs"]
A05, A025, A0125 = "fix:analog0.050", "fix:analog0.025", "fix:analog0.013"   # 0.0125 is printed as 0.013


def med_ci(x, rng):
    x = np.asarray(x)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return np.median(x), *np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(2026)
    b = pd.read_csv(os.path.join(R, "b_e7b_u98.csv")).drop_duplicates(KEYS + ["scheme"], keep="last")
    e7 = pd.read_csv(os.path.join(R, "b_e7_u98.csv"))
    e7 = e7[(e7.channel == "deph_sweep") & (e7.family == "ising")].drop_duplicates(KEYS + ["scheme"], keep="last")
    rows = []
    for k, g in b.groupby(KEYS):
        fam, var, n, L, gt, o = k
        g7 = e7.set_index(KEYS).loc[[k]] if k in e7.set_index(KEYS).index else None
        if g7 is None or "greedy" not in set(g.scheme):
            continue
        tag = f"_{fam}_{var}_n{n}_L{L}_gt{gt}.npz"
        zb = np.load(os.path.join(R, "b_e7b_raw", "deph_small" + tag))
        z7 = np.load(os.path.join(R, "b_e7_raw", "deph_sweep" + tag))
        vb = dict(zip(g.scheme, g["var"]))
        v7 = dict(zip(g7.scheme, g7["var"]))
        arr = {("b", s): zb[f"{o}_{s}"].astype(np.float64) for s in vb}
        arr.update({("7", s): z7[f"{o}_{s}"].astype(np.float64) for s in v7})
        var_ = {("b", s): vb[s] for s in vb} | {("7", s): v7[s] for s in v7}
        fixed = [key for key in var_ if key[1].startswith("fix:")]
        best = min(fixed, key=lambda s: var_[s])
        r = dict(zip(KEYS, k), best_ext=f"{best[0]}:{best[1]}", deterministic=var_[best] < EPS,
                 maxabsz=float(g[g["var"] > EPS].z.abs().max()) if (g["var"] > EPS).any() else 0.0)
        if r["deterministic"]:
            rows.append(r)
            continue
        bs = {}

        def bv(key):
            if key not in bs:
                x = arr[key]
                idx = rng.integers(0, len(x), size=(NB, len(x)))
                bs[key] = x[idx].var(axis=1, ddof=1)
            return bs[key]

        def ratio(num, den):
            pt = min(var_[s] for s in num) / max(var_[den], 1e-300)
            bb = np.min([bv(s) for s in num], axis=0) / np.maximum(bv(den), 1e-300)
            return (pt, *np.percentile(bb, [2.5, 97.5]))

        r["r1"], r["r1_lo"], r["r1_hi"] = ratio([("b", A05)], ("b", A025))
        r["r2"], r["r2_lo"], r["r2_hi"] = ratio([("b", A025)], ("b", A0125))
        f7 = [s for s in fixed if s[0] == "7"]
        r["G7"], r["G7_lo"], r["G7_hi"] = ratio(f7, ("7", "greedy"))
        r["Gp"], r["Gp_lo"], r["Gp_hi"] = ratio(fixed, ("7", "greedy"))
        r["Gpp"], r["Gpp_lo"], r["Gpp_hi"] = ratio(fixed, ("b", "greedy"))
        r["base_drop"] = min(var_[s] for s in f7) / var_[best]          # E7 best-fixed / extended best-fixed
        rows.append(r)
    S = pd.DataFrame(rows)
    out = os.path.join(R, "b_e7b_summary.csv")
    S.to_csv(out, index=False)
    nd = S[~S.deterministic]
    print(f"groups {len(S)}, deterministic (excluded) {int(S.deterministic.sum())} -> {out}")
    print(f"bias: max|z| (E7b rows, var>1e-12) = {nd.maxabsz.max():.2f}")
    for met in ("r1", "r2", "G7", "Gp", "Gpp", "base_drop"):
        m, lo, hi = med_ci(nd[met], rng)
        extra = ""
        if met + "_lo" in nd:
            extra = f"  CI>1 {int((nd[met + '_lo'] > 1).sum())}  CI<1 {int((nd[met + '_hi'] < 1).sum())}"
        print(f"  {met:9s} median {m:.4f} [{lo:.4f}, {hi:.4f}]  n_groups {len(nd)}{extra}")
    print("  median by gt:")
    print(nd.groupby("gt")[["r1", "r2", "G7", "Gp", "base_drop"]].median().round(4))
    print("  median by variant:")
    print(nd.groupby("variant")[["r1", "r2", "G7", "Gp"]].median().round(4))
    print("  extended best-fixed counts:", nd.best_ext.value_counts().to_dict())
    m = np.median(nd.r2)
    print(f"  DECISION (pre-registered): median r2 = {m:.4f} -> "
          f"{'CONVERGED in theta (<= 1.02)' if m <= 1.02 else 'NOT converged in theta (> 1.02)'}")


if __name__ == "__main__":
    main()
