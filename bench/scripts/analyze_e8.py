"""Analysis of E8 (pre-registered in docs/plan_B_characterization_2026-11-06.md).

R_M = Var_bestfixed / Var_greedy(M); PP_err_M = |pp_mean - exact| / sqrt(Var_bestfixed / shots).
Decision per channel at n = 10: some M <= 100 with median R_M >= 1.2 and CI lo >= 1.1; "useful for trajectories"
only if also median PP_err_M > 1 at that M. n = 10 baselines (fixed, exact greedy) come from E7 (b_e7_*), n = 6, 8 from E8.
Exclusion: Var_bestfixed < 1e-12. Per-group CI: bootstrap of raw trajectories (500). Median CI: bootstrap over groups (2000).

Usage: python bench/scripts/analyze_e8.py   -> bench/results/b_e8_summary.csv
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "bench/results")
EPS = 1e-12
NB = 500
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]


def load(pattern):
    fs = sorted(glob.glob(os.path.join(RES, pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def boot_var(x, rng):
    idx = rng.integers(0, len(x), size=(NB, len(x)))
    return x[idx].var(axis=1, ddof=1)


def med_ci(x, rng):
    x = np.asarray(x)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return np.median(x), *np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(2468)
    e8 = load("b_e8_u*.csv")
    e7 = load("b_e7_u*.csv")
    e7 = e7[e7.n == 10] if len(e7) else e7
    base = pd.concat([e8[~e8.scheme.str.startswith("greedyM")], e7], ignore_index=True)
    rows = []
    for k, g in e8[e8.scheme.str.startswith("greedyM")].groupby(KEYS):
        ch, fam, var, n, L, gt, o = k
        b = base[(base.channel == ch) & (base.family == fam) & (base.variant == var) & (base.n == n)
                 & (base.L == L) & (base["gt"] == gt) & (base.obs == o)]
        fx = b[b.scheme.str.startswith("fix:")]
        ex = b[b.scheme == "greedy"]
        if fx.empty or ex.empty:
            continue
        vb = fx["var"].min()
        if vb < EPS:
            rows.append(dict(zip(KEYS, k), deterministic=True))
            continue
        rawb = np.load(os.path.join(RES, "b_e8_raw" if n < 10 else "b_e7_raw", f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}.npz"))
        rawm = np.load(os.path.join(RES, "b_e8_raw", f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}.npz"))
        bfix = np.min([boot_var(rawb[f"{o}_{s}"].astype(np.float64), rng) for s in fx.scheme], axis=0)
        shots = int(fx.shots.iloc[0])
        r0 = dict(zip(KEYS, k), deterministic=False, var_bestfixed=vb, best_fixed=fx.loc[fx["var"].idxmin(), "scheme"],
                  G_exact=vb / ex["var"].iloc[0])
        for _, m in g.iterrows():
            bm = boot_var(rawm[f"{o}_{m.scheme}"].astype(np.float64), rng)
            lo, hi = np.percentile(bfix / np.maximum(bm, 1e-300), [2.5, 97.5])
            rows.append({**r0, "M": int(m.M), "R": vb / max(m["var"], 1e-300), "R_lo": lo, "R_hi": hi, "z": m.z,
                         "pp_err": abs(m.pp_mean - m.exact) / np.sqrt(vb / shots), "pp_abs": abs(m.pp_mean - m.exact)})
    S = pd.DataFrame(rows)
    out = os.path.join(RES, "b_e8_summary.csv")
    S.to_csv(out, index=False)
    print(f"rows {len(S)}, deterministic groups {int((S.deterministic == True).sum())} -> {out}")  # noqa: E712
    nd = S[S.deterministic == False]  # noqa: E712
    print(f"bias: max|z| = {nd.z.abs().max():.2f}")
    for ch, C in nd.groupby("channel"):
        print(f"\n===== {ch} =====")
        passed = []
        for (n, M), g in C.groupby(["n", "M"]):
            m, lo, hi = med_ci(g.R, rng)
            pe = g.pp_err.median()
            print(f"  n={int(n):2d} M={int(M):6d} groups {len(g):3d} median R {m:.3f} [{lo:.3f},{hi:.3f}] "
                  f"CI>1 {int((g.R_lo > 1).sum())} CI<1 {int((g.R_hi < 1).sum())} median PP_err {pe:.2f}")
            if n == 10 and M <= 100 and m >= 1.2 and lo >= 1.1:
                passed.append((int(M), m, lo, pe))
        print("  median G_exact per n:", C.drop_duplicates(KEYS).groupby("n").G_exact.median().round(3).to_dict())
        print("  by family at n=10, median R:")
        t = C[C.n == 10]
        if len(t):
            print(t.pivot_table(index="family", columns="M", values="R", aggfunc="median").round(3))
        print("  DECISION (pre-registered):")
        if not len(t):
            print("    incomplete: n = 10 required")
        elif passed:
            M, m, lo, pe = passed[0]
            print(f"    KEEPS the gain at M={M}: median R {m:.3f} (CI lo {lo:.3f}); "
                  f"PP_err {pe:.2f} -> {'useful for trajectories' if pe > 1 else 'truncated propagation alone suffices'}")
        else:
            print("    does NOT keep the gain at any M <= 100")


if __name__ == "__main__":
    main()
