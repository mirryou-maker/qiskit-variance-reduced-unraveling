"""Analysis of E2 and E3 (pre-registered in docs/plan_B_characterization_2026-11-06.md).

E2 (per channel depol / ad): G = Var_bestfixed / Var_greedy, bestfixed = min over ALL fixed candidates.
   Decision: median G >= 1.2 at n = 10 and n = 10 CI lower bound >= 1.1.
E3 (dephasing, TJM + myopic):
   Q1  M = Var_myopic / Var_greedy;   back-propagation matters iff median M >= 1.1 at n = 10, CI lo > 1.0.
   Q2  T = min(proj, analog*) / Var_greedy (reported); fraction of groups where each TJM scheme is worse than std.
   Q3  A = min(4 fixed) / Var_greedy_all;  holds iff median A >= 1.2 at n = 10, CI lo >= 1.1.
Exclusion: groups whose best fixed variance < 1e-12 (deterministic estimator exists).
Per-group CIs: bootstrap of the raw trajectory arrays (independent runs, B = 500).
Median CIs: bootstrap over groups (2000).

Usage: python bench/scripts/analyze_e23.py e2|e3|e7  (e7 is analysed like e2)
Writes bench/results/b_<e2|e3>_summary.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EPS = 1e-12
NB = 500
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]


def boot_var(x, rng):
    idx = rng.integers(0, len(x), size=(NB, len(x)))
    return x[idx].var(axis=1, ddof=1)


def ratio(z, o, num, den, v, rng, cache):
    """point and 95% CI of min_{s in num} Var_s / Var_den."""
    def bv(s):
        if s not in cache:
            cache[s] = boot_var(z[f"{o}_{s}"].astype(np.float64), rng)
        return cache[s]
    pt = min(v[s] for s in num) / max(v[den], 1e-300)
    b = np.min([bv(s) for s in num], axis=0) / np.maximum(bv(den), 1e-300)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return pt, lo, hi


def med_ci(x, rng):
    x = np.asarray(x)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return np.median(x), *np.percentile(b, [2.5, 97.5])


def main():
    exp = sys.argv[1]
    raw = os.path.join(ROOT, f"bench/results/b_{exp}_raw")
    files = sorted(glob.glob(os.path.join(ROOT, f"bench/results/b_{exp}_u*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(KEYS + ["scheme"], keep="last")
    rng = np.random.default_rng(12345)
    rows = []
    for k, g in df.groupby(KEYS):
        ch, fam, var, n, L, gt, o = k
        v = dict(zip(g["scheme"], g["var"]))
        fixed = sorted(s for s in v if s.startswith("fix:"))
        std = g[g.scheme.str.startswith("fix:") & (g.frac_std == 1.0)].scheme.iloc[0]
        need = {"greedy"} | ({"myopic", "greedy_all"} if exp == "e3" else set())
        if not need <= set(v):
            continue
        z = np.load(os.path.join(raw, f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}.npz"))
        best = min(fixed, key=lambda s: v[s])
        det = v[best] < EPS
        nz = g[g["var"] > EPS]
        r = dict(zip(KEYS, k), deterministic=det, best_fixed=best, std=std,
                 maxabsz=float(nz.z.abs().max()) if len(nz) else 0.0,
                 **{f"var_{s}": v[s] for s in v})
        if not det:
            c = {}
            if exp != "e3":
                r["G"], r["G_lo"], r["G_hi"] = ratio(z, o, fixed, "greedy", v, rng, c)
                r["G_vs_std"] = v[std] / max(v["greedy"], 1e-300)
            else:
                four = ["fix:std", "fix:proj"] + [s for s in fixed if "analog" in s]
                tjm = [s for s in four if s != "fix:std"]
                r["G"], r["G_lo"], r["G_hi"] = ratio(z, o, ["fix:std", "fix:proj"], "greedy", v, rng, c)
                r["M"], r["M_lo"], r["M_hi"] = ratio(z, o, ["myopic"], "greedy", v, rng, c)
                r["T"], r["T_lo"], r["T_hi"] = ratio(z, o, tjm, "greedy", v, rng, c)
                r["A"], r["A_lo"], r["A_hi"] = ratio(z, o, four, "greedy_all", v, rng, c)
                for s in tjm:  # >1: this TJM unraveling beats std
                    pt, lo, hi = ratio(z, o, ["fix:std"], s, v, rng, c)
                    r[f"R_{s[4:]}"], r[f"R_{s[4:]}_lo"], r[f"R_{s[4:]}_hi"] = pt, lo, hi
        rows.append(r)
    S = pd.DataFrame(rows)
    out = os.path.join(ROOT, f"bench/results/b_{exp}_summary.csv")
    S.to_csv(out, index=False)
    nd = S[~S.deterministic]
    print(f"groups {len(S)}, deterministic (excluded) {int(S.deterministic.sum())} -> {out}")
    print(f"bias: max|z| over var>1e-12 rows = {nd.maxabsz.max():.2f}")

    for ch, C in nd.groupby("channel"):
        prim = C[C["gt"] <= 0.1]
        print(f"\n===== channel {ch}: PRIMARY gt <= 0.1 =====")
        mets = ["G"] if exp != "e3" else ["G", "M", "T", "A"]
        meds = {}
        for met in mets:
            print(f"  {met}:")
            for n, g in prim.groupby("n"):
                m, lo, hi = med_ci(g[met], rng)
                meds[(met, n)] = (m, lo, hi)
                print(f"    n={n:2d} groups {len(g):3d} median {m:.3f} [{lo:.3f}, {hi:.3f}]  "
                      f"CI>1 {int((g[met + '_lo'] > 1).sum())}  CI<1 {int((g[met + '_hi'] < 1).sum())}")
        if exp != "e3":
            print("  median G_vs_std per n:", prim.groupby("n").G_vs_std.median().round(3).to_dict())
            print("  best-fixed candidate counts:", prim.best_fixed.value_counts().to_dict())
        else:
            for s in [c for c in prim.columns if c.startswith("R_") and not c.endswith(("_lo", "_hi"))]:
                print(f"  {s[2:]} vs std: better(CI>1) {int((prim[s + '_lo'] > 1).sum())}"
                      f" worse(CI<1) {int((prim[s + '_hi'] < 1).sum())} of {len(prim)}; median {prim[s].median():.3f}")
        print("  by family/variant (median G):")
        print(prim.pivot_table(index=["family", "variant"], columns="n", values="G", aggfunc="median").round(3))
        print("  by observable (median G):")
        print(prim.pivot_table(index="obs", columns="n", values="G", aggfunc="median").round(3))
        sec = C[C["gt"] > 0.1]
        if len(sec):
            print("  SECONDARY gt=0.3 median per n:", {m_: sec.groupby("n")[m_].median().round(3).to_dict()
                                                   for m_ in mets})
        print("  DECISION (pre-registered):")
        if (mets[0], 10) not in meds:
            print("    incomplete: n = 10 required")
            continue
        if exp != "e3":
            m, lo, _ = meds[("G", 10)]
            print(f"    median G n10 {m:.3f} (CI lo {lo:.3f}) -> adaptive gain "
                  f"{'EXTENDS' if m >= 1.2 and lo >= 1.1 else 'does NOT extend'} to {ch}")
        else:
            m, lo, _ = meds[("M", 10)]
            print(f"    Q1 median M n10 {m:.3f} (CI lo {lo:.3f}) -> back-propagation "
                  f"{'MATTERS' if m >= 1.1 and lo > 1.0 else 'does NOT matter (myopic suffices)'}")
            m, lo, _ = meds[("A", 10)]
            print(f"    Q3 median A n10 {m:.3f} (CI lo {lo:.3f}) -> adaptive gain vs TJM unravelings "
                  f"{'HOLDS' if m >= 1.2 and lo >= 1.1 else 'does NOT hold'}")


if __name__ == "__main__":
    main()
