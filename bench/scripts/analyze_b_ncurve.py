"""Analysis of E1 (pre-registered in docs/plan_B_characterization_2026-11-06.md).

G = min(Var_std, Var_proj) / Var_greedy per group (oracle best-fixed; conservative for us).
Per-group CI: bootstrap of the three raw trajectory arrays (independent runs).
Median-G CI per n: bootstrap over groups.
Decision (primary ensemble gamma*t <= 0.1): H_adapt supported iff median G >= 1.2 at n = 10 and 12
and the CI lower bound of the median at n = 12 >= 1.1.

Usage: python bench/scripts/analyze_b_ncurve.py [csv ...]  (default: bench/results/b_ncurve*.csv)
Writes bench/results/b_ncurve_summary.csv (one row per group).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "bench/results/b_ncurve_raw")
EPS = 1e-12
B = 500


def boot_var(x, rng, B=B):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return x[idx].var(axis=1, ddof=1)


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "bench/results/b_ncurve_u*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["family", "variant", "n", "L", "gt", "obs", "scheme"], keep="last")
    rng = np.random.default_rng(12345)
    rows = []
    for k, g in df.groupby(["family", "variant", "n", "L", "gt", "obs"]):
        fam, var, n, L, gt, o = k
        v = dict(zip(g["scheme"], g["var"]))
        if not {"std", "proj", "greedy"} <= set(v):
            continue
        key = f"{fam}_{var}_n{n}_L{L}_gt{gt}"
        z = np.load(os.path.join(RAW, key + ".npz"))
        xs, xp_, xg = (z[f"{o}_{s}"].astype(np.float64) for s in ("std", "proj", "greedy"))
        det = v["std"] < EPS and v["proj"] < EPS
        r = dict(family=fam, variant=var, n=n, L=L, gt=gt, obs=o, deterministic=det,
                 var_std=v["std"], var_proj=v["proj"], var_greedy=v["greedy"],
                 var_pi=v.get("pi", np.nan),
                 maxabsz=float(np.max(np.abs(g["z"][g["var"] > EPS]))) if (g["var"] > EPS).any() else 0.0)
        if not det:
            bs, bp, bg = boot_var(xs, rng), boot_var(xp_, rng), boot_var(xg, rng)
            bf = np.minimum(bs, bp)
            r["G"] = min(v["std"], v["proj"]) / max(v["greedy"], 1e-300)
            Gb = bf / np.maximum(bg, 1e-300)
            r["G_lo"], r["G_hi"] = np.percentile(Gb, [2.5, 97.5])
            r["R_proj"] = v["std"] / max(v["proj"], 1e-300)  # >1: projector beats std
            Rb = bs / np.maximum(bp, 1e-300)
            r["R_proj_lo"], r["R_proj_hi"] = np.percentile(Rb, [2.5, 97.5])
            r["G_vs_std"] = v["std"] / max(v["greedy"], 1e-300)
            if "pi" in v:
                xpi = z[f"{o}_pi"].astype(np.float64)
                r["G_pi"] = min(v["std"], v["proj"]) / max(v["pi"], 1e-300)
                Pb = bf / np.maximum(boot_var(xpi, rng), 1e-300)
                r["G_pi_lo"], r["G_pi_hi"] = np.percentile(Pb, [2.5, 97.5])
        rows.append(r)
    S = pd.DataFrame(rows)
    out = os.path.join(ROOT, "bench/results/b_ncurve_summary.csv")
    S.to_csv(out, index=False)
    print(f"groups: {len(S)}  deterministic (excluded): {int(S.deterministic.sum())}  -> {out}")
    nd = S[~S.deterministic]
    print(f"bias: max|z| over non-deterministic rows = {nd.maxabsz.max():.2f}")

    def med_ci(x):
        x = np.asarray(x)
        b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
        return np.median(x), *np.percentile(b, [2.5, 97.5])

    prim = nd[nd["gt"] <= 0.1]
    print("\nPRIMARY (gt <= 0.1): median G = Var_bestfixed / Var_greedy  [95% CI over groups]")
    meds = {}
    for n, g in prim.groupby("n"):
        m, lo, hi = med_ci(g.G)
        meds[n] = (m, lo, hi)
        worse = int((g.G_hi < 1).sum())
        better = int((g.G_lo > 1).sum())
        pb = int((g.R_proj_lo > 1).sum())
        pw = int((g.R_proj_hi < 1).sum())
        print(f"  n={n:2d}: groups {len(g):3d}  median G {m:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"greedy better/worse (CI) {better}/{worse}  | proj vs std better/worse {pb}/{pw}  "
              f"median G_vs_std {np.median(g.G_vs_std):.2f}")
    print("\n  by family/variant (median G per n):")
    print(prim.pivot_table(index=["family", "variant"], columns="n", values="G", aggfunc="median").round(3))
    print("\n  by observable (median G per n):")
    print(prim.pivot_table(index="obs", columns="n", values="G", aggfunc="median").round(3))
    if "G_pi" in prim:
        p6 = prim[prim.n == 6].dropna(subset=["G_pi"])
        if len(p6):
            m, lo, hi = med_ci(p6.G_pi)
            print(f"\n  PI vs best-fixed at n=6: median {m:.3f} [{lo:.3f}, {hi:.3f}], "
                  f"CI>1 {int((p6.G_pi_lo > 1).sum())}/{len(p6)}, CI<1 {int((p6.G_pi_hi < 1).sum())}")
    sec = nd[nd["gt"] > 0.1]
    if len(sec):
        print("\nSECONDARY (gt = 0.3): median G per n:",
              sec.groupby("n").G.median().round(3).to_dict())

    print("\nDECISION (pre-registered):")
    if 10 in meds and 12 in meds:
        ok = meds[10][0] >= 1.2 and meds[12][0] >= 1.2 and meds[12][1] >= 1.1
        print(f"  median G n10 = {meds[10][0]:.3f}, n12 = {meds[12][0]:.3f} (CI lo {meds[12][1]:.3f})"
              f" -> H_adapt {'SUPPORTED' if ok else 'FALSIFIED'}")
    else:
        print("  incomplete: n = 10 and n = 12 both required")


if __name__ == "__main__":
    main()
