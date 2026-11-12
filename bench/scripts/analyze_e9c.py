"""Analysis of E9c (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E9c pre-registration").

Inputs: bench/results/b_e9a_summary.csv (E9a, all groups) and bench/results/b_e9c_main_p*.csv (PP at 3e6, 1e7, 3e7 on E9a TN groups).
Grid G = {1e2, 1e3, 1e4, 1e5, 1e6, 3e6}, partner(M) = 10 M (3e6 -> 3e7).
  solved(C) = some M in G, M <= C, partner available, |PP_M - PP_partner| <= eps, and |PP_top - traj mean| / SE_ref < 4
              (PP_top = PP at the largest M available for the group)
  TN(C) = not solved(C),  C in {1e5, 1e6, 3e6}.   E9a-solved groups stay solved (criterion only gains M values).
n = 12: true_TN(C) = no M in G, M <= C, with |PP_M - exact| <= eps; false-solved / false-TN rates per cap.
Decision: TN(1e6) > 5% of n 20/24 groups -> regime survives (report the curve); <= 5% -> claim withdrawn.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = KEYS[:-1]
GRID = [100, 1000, 10000, 100000, 1000000, 3000000]
PART = {m: 10 * m for m in GRID}
CAPS = [100000, 1000000, 3000000]
NEW = [3000000, 10000000, 30000000]


def cluster_ci(d, col, rng, B=4000):
    ids = d[CFG].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    x = d[col].to_numpy(float)
    b = [x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])].mean() for _ in range(B)]
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(9)
    S = pd.read_csv(os.path.join(R, "b_e9a_summary.csv"))
    S = S[~S.deterministic.astype(str).eq("True")].copy()
    files = sorted(glob.glob(os.path.join(R, "b_e9c_main_p*.csv")))
    E = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=KEYS + ["M"])
    E = E.drop_duplicates(KEYS + ["M"], keep="last")
    print(f"E9c rows {len(E)} from {len(files)} files; status counts {E.status.value_counts().to_dict() if len(E) else {}}")
    print("hosts/gpus/idle:", E.groupby(["host", "gpu"]).gpu_other.agg(lambda x: sorted(set(x))).to_dict() if len(E) else {})
    ok = E[E.status == "ok"]
    for m in NEW:
        sub = ok[ok.M == m][KEYS + ["value", "wall_s", "peak"]].rename(
            columns={"value": f"pp{m}", "wall_s": f"t{m}", "peak": f"peak{m}"})
        S = S.merge(sub, on=KEYS, how="left")
    S["e9c_tried"] = S.set_index(KEYS).index.isin(E.set_index(KEYS).index) if len(E) else False
    se = np.sqrt(S.var_std.clip(lower=1e-300) / S.nref)
    allM = [100, 1000, 10000, 100000, 1000000] + NEW
    top = S[[f"pp{m}" for m in allM]].ffill(axis=1).iloc[:, -1]
    S["ppTop"] = top
    S["z_top"] = (top - S.traj_mean) / se
    for C in CAPS:
        conv = np.zeros(len(S), bool)
        for m in GRID:
            if m > C:
                continue
            d = (S[f"pp{m}"] - S[f"pp{PART[m]}"]).abs()
            conv |= (d <= S.eps).fillna(False).to_numpy()
        solved = conv & (S.z_top.abs() < 4).to_numpy()
        solved |= ~S.TN.astype(str).eq("True").to_numpy()            # E9a-solved stay solved
        S[f"TN{C}"] = (~solved).astype(float)
        if "exact" in S:
            tr = np.zeros(len(S), bool)
            for m in GRID:
                if m <= C:
                    tr |= ((S[f"pp{m}"] - S.exact).abs() <= S.eps).fillna(False).to_numpy()
            S[f"trueTN{C}"] = np.where(S.exact.notna(), (~tr).astype(float), np.nan)
    out = os.path.join(R, "b_e9c_summary.csv")
    S.to_csv(out, index=False)
    print(f"-> {out}")

    v = S[S.n == 12]
    print("\n== n = 12 validation (360 groups) ==")
    for C in CAPS:
        tn, tt = v[f"TN{C}"].astype(bool), v[f"trueTN{C}"].astype(bool)
        print(f"  cap {C:>8.0e}: classified TN {int(tn.sum()):3d}, true TN {int(tt.sum()):3d}; "
              f"false-solved {int((~tn & tt).sum())} ({(~tn & tt).sum() / max((~tn).sum(), 1):.1%} of solved), "
              f"false-TN {int((tn & ~tt).sum())}")
    big = S[S.n.isin([20, 24])]
    tried = big[big.TN.astype(str).eq("True")]
    print(f"\n== n 20/24: {len(big)} groups, {len(tried)} E9a-TN; E9c rows present for {int(tried.e9c_tried.sum())}; "
          f"ok at 3e6/1e7/3e7: {[int(tried[f'pp{m}'].notna().sum()) for m in NEW]}")
    for C in CAPS:
        m, lo, hi = cluster_ci(big, f"TN{C}", rng)
        print(f"  TN(cap {C:.0e}) all: {int(big[f'TN{C}'].sum())}/{len(big)} = {m:.1%} [{lo:.1%}, {hi:.1%}] (config-cluster 95%)")
        for ch, g in big.groupby("channel"):
            mm, l2, h2 = cluster_ci(g, f"TN{C}", rng)
            print(f"      {ch:5s} {int(g[f'TN{C}'].sum())}/{len(g)} = {mm:.1%} [{l2:.1%}, {h2:.1%}]")
    print("\n== where (fraction TN at cap 1e6 / 3e6) ==")
    for col in ("family", "variant", "n", "L", "gt", "obs"):
        print(f"  by {col}:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in
                               big.groupby(col)[["TN1000000", "TN3000000"]].mean().iterrows()})
    print("  channel x family at cap 1e6:")
    print(big.pivot_table(index="channel", columns="family", values="TN1000000", aggfunc="mean").round(3))
    print("\n== cost ==")
    for m in NEW:
        r = (tried[f"t{m}"] / tried.t_traj20k).dropna()
        if len(r):
            print(f"  t_PP({m:.0e}) / t_traj20k: median {r.median():.2e}, max {r.max():.2e};  "
                  f"t_PP median {tried[f't{m}'].median():.1f} s;  peak terms median {tried[f'peak{m}'].median():.3g}")
    print(f"  |z_top| >= 4 among n 20/24 E9a-TN: {int((tried.z_top.abs() >= 4).sum())}")
    k = big["TN1000000"].mean()
    print(f"\nDECISION (pre-registered): TN(cap 1e6) = {k:.1%} -> "
          + ("trajectory-needing regime SURVIVES (> 5%): report TN vs cap" if k > 0.05
             else "claim WITHDRAWN (<= 5%): E9b information only"))


if __name__ == "__main__":
    main()
