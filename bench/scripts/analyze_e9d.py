"""Analysis of E9d (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E9d pre-registration").

Inputs: bench/results/b_e9c_summary.csv (E9a + E9c, per group, caps 1e5/1e6/3e6) and bench/results/b_e9d_main_p*.csv
(PP at M = 1e8 on groups with TN(3e6) = 1).
  solved(1e7) = solved(3e6) OR (|PP_1e7 - PP_1e8| <= eps AND |PP_1e8 - traj mean| / SE_ref < 4)
  TN(1e7) over the 720 n 20/24 groups; config-cluster 95% CI. Missing groups: pessimistic (TN) / optimistic (solved) bounds.
n = 12: true_TN(1e7) = true_TN(3e6) AND |PP_1e7 - exact| > eps.
Decision: TN(1e7) <= 5% -> claim withdrawn; > 5% -> regime survives to 1e7; bounds straddling 5% -> undetermined.
Post hoc (labelled, not a decision input): partner 3M at cap 1e7 (|PP_1e7 - PP_3e7| <= eps, z with PP_3e7).
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = KEYS[:-1]
BIG = 100_000_000


def cluster_ci(d, col, rng, B=4000):
    ids = d[CFG].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    x = d[col].to_numpy(float)
    b = [x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])].mean() for _ in range(B)]
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(13)
    S = pd.read_csv(os.path.join(R, "b_e9c_summary.csv"))
    files = sorted(glob.glob(os.path.join(R, "b_e9d_main_p*.csv")))
    E = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=KEYS + ["M", "status"])
    E = E.drop_duplicates(KEYS + ["M"], keep="last")
    print(f"E9d rows {len(E)} from {len(files)} files; status {E.status.value_counts().to_dict() if len(E) else {}}")
    if len(E):
        print("hosts/gpus:", E.groupby(["host", "gpu"]).size().to_dict())
    ok = E[(E.status == "ok") & (E.M == BIG)][KEYS + ["value", "wall_s", "peak", "uniq"]].rename(
        columns={"value": "pp1e8", "wall_s": "t1e8", "peak": "peak1e8", "uniq": "uniq1e8"})
    S = S.merge(ok, on=KEYS, how="left")
    se = np.sqrt(S.var_std.clip(lower=1e-300) / S.nref)
    need = S.TN3000000 == 1
    S["missing"] = need & S.pp1e8.isna()
    z8 = (S.pp1e8 - S.traj_mean) / se
    new = ((S.pp10000000 - S.pp1e8).abs() <= S.eps) & (z8.abs() < 4)
    solved = (S.TN3000000 == 0) | new.fillna(False)
    S["TN10000000"] = (~solved).astype(float)                       # missing counted as TN (pessimistic)
    S["TN10000000_opt"] = np.where(S.missing, 0.0, S.TN10000000)     # missing counted as solved
    z3 = (S.pp30000000 - S.traj_mean) / se
    alt = ((S.pp10000000 - S.pp30000000).abs() <= S.eps) & (z3.abs() < 4)
    S["TN10000000_p3"] = (~((S.TN3000000 == 0) | alt.fillna(False))).astype(float)
    if "exact" in S:
        tr = (S.pp10000000 - S.exact).abs() <= S.eps
        S["trueTN10000000"] = np.where(S.exact.notna(), ((S.trueTN3000000 == 1) & ~tr.fillna(False)).astype(float), np.nan)
    S.to_csv(os.path.join(R, "b_e9d_summary.csv"), index=False)
    print(f"-> {os.path.join(R, 'b_e9d_summary.csv')}")

    v = S[S.n == 12]
    if len(v):
        tn, tt = v.TN10000000.astype(bool), v.trueTN10000000.astype(bool)
        print(f"\n== n = 12 (360): cap 1e7 classified TN {int(tn.sum())}, true TN {int(tt.sum())}; false-solved "
              f"{int((~tn & tt).sum())} ({(~tn & tt).sum() / max((~tn).sum(), 1):.1%} of solved), false-TN {int((tn & ~tt).sum())}")
        vv = v[v.TN3000000 == 1]
        print(f"   n12 PP_1e8 vs exact (PP_1e8 should be untruncated, 4^12 < 1e8): max |diff| "
              f"{(vv.pp1e8 - vv.exact).abs().max():.2e} over {int(vv.pp1e8.notna().sum())}")
    big = S[S.n.isin([20, 24])]
    print(f"\n== n 20/24: {len(big)} groups; TN(3e6) {int(big.TN3000000.sum())}; PP_1e8 present "
          f"{int(big.pp1e8.notna().sum())}; missing {int(big.missing.sum())}")
    for col, lab in [("TN100000", "cap 1e5"), ("TN1000000", "cap 1e6"), ("TN3000000", "cap 3e6"),
                     ("TN10000000", "cap 1e7 (pessimistic)"), ("TN10000000_opt", "cap 1e7 (optimistic)")]:
        m, lo, hi = cluster_ci(big, col, rng)
        print(f"  TN {lab:22s}: {int(big[col].sum()):3d}/{len(big)} = {m:.1%} [{lo:.1%}, {hi:.1%}]  "
              + "  ".join(f"{ch} {g[col].mean():.1%}" for ch, g in big.groupby("channel")))
    print("\n== where at cap 1e7 (pessimistic) ==")
    for col in ("family", "variant", "n", "L", "gt", "obs"):
        print(f"  by {col}:", {k: round(x, 3) for k, x in big.groupby(col).TN10000000.mean().items()})
    print(big.pivot_table(index="channel", columns="family", values="TN10000000", aggfunc="mean").round(3))
    t = big[need.loc[big.index]]
    r = (t.t1e8 / t.t_traj20k).dropna()
    if len(r):
        print(f"\n== cost: t_PP(1e8) median {t.t1e8.median():.0f} s, max {t.t1e8.max():.0f} s; t_PP/t_traj20k median {r.median():.2e},"
              f" max {r.max():.2e}; distinct terms before truncation median {t.uniq1e8.median():.3g}, max {t.uniq1e8.max():.3g}")
        print(f"   |PP_1e8 - PP_3e7| median {(t.pp1e8 - t.pp30000000).abs().median():.2e}; "
              f"|PP_1e7 - PP_1e8| / eps median {((t.pp10000000 - t.pp1e8).abs() / t.eps).median():.2f}; |z_1e8| >= 4: {int((z8[t.index].abs() >= 4).sum())}")
    m, lo, hi = cluster_ci(big, "TN10000000_p3", rng)
    print(f"\nPOST HOC (not pre-registered as a decision input): cap 1e7 with partner 3M: {m:.1%} [{lo:.1%}, {hi:.1%}]")
    pes, opt = big.TN10000000.mean(), big.TN10000000_opt.mean()
    if (pes > 0.05) == (opt > 0.05):
        verdict = ("trajectory-needing regime SURVIVES to cap 1e7 (> 5%)" if pes > 0.05
                   else "claim WITHDRAWN (TN(1e7) <= 5%)")
    else:
        verdict = "UNDETERMINED (missing groups straddle 5%)"
    print(f"\nDECISION (pre-registered): TN(cap 1e7) = {pes:.1%} (pessimistic) / {opt:.1%} (optimistic) -> {verdict}")


if __name__ == "__main__":
    main()
