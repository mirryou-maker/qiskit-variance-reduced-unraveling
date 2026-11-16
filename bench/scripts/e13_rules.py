"""E13 group classification (docs/plan_B_characterization_2026-11-06.md, "E13 pre-registration").

The cap-1e7 rule is the E9a -> E9c -> E9d chain, unchanged:
  E9a  (analyze_e11.classify == analyze_e9a.py): solved iff some M in {1e2..1e5} has |PP_M - PP_10M| <= eps and
       |PP_1e6 - traj| / SE_ref < 4; TN = not solved or t_PP(M*) > t_traj20k. eps = sqrt(Var_std / 20000).
  E9c  (analyze_e9c.py): grid {1e2..1e6, 3e6}, partner 10 M; solved(C) iff some grid M <= C with partner available has
       |PP_M - PP_partner| <= eps AND |PP_top - traj| / SE_ref < 4 (PP_top = PP at the largest M available among
       1e2..3e7); E9a-solved stay solved.  TN(C) for C in {1e5, 1e6, 3e6}.
  E9d  (analyze_e9d.py): solved(1e7) = solved(3e6) OR (|PP_1e7 - PP_1e8| <= eps AND |PP_1e8 - traj| / SE_ref < 4).
caps(S) adds TN100000, TN1000000, TN3000000, TN10000000 to a per-group frame with columns pp<M>, traj_mean, var_std,
nref, eps, TN (E9a) and pp100000000 (NaN where not run). check() verifies that caps() reproduces
bench/results/b_e9d_summary.csv (the E9 data) exactly.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MS_A = [100, 1000, 10000, 100000, 1000000]
MS_B = [3000000, 10000000, 30000000]
M_C = 100000000
GRID = MS_A + [3000000]
CAPS = [100000, 1000000, 3000000]


def _true(x):
    return x.astype(str).isin(["True", "1", "1.0"])


def caps(S):
    S = S.copy()
    for m in MS_A + MS_B + [M_C]:
        if f"pp{m}" not in S:
            S[f"pp{m}"] = np.nan
    se = np.sqrt(S.var_std.clip(lower=1e-300) / S.nref)
    allM = MS_A + MS_B
    top = S[[f"pp{m}" for m in allM]].astype(float).ffill(axis=1).iloc[:, -1]
    z_top = (top - S.traj_mean) / se
    e9a_tn = _true(S.TN).to_numpy()
    for C in CAPS:
        conv = np.zeros(len(S), bool)
        for m in GRID:
            if m > C:
                continue
            d = (S[f"pp{m}"] - S[f"pp{10 * m}"]).abs()
            conv |= (d <= S.eps).fillna(False).to_numpy()
        solved = conv & (z_top.abs() < 4).to_numpy()
        solved |= ~e9a_tn
        S[f"TN{C}"] = (~solved).astype(float)
    z8 = (S[f"pp{M_C}"] - S.traj_mean) / se
    new = ((S["pp10000000"] - S[f"pp{M_C}"]).abs() <= S.eps) & (z8.abs() < 4)
    S["TN10000000"] = (~((S.TN3000000 == 0) | new.fillna(False).to_numpy())).astype(float)
    S["need_B"] = e9a_tn                                    # runs at 3e6 / 1e7 / 3e7 are needed
    S["need_C"] = S.TN3000000 == 1                          # run at 1e8 is needed
    S["missing_C"] = S.need_C & S[f"pp{M_C}"].isna()
    S["z_top"], S["z_1e8"] = z_top, z8
    return S


def check():
    """caps() on the E9 per-group data must reproduce b_e9d_summary.csv (all non-deterministic groups, all n)."""
    ref = pd.read_csv(os.path.join(ROOT, "bench/results/b_e9d_summary.csv"))
    ref = ref.rename(columns={"pp1e8": f"pp{M_C}"})
    got = caps(ref.drop(columns=["TN100000", "TN1000000", "TN3000000", "TN10000000"]))
    bad = {}
    for c in ("TN100000", "TN1000000", "TN3000000", "TN10000000"):
        bad[c] = int((got[c].to_numpy() != ref[c].to_numpy()).sum())
    return len(ref), bad
