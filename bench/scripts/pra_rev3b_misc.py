"""Review 3 (2026-11-14) supporting numbers. Reads summary CSVs only (no simulation, no raw arrays).

(A) Decision-line intervals of the pre-specified decision cells (reproduction, not post hoc): the decision lines of
    analyze_e10.py / analyze_e11.py print only the lower bound; here the same draw is repeated to give both bounds.
    E10 and E10-V100: med_ci(<t>_W_bf of channel, default_rng(7), B = 2000) per cell, as analyze_e10.main (decision).
    E11 Q1: med_ci(single_W_bf, default_rng(7)); Q2: med_ci(W_multi A20k/all, default_rng(20261113), B = 4000).
(B) POST HOC family-wise view: the E10 ad pilot cell at a Bonferroni level over the 12 cells of E9b, E9e and E10
    (1 - 0.05/12, two-sided percentile interval; B = 20000, default_rng(7)).
(C) POST HOC practical weight of PP on the 98 E9b/E10 groups (b_e9d_summary.csv, n = 20): groups not TN at cap 1e7,
    and for them the PP cost that establishes convergence, t_PP(M*) + t_PP(partner of M*), relative to the
    20000-trajectory reference time t_traj20k (same columns as analyze_e9d; GPUs and hosts differ, indicative only).
    M* = smallest M on {1e2, 1e3, 1e4, 1e5, 1e6, 3e6, 1e7} with |PP(M) - PP(partner)| <= eps (partner 10 M; 3e7 for 3e6;
    1e8 for 1e7). For the groups still TN at 1e7, the classification cost t(1e7) + t(1e8) is given as well.
(D) POST HOC E11 selection: TN groups whose PP differences converged (some d_M <= eps, M <= 1e5) but which are TN only
    through the |z| >= 4 clause (b_e11_select_summary.csv).
(E) POST HOC hard proxy (r = |PP_1e5 - PP_1e6| / eps > 10): counts in E11 and calibration on the 98 E9b groups.
Usage (ubuntu94, env sci): python bench/scripts/pra_rev3b_misc.py > bench/results/pra_rev3b_misc.txt
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench", "results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
LO, HI = 0.625, 99.375


def med_ci(x, rng, B=2000, lo=LO, hi=HI):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [lo, hi])


print("(A) decision-line intervals (same draws as the pre-specified decision lines; 98.75%)")
for name, f in (("E10", "b_e10_summary.csv"), ("E10-V100", "b_e10_summary_v100.csv")):
    S = pd.read_csv(os.path.join(R, f))
    for t in ("pilot", "ideal"):
        for ch, C in S.groupby("channel"):
            m, lo, hi = med_ci(C[f"{t}_W_bf"], np.random.default_rng(7))
            print(f"  {name:8s} {t:5s} {ch:4s} groups {len(C):2d}: median {m:.3f} decision [{lo:.3f}, {hi:.3f}]")
S11 = pd.read_csv(os.path.join(R, "b_e11_summary.csv"))
M11 = pd.read_csv(os.path.join(R, "b_e11_multi.csv"))
M11 = M11[(M11.edef == "A20k") & (M11.subset == "all")]
for ch in ("ad", "deph"):
    m, lo, hi = med_ci(S11[S11.channel == ch].single_W_bf, np.random.default_rng(7))
    print(f"  E11 Q1   pilot {ch:4s} groups {int((S11.channel == ch).sum()):2d}: median {m:.3f} decision [{lo:.3f}, {hi:.3f}]")
    m, lo, hi = med_ci(M11[M11.channel == ch].W_multi, np.random.default_rng(20261113), 4000)
    print(f"  E11 Q2   multi {ch:4s} configs {int((M11.channel == ch).sum()):2d}: median {m:.3f} decision [{lo:.3f}, {hi:.3f}]")

print("\n(B) POST HOC family-wise: E10 ad pilot at 1 - 0.05/12 (12 cells: E9b 4, E9e 4, E10 4)")
S = pd.read_csv(os.path.join(R, "b_e10_summary.csv"))
a = 100 * 0.025 / 12
m, lo, hi = med_ci(S[S.channel == "ad"].pilot_W_bf, np.random.default_rng(7), 20000, a, 100 - a)
print(f"  level {1 - 0.05 / 12:.5f}: median {m:.3f} [{lo:.3f}, {hi:.3f}]")
m, lo, hi = med_ci(S11[S11.channel == "ad"].single_W_bf, np.random.default_rng(7), 20000, a, 100 - a)
print(f"  (same level, E11 Q1 ad, for comparison): median {m:.3f} [{lo:.3f}, {hi:.3f}]")

print("\n(C) POST HOC PP practical weight on the 98 E9b/E10 groups")
D = pd.read_csv(os.path.join(R, "b_e9d_summary.csv"))
D = D[D.n == 20].merge(S[KEYS], on=KEYS)
grid = [(100, "pp100", "t100", "pp1000", "t1000"), (1000, "pp1000", "t1000", "pp10000", "t10000"),
        (10000, "pp10000", "t10000", "pp100000", "t100000"), (100000, "pp100000", "t100000", "pp1000000", "t1000000"),
        (1000000, "pp1000000", "t1000000", "pp10000000", "t10000000"),
        (3000000, "pp3000000", "t3000000", "pp30000000", "t30000000"),
        (10000000, "pp10000000", "t10000000", "pp1e8", "t1e8")]
rows = []
for r in D.itertuples():
    Ms, cost = None, np.nan
    for M, a_, ta, b_, tb in grid:
        va, vb = getattr(r, a_), getattr(r, b_)
        if np.isfinite(va) and np.isfinite(vb) and abs(va - vb) <= r.eps:
            Ms, cost = M, getattr(r, ta) + getattr(r, tb)
            break
    rows.append(dict(channel=r.channel, TN1e7=int(r.TN10000000), Mstar=Ms, cost=cost, t_traj20k=r.t_traj20k,
                     cls_cost=r.t10000000 + r.t1e8))
P = pd.DataFrame(rows)
sol = P[P.TN1e7 == 0]
print(f"  groups {len(P)} (by channel {P.channel.value_counts().to_dict()}); not TN at cap 1e7: {len(sol)} "
      f"(by channel {sol.channel.value_counts().to_dict()}); TN at 1e7: {int(P.TN1e7.sum())}")
print(f"  solved groups with an M* on the grid: {int(sol.Mstar.notna().sum())}/{len(sol)}; M* distribution "
      f"{sol.Mstar.value_counts().sort_index().to_dict()}")
ratio = (sol.cost / sol.t_traj20k).dropna()
print(f"  t_PP(M*) + t_PP(partner) over t_traj20k, solved groups: median {ratio.median():.3e} (1/{1 / ratio.median():.0f}),"
      f" IQR [{ratio.quantile(.25):.3e}, {ratio.quantile(.75):.3e}], max {ratio.max():.3e}, < 1 in "
      f"{int((ratio < 1).sum())}/{len(ratio)}")
tn = P[P.TN1e7 == 1]
rt = (tn.cls_cost / tn.t_traj20k).dropna()
print(f"  TN@1e7 groups: classification cost t(1e7) + t(1e8) over t_traj20k: median {rt.median():.3e}, max {rt.max():.3e} "
      f"({len(rt)} groups)")

print("\n(D) POST HOC E11 selection: TN only through the |z| >= 4 clause")
E = pd.read_csv(os.path.join(R, "b_e11_select_summary.csv"))
E = E[~E.deterministic & E.TN]
conv = (E[["d100", "d1000", "d10000", "d100000"]].le(E.eps, axis=0)).any(axis=1)
Z = E[conv & (E.z_pp1e6.abs() >= 4)]
run = S11[KEYS].assign(run=True)
Z = Z.merge(run, on=KEYS, how="left")
print(f"  TN groups (non-deterministic) {len(E)}; converged PP differences but |z_pp1e6| >= 4: {len(Z)}; "
      f"of these in the 95 E11 groups run: {int(Z.run.fillna(False).sum())}")
for r in Z.itertuples():
    print(f"    {r.channel} {r.family} {r.variant} L{r.L} gt{r.gt} {r.obs}: z = {r.z_pp1e6:.1f}, r_hard {r.r_hard:.2f}, "
          f"run {bool(r.run) if r.run == r.run else False}")

print("\n(E) POST HOC hard proxy r > 10")
G = pd.read_csv(os.path.join(R, "b_e11_groups.csv"))
print(f"  E11 selected TN groups {len(G)}: hard {int(G.hard.sum())}; of the 95 run: hard {int(S11.hard.sum())} "
      f"(ad {int(S11[S11.channel == 'ad'].hard.sum())}, deph {int(S11[S11.channel == 'deph'].hard.sum())}), "
      f"not hard {int((~S11.hard).sum())}")
D["r"] = D.d100000 / D.eps
D["hard"] = D.r > 10
for h, g in D.groupby("hard"):
    print(f"  E9b calibration, hard={h}: groups {len(g)}, TN@1e7 {int(g.TN10000000.sum())}, not TN@1e7 "
          f"{int((g.TN10000000 == 0).sum())} ({(g.TN10000000 == 0).mean():.1%})")
