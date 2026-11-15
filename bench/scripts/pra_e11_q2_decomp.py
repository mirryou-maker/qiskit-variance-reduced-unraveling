"""POST HOC (review 3, 2026-11-14): where does the E11 Q2 gain W_multi come from?

Per configuration k (E11, eps_o^2 = Var_bf(o)/20000, i.e. analyze_e11.q2_config with edef A20k, subset all):
  F1      = min_c t_c max_{o in O_k} Var_c(o)/eps_o^2       (one oracle fixed run meeting every target; as analyze_e11)
  F_ideal = max_{o in O_k} t_bf(o) Var_bf(o)/eps_o^2 = 20000 max_o t_bf(o)
            (cost if one fixed candidate were the per-observable oracle for every o at once; no fixed run can do better
             than the hardest observable's own oracle, up to the choice of bf by variance rather than by Var*t)
  P_F     = F1 / F_ideal  (>= 1 up to that caveat): penalty of serving all observables with ONE fixed candidate
  G_tab   = F_ideal / S_multi: the multi-target table against that penalty-free reference
  W_multi = F1 / S_multi = P_F * G_tab (identity; checked against b_e11_multi.csv)
Per observable: multi_W1_bf = Var_bf t_f / (Var_m t_m) is in b_e11_summary.csv (per group).
Also: |O_k| distribution per channel; W_multi median for |O_k| = 1, >= 2, = 4 (98.75% config bootstrap, B = 4000).
Reads b_e11_u94.csv, b_e11_raw/ (via analyze_e11.load), b_e11_multi.csv, b_e11_summary.csv.
Output: bench/results/pra_e11_q2_decomp.txt (stdout) and pra_e11_q2_decomp.csv (per configuration).
Usage (ubuntu94, env sci): python bench/scripts/pra_e11_q2_decomp.py > bench/results/pra_e11_q2_decomp.txt
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("analyze_e11", os.path.join(HERE, "analyze_e11.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
R = A.R
LO, HI = 0.625, 99.375


def med_ci(x, seed=20261113, B=4000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [LO, HI])


C = A.load(os.path.join(R, "b_e11_u94.csv"), os.path.join(R, "b_e11_raw"), os.path.join(R, "b_e11_groups.csv"))
ref = pd.read_csv(os.path.join(R, "b_e11_multi.csv"))
ref = ref[(ref.edef == "A20k") & (ref.subset == "all")].set_index(A.CFG)
rows = []
for c in C:
    obs = c["obs"]
    r, _ = A.q2_config(c, obs, "A20k")
    V = np.array([c["X"][o].var(axis=1, ddof=1) for o in obs])
    vbf, ibf = V.min(axis=1), V.argmin(axis=1)
    eps2 = vbf / A.NPROD
    t = c["t"]
    F1 = float(np.min(t * (V / eps2[:, None]).max(axis=0)))
    Fid = float(np.max(t[ibf] * vbf / eps2))
    jstar = int(np.argmin(t * (V / eps2[:, None]).max(axis=0)))
    assert abs(F1 - r["F1"]) < 1e-9 * F1
    w_ref = float(ref.loc[tuple(c["key"]), "W_multi"])
    assert abs(r["W_multi"] - w_ref) < 1e-9 * w_ref
    nbest = len(set(ibf.tolist()))
    rows.append(dict(zip(A.CFG, c["key"]), m=len(obs), n_distinct_bf=nbest, fixed_choice=c["labels"][jstar],
                     F1=F1, F_ideal=Fid, S_multi=r["Smulti"], P_F=F1 / Fid, G_tab=Fid / r["Smulti"],
                     W_multi=r["W_multi"]))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(R, "pra_e11_q2_decomp.csv"), index=False)
S = pd.read_csv(os.path.join(R, "b_e11_summary.csv"))
print("POST HOC decomposition of E11 Q2 (A20k, all TN observables of each configuration); W_multi = P_F * G_tab")
for ch, d in D.groupby("channel"):
    print(f"\n== {ch}: {len(d)} configurations; |O_k| distribution {d.m.value_counts().sort_index().to_dict()}")
    for lab, sub in [("all", d), ("|O_k|=1", d[d.m == 1]), ("|O_k|>=2", d[d.m >= 2]), ("|O_k|=4", d[d.m == 4])]:
        if len(sub) == 0:
            continue
        for col in ("W_multi", "P_F", "G_tab"):
            md, lo, hi = med_ci(sub[col])
            print(f"  {lab:9s} {col:8s} configs {len(sub):2d}  median {md:.3f} [{lo:.3f}, {hi:.3f}]  >1 "
                  f"{int((sub[col] > 1 + 1e-12).sum())}/{len(sub)}")
    m2 = d[d.m >= 2]
    print(f"  |O_k|>=2: configs where the per-observable oracles differ (n_distinct_bf > 1): "
          f"{int((m2.n_distinct_bf > 1).sum())}/{len(m2)}; median log P_F / log W_multi over configs with W_multi > 1: "
          f"{np.median(np.log(m2[m2.W_multi > 1].P_F) / np.log(m2[m2.W_multi > 1].W_multi)):.3f}")
    g = S[S.channel == ch]
    md, lo, hi = med_ci(g.multi_W1_bf, seed=20261114, B=2000)
    print(f"  per observable (groups {len(g)}): multi_W1_bf median {md:.3f} [{lo:.3f}, {hi:.3f}] (B 2000, new draw); "
          f"single_W_bf median {g.single_W_bf.median():.3f}")
