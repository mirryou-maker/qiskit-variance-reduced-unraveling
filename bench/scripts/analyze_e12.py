"""Analysis of E12 (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E12 pre-registration"): sensitivity of
the static pilot table to the pilot size P and the truncation M. A sensitivity study, NOT a decision test.

Per group (98 E9b groups, n = 20) and setting p<P>_M<M>:
  Var_fixed(c): E9b raw arrays (N = 2000), valid iff the E12 re-timed fixed runs match E9b (>= 99%).
  t_f(c)      : E12 fixed wall / N_t (same session as the static runs).
  Var_static  : E12 production run (N = 2000);  t_static = wall / N + (bp_s + build_s) / 20000 (as E10).
  W_bf = Var_bf t_f / (Var_static t_static),  R_bf = Var_bf / Var_static (best fixed = argmin Var over candidates).
Per channel x setting: median over groups with the group-bootstrap 98.75% interval and the config-cluster 98.75% interval
(analyze_e10.med_ci / cluster_ci). Also: paired ratio W(setting) / W(p74_M100) per group (median, group bootstrap),
table agreement with the p74_M100 table, non-std fraction, build_s, bias |z| vs the E9b best-fixed mean.
Usage: python bench/scripts/analyze_e12.py [b_e12 csv [raw dir [tag]]] > bench/results/b_e12_analysis.txt
Outputs b_e12_summary<tag>.csv (per group x setting) and b_e12_medians<tag>.csv (per channel x setting).
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
_spec = importlib.util.spec_from_file_location("analyze_e10", os.path.join(os.path.dirname(__file__), "analyze_e10.py"))
A10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A10)
KEYS, CFG, NCAND, NPROD = A10.KEYS, A10.CFG, A10.NCAND, A10.NPROD
REF = "p74_M100"
ORDER = ["p37_M100", "p74_M100", "p296_M100", "p74_M10", "p74_M1000"]


def main():
    f12 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "b_e12_v100.csv")
    rawdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "b_e12_raw")
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    rng = np.random.default_rng(2026)
    d = pd.read_csv(f12).drop_duplicates(KEYS + ["kind", "scheme"], keep="last")
    rows = []
    for k, g in d.groupby(KEYS):
        ch, fam, var, n, L, gt, o = k
        fx = g[g.kind == "fixed"]
        st = g[g.kind == "static"].set_index("scheme")
        if len(fx) != NCAND[ch] or not len(st):
            continue
        ks = A10.key_str(ch, fam, var, n, L, gt)
        raw = np.load(os.path.join(R, "b_e9b_raw", ks + ".npz"))
        labels = list(fx.scheme)
        arr = {s: raw[f"fixed_{s}_{o}"].astype(np.float64) for s in labels}
        vf = {s: arr[s].var(ddof=1) for s in labels}
        tf = dict(zip(fx.scheme, fx.wall_s / fx.shots))
        best = min(vf, key=vf.get)
        N = len(arr[best])
        tabs = {}
        for t in st.index:
            tabs[t] = np.load(os.path.join(rawdir, f"{ks}_{t}_{o}.npz"))["tab"].astype(int)
        for t in st.index:
            s = st.loc[t]
            x = np.load(os.path.join(rawdir, f"{ks}_{t}_{o}.npz"))["x"].astype(np.float64)
            vs = x.var(ddof=1)
            ts = float(s.wall_s) / float(s.shots) + (float(s.bp_s) + float(s.build_s)) / NPROD
            rows.append(dict(zip(KEYS, k), setting=t, P=int(s.pilot_shots), M=int(s.M), best=best,
                             match_min=float(fx.match.min()), var_bf=vf[best], t_f=tf[best], var=vs, t_static=ts,
                             tratio=ts / tf[best], build_s=float(s.build_s), bp_s=float(s.bp_s), nonstd=float(s.nonstd),
                             agree_best=float(np.mean(tabs[t] == labels.index(best))),
                             agree_ref=float(np.mean(tabs[t] == tabs[REF])) if REF in tabs else np.nan,
                             z=(x.mean() - arr[best].mean()) / np.sqrt(vs / len(x) + vf[best] / N),
                             R_bf=vf[best] / vs, W_bf=vf[best] * tf[best] / (vs * ts),
                             gpu_other=";".join(sorted(set(g.gpu_other.astype(str)))),
                             other_py=";".join(sorted(set(g.other_py.astype(str)))),
                             gpu_util_max=float(pd.to_numeric(g.gpu_util, errors="coerce").max()),
                             load1_max=float(g.load1.max())))
    S = pd.DataFrame(rows)
    ref = S[S.setting == REF].set_index(KEYS)
    S["W_rel"] = [r.W_bf / ref.loc[tuple(r[KEYS]), "W_bf"] if tuple(r[KEYS]) in ref.index else np.nan
                  for _, r in S.iterrows()]
    out = os.path.join(R, f"b_e12_summary{tag}.csv")
    S.to_csv(out, index=False)
    ng = len(S.groupby(KEYS))
    print(f"E12 analysis of {f12}\ngroups {ng} (configs {len(S.groupby(CFG))}), rows {len(S)} -> {out}")
    print(f"identity check: min match {S.match_min.min():.5f}; groups < 0.99: "
          f"{int((S.groupby(KEYS).match_min.min() < 0.99).sum())}")
    print(f"idle record: gpu_other {sorted(set(';'.join(S.gpu_other).split(';')))[:20]}; other_py "
          f"{sorted(set(S.other_py))}; gpu_util max {S.gpu_util_max.max()} %; load1 max {S.load1_max.max()}")
    zz = S.z.abs()
    print(f"bias |z| vs E9b best-fixed mean: max {zz.max():.2f}, n(|z|>4) {int((zz > 4).sum())} of {len(S)}")
    print("\nSENSITIVITY (not a decision test). Median over groups [98.75% group bootstrap] {98.75% config-cluster}")
    recs = []
    for ch, C0 in S.groupby("channel"):
        for t in [x for x in ORDER if x in set(C0.setting)] + sorted(set(C0.setting) - set(ORDER)):
            C = C0[C0.setting == t]
            rec = dict(channel=ch, setting=t, P=int(C.P.iloc[0]), M=int(C.M.iloc[0]), groups=len(C),
                       configs=len(C.groupby(CFG)))
            for col in ("W_bf", "R_bf", "W_rel"):
                v = C[col].dropna()
                m, lo, hi = A10.med_ci(v, rng)
                clo, chi = A10.cluster_ci(C.dropna(subset=[col]), col, rng) if len(v) > 1 else (np.nan, np.nan)
                rec.update({col: m, f"{col}_lo": lo, f"{col}_hi": hi, f"{col}_clo": clo, f"{col}_chi": chi})
            rec.update(tratio=C.tratio.median(), build_s=C.build_s.median(), bp_s=C.bp_s.median(),
                       nonstd=C.nonstd.median(), agree_best=C.agree_best.median(), agree_ref=C.agree_ref.median())
            recs.append(rec)
            print(f"  {ch:4s} {t:10s} groups {len(C):2d}  W_bf {rec['W_bf']:.3f} [{rec['W_bf_lo']:.3f}, {rec['W_bf_hi']:.3f}] "
                  f"{{{rec['W_bf_clo']:.3f}, {rec['W_bf_chi']:.3f}}}  R_bf {rec['R_bf']:.3f} [{rec['R_bf_lo']:.3f}, "
                  f"{rec['R_bf_hi']:.3f}] {{{rec['R_bf_clo']:.3f}, {rec['R_bf_chi']:.3f}}}  W/W_ref {rec['W_rel']:.3f} "
                  f"[{rec['W_rel_lo']:.3f}, {rec['W_rel_hi']:.3f}]  t_s/t_f {rec['tratio']:.3f} build_s "
                  f"{rec['build_s']:.2f} bp_s {rec['bp_s']:.2f} nonstd {rec['nonstd']:.3f} agree_ref {rec['agree_ref']:.3f}")
    pd.DataFrame(recs).to_csv(os.path.join(R, f"b_e12_medians{tag}.csv"), index=False)
    print("\n== breakdown (POST HOC): median W_bf by L ==")
    print(S.pivot_table(index=["channel", "setting"], columns="L", values="W_bf", aggfunc="median").round(3))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
