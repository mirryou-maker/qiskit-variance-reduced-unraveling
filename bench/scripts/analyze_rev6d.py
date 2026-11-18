"""Analysis of Rev6 D (pre-registered in docs/plan_B_characterization_2026-11-06.md, "Rev6 C/D pre-registration").

High-accuracy trajectory references (b_rev6d.py traj, default unraveling, 1e6 trajectories per config) for
ad_brickwork_s24_n20_L16_gt0.01 (primary) and deph_brickwork_s17_n20_L16_gt0.01 (secondary), compared with
  - Pauli propagation at caps 1e5 .. 1e8 (E13 selection, b_e13_select_summary.csv) and 3e8 / 1e9 (b_rev6d.py pp, if run),
  - every E13 estimate of the config (fixed candidates, ST, CV0 / CVbf / ST+CV cross-fitted; selection 1000-trajectory mean),
  - the Rev6 C partially-mixed fixed candidates (ad, if available).
Labels per group and cap: PP_M inaccurate iff |PP_M - ref| > eps + 2 SE_ref; accurate iff < eps - 2 SE_ref; else ambiguous.

  python analyze_rev6d.py [rev6d main dir] [e13 main dir] [e13 raw dir]
     -> bench/results/b_rev6d_summary.csv, b_rev6d_estimates.csv, b_rev6d_analysis.txt (stdout)
CPU only.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
sys.path.insert(0, os.path.dirname(__file__))
import analyze_e13 as A13  # noqa: E402

CFG = A13.CFG
KEYS = A13.KEYS
CAPS = [100000, 1000000, 3000000, 10000000, 30000000, 100000000]
REF_KEYS = ["ad_brickwork_s24_n20_L16_gt0.01", "deph_brickwork_s17_n20_L16_gt0.01"]


def label(d, eps, se):
    if not np.isfinite(d):
        return "n/a"
    return "inaccurate" if d > eps + 2 * se else ("accurate" if d < eps - 2 * se else "ambiguous")


def main():
    a = sys.argv[1:] + [None] * 5
    rdir = a[0] or os.path.join(R, "b_rev6d_main")
    e13main = a[1] or os.path.join(R, "b_e13_main")
    e13raw = a[2] or os.path.join(R, "b_e13_raw")
    T = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(rdir, "rev6d_traj_*.csv")) if os.path.getsize(f) > 0])
    T = T[T.status == "ok"].drop_duplicates(KEYS + ["block"], keep="first")
    ppf = [f for f in glob.glob(os.path.join(rdir, "rev6d_pp_*.csv")) if os.path.getsize(f) > 0]
    PP = pd.concat([pd.read_csv(f) for f in ppf]) if ppf else pd.DataFrame(columns=KEYS + ["M", "value", "status", "wall_s"])
    Sel = pd.read_csv(os.path.join(R, "b_e13_select_summary.csv"))
    S13 = pd.read_csv(os.path.join(R, "b_e13_summary.csv"))
    G = pd.read_csv(os.path.join(R, "b_e13_groups.csv"))
    C13, _ = A13.load(e13main, e13raw)
    C13 = {A13.key_str(*c["key"]): c for c in C13}
    import analyze_rev6c as A6
    N6 = {}
    for name, md, rd, new in A6.SETS:
        try:
            got, _ = A6.load_rev6c(os.path.join(R, md), os.path.join(R, rd), new)
        except Exception as e:  # noqa: BLE001
            print(f"({name} candidates not available: {type(e).__name__}: {e})")
            continue
        for kk, v in got.items():
            N6.setdefault(A13.key_str(*kk), []).append((name, new, v))
    rows, est = [], []
    for ks in REF_KEYS:
        c = C13[ks]
        k = c["key"]
        tn = set(c["obs"])
        for o in A13.OBS4:
            t = T[(T[CFG].astype(str).agg("|".join, axis=1) == "|".join(map(str, k))) & (T.obs == o)]
            if not len(t):
                continue
            n = float(t.shots.sum())
            m = float(t.value.sum()) / n
            var = (float(t.sumsq.sum()) / n - m * m) * n / (n - 1)
            se = np.sqrt(var / n)
            s = Sel[(Sel[CFG + ["obs"]].astype(str) == pd.Series(dict(zip(CFG + ["obs"], [str(x) for x in k + (o,)])))).all(axis=1)]
            s = s.iloc[0]
            eps = float(s.eps)
            row = dict(zip(KEYS, k + (o,)), e13_group=o in tn, blocks=len(t), N_ref=int(n), ref=m, se_ref=se,
                       var_ref=var, eps=eps, se_ref_over_eps=se / eps, sel_traj_mean=float(s.traj_mean),
                       z_sel=(float(s.traj_mean) - m) / np.sqrt(float(s.var_std) / float(s.nref) + se ** 2),
                       t_pp1e8=float(s.t100000000) if pd.notna(s.t100000000) else np.nan)
            sb = S13[(S13[CFG + ["obs"]].astype(str) == pd.Series(dict(zip(CFG + ["obs"], [str(x) for x in k + (o,)])))).all(axis=1)]
            row["traj_cost_eps_s"] = float(20000 * sb.t_bf.iloc[0]) if len(sb) else np.nan
            caps = {M: float(s[f"pp{M}"]) if pd.notna(s.get(f"pp{M}", np.nan)) else np.nan for M in CAPS}
            p = PP[(PP.variant == k[2]) & (PP.channel == k[0]) & (PP.L == k[4]) & (PP.obs == o) & (PP.status == "ok")]
            for _, r in p.iterrows():
                caps[int(float(r.M))] = float(r.value)
                row[f"t_pp{int(float(r.M)):.0e}"] = float(r.wall_s)
            for M, v in sorted(caps.items()):
                d = abs(v - m) if np.isfinite(v) else np.nan
                row[f"pp{M:.0e}"] = v
                row[f"d{M:.0e}_eps"] = d / eps
                row[f"lab{M:.0e}"] = label(d, eps, se)
            rows.append(row)
            # estimates
            X = c["X"][o]
            for j, lab in enumerate(c["labels"]):
                x = X[j]
                est.append(dict(zip(KEYS, k + (o,)), estimator=f"fixed {lab}", source="E13", N=len(x), mean=x.mean(),
                                se=x.std(ddof=1) / np.sqrt(len(x))))
            for name, new, nv in N6.get(ks, []):
                for jj, j in enumerate(new):
                    x = nv["X"][o][jj]
                    est.append(dict(zip(KEYS, k + (o,)), estimator=f"fixed {nv['labels'][j]}", source=name,
                                    N=len(x), mean=x.mean(), se=x.std(ddof=1) / np.sqrt(len(x))))
            for oo, e in c["per"].items():               # ST / CV runs targeted at oo; plain estimates of o
                y = e["single"]["Y"][o]
                est.append(dict(zip(KEYS, k + (o,)), estimator=f"ST(target {oo}) plain", source="E13", N=len(y),
                                mean=y.mean(), se=y.std(ddof=1) / np.sqrt(len(y))))
                if oo == o:
                    for kd in ("cv0", "cvbf", "stcv"):
                        if kd == "cvbf" and e[kd]["same_as_cv0"]:
                            continue
                        f, S = e[kd]["f"], e[kd]["S"]
                        mm = A13.cv_mean(f, S, "crossfit")
                        v = float(A13.cv_var(f, S, "crossfit")[0])
                        est.append(dict(zip(KEYS, k + (o,)), estimator=f"{kd.upper()} crossfit", source="E13", N=len(f),
                                        mean=mm, se=np.sqrt(v / len(f))))
    D = pd.DataFrame(rows)
    E = pd.DataFrame(est)
    ref = D.set_index("obs")
    E["ref"] = [float(D[(D.channel == r.channel) & (D.variant == r.variant) & (D.obs == r.obs)].ref.iloc[0]) for r in E.itertuples()]
    E["se_ref"] = [float(D[(D.channel == r.channel) & (D.variant == r.variant) & (D.obs == r.obs)].se_ref.iloc[0]) for r in E.itertuples()]
    E["z"] = (E["mean"] - E.ref) / np.sqrt(E.se ** 2 + E.se_ref ** 2)
    D.to_csv(os.path.join(R, "b_rev6d_summary.csv"), index=False)
    E.to_csv(os.path.join(R, "b_rev6d_estimates.csv"), index=False)
    del ref
    idle = T.gpu_other.astype(str).value_counts().to_dict()
    print(f"Rev6 D: trajectory blocks {len(T) // 4} (rows {len(T)}); workers {T.worker.nunique()} on {sorted(set(T.host))}; "
          f"gpu_other {idle} (not a timing run)")
    if len(PP):
        print("PP beyond 1e8 (b_rev6d.py pp, propagate_big): " + "; ".join(
            f"{r.obs} M={float(r.M):.0e} {r.status} ({float(r.wall_s):.0f} s, {r.gpu})" for r in PP.itertuples()))
    capcols = [f"{M:.0e}" for M in sorted(set(CAPS) | set(int(float(x)) for x in PP.M.dropna().unique()))]
    for ks in REF_KEYS:
        sub = D[D[CFG].astype(str).agg("_".join, axis=1).str.len() > 0]
        sub = D[[A13.key_str(*tuple(r[c] for c in CFG)) == ks for _, r in D.iterrows()]]
        print(f"\n== {ks}")
        for _, r in sub.iterrows():
            print(f"  {r.obs:3s} {'E13 group' if r.e13_group else 'not E13 '}: ref {r.ref:+.7f} +/- {r.se_ref:.2e} "
                  f"(N {r.N_ref}, SE/eps {r.se_ref_over_eps:.3f}); eps {r.eps:.3e}; selection 1000-traj mean z {r.z_sel:+.2f}")
            print("      |PP_M - ref|/eps: " + "  ".join(
                f"{c} {r[f'd{c}_eps']:.2f}({r[f'lab{c}'][:4]})" for c in capcols if f"d{c}_eps" in r and np.isfinite(r[f"d{c}_eps"])))
            print(f"      PP 1e8 time {r.t_pp1e8:.0f} s (P100/V100, E13 selection); trajectory cost to eps (20000 t_bf) "
                  f"{r.traj_cost_eps_s:.0f} s")
    print("\n== E13 / rev6c estimators vs reference: |z| summary")
    E["az"] = E.z.abs()
    print(E.groupby(["channel", "source"]).az.agg(["count", "max", lambda x: int((x > 4).sum())]).rename(
        columns={"<lambda_0>": "n(|z|>4)"}).round(2).to_string())
    print(E[E.estimator.str.contains("crossfit") | E.estimator.str.startswith("ST(target")].query("az > 3").to_string()
          if (E.az > 3).any() else "  no estimator with |z| > 3")
    print("\n######## Labels (pre-registered) over E13 groups (TN at cap 1e7)")
    for ch, g in D[D.e13_group].groupby("channel"):
        for c in capcols:
            if f"lab{c}" in g:
                vc = g[f"lab{c}"].value_counts().to_dict()
                print(f"  {ch:4s} PP {c}: {vc}")
        n7 = int((g["lab1e+07"] == "accurate").sum())
        n8 = int((g["lab1e+08"] == "accurate").sum())
        print(f"  {ch}: 'PP unconverged at 1e7' CONFIRMED as accuracy failure in {int((g['lab1e+07'] == 'inaccurate').sum())}/"
              f"{len(g)} groups; PP_1e7 accurate in {n7}/{len(g)} -> "
              + ("statement triggered: PP_1e7 already accurate in >= half (weakens practical significance)"
                 if n7 * 2 >= len(g) else "fewer than half accurate (label holds)")
              + f"; PP_1e8 accurate in {n8}/{len(g)}"
              + (" -> statement triggered: PP_1e8 accurate in all groups (viable competitor)" if n8 == len(g) else ""))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
