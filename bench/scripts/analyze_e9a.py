"""Analysis of E9a (pre-registered in docs/plan_B_characterization_2026-11-06.md, with the 2026-11-08 amendments).

Per group (channel, family, variant, n, L, gt, obs):
  eps      = sqrt(Var_std / 20000)                      (Var_std from the N_ref trajectory run)
  delta_M  = |PP_M - PP_10M|,  M in {1e2, 1e3, 1e4, 1e5}
  solved   = some M <= 1e5 has delta_M <= eps  and  |PP_1e6 - traj mean| / SE_ref < 4
  M*       = smallest such M;  t_traj20k = traj wall * 20000 / N_ref  (extrapolated)
  TN       = not solved  or  t_PP(M*) > t_traj20k
Decision on n in {20, 24}: TN <= 5% of groups -> trajectories not needed in this ensemble.
Secondary: groups with |traj mean| >= 0.01. Validation at n = 12 against the exact answer.
Groups with Var_std < 1e-12 are deterministic and reported separately.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
MS = [100, 1000, 10000, 100000, 1000000]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(S, label):
    k, n = int(S.TN.sum()), len(S)
    lo, hi = wilson(k, n)
    print(f"  {label:34s} TN {k:4d}/{n:4d} = {k / max(n, 1):6.1%}  [{lo:.1%}, {hi:.1%}]")


def main():
    files = sorted(glob.glob(os.path.join(R, "b_e9a_u9*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(KEYS + ["kind", "M"], keep="last")
    rows = []
    for k, g in df.groupby(KEYS):
        pp = g[g.kind == "pp"].set_index("M")
        tr = g[g.kind == "traj"]
        ex = g[g.kind == "exact"]
        if len(tr) == 0 or not all(float(m) in pp.index for m in MS):
            continue
        tr = tr.iloc[0]
        v = {m: float(pp.loc[float(m), "value"]) for m in MS}
        t = {m: float(pp.loc[float(m), "wall_s"]) for m in MS}
        var, N, mean = float(tr["var"]), int(tr["shots"]), float(tr["value"])
        r = dict(zip(KEYS, k), traj_mean=mean, var_std=var, nref=N, t_traj20k=float(tr.wall_s) * 20000 / N,
                 gpu_other=str(tr.gpu_other), host=tr.host, **{f"pp{m}": v[m] for m in MS},
                 **{f"t{m}": t[m] for m in MS})
        r["deterministic"] = var < 1e-12
        eps = np.sqrt(max(var, 0.0) / 20000)
        r["eps"] = eps
        se_ref = np.sqrt(max(var, 1e-300) / N)
        r["z_pp1e6"] = (v[MS[-1]] - mean) / se_ref
        Mstar = None
        for a, b in zip(MS[:-1], MS[1:]):
            r[f"d{a}"] = abs(v[a] - v[b])
            if Mstar is None and r[f"d{a}"] <= eps:
                Mstar = a
        r["solved"] = (Mstar is not None) and abs(r["z_pp1e6"]) < 4
        r["Mstar"] = Mstar if r["solved"] else np.nan
        r["tPP_Mstar"] = t[Mstar] if r["solved"] else np.nan
        r["TN"] = (not r["solved"]) or (r["tPP_Mstar"] > r["t_traj20k"])
        if len(ex):
            e = float(ex.iloc[0].value)
            r["exact"] = e
            r["err_traj_z"] = (mean - e) / se_ref
            r["err1e6_over_eps"] = abs(v[MS[-1]] - e) / max(eps, 1e-300)
            r["errMstar_over_eps"] = abs(v[Mstar] - e) / max(eps, 1e-300) if r["solved"] else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    out = os.path.join(R, "b_e9a_summary.csv")
    S.to_csv(out, index=False)
    print(f"groups {len(S)} (deterministic {int(S.deterministic.sum())}) -> {out}")
    print("hosts / idle:", S.groupby("host").gpu_other.agg(lambda x: sorted(set(x))).to_dict())
    nd = S[~S.deterministic]

    v12 = nd[nd.n == 12]
    if len(v12):
        print("\n== n = 12 validation against exact ==")
        print(f"  trajectory bias: max|z| {v12.err_traj_z.abs().max():.2f}, rms {np.sqrt((v12.err_traj_z ** 2).mean()):.3f}")
        sv = v12[v12.solved]
        bad = int((sv.errMstar_over_eps > 1).sum())
        print(f"  solved {len(sv)}/{len(v12)}; of these, true error at M* > eps in {bad} "
              f"({bad / max(len(sv), 1):.1%})  <- criterion reliability")
        print(f"  |PP_1e6 - exact| / eps: median {v12.err1e6_over_eps.median():.3g}, "
              f"max {v12.err1e6_over_eps.max():.3g}")
        summarize(v12, "n=12 (context)")

    big = nd[nd.n.isin([20, 24])]
    if not len(big):
        print("no n = 20/24 groups yet")
        return
    print("\n== n in {20, 24}: PRIMARY ==")
    summarize(big, "all")
    for ch, C in big.groupby("channel"):
        summarize(C, f"channel {ch}")
    for n, C in big.groupby("n"):
        summarize(C, f"n = {n}")
    sec = big[big.traj_mean.abs() >= 0.01]
    print("\n== SECONDARY: |traj mean| >= 0.01 ==")
    summarize(sec, "all")
    for ch, C in sec.groupby("channel"):
        summarize(C, f"channel {ch}")
    print("\n== where TN groups are (fraction TN) ==")
    for col in ("family", "variant", "L", "gt", "obs"):
        print(f"  by {col}:", big.groupby(col).TN.mean().round(3).to_dict())
    print("  by L x gt:")
    print(big.pivot_table(index="L", columns="gt", values="TN", aggfunc="mean").round(2))
    print("  by family x L (secondary subset):")
    print(sec.pivot_table(index="family", columns="L", values="TN", aggfunc="mean").round(2))
    print("\n== M* (solved groups) ==")
    sv = nd[nd.solved]
    print("  median M* by n:", sv.groupby("n").Mstar.median().to_dict())
    print(sv.pivot_table(index="L", columns="n", values="Mstar", aggfunc="median"))
    print("\n== |PP_1e6 - traj| z (n 20/24): ", f"rms {np.sqrt((big.z_pp1e6 ** 2).mean()):.3f}, "
          f"max {big.z_pp1e6.abs().max():.2f}, |z|>=4: {int((big.z_pp1e6.abs() >= 4).sum())}")
    print("  time: median t_PP(1e6) / t_traj20k(extrap.):",
          f"{(big.t1000000 / big.t_traj20k).median():.4f}")
    k, n = int(big.TN.sum()), len(big)
    print(f"\nDECISION (pre-registered, n 20/24 all groups): TN {k}/{n} = {k / n:.1%} -> "
          + ("trajectories NOT needed in this ensemble (<= 5%)" if k / n <= 0.05
             else "a trajectory-needing regime EXISTS (> 5%): report where, propose E9b to the user"))


if __name__ == "__main__":
    main()
