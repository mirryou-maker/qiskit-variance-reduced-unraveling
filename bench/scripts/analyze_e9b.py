"""Analysis of E9b (pre-registered in docs/plan_B_characterization_2026-11-06.md, written before the main runs).

Per group (channel, family, variant, L, gt, obs) and M in {10, 100}:
  Var_bf = min over fixed candidates;  t_f = that candidate's run wall / N
  t_g    = (greedy run wall + bp_s * N / 20000) / N
  R_M = Var_bf / Var_g,  W_M = Var_bf t_f / (Var_g t_g)
Per-group CIs: bootstrap of raw arrays (B = 500; best candidate re-chosen in each replicate).
Decision: some channel and M with median W_M > 1 and bootstrap-over-groups lower bound (98.75%, Bonferroni 4) > 1.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
MS = [10, 100]
NB = 500
NPROD = 20000


def med_ci(x, rng, lo=0.625, hi=99.375):
    x = np.asarray(x)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return np.median(x), *np.percentile(b, [lo, hi])


def main():
    rng = np.random.default_rng(2026)
    # sources: (csv, raw dir). Main parts use b_e9b_raw; a takeover run (second host, --reverse) has its own raw dir.
    # Every group is taken from ONE source that has it complete, so t_f and t_g come from the same host.
    srcs = [(f, os.path.join(R, "b_e9b_raw")) for f in sorted(glob.glob(os.path.join(R, "b_e9b_u9?.csv")))]
    srcs += [(f, os.path.join(R, "b_e9b_raw_" + os.path.basename(f)[6:-4]))
             for f in sorted(glob.glob(os.path.join(R, "b_e9b_u9?p*.csv")))]
    parts = []
    for j, (f, rd) in enumerate(srcs):
        d = pd.read_csv(f).drop_duplicates(KEYS + ["kind", "scheme", "M"], keep="last")
        d["_src"], d["_raw"] = j, rd
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    ncand = {"deph": 7, "ad": 5}
    rows = []
    for k, gall in df.groupby(KEYS):
        ch, fam, var, n, L, gt, o = k
        g = None
        for _, gs in gall.groupby("_src"):
            gm = gs[gs.kind == "greedy"]
            if (gs.kind == "fixed").sum() == ncand[ch] and all(float(m) in set(gm.M.astype(float)) for m in MS):
                g = gs
                break
        if g is None:
            continue
        fx = g[g.kind == "fixed"]
        gr = g[g.kind == "greedy"].set_index("M")
        raw = np.load(os.path.join(g._raw.iloc[0], f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}.npz"))
        N = int(fx.shots.iloc[0])
        tf = dict(zip(fx.scheme, fx.wall_s / fx.shots))
        vf = dict(zip(fx.scheme, fx["var"]))
        best = min(vf, key=vf.get)
        arr = {s: raw[f"fixed_{s}_{o}"].astype(np.float64) for s in vf}
        idx = rng.integers(0, N, size=(NB, N))
        bvf = {s: arr[s][idx].var(axis=1, ddof=1) for s in arr}
        stack = np.array([bvf[s] for s in vf])                         # (C, NB)
        tvec = np.array([tf[s] for s in vf])
        jb = stack.argmin(axis=0)
        b_var_bf = stack.min(axis=0)
        b_t_f = tvec[jb]
        r = dict(zip(KEYS, k), best=best, var_bf=vf[best], t_f=tf[best], N=N, host=fx.host.iloc[0],
                 gpu_other=";".join(sorted(set(g.gpu_other.astype(str)))),
                 mean_bf=float(fx.set_index("scheme").loc[best, "value"]))
        for m in MS:
            row = gr.loc[float(m)]
            vg, tg = float(row["var"]), (float(row.wall_s) + float(row.bp_s) * N / NPROD) / N
            x = raw[f"greedy_M{m}_{o}"].astype(np.float64)
            bvg = x[rng.integers(0, len(x), size=(NB, len(x)))].var(axis=1, ddof=1)
            Rm = vf[best] / vg
            Wm = vf[best] * tf[best] / (vg * tg)
            bR = b_var_bf / np.maximum(bvg, 1e-300)
            bW = b_var_bf * b_t_f / (np.maximum(bvg, 1e-300) * tg)
            r.update({f"var_g{m}": vg, f"t_g{m}": tg, f"tratio{m}": tg / tf[best], f"R{m}": Rm, f"W{m}": Wm,
                      f"R{m}_lo": np.percentile(bR, 2.5), f"R{m}_hi": np.percentile(bR, 97.5),
                      f"W{m}_lo": np.percentile(bW, 2.5), f"W{m}_hi": np.percentile(bW, 97.5),
                      f"nonstd{m}": float(row.nonstd),
                      f"z{m}": (float(row.value) - r["mean_bf"]) / np.sqrt(vg / len(x) + vf[best] / N)})
        rows.append(r)
    S = pd.DataFrame(rows)
    out = os.path.join(R, "b_e9b_summary.csv")
    S.to_csv(out, index=False)
    print(f"groups {len(S)} -> {out};  idle: {sorted(set(S.gpu_other))}")
    print("best fixed counts:", S.groupby("channel").best.value_counts().to_dict())
    for m in MS:
        print(f"\n== M = {m} ==")
        print(f"  bias: max|z| {S[f'z{m}'].abs().max():.2f}, |z|>=4: {int((S[f'z{m}'].abs() >= 4).sum())}")
        for ch, C in S.groupby("channel"):
            mr, rl, rh = med_ci(C[f"R{m}"], rng, 2.5, 97.5)
            mw, wl, wh = med_ci(C[f"W{m}"], rng)
            print(f"  {ch:5s} groups {len(C):3d}  R median {mr:.3f} [{rl:.3f}, {rh:.3f}] (95%)   "
                  f"W median {mw:.3f} [{wl:.3f}, {wh:.3f}] (98.75%)   t_g/t_f median {C[f'tratio{m}'].median():.2f}")
            print(f"        W CI>1: {int((C[f'W{m}_lo'] > 1).sum())}, W CI<1: {int((C[f'W{m}_hi'] < 1).sum())};  "
                  f"R CI>1: {int((C[f'R{m}_lo'] > 1).sum())}, R CI<1: {int((C[f'R{m}_hi'] < 1).sum())};  "
                  f"nonstd median {C[f'nonstd{m}'].median():.3f}")
    print("\n== breakdown: median W ==")
    for col in ("family", "variant", "L", "gt", "obs"):
        print(f"  by channel x {col}:")
        print(S.pivot_table(index="channel", columns=col, values=["W10", "W100"], aggfunc="median").round(3))
    passed = []
    for m in MS:
        for ch, C in S.groupby("channel"):
            mw, wl, _ = med_ci(C[f"W{m}"], np.random.default_rng(7))
            if mw > 1 and wl > 1:
                passed.append(f"{ch} M={m} (median W {mw:.3f}, lower {wl:.3f})")
    print("\nDECISION (pre-registered): " + ("USABLE adaptive policy in the TN regime: " + "; ".join(passed) if passed
                                          else "NO wall-clock advantage in any channel / M -> PRA characterization"))


if __name__ == "__main__":
    main()
