"""Analysis of E9e (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E9e pre-registration").

Variances (and their bootstrap) come from the E9b raw arrays (N = 2000), exactly as analyze_e9b.py; per-trajectory times
come from the E9e re-timing (optimized implementation, one host):
  t_f(c) = E9e fixed wall / N_t,   t_g = E9e greedy wall / N_t + bp_s / 20000
Identity check: a group is valid for E9b variances only if its greedy runs and all its fixed runs match the E9b
trajectories in >= 99% (match column of b_e9e_*.csv); otherwise it is listed for the full re-run fallback.
Decision (same rule as E9b): some channel and M with median W_M > 1 and bootstrap-over-groups lower bound (98.75%) > 1.
Usage: python analyze_e9e.py [e9e csv ...]   (default bench/results/b_e9e_u94.csv)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
MS = [10, 100]
NB = 500
NPROD = 20000
NCAND = {"deph": 7, "ad": 5}


def med_ci(x, rng, lo=0.625, hi=99.375):
    x = np.asarray(x)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return np.median(x), *np.percentile(b, [lo, hi])


def e9b_table():
    """E9b rows (variances), one complete source per group, as analyze_e9b.py."""
    parts = []
    for j, f in enumerate(sorted(p for p in os.listdir(R) if p.startswith("b_e9b_u9") and p.endswith(".csv"))):
        d = pd.read_csv(os.path.join(R, f)).drop_duplicates(KEYS + ["kind", "scheme", "M"], keep="last")
        d["_src"] = j
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    files = sys.argv[1:] or [os.path.join(R, "b_e9e_u94.csv")]
    tag = os.path.basename(files[0])[6:-4]
    rng = np.random.default_rng(2026)
    old = e9b_table()
    new = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    new = new.drop_duplicates(KEYS + ["kind", "scheme", "M"], keep="last")
    old_sum = pd.read_csv(os.path.join(R, "b_e9b_summary.csv")).set_index(KEYS)
    old_i = old.set_index(KEYS).sort_index()
    # pre-registered fallback: greedy runs with match < 0.99 re-run in full (N = 2000) with the new code
    fbf = [os.path.join(R, p) for p in os.listdir(R) if p.startswith("b_e9e_fb_") and p.endswith(".csv")]
    fb = {}
    for p in fbf:
        for _, q in pd.read_csv(p).iterrows():
            fb[(q["channel"], q["family"], q["variant"], int(q["n"]), int(q["L"]), float(q["gt"]), q["obs"],
                int(q["M"]))] = q
    rows, bad = [], []
    for k, gn in new.groupby(KEYS):
        ch, fam, var, n, L, gt, o = k
        fx_n = gn[gn.kind == "fixed"]
        gr_n = gn[gn.kind == "greedy"].set_index("M")
        if len(fx_n) != NCAND[ch] or not all(float(m) in set(gr_n.index.astype(float)) for m in MS):
            continue
        go = old_i.loc[k].reset_index()
        go = go[go._src == go._src.iloc[0]] if go._src.nunique() == 1 else go
        fx_o = go[go.kind == "fixed"].drop_duplicates("scheme")
        gr_o = go[go.kind == "greedy"].drop_duplicates("M").set_index("M")
        raw = np.load(os.path.join(R, "b_e9b_raw", f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}.npz"))
        N = int(fx_o.shots.iloc[0])
        vf = dict(zip(fx_o.scheme, fx_o["var"]))
        tf = dict(zip(fx_n.scheme, fx_n.wall_s / fx_n.shots))
        best = min(vf, key=vf.get)
        match_min = min(fx_n.match.min(), gr_n.match.min())
        fb_used = []
        if match_min < 0.99:
            miss = [m for m in MS if float(gr_n.loc[float(m), "match"]) < 0.99]
            if fx_n.match.min() < 0.99 or any((ch, fam, var, int(n), int(L), float(gt), o, m) not in fb for m in miss):
                bad.append((k, match_min))
            fb_used = miss
        arr = {s: raw[f"fixed_{s}_{o}"].astype(np.float64) for s in vf}
        idx = rng.integers(0, N, size=(NB, N))
        stack = np.array([arr[s][idx].var(axis=1, ddof=1) for s in vf])
        tvec = np.array([tf[s] for s in vf])
        jb = stack.argmin(axis=0)
        b_var_bf, b_t_f = stack.min(axis=0), tvec[jb]
        r = dict(zip(KEYS, k), best=best, var_bf=vf[best], t_f=tf[best], t_f_e9b=float(old_sum.loc[k, "t_f"]),
                 host=fx_n.host.iloc[0], gpu_other=";".join(sorted(set(gn.gpu_other.astype(str)))),
                 match_min=match_min)
        for m in MS:
            rn = gr_n.loc[float(m)] if float(m) in gr_n.index else gr_n.loc[m]
            fk = (ch, fam, var, int(n), int(L), float(gt), o, m)
            tg = float(rn.wall_s) / float(rn.shots) + float(rn.bp_s) / NPROD
            if m in fb_used and fk in fb:
                vg = float(fb[fk]["var"])
                x = np.load(os.path.join(R, "b_e9e_fb_raw", f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}_M{m}_{o}.npz"))["x"]
                x = x.astype(np.float64)
                r[f"fb{m}"] = 1
            else:
                vg = float(gr_o.loc[float(m), "var"] if float(m) in gr_o.index else gr_o.loc[m, "var"])
                x = raw[f"greedy_M{m}_{o}"].astype(np.float64)
                r[f"fb{m}"] = 0
            bvg = x[rng.integers(0, len(x), size=(NB, len(x)))].var(axis=1, ddof=1)
            bW = b_var_bf * b_t_f / (np.maximum(bvg, 1e-300) * tg)
            r.update({f"t_g{m}": tg, f"tratio{m}": tg / tf[best], f"tratio{m}_e9b": float(old_sum.loc[k, f"tratio{m}"]),
                      f"R{m}": vf[best] / vg, f"W{m}": vf[best] * tf[best] / (vg * tg),
                      f"W{m}_e9b": float(old_sum.loc[k, f"W{m}"]),
                      f"W{m}_lo": np.percentile(bW, 2.5), f"W{m}_hi": np.percentile(bW, 97.5)})
        rows.append(r)
    S = pd.DataFrame(rows)
    out = os.path.join(R, f"b_e9e_summary{'' if tag == 'u94' else '_' + tag}.csv")
    S.to_csv(out, index=False)
    print(f"groups {len(S)} -> {out};  hosts {sorted(set(S.host))};  idle: {sorted(set(S.gpu_other))}")
    print(f"identity check: min match over groups {S.match_min.min():.4f}; groups < 0.99 WITHOUT fallback "
          f"(variance still from E9b, INVALID): {len(bad)} {bad}")
    print(f"fallback variances used: M10 {int(S.fb10.sum())} runs, M100 {int(S.fb100.sum())} runs (from {fbf})")
    print(f"t_f (new / E9b) median {np.median(S.t_f / S.t_f_e9b):.3f} (E9b mixes RTX 3070 Ti and RTX 2060 hosts)")
    for m in MS:
        print(f"\n== M = {m} ==")
        for ch, C in S.groupby("channel"):
            mw, wl, wh = med_ci(C[f"W{m}"], rng)
            print(f"  {ch:5s} groups {len(C):3d}  t_g/t_f median {C[f'tratio{m}'].median():.2f} "
                  f"(E9b {C[f'tratio{m}_e9b'].median():.2f})   R median {C[f'R{m}'].median():.3f}   "
                  f"W median {mw:.3f} [{wl:.3f}, {wh:.3f}] (98.75%)  (E9b W median {C[f'W{m}_e9b'].median():.3f})")
            print(f"        W CI>1: {int((C[f'W{m}_lo'] > 1).sum())}, W CI<1: {int((C[f'W{m}_hi'] < 1).sum())}, "
                  f"W point > 1: {int((C[f'W{m}'] > 1).sum())}")
    print("\n== breakdown: median W (optimized) ==")
    for col in ("family", "L", "gt", "obs"):
        print(f"  by channel x {col}:")
        print(S.pivot_table(index="channel", columns=col, values=["W10", "W100"], aggfunc="median").round(3))
    passed = []
    for m in MS:
        for ch, C in S.groupby("channel"):
            mw, wl, _ = med_ci(C[f"W{m}"], np.random.default_rng(7))
            if mw > 1 and wl > 1:
                passed.append(f"{ch} M={m} (median W {mw:.3f}, lower {wl:.3f})")
    print("\nDECISION (pre-registered): " + ("USABLE adaptive policy with the optimized implementation: "
                                          + "; ".join(passed) if passed
                                          else "NO wall-clock advantage in any channel / M even with the optimized "
                                               "implementation -> negative result stands"))


if __name__ == "__main__":
    main()
