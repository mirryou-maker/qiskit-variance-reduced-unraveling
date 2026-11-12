"""Analysis of E10 (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E10 pre-registration").

Static per-location table vs best fixed / default / greedy (M = 100) on the 98 E9b groups at n = 20.
  Var_fixed(c): E9b raw arrays (N = 2000; bench/results/b_e9b_raw/<cfg>.npz), valid iff the E10 re-timed fixed runs match
                the E9b trajectories (>= 99%).
  t_f(c)      : E10 fixed wall / N_t (same session as the static runs).
  Var_static  : E10 production run (N = 2000; bench/results/b_e10_raw/<cfg>_<table>_<obs>.npz).
  t_static    : wall / N + (bp_s + build_s) / 20000.
  greedy M100 : Var and t_g from E9e (b_e9e_summary.csv; fallback raw where fb100 = 1, else E9b raw).
W = Var_ref t_ref / (Var_static t_static), R = Var_ref / Var_static. Per-group bootstrap B = 500 (best fixed re-chosen per
replicate). Median over groups with group-bootstrap 98.75% interval (primary) and config-cluster 98.75% (secondary).
Decision: some channel and table type with median W_bf > 1 and lower bound > 1.
Usage (ubuntu94, env sci): python bench/scripts/analyze_e10.py [b_e10 csv [raw dir]] > bench/results/b_e10_analysis.txt
Outputs b_e10_summary.csv (per group) and b_e10_medians.csv (per subset x table x channel); other csv names get a suffix.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = ["channel", "family", "variant", "n", "L", "gt"]
TABLES = ["pilot", "ideal"]
NB = 500
NPROD = 20000
NCAND = {"deph": 7, "ad": 5}
LO, HI = 0.625, 99.375


def med_ci(x, rng):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return np.median(x), *np.percentile(b, [LO, HI])


def cluster_ci(d, col, rng, B=2000):
    x = d[col].to_numpy(float)
    ids = d[CFG].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return np.percentile(cb, [LO, HI])


def key_str(ch, fam, var, n, L, gt):
    return f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"


def main():
    f10 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "b_e10_u94.csv")
    rawdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "b_e10_raw")
    tag = "" if os.path.basename(f10) == "b_e10_u94.csv" else "_" + os.path.basename(f10)[6:-4]   # smoke -> _smoke
    rng = np.random.default_rng(2026)
    d = pd.read_csv(f10).drop_duplicates(KEYS + ["kind", "scheme"], keep="last")
    e9e = pd.read_csv(os.path.join(R, "b_e9e_summary.csv")).set_index(KEYS)
    rows = []
    for k, g in d.groupby(KEYS):
        ch, fam, var, n, L, gt, o = k
        fx = g[g.kind == "fixed"]
        st = g[g.kind == "static"].set_index("scheme")
        if len(fx) != NCAND[ch] or not set(TABLES) & set(st.index):
            continue
        ks = key_str(ch, fam, var, n, L, gt)
        raw = np.load(os.path.join(R, "b_e9b_raw", ks + ".npz"))
        labels = list(fx.scheme)
        arr = {s: raw[f"fixed_{s}_{o}"].astype(np.float64) for s in labels}
        vf = {s: arr[s].var(ddof=1) for s in labels}
        tf = dict(zip(fx.scheme, fx.wall_s / fx.shots))
        best = min(vf, key=vf.get)
        std = labels[0] if labels[0] in ("std", "jump") else None
        assert std is not None, labels
        N = len(arr[best])
        idx = rng.integers(0, N, size=(NB, N))
        stack = np.array([arr[s][idx].var(axis=1, ddof=1) for s in labels])
        tvec = np.array([tf[s] for s in labels])
        jb = stack.argmin(axis=0)
        b_vbf, b_tf = stack.min(axis=0), tvec[jb]
        b_v0 = stack[0]
        # greedy M100 (E9e)
        q = e9e.loc[k]
        vg = float(q.var_bf) / float(q.R100)
        tg = float(q.t_g100)
        if int(q.fb100):
            xg = np.load(os.path.join(R, "b_e9e_fb_raw", f"{ks}_M100_{o}.npz"))["x"].astype(np.float64)
        else:
            xg = raw[f"greedy_M100_{o}"].astype(np.float64)
        b_vg = xg[rng.integers(0, len(xg), size=(NB, len(xg)))].var(axis=1, ddof=1)
        r = dict(zip(KEYS, k), best=best, var_bf=vf[best], var_std=vf[std], t_f=tf[best], t_std=tf[std],
                 t_f_e9e=float(q.t_f), var_g=vg, t_g=tg, match_min=float(fx.match.min()),
                 gpu_other=";".join(sorted(set(g.gpu_other.astype(str)))),
                 other_py=";".join(sorted(set(g.other_py.astype(str)))), runner_max=int(g.runner.max()),
                 gpu_util_max=float(pd.to_numeric(g.gpu_util, errors="coerce").max()),
                 gpu_mem_max=float(pd.to_numeric(g.gpu_mem, errors="coerce").max()), load1_max=float(g.load1.max()))
        mean_bf = arr[best].mean()
        for t in TABLES:
            if t not in st.index:
                continue
            s = st.loc[t]
            z = np.load(os.path.join(rawdir, f"{ks}_{t}_{o}.npz"))
            x = z["x"].astype(np.float64)
            vs = x.var(ddof=1)
            ts = float(s.wall_s) / float(s.shots) + (float(s.bp_s) + float(s.build_s)) / NPROD
            b_vs = np.maximum(x[rng.integers(0, len(x), size=(NB, len(x)))].var(axis=1, ddof=1), 1e-300)
            tab = z["tab"].astype(int)
            bW = b_vbf * b_tf / (b_vs * ts)
            r.update({f"{t}_var": vs, f"{t}_t": ts, f"{t}_tratio": ts / tf[best], f"{t}_nonstd": float(s.nonstd),
                      f"{t}_agree_best": float(np.mean(tab == labels.index(best))),
                      f"{t}_build_s": float(s.build_s), f"{t}_bp_s": float(s.bp_s),
                      f"{t}_z": (x.mean() - mean_bf) / np.sqrt(vs / len(x) + vf[best] / N),
                      f"{t}_R_bf": vf[best] / vs, f"{t}_W_bf": vf[best] * tf[best] / (vs * ts),
                      f"{t}_W_bf_lo": np.percentile(bW, 2.5), f"{t}_W_bf_hi": np.percentile(bW, 97.5),
                      f"{t}_R_std": vf[std] / vs, f"{t}_W_std": vf[std] * tf[std] / (vs * ts),
                      f"{t}_R_g": vg / vs, f"{t}_W_g": vg * tg / (vs * ts)})
            bW0 = b_v0 * tf[std] / (b_vs * ts)
            bWg = b_vg * tg / (b_vs * ts)
            r.update({f"{t}_W_std_lo": np.percentile(bW0, 2.5), f"{t}_W_std_hi": np.percentile(bW0, 97.5),
                      f"{t}_W_g_lo": np.percentile(bWg, 2.5), f"{t}_W_g_hi": np.percentile(bWg, 97.5)})
        rows.append(r)
    S = pd.DataFrame(rows)
    # TN subsets (b_e9d_summary.csv, as pra_revision.py C4)
    e9d = pd.read_csv(os.path.join(R, "b_e9d_summary.csv"))
    e9d = e9d[e9d.n == 20][KEYS + ["TN1000000", "TN10000000"]]
    S = S.merge(e9d, on=KEYS, how="left")
    out = os.path.join(R, f"b_e10_summary{tag}.csv")
    S.to_csv(out, index=False)

    print(f"E10 analysis of {f10}\ngroups {len(S)} (configs {len(S.groupby(CFG))}) -> {out}")
    print(f"identity check (re-timed fixed vs E9b raw, first N_t): min match {S.match_min.min():.5f}; "
          f"groups < 0.99: {int((S.match_min < 0.99).sum())}")
    print(f"idle record: gpu_other {sorted(set(S.gpu_other))}; other_py {sorted(set(S.other_py))}; "
          f"runner max {S.runner_max.max()}; gpu_util max {S.gpu_util_max.max()} %; gpu_mem max {S.gpu_mem_max.max()} MiB; "
          f"load1 max {S.load1_max.max()}")
    print(f"t_f(best) E10 / E9e: median {np.median(S.t_f / S.t_f_e9e):.3f} "
          f"[{(S.t_f / S.t_f_e9e).min():.3f}, {(S.t_f / S.t_f_e9e).max():.3f}]")
    for t in TABLES:
        if f"{t}_var" not in S:
            continue
        zz = S[f"{t}_z"].abs()
        print(f"{t}: t_static/t_f(best) median {S[f'{t}_tratio'].median():.3f}; build_s median {S[f'{t}_build_s'].median():.2f} s, "
              f"bp_s median {S[f'{t}_bp_s'].median():.2f} s; table non-std median {S[f'{t}_nonstd'].median():.3f}; "
              f"table = best fixed median {S[f'{t}_agree_best'].median():.3f}; bias |z| vs E9b best-fixed mean max {zz.max():.2f}, "
              f"n(|z|>4) {int((zz > 4).sum())}")
    subsets = [("all98", S), ("TN1e6", S[S.TN1000000 == 1]), ("TN1e7", S[S.TN10000000 == 1])]
    tab_rows = []
    print("\nMedian over groups [98.75% group bootstrap] {98.75% config-cluster}; per-group CI>1 / CI<1 counts")
    for lab, sub in subsets:
        print(f"\n== subset {lab} ({len(sub)} groups) ==")
        for t in TABLES:
            if f"{t}_var" not in sub:
                continue
            for ch, C in sub.groupby("channel"):
                line = []
                rec = dict(subset=lab, table=t, channel=ch, groups=len(C), configs=len(C.groupby(CFG)))
                for ref in ("bf", "std", "g"):
                    m, lo, hi = med_ci(C[f"{t}_W_{ref}"], rng)
                    clo, chi = cluster_ci(C, f"{t}_W_{ref}", rng) if len(C) > 1 else (np.nan, np.nan)
                    rm, rlo, rhi = med_ci(C[f"{t}_R_{ref}"], rng)
                    rec.update({f"W_{ref}": m, f"W_{ref}_lo": lo, f"W_{ref}_hi": hi, f"W_{ref}_clo": clo,
                                f"W_{ref}_chi": chi, f"R_{ref}": rm, f"R_{ref}_lo": rlo, f"R_{ref}_hi": rhi,
                                f"n_W_{ref}_gt1": int((C[f"{t}_W_{ref}_lo"] > 1).sum()),
                                f"n_W_{ref}_lt1": int((C[f"{t}_W_{ref}_hi"] < 1).sum())})
                    line.append(f"W_{ref} {m:.3f} [{lo:.3f}, {hi:.3f}] {{{clo:.3f}, {chi:.3f}}} "
                                f"(CI>1 {rec[f'n_W_{ref}_gt1']}, CI<1 {rec[f'n_W_{ref}_lt1']}); "
                                f"R_{ref} {rm:.3f} [{rlo:.3f}, {rhi:.3f}]")
                tab_rows.append(rec)
                print(f"  {t:5s} {ch:4s} groups {len(C):2d} t_s/t_f {C[f'{t}_tratio'].median():.3f}")
                for li in line:
                    print("      " + li)
    pd.DataFrame(tab_rows).to_csv(os.path.join(R, f"b_e10_medians{tag}.csv"), index=False)
    print("\n== breakdown (all 98, POST HOC): median W_bf ==")
    for col in ("family", "L", "gt", "obs"):
        cols = [f"{t}_W_bf" for t in TABLES if f"{t}_W_bf" in S]
        print(S.pivot_table(index="channel", columns=col, values=cols, aggfunc="median").round(3))
    passed = []
    for t in TABLES:
        if f"{t}_W_bf" not in S:
            continue
        for ch, C in S.groupby("channel"):
            m, lo, _ = med_ci(C[f"{t}_W_bf"], np.random.default_rng(7))
            if m > 1 and lo > 1:
                passed.append(f"{ch} {t} (median W {m:.3f}, lower {lo:.3f})")
    print("\nDECISION (pre-registered, all 98 groups): " + (
        "USABLE static table: " + "; ".join(passed) if passed else
        "NO channel / table type has median W_static > 1 with 98.75% lower bound > 1 -> the static per-location table "
        "does not recover a wall-clock advantage over the best fixed unraveling; the negative result extends to it"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
