"""Analysis of E11 (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E11 pre-registration").

  python analyze_e11.py select [b_e11_select.csv]   -> b_e11_groups.csv (E9a rule; checks it reproduces b_e9a_summary.csv)
  python analyze_e11.py scope  <smoke csv>          -> b_e11_scope.txt (config order + projected cost, pre-registered rule)
  python analyze_e11.py main   [b_e11_u94.csv [raw dir [groups csv]]] -> b_e11_summary.csv, b_e11_multi.csv, b_e11_medians.csv
Usage on ubuntu94 (CPU, env sci).

Q1 (per group, as analyze_e10.py): W_bf = Var_bf t_f / (Var_s t_s), t_s = wall/2000 + (bp_s + build_s)/20000;
   Var and t_f of the fixed candidates from the E11 N = 2000 runs. Median over groups, 98.75% group bootstrap (B = 2000);
   decision line rng 7 (as E10). Config-cluster 98.75% secondary.
Q2 (per config): eps_o^2 = Var_bf(o)/20000 for o in O_k (TN observables of the config);
   F1 = min_c t_c max_o Var_c(o)/eps_o^2 ; S_multi = tau_m max_o Var_m(o)/eps_o^2 + bp_m + build_m (one-time paid once);
   W_multi = F1 / S_multi. Median over configs, 98.75% config bootstrap (B = 4000, default_rng(20261113)).
Decision for both: median > 1 and 98.75% lower bound > 1 (Bonferroni over 4 tests: Q1, Q2 x ad, deph; ad primary).
"""
import glob
import importlib.util
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = ["channel", "family", "variant", "n", "L", "gt"]
MS = [100, 1000, 10000, 100000, 1000000]
NCAND = {"deph": 7, "ad": 5}
LO, HI = 0.625, 99.375
NB = 500
NPROD = 20000
HARD_R = 10.0          # hard subset: |PP_1e5 - PP_1e6| / eps > 10 (calibrated on E9 data before E11, see plan doc)
BUDGET_H = 25.0        # main-run budget of the scope rule


def key_str(ch, fam, var, n, L, gt):
    return f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"


# ------------------------------------------------------------------------------------------------------ selection
def classify(df):
    """E9a rule (analyze_e9a.py, verbatim logic)."""
    df = df.drop_duplicates(KEYS + ["kind", "M"], keep="last")
    rows = []
    for k, g in df.groupby(KEYS):
        pp = g[g.kind == "pp"].set_index("M")
        tr = g[g.kind == "traj"]
        if len(tr) == 0 or not all(float(m) in pp.index for m in MS):
            continue
        tr = tr.iloc[0]
        v = {m: float(pp.loc[float(m), "value"]) for m in MS}
        t = {m: float(pp.loc[float(m), "wall_s"]) for m in MS}
        var, N, mean = float(tr["var"]), int(tr["shots"]), float(tr["value"])
        r = dict(zip(KEYS, k), traj_mean=mean, var_std=var, nref=N, t_traj20k=float(tr.wall_s) * 20000 / N,
                 gpu_other=str(tr.gpu_other), **{f"pp{m}": v[m] for m in MS})
        r["deterministic"] = var < 1e-12
        eps = np.sqrt(max(var, 0.0) / 20000)
        r["eps"] = eps
        se_ref = np.sqrt(max(var, 1e-300) / N)
        r["z_pp1e6"] = (v[MS[-1]] - mean) / se_ref
        Mstar = None
        for a_, b_ in zip(MS[:-1], MS[1:]):
            r[f"d{a_}"] = abs(v[a_] - v[b_])
            if Mstar is None and r[f"d{a_}"] <= eps:
                Mstar = a_
        r["solved"] = (Mstar is not None) and abs(r["z_pp1e6"]) < 4
        r["Mstar"] = Mstar if r["solved"] else np.nan
        r["tPP_Mstar"] = t[Mstar] if r["solved"] else np.nan
        r["TN"] = (not r["solved"]) or (r["tPP_Mstar"] > r["t_traj20k"])
        rows.append(r)
    return pd.DataFrame(rows)


def select(path):
    old = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "b_e9a_u9*.csv")))], ignore_index=True)
    ref = pd.read_csv(os.path.join(R, "b_e9a_summary.csv"))
    chk = classify(old).merge(ref[KEYS + ["TN", "deterministic"]], on=KEYS, suffixes=("", "_ref"))
    nbad = int((chk.TN != chk.TN_ref).sum() + (chk.deterministic != chk.deterministic_ref).sum())
    print(f"reproduction of b_e9a_summary.csv by classify(): {len(chk)} groups compared, mismatches {nbad}")
    assert nbad == 0 and len(chk) == len(ref)
    S = classify(pd.read_csv(path))
    S = S[S.n == 20]
    S["r_hard"] = S.d100000 / S.eps
    S["hard"] = S.r_hard > HARD_R
    S.to_csv(os.path.join(R, "b_e11_select_summary.csv"), index=False)
    nd = S[~S.deterministic]
    print(f"fresh groups {len(S)} (deterministic {int(S.deterministic.sum())}); non-deterministic TN "
          f"{int(nd.TN.sum())}/{len(nd)} = {nd.TN.mean():.1%}; gpu_other {sorted(set(S.gpu_other))}")
    print(nd.groupby(["channel", "family", "variant"]).TN.agg(["sum", "count"]))
    g = nd[nd.TN][KEYS + ["traj_mean", "var_std", "r_hard", "hard"]]
    g.to_csv(os.path.join(R, "b_e11_groups.csv"), index=False)
    print(f"-> b_e11_groups.csv: {len(g)} groups, {len(g.groupby(CFG))} configs "
          f"(by channel {g.groupby('channel').size().to_dict()}, configs "
          f"{g.drop_duplicates(CFG).groupby('channel').size().to_dict()}); hard (r > {HARD_R}) {int(g.hard.sum())}")
    print(f"all-z |PP_1e6 - traj| >= 4 among TN: {int((S[S.TN & ~S.deterministic].z_pp1e6.abs() >= 4).sum())}")


# --------------------------------------------------------------------------------------------------------- scope
def tmodel():
    """Per-trajectory time of candidate c as a function of (channel, family, L): E10 fixed re-timings (ubuntu94),
    least-squares t = a L + b per (channel, family, candidate index); fallback per (channel, candidate)."""
    d = pd.read_csv(os.path.join(R, "b_e10_u94.csv"))
    fx = d[d.kind == "fixed"].drop_duplicates(CFG + ["scheme"]).copy()
    fx["t"] = fx.wall_s / fx.shots
    fx["c"] = fx.groupby(CFG).cumcount()
    fits = {}
    for key, g in itertools.chain(fx.groupby(["channel", "family", "c"]), fx.groupby(["channel", "c"])):
        A = np.c_[g.L.to_numpy(float), np.ones(len(g))]
        fits[key] = np.linalg.lstsq(A, g.t.to_numpy(float), rcond=None)[0]
    return fits


def proj_cost(k, nobs, fits, N=2000, pilot=74):
    ch, fam, L = k[0], k[1], k[4]
    tc = []
    for c in range(NCAND[ch]):
        ab = fits.get((ch, fam, c), fits[(ch, c)])
        tc.append(ab[0] * L + ab[1])
    tc = np.array(tc)
    single = nobs * (N * tc[0] + pilot * 2.5 * tc[0] + 1.0)
    multi = N * tc[0] + pilot * (1 + 1.5 * nobs) * tc[0] + 0.5 * nobs
    pauses = (NCAND[ch] + nobs + 1) * 2.5
    return N * tc.sum() + single + multi + pauses


def scope(smoke_csv):
    sys.path.insert(0, os.path.join(ROOT, "bench/scripts"))
    spec = importlib.util.spec_from_file_location("b_e11", os.path.join(ROOT, "bench/scripts/b_e11.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    fits = tmodel()
    sm = pd.read_csv(smoke_csv)
    meas, pred = 0.0, 0.0
    for k, g in sm.groupby(CFG):
        shots = int(g[g.kind == "fixed"].shots.iloc[0])
        nobs = g[g.kind == "single"].obs.nunique()
        m = (g[g.kind == "fixed"].drop_duplicates("scheme").wall_s.sum()
             + (g[g.kind == "single"].wall_s + g[g.kind == "single"].build_s + g[g.kind == "single"].bp_s).sum()
             + g[g.kind == "multi"].iloc[0][["wall_s", "build_s"]].sum())
        p = proj_cost(k, nobs, fits, N=shots)
        print(f"smoke {key_str(*k)}: measured {m:.1f} s, model {p:.1f} s (N {shots})")
        meas, pred = meas + m, pred + p
    cal = meas / pred
    G = pd.read_csv(os.path.join(R, "b_e11_groups.csv"))
    cfg = {k: len(g) for k, g in G.groupby(CFG)}
    order = B.hash_order(list(cfg))
    tot, keep, lines = 0.0, [], []
    for k in order:
        c = cal * proj_cost(k, cfg[k], fits) / 3600
        inc = tot + c <= BUDGET_H
        if inc:
            keep.append(k)
            tot += c
        lines.append(f"# {key_str(*k):40s} obs {cfg[k]} proj {c:6.2f} h {'IN' if inc else 'OUT'}")
        if not inc:
            break                                  # longest prefix in hash order (rule: stop at the first misfit)
    full = cal * sum(proj_cost(k, cfg[k], fits) for k in cfg) / 3600
    out = os.path.join(R, "b_e11_scope.txt")
    with open(out, "w") as fh:
        fh.write(f"# calibration measured/model on smoke = {cal:.3f}; projected full main run {full:.1f} h; "
                 f"budget {BUDGET_H} h; kept {len(keep)}/{len(cfg)} configs ({sum(cfg[k] for k in keep)} groups), "
                 f"projected {tot:.1f} h\n")
        fh.write("\n".join(lines) + "\n")
        fh.write("\n".join(key_str(*k) for k in keep) + "\n")
    print(open(out).read().splitlines()[0])


# ---------------------------------------------------------------------------------------------------------- main
def med_ci(x, rng, B=2000):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [LO, HI])


def cluster_ci(x, ids, rng, B=2000):
    x = np.asarray(x, float)
    u, inv = np.unique(np.asarray(ids), return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return tuple(np.percentile(cb, [LO, HI]))


def best_assign(n_obs, n_opt, cost_of_group):
    best = np.inf
    for a in itertools.product(range(n_opt), repeat=n_obs):
        tot = 0.0
        for c in set(a):
            tot += cost_of_group(c, [i for i in range(n_obs) if a[i] == c])
            if tot >= best:
                break
        best = min(best, tot)
    return best


def load(f11, rawdir, groups):
    d = pd.read_csv(f11).drop_duplicates(KEYS + ["kind", "scheme"], keep="last")
    G = pd.read_csv(groups)
    if "hard" not in G:
        G["hard"] = False
    hard = {tuple(r[:-1]): bool(r[-1]) for r in G[KEYS + ["hard"]].itertuples(index=False)}
    C = []
    for k, g in d.groupby(CFG):
        ch = k[0]
        fx = g[g.kind == "fixed"]
        obs = sorted(set(G.merge(pd.DataFrame([dict(zip(CFG, k))]), on=CFG).obs))
        labels = list(fx[fx.obs == obs[0]].scheme)
        if len(labels) != NCAND[ch] or set(g[g.kind == "multi"].obs) != set(obs) or \
                set(g[g.kind == "single"].obs) != set(obs):
            print(f"incomplete config skipped: {key_str(*k)}")
            continue
        ks = key_str(*k)
        fr = np.load(os.path.join(rawdir, f"{ks}_fixed.npz"))
        X = {q: np.array([fr[f"fixed_{s}_{q}"].astype(np.float64) for s in labels]) for q in ("x", "zz", "xc", "zc")}
        f0 = fx[fx.obs == obs[0]].set_index("scheme")
        t = np.array([float(f0.loc[s, "wall_s"]) / float(f0.loc[s, "shots"]) for s in labels])
        sing = {}
        for o in obs:
            r = g[(g.kind == "single") & (g.obs == o)].iloc[0]
            z = np.load(os.path.join(rawdir, f"{ks}_single_{o}.npz"))
            sing[o] = dict(tau=float(r.wall_s) / float(r.shots), bp=float(r.bp_s), build=float(r.build_s),
                           nonstd=float(r.nonstd), tab=z["tab"].astype(int),
                           Y={q: z[f"all_{q}"].astype(np.float64) for q in ("x", "zz", "xc", "zc")})
        mr = g[g.kind == "multi"].iloc[0]
        zm = np.load(os.path.join(rawdir, f"{ks}_multi.npz"))
        multi = dict(tau=float(mr.wall_s) / float(mr.shots), bp=float(mr.bp_s), build=float(mr.build_s),
                     nonstd=float(mr.nonstd), tab=zm["tab"].astype(int),
                     Y={q: zm[f"all_{q}"].astype(np.float64) for q in ("x", "zz", "xc", "zc")})
        idle = dict(gpu_other=";".join(sorted(set(g.gpu_other.astype(str)))),
                    other_py=";".join(sorted(set(g.other_py.astype(str)))), runner_max=int(g.runner.max()),
                    gpu_util_max=float(pd.to_numeric(g.gpu_util, errors="coerce").max()),
                    gpu_mem_max=float(pd.to_numeric(g.gpu_mem, errors="coerce").max()), load1_max=float(g.load1.max()))
        C.append(dict(key=k, ch=ch, obs=obs, labels=labels, t=t, X=X, sing=sing, multi=multi, idle=idle,
                      hard={o: hard.get(k + (o,), False) for o in obs}))
    return C


def q1(C, rng):
    rows = []
    for c in C:
        k, labels, t = c["key"], c["labels"], c["t"]
        for o in c["obs"]:
            arr = c["X"][o]
            vf = arr.var(axis=1, ddof=1)
            ib = int(np.argmin(vf))
            N = arr.shape[1]
            idx = rng.integers(0, N, size=(NB, N))
            stack = np.array([arr[j][idx].var(axis=1, ddof=1) for j in range(len(labels))])
            jb = stack.argmin(axis=0)
            b_vbf, b_tf, b_v0 = stack.min(axis=0), t[jb], stack[0]
            s = c["sing"][o]
            x = s["Y"][o]
            vs = x.var(ddof=1)
            ts = s["tau"] + (s["bp"] + s["build"]) / NPROD
            b_vs = np.maximum(x[rng.integers(0, len(x), size=(NB, len(x)))].var(axis=1, ddof=1), 1e-300)
            bW = b_vbf * b_tf / (b_vs * ts)
            bW0 = b_v0 * t[0] / (b_vs * ts)
            m = c["multi"]
            xm = m["Y"][o]
            vm = xm.var(ddof=1)
            tm = m["tau"] + (m["bp"] + m["build"]) / NPROD
            rows.append(dict(zip(KEYS, k + (o,)), best=labels[ib], var_bf=vf[ib], var_std=vf[0], t_f=t[ib], t_std=t[0],
                             single_var=vs, single_tratio=ts / t[ib], single_build_s=s["build"], single_bp_s=s["bp"],
                             single_nonstd=s["nonstd"], single_agree_best=float(np.mean(s["tab"] == ib)),
                             single_z=(x.mean() - arr[ib].mean()) / np.sqrt(vs / len(x) + vf[ib] / N),
                             single_R_bf=vf[ib] / vs, single_W_bf=vf[ib] * t[ib] / (vs * ts),
                             single_W_bf_lo=np.percentile(bW, 2.5), single_W_bf_hi=np.percentile(bW, 97.5),
                             single_R_std=vf[0] / vs, single_W_std=vf[0] * t[0] / (vs * ts),
                             single_W_std_lo=np.percentile(bW0, 2.5), single_W_std_hi=np.percentile(bW0, 97.5),
                             multi_R_bf=vf[ib] / vm, multi_W1_bf=vf[ib] * t[ib] / (vm * tm),
                             multi_vs_single_var=vm / vs,
                             multi_z=(xm.mean() - arr[ib].mean()) / np.sqrt(vm / len(xm) + vf[ib] / N),
                             hard=c["hard"][o], **c["idle"]))
    return pd.DataFrame(rows)


def q2_config(c, obs, edef):
    V = np.array([c["X"][o].var(axis=1, ddof=1) for o in obs])
    vbf, ibf = V.min(axis=1), V.argmin(axis=1)
    mu = np.array([c["X"][o][ibf[i]].mean() for i, o in enumerate(obs)])
    if edef in ("A20k", "A100k"):
        eps2 = vbf / (20000 if edef == "A20k" else 100000)
        keep = np.ones(len(obs), bool)
    else:                                                       # Brel (E10 post hoc definition)
        keep = np.abs(mu) > 4 * np.sqrt(vbf / 2000)
        if not keep.any():
            return None, int((~keep).sum())
        r2 = np.max(vbf[keep] / mu[keep] ** 2) / 20000
        eps2 = r2 * mu ** 2
    obs = [o for o, kk in zip(obs, keep) if kk]
    V, vbf, ibf, eps2 = V[keep], vbf[keep], ibf[keep], eps2[keep]
    m, tf = len(obs), c["t"]
    need = V / eps2[:, None]
    F1 = float(np.min(tf * need.max(axis=0)))
    F2 = float(np.sum(tf[ibf] * vbf / eps2))
    F3 = best_assign(m, len(tf), lambda cc, ii: tf[cc] * need[ii, cc].max())
    M = c["multi"]
    vm = np.array([M["Y"][o].var(ddof=1) for o in obs])
    Sm = float(M["tau"] * np.max(vm / eps2) + M["bp"] + M["build"])
    Smp = float(np.max(vm / eps2) * (M["tau"] + (M["bp"] + M["build"]) / NPROD))
    Ss = [c["sing"][o] for o in obs]
    vs = np.array([Ss[i]["Y"][o].var(ddof=1) for i, o in enumerate(obs)])
    S1 = float(np.sum([Ss[i]["tau"] * vs[i] / eps2[i] + Ss[i]["bp"] + Ss[i]["build"] for i in range(m)]))
    return dict(m=m, F1=F1, F2=F2, F3=F3, Smulti=Sm, S1=S1, W_multi=F1 / Sm, W_F2_Sm=F2 / Sm, W_F3_Sm=F3 / Sm,
                W_F1_Smp=F1 / Smp, W_F1_S1=F1 / S1, W_S1_Sm=S1 / Sm), int((~keep).sum())


def main_analysis(f11, rawdir, groups):
    C = load(f11, rawdir, groups)
    print(f"E11 analysis of {f11}: configs {len(C)} "
          f"({pd.Series([c['ch'] for c in C]).value_counts().to_dict()}), groups {sum(len(c['obs']) for c in C)}")
    rng = np.random.default_rng(2026)
    Q = q1(C, rng)
    Q.to_csv(os.path.join(R, "b_e11_summary.csv"), index=False)
    print(f"idle record: gpu_other {sorted(set(Q.gpu_other))}; other_py {sorted(set(Q.other_py))}; runner max "
          f"{Q.runner_max.max()}; gpu_util max {Q.gpu_util_max.max()} %; gpu_mem max {Q.gpu_mem_max.max()} MiB; "
          f"load1 max {Q.load1_max.max()}")
    for p in ("single", "multi"):
        zz = Q[f"{p}_z"].abs()
        print(f"bias |z| vs best-fixed mean ({p}, target observables): max {zz.max():.2f}, n(|z|>4) {int((zz > 4).sum())}")
    zall = []
    for c in C:
        for q in ("x", "zz", "xc", "zc"):
            V = c["X"][q].var(axis=1, ddof=1)
            ib = int(np.argmin(V))
            if V[ib] < 1e-14:
                continue
            for y in [c["multi"]["Y"][q]] + [s["Y"][q] for s in c["sing"].values()]:
                vy = y.var(ddof=1)
                zall.append((y.mean() - c["X"][q][ib].mean()) / np.sqrt(vy / len(y) + V[ib] / c["X"][q].shape[1]))
    zall = np.abs(zall)
    print(f"bias |z| over ALL 4 observables of every static run: max {zall.max():.2f}, n(|z|>4) {int((zall > 4).sum())} "
          f"of {len(zall)}")
    print(f"single: t_s/t_f(best) median {Q.single_tratio.median():.3f}; build_s median {Q.single_build_s.median():.2f}"
          f" s; bp_s median {Q.single_bp_s.median():.2f} s; non-std median {Q.single_nonstd.median():.3f}; "
          f"table = best fixed median {Q.single_agree_best.median():.3f}")
    meds = []
    print("\n######## Q1 (single-target E10 pilot table vs oracle best fixed, per target observable)")
    print("median [98.75% group bootstrap] {98.75% config-cluster}; per-group 95% CI>1 / CI<1 counts")
    subsets = [("all", Q), ("hard", Q[Q.hard])]
    for lab, sub in subsets:
        print(f"\n== subset {lab} ({len(sub)} groups) ==")
        for ch, D in sub.groupby("channel"):
            ids = D[CFG].astype(str).agg("|".join, axis=1)
            for col in ("single_W_bf", "single_R_bf", "single_W_std", "multi_W1_bf", "multi_R_bf"):
                md, lo, hi = med_ci(D[col], rng)
                clo, chi = cluster_ci(D[col], ids, rng) if ids.nunique() > 1 else (np.nan, np.nan)
                gt1 = int((D[f"{col}_lo"] > 1).sum()) if f"{col}_lo" in D else ""
                lt1 = int((D[f"{col}_hi"] < 1).sum()) if f"{col}_hi" in D else ""
                meds.append(dict(q="Q1", subset=lab, channel=ch, metric=col, units=len(D), configs=ids.nunique(),
                                 median=md, lo=lo, hi=hi, clo=clo, chi=chi, n_ci_gt1=gt1, n_ci_lt1=lt1,
                                 n_gt1=int((D[col] > 1).sum())))
                print(f"  {ch:4s} {col:14s} groups {len(D):3d}  {md:.3f} [{lo:.3f}, {hi:.3f}] {{{clo:.3f}, {chi:.3f}}}"
                      f"  >1 {int((D[col] > 1).sum())}/{len(D)}" + (f"  CI>1 {gt1} CI<1 {lt1}" if gt1 != "" else ""))
    print("\n== Q1 breakdown (POST HOC): median single_W_bf ==")
    for col in ("family", "variant", "L", "gt", "obs"):
        print(Q.pivot_table(index="channel", columns=col, values="single_W_bf", aggfunc="median").round(3))

    print("\n######## Q2 (multi-target table, one run for all observables of the config)")
    rows, nex = [], {}
    for edef in ("A20k", "A100k", "Brel"):
        for sub in ("all", "hard_r"):
            ne = 0
            for c in C:
                obs = c["obs"]
                if sub == "hard_r":
                    obs = [o for o in obs if c["hard"][o]]
                    if not obs:
                        continue
                r, ex = q2_config(c, obs, edef)
                ne += ex
                if r is None:
                    continue
                rows.append(dict(zip(CFG, c["key"]), edef=edef, subset=sub, nonstd=c["multi"]["nonstd"],
                                 build_s=c["multi"]["build"], bp_s=c["multi"]["bp"],
                                 tratio=c["multi"]["tau"] / c["t"][0], **r))
            nex[(edef, sub)] = ne
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(R, "b_e11_multi.csv"), index=False)
    A = M[(M.edef == "A20k") & (M.subset == "all")]
    print(f"multi table: tau/t_std median {A.tratio.median():.3f}; build_s median {A.build_s.median():.2f} s; "
          f"bp_s median {A.bp_s.median():.2f} s; non-std median {A.nonstd.median():.3f}; |O_k| distribution "
          f"{A.m.value_counts().sort_index().to_dict()}")
    for (edef, sub), D0 in M.groupby(["edef", "subset"], sort=False):
        print(f"\n== eps={edef} subset={sub}" + (f" (observables excluded, unresolved mean: {nex[(edef, sub)]})"
                                                 if edef == "Brel" else ""))
        for ch, D in D0.groupby("channel"):
            for col in ("W_multi", "W_F1_S1", "W_S1_Sm", "W_F2_Sm", "W_F3_Sm", "W_F1_Smp"):
                md, lo, hi = med_ci(D[col], np.random.default_rng(20261113), 4000)
                meds.append(dict(q="Q2", subset=f"{edef}/{sub}", channel=ch, metric=col, units=len(D),
                                 configs=len(D), median=md, lo=lo, hi=hi, n_gt1=int((D[col] > 1).sum())))
                print(f"  {ch:4s} {col:9s} configs {len(D):2d}  {md:.3f} [{lo:.3f}, {hi:.3f}]  >1 "
                      f"{int((D[col] > 1).sum())}/{len(D)}")
    pd.DataFrame(meds).to_csv(os.path.join(R, "b_e11_medians.csv"), index=False)

    print("\n######## DECISIONS (pre-registered; 98.75% = Bonferroni over 4 tests; ad primary, deph secondary)")
    for ch in ("ad", "deph"):
        D = Q[Q.channel == ch]
        if len(D):
            md, lo, hi = med_ci(D.single_W_bf, np.random.default_rng(7))
            print(f"  Q1 {ch:4s}: median W_bf {md:.3f} [{lo:.3f}, {hi:.3f}] groups {len(D)} -> "
                  + ("REPLICATED (median > 1, lower > 1)" if md > 1 and lo > 1 else "NOT replicated"))
        D = A[A.channel == ch]
        if len(D):
            md, lo, hi = med_ci(D.W_multi, np.random.default_rng(20261113), 4000)
            print(f"  Q2 {ch:4s}: median W_multi {md:.3f} [{lo:.3f}, {hi:.3f}] configs {len(D)} -> "
                  + ("PASS (multi-target table beats one oracle fixed run on all observables)" if md > 1 and lo > 1
                     else "FAIL"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    mode = sys.argv[1]
    if mode == "select":
        select(sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "b_e11_select.csv"))
    elif mode == "scope":
        scope(sys.argv[2])
    else:
        main_analysis(sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "b_e11_u94.csv"),
                      sys.argv[3] if len(sys.argv) > 3 else os.path.join(R, "b_e11_raw"),
                      sys.argv[4] if len(sys.argv) > 4 else os.path.join(R, "b_e11_groups.csv"))
