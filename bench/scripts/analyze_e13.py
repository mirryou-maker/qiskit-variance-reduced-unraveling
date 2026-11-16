"""Analysis of E13 (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E13 pre-registration").

  python analyze_e13.py check                       -> e13_rules.caps() reproduces b_e9d_summary.csv (E9 data)
  python analyze_e13.py select [sel dir]            -> b_e13_select_summary.csv, b_e13_groups.csv (TN at cap 1e7)
  python analyze_e13.py scope <smoke main dir>      -> b_e13_scope.txt (config order + projected node-hours)
  python analyze_e13.py main [main dir [raw dir]]   -> b_e13_summary.csv, b_e13_multi.csv, b_e13_medians.csv
CPU only (ubuntu94 env sci, or an iREMB compute node).

Per group (config k, target o), t in s per trajectory, NPROD = 20000:
  bf      : oracle best fixed candidate, argmin_c Var_c(o) over the N = 2000 fixed runs; t_bf = its wall / N.
  ST      : E10 pilot table, t = wall/N + (bp_s + build_s)/NPROD.
  CV0     : control variate on candidate 0, cross-fitted coefficient (primary), t = wall/N + bp_s/NPROD.
  CVbf    : control variate on the oracle best fixed candidate (= CV0 run if bf = 0).
  ST+CV   : control variate on the ST table, t = wall/N + (bp_s + build_s)/NPROD.
  Q1  : W_ST_bf    = Var_bf t_bf / (Var_ST t_ST)
  Q2a : W_CV0_bf   = Var_bf t_bf / (Var_CV0 t_CV0)
  Q2b : W_STCV_CV0 = Var_CV0 t_CV0 / (Var_STCV t_STCV)
Decision (4 tests, Bonferroni 98.75%): Q1 ad, Q2a ad, Q2b ad, Q2a deph: median over groups > 1 and 98.75% group
bootstrap lower bound > 1 (B = 2000, default_rng(7) for the decision line, as E11). Everything else is secondary.
"""
import glob
import importlib.util
import os
import sys
from math import comb

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
sys.path.insert(0, os.path.dirname(__file__))
import e13_rules as RU  # noqa: E402


def _load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = KEYS[:-1]
NCAND = {"deph": 7, "ad": 5}
LO, HI = 0.625, 99.375
NB = 500
NPROD = 20000
NODEH_BUDGET = 120.0
OBS4 = ("x", "zz", "xc", "zc")


def key_str(ch, fam, var, n, L, gt):
    return f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"


# ------------------------------------------------------------------------------------------------ check / select
def check():
    n, bad = RU.check()
    print(f"e13_rules.caps() vs b_e9d_summary.csv: {n} groups; mismatches {bad}")
    assert all(v == 0 for v in bad.values())


def summarize_selection(seldir):
    A11 = _load("analyze_e11")
    fs = sorted(glob.glob(os.path.join(seldir, "sel_*.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.status == "ok"].copy()
    d["M"] = pd.to_numeric(d.M, errors="coerce")
    a = d[(d.kind == "traj") | d.M.isin([float(m) for m in RU.MS_A])]
    Sm = A11.classify(a)
    hi = d[(d.kind == "pp") & d.M.isin([float(m) for m in RU.MS_B + [RU.M_C]])].drop_duplicates(KEYS + ["M"], keep="last")
    for m in RU.MS_B + [RU.M_C]:
        sub = hi[hi.M == float(m)][KEYS + ["value", "wall_s"]].rename(columns={"value": f"pp{m}", "wall_s": f"t{m}"})
        Sm = Sm.merge(sub, on=KEYS, how="left")
    nd = Sm[~Sm.deterministic.astype(bool)].copy()
    nd = RU.caps(nd)
    det = Sm[Sm.deterministic.astype(bool)]
    return d, Sm, nd, det, fs


def select(seldir):
    check()
    d, Sm, nd, det, fs = summarize_selection(seldir)
    oom = d.status.eq("oom").sum() if "status" in d else 0
    print(f"selection ledgers {len(fs)}; groups {len(Sm)} (deterministic {len(det)}); workers "
          f"{sorted(set(d.worker))}; gpus {sorted(set(d.gpu))}; gpu_other {sorted(set(d.gpu_other.astype(str)))[:6]}")
    missB = nd.need_B & nd[[f"pp{m}" for m in RU.MS_B]].isna().any(axis=1)
    print(f"stage B needed {int(nd.need_B.sum())}, missing {int(missB.sum())}; stage C needed {int(nd.need_C.sum())}, "
          f"missing {int(nd.missing_C.sum())}; oom rows {int(oom)}")
    nd.to_csv(os.path.join(R, "b_e13_select_summary.csv"), index=False)
    for col in ("TN", "TN100000", "TN1000000", "TN3000000", "TN10000000"):
        x = nd[col].astype(float)
        print(f"  {col:11s}: {int(x.sum()):3d}/{len(nd)}  " + "  ".join(
            f"{ch} {int(g[col].astype(float).sum())}/{len(g)}" for ch, g in nd.groupby("channel")))
    g = nd[(nd.TN10000000 == 1) & ~nd.missing_C & ~missB]
    g = g[KEYS + ["traj_mean", "var_std", "eps"]]
    g.to_csv(os.path.join(R, "b_e13_groups.csv"), index=False)
    cf = g.drop_duplicates(CFG)
    print(f"-> b_e13_groups.csv: {len(g)} groups TN at cap 1e7, {len(cf)} configs; by channel groups "
          f"{g.groupby('channel').size().to_dict()}, configs {cf.groupby('channel').size().to_dict()}")
    print(g.groupby(["channel", "L"]).size().rename("groups").to_frame().join(
        cf.groupby(["channel", "L"]).size().rename("configs")))
    print("by variant:", g.groupby(["channel", "variant"]).size().to_dict())
    print("by obs:", g.groupby(["channel", "obs"]).size().to_dict())
    t = nd[nd.need_C]
    if len(t) and "t100000000" in t:
        print(f"t_PP(1e8): median {t.t100000000.median():.0f} s, max {t.t100000000.max():.0f} s; |PP_1e7 - PP_1e8|/eps "
              f"over TN@1e7 groups median {((nd.pp10000000 - nd.pp100000000).abs() / nd.eps)[nd.TN10000000 == 1].median():.2f}")


# ------------------------------------------------------------------------------------------------------- scope
def load_runs(maindir):
    fs = sorted(glob.glob(os.path.join(maindir, "main_*.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 0], ignore_index=True)
    return d


def scope(smokedir):
    B11 = _load("analyze_e11")
    fits = B11.tmodel()                                     # E10 ubuntu94 fixed re-timings: t = a L + b
    sm = load_runs(smokedir)
    cal, rcv, tbs = {}, {}, {}
    for k, g in sm.groupby(CFG):
        ch, L = k[0], k[4]
        fx = g[g.kind == "fixed"].drop_duplicates("scheme")
        t = (fx.wall_s / fx.shots).to_numpy()
        mod = np.array([np.dot(fits.get((ch, k[1], c), fits[(ch, c)]), [L, 1]) for c in range(NCAND[ch])])
        cal[ch] = float(np.sum(t) / np.sum(mod))
        cvr = g[g.kind.isin(["cv0", "cvbf", "stcv"]) & g.wall_s.notna()]
        rcv[ch] = float((cvr.wall_s / cvr.shots).median() / t[0])
        si = g[g.kind == "single"]
        tbs[ch] = (float((si.build_s / (fx.wall_s.iloc[0] / fx.shots.iloc[0])).median()),
                   float(si.bp_s.median()))
        print(f"smoke {key_str(*k)}: t_f {np.round(t, 4)} s; calibration measured/model {cal[ch]:.3f}; "
              f"t_cv/t_0 {rcv[ch]:.2f}; build_s/t_0 {tbs[ch][0]:.0f}; bp_s {tbs[ch][1]:.2f}")

    def proj(k, nobs, N=2000):
        ch, fam, L = k[0], k[1], k[4]
        tc = cal[ch] * np.array([np.dot(fits.get((ch, fam, c), fits[(ch, c)]), [L, 1]) for c in range(NCAND[ch])])
        per_obs = N * tc[0] + tbs[ch][0] * tc[0] + tbs[ch][1] + 3 * N * rcv[ch] * tc[0]
        return N * tc.sum() + nobs * per_obs + (NCAND[ch] + 4 * nobs) * 2.5

    G = pd.read_csv(os.path.join(R, "b_e13_groups.csv"))
    cfg = {k: len(g) for k, g in G.groupby(CFG)}
    SEL = _load("b_e13_select")
    order = [k for ch in ("ad", "deph") for k in SEL.hash_order([k for k in cfg if k[0] == ch], "e13-order")]
    tot, keep, lines = 0.0, [], []
    for k in order:
        c = proj(k, cfg[k]) / 3600
        inc = tot + c <= NODEH_BUDGET
        if inc:
            keep.append(k)
            tot += c
        lines.append(f"# {key_str(*k):40s} obs {cfg[k]} proj {c:6.2f} node-h {'IN' if inc else 'OUT'}")
        if not inc:
            break
    full = sum(proj(k, cfg[k]) for k in cfg) / 3600
    out = os.path.join(R, "b_e13_scope.txt")
    with open(out, "w") as fh:
        fh.write(f"# calibration {cal}; t_cv/t_0 {rcv}; projected full main run {full:.1f} node-h; budget "
                 f"{NODEH_BUDGET} node-h; kept {len(keep)}/{len(cfg)} configs ({sum(cfg[k] for k in keep)} groups), "
                 f"projected {tot:.1f} node-h\n")
        fh.write("\n".join(lines) + "\n")
        fh.write("\n".join(key_str(*k) for k in keep) + "\n")
    print(open(out).read().splitlines()[0])


# -------------------------------------------------------------------------------------------------------- main
def med_ci(x, rng, B=2000):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [LO, HI])


def cluster_ci(x, ids, rng, B=2000):
    x = np.asarray(x, float)
    u, inv = np.unique(np.asarray(ids), return_inverse=True)
    if len(u) < 2:
        return np.nan, np.nan
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return tuple(np.percentile(cb, [LO, HI]))


def cv_var(f, S, rule="crossfit"):
    """Variance of the per-trajectory CV estimates; f, S (..., N) arrays (bootstrap replicates along axis 0)."""
    f, S = np.atleast_2d(f), np.atleast_2d(S)
    if rule == "beta1":
        return (f - S).var(axis=-1, ddof=1)
    if rule == "insample":
        fc, sc = f - f.mean(-1, keepdims=True), S - S.mean(-1, keepdims=True)
        b = (fc * sc).sum(-1) / np.maximum((sc * sc).sum(-1), 1e-300)
        return (f - b[:, None] * S).var(axis=-1, ddof=1)
    ev = np.arange(f.shape[-1]) % 2 == 0

    def beta(m):
        fm, sm = f[:, m], S[:, m]
        fc, sc = fm - fm.mean(-1, keepdims=True), sm - sm.mean(-1, keepdims=True)
        return (fc * sc).sum(-1) / np.maximum((sc * sc).sum(-1), 1e-300)
    be, bo = beta(ev), beta(~ev)
    coef = np.where(ev[None, :], bo[:, None], be[:, None])
    return (f - coef * S).var(axis=-1, ddof=1)


def cv_mean(f, S, rule="crossfit"):
    ev = np.arange(len(f)) % 2 == 0
    if rule == "beta1":
        return float((f - S).mean())

    def beta(m):
        return np.cov(f[m], S[m], ddof=1)[0, 1] / max(S[m].var(ddof=1), 1e-300)
    if rule == "insample":
        return float((f - beta(np.ones(len(f), bool)) * S).mean())
    coef = np.where(ev, beta(~ev), beta(ev))
    return float((f - coef * S).mean())


def load(maindir, rawdir):
    d = load_runs(maindir)
    G = pd.read_csv(os.path.join(R, "b_e13_groups.csv"))
    C, dups = [], []
    for k, gk in d.groupby(CFG):
        ch = k[0]
        obs = sorted(set(G.merge(pd.DataFrame([dict(zip(CFG, k))]), on=CFG).obs))
        complete = []
        for wk, g in gk.groupby("worker"):
            fx = g[g.kind == "fixed"]
            ok = (fx.scheme.nunique() == NCAND[ch]
                  and all(set(g[g.kind == kd].obs) >= set(obs) for kd in ("single", "cv0", "cvbf", "stcv")))
            if ok:
                complete.append((g.timestamp.max(), wk, g))
        if not complete:
            print(f"incomplete config skipped: {key_str(*k)}")
            continue
        complete.sort()
        if len(complete) > 1:
            dups.append(key_str(*k))
        _, wk, g = complete[0]
        g = g.drop_duplicates(["obs", "kind", "scheme"], keep="last")
        ks = key_str(*k)
        pre = os.path.join(rawdir, f"{ks}__{wk}")
        fr = np.load(pre + "_fixed.npz")
        fx = g[g.kind == "fixed"]
        labels = list(fx[fx.obs == obs[0]].sort_values("cand").scheme)
        X = {q: np.array([fr[f"fixed_{s}_{q}"].astype(np.float64) for s in labels]) for q in OBS4}
        f0 = fx[fx.obs == obs[0]].set_index("scheme")
        t = np.array([float(f0.loc[s, "wall_s"]) / float(f0.loc[s, "shots"]) for s in labels])
        per = {}
        for o in obs:
            r = g[(g.kind == "single") & (g.obs == o)].iloc[0]
            z = np.load(pre + f"_single_{o}.npz")
            e = dict(single=dict(tau=r.wall_s / r.shots, bp=float(r.bp_s), build=float(r.build_s), nonstd=float(r.nonstd),
                                 tab=z["tab"].astype(int), Y={q: z[f"all_{q}"].astype(np.float64) for q in OBS4}))
            for kd in ("cv0", "cvbf", "stcv"):
                rr = g[(g.kind == kd) & (g.obs == o)].iloc[0]
                src = "cv0" if (kd == "cvbf" and rr.scheme == "cv0") else kd
                rs = g[(g.kind == src) & (g.obs == o)].iloc[0]
                zz = np.load(pre + f"_{src}_{o}.npz")
                build = float(rs.build_s) if src == "stcv" else 0.0
                e[kd] = dict(tau=rs.wall_s / rs.shots, bp=float(rs.bp_s), build=build, cand=int(rs.cand),
                             f=zz["f"].astype(np.float64), S=zz["S"].astype(np.float64), same_as_cv0=(src != kd),
                             Y={q: zz[f"all_{q}"].astype(np.float64) for q in OBS4})
            per[o] = e
        idle = dict(gpu_other=";".join(sorted(set(g.gpu_other.dropna().astype(str)))),
                    other_py=";".join(sorted(set(g.other_py.dropna().astype(str)))),
                    gpu_util_max=float(pd.to_numeric(g.gpu_util, errors="coerce").max()),
                    gpu_mem_max=float(pd.to_numeric(g.gpu_mem, errors="coerce").max()),
                    load1_max=float(pd.to_numeric(g.load1, errors="coerce").max()), host=str(g.host.iloc[0]),
                    gpu=str(g.gpu.iloc[0]), worker=wk)
        C.append(dict(key=k, ch=ch, obs=obs, labels=labels, t=t, X=X, per=per, idle=idle))
    return C, dups


def group_rows(C, rng):
    rows = []
    for c in C:
        k, labels, t = c["key"], c["labels"], c["t"]
        for o in c["obs"]:
            arr = c["X"][o]
            vf = arr.var(axis=1, ddof=1)
            ib = int(np.argmin(vf))
            N = arr.shape[1]
            e = c["per"][o]
            s = e["single"]
            xs = s["Y"][o]
            vs = xs.var(ddof=1)
            ts = s["tau"] + (s["bp"] + s["build"]) / NPROD
            tm = {kd: e[kd]["tau"] + (e[kd]["bp"] + e[kd]["build"]) / NPROD for kd in ("cv0", "cvbf", "stcv")}
            vcv = {(kd, rule): float(cv_var(e[kd]["f"], e[kd]["S"], rule)[0])
                   for kd in ("cv0", "cvbf", "stcv") for rule in ("crossfit", "beta1", "insample")}
            row = dict(zip(KEYS, k + (o,)), best=labels[ib], cand_bf=ib, var_bf=vf[ib], var_std=vf[0], t_bf=t[ib],
                       t_std=t[0], var_st=vs, t_st=ts, st_nonstd=s["nonstd"], st_build_s=s["build"], st_bp_s=s["bp"],
                       st_agree_best=float(np.mean(s["tab"] == ib)), cvbf_is_cv0=e["cvbf"]["same_as_cv0"])
            for kd in ("cv0", "cvbf", "stcv"):
                row[f"t_{kd}"] = tm[kd]
                row[f"tratio_{kd}"] = e[kd]["tau"] / t[0]
                row[f"var_plain_{kd}"] = e[kd]["f"].var(ddof=1)
                for rule in ("crossfit", "beta1", "insample"):
                    row[f"var_{kd}_{rule}"] = vcv[(kd, rule)]
                row[f"Rin_{kd}"] = row[f"var_plain_{kd}"] / vcv[(kd, "crossfit")]
            Cbf = vf[ib] * t[ib]
            row["W_ST_bf"] = Cbf / (vs * ts)
            row["R_ST_bf"] = vf[ib] / vs
            row["W_ST_std"] = vf[0] * t[0] / (vs * ts)
            for rule in ("crossfit", "beta1", "insample"):
                sfx = "" if rule == "crossfit" else f"_{rule}"
                c0 = vcv[("cv0", rule)] * tm["cv0"]
                cb = vcv[("cvbf", rule)] * tm["cvbf"]
                cs = vcv[("stcv", rule)] * tm["stcv"]
                row[f"W_CV0_bf{sfx}"] = Cbf / c0
                row[f"W_STCV_CV0{sfx}"] = c0 / cs
                row[f"W_CVbf_bf{sfx}"] = Cbf / cb
                row[f"W_STCV_bf{sfx}"] = Cbf / cs
                row[f"W_ST_CV0{sfx}"] = c0 / (vs * ts)
                row[f"W_STCV_CVbest{sfx}"] = min(c0, cb) / cs
            row["R_CV0_bf"] = vf[ib] / vcv[("cv0", "crossfit")]
            row["R_STCV_CV0"] = vcv[("cv0", "crossfit")] / vcv[("stcv", "crossfit")]
            # per-group bootstrap (B = NB; bf re-chosen per replicate; cross-fit recomputed per replicate)
            idx = rng.integers(0, N, size=(NB, N))
            stack = np.array([arr[j][idx].var(axis=1, ddof=1) for j in range(len(labels))])
            jb = stack.argmin(axis=0)
            bC = stack.min(axis=0) * t[jb]
            bs = xs[rng.integers(0, len(xs), size=(NB, len(xs)))].var(axis=1, ddof=1) * ts
            bcv = {}
            for kd in ("cv0", "stcv"):
                ii = rng.integers(0, len(e[kd]["f"]), size=(NB, len(e[kd]["f"])))
                bcv[kd] = cv_var(e[kd]["f"][ii], e[kd]["S"][ii]) * tm[kd]
            for lab, num, den in (("W_ST_bf", bC, bs), ("W_CV0_bf", bC, bcv["cv0"]), ("W_STCV_CV0", bcv["cv0"], bcv["stcv"])):
                q = num / np.maximum(den, 1e-300)
                row[f"{lab}_lo"], row[f"{lab}_hi"] = np.percentile(q, [2.5, 97.5])
            # bias: z vs the best-fixed mean
            mbf, se_bf2 = arr[ib].mean(), vf[ib] / N
            row["z_st"] = (xs.mean() - mbf) / np.sqrt(vs / len(xs) + se_bf2)
            for kd in ("cv0", "cvbf", "stcv"):
                for rule in ("crossfit", "beta1", "insample"):
                    m = cv_mean(e[kd]["f"], e[kd]["S"], rule)
                    row[f"z_{kd}_{rule}"] = (m - mbf) / np.sqrt(vcv[(kd, rule)] / len(e[kd]["f"]) + se_bf2)
                row[f"z_S_{kd}"] = e[kd]["S"].mean() / (e[kd]["S"].std(ddof=1) / np.sqrt(len(e[kd]["S"])))
            row.update(c["idle"])
            rows.append(row)
    return pd.DataFrame(rows)


def sign_p(k, n):
    """Two-sided exact binomial (sign) test p-value for k successes of n at p = 1/2."""
    if n == 0:
        return np.nan
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main_analysis(maindir, rawdir):
    C, dups = load(maindir, rawdir)
    print(f"E13 analysis: configs {len(C)} ({pd.Series([c['ch'] for c in C]).value_counts().to_dict()}), groups "
          f"{sum(len(c['obs']) for c in C)}; configs with a duplicate complete run (first used): {dups}")
    rng = np.random.default_rng(2026)
    Q = group_rows(C, rng)
    Q.to_csv(os.path.join(R, "b_e13_summary.csv"), index=False)
    print(f"hosts {sorted(set(Q.host))}; gpus {sorted(set(Q.gpu))}")
    print(f"idle record: gpu_other {sorted(set(Q.gpu_other))}; other_py {sorted(set(Q.other_py))}; gpu_util max "
          f"{Q.gpu_util_max.max()} %; gpu_mem max {Q.gpu_mem_max.max()} MiB; load1 max {Q.load1_max.max()}")
    zc = [c for c in Q.columns if c.startswith("z_")]
    for c in zc:
        print(f"  bias |{c}| max {Q[c].abs().max():.2f}, n(>4) {int((Q[c].abs() > 4).sum())}")
    zall = []
    for c in C:
        for q in OBS4:
            V = c["X"][q].var(axis=1, ddof=1)
            ib = int(np.argmin(V))
            if V[ib] < 1e-14:
                continue
            for e in c["per"].values():
                ys = [e["single"]["Y"][q]] + [e[kd]["Y"][q] for kd in ("cv0", "cvbf", "stcv") if not e[kd]["same_as_cv0"]]
                for y in ys:
                    zall.append((y.mean() - c["X"][q][ib].mean()) / np.sqrt(y.var(ddof=1) / len(y) + V[ib] / c["X"][q].shape[1]))
    zall = np.abs(zall)
    print(f"bias |z| of the PLAIN estimates of all 4 observables in every ST / CV run vs best fixed: max {zall.max():.2f}, "
          f"n(>4) {int((zall > 4).sum())} of {len(zall)}")
    for ch, D in Q.groupby("channel"):
        print(f"[{ch}] t_ST/t_bf {(D.t_st / D.t_bf).median():.3f}; t_cv0/t_0 {D.tratio_cv0.median():.2f}; t_stcv/t_0 "
              f"{D.tratio_stcv.median():.2f}; ST build_s {D.st_build_s.median():.1f} s; bp_s {D.st_bp_s.median():.2f} s; "
              f"ST non-std {D.st_nonstd.median():.3f}; CVbf = CV0 in {int(D.cvbf_is_cv0.sum())}/{len(D)}; "
              f"best fixed {D.best.value_counts().to_dict()}; within-run CV variance reduction Var(f)/Var(Y) CV0 "
              f"{D.Rin_cv0.median():.2f}, CVbf {D.Rin_cvbf.median():.2f}, STCV {D.Rin_stcv.median():.2f}")

    meds = []
    metrics = [("W_ST_bf", "Q1"), ("W_CV0_bf", "Q2a"), ("W_STCV_CV0", "Q2b"), ("R_ST_bf", "sec"), ("R_CV0_bf", "sec"),
               ("R_STCV_CV0", "sec"), ("W_ST_std", "sec"), ("W_CVbf_bf", "sec"), ("W_STCV_bf", "sec"),
               ("W_ST_CV0", "sec"), ("W_STCV_CVbest", "sec"), ("W_CV0_bf_beta1", "sec"), ("W_STCV_CV0_beta1", "sec"),
               ("W_CV0_bf_insample", "sec"), ("W_STCV_CV0_insample", "sec"), ("W_CVbf_bf_beta1", "sec")]
    print("\n######## per-group metrics: median [98.75% group bootstrap] {98.75% config-cluster} <98.75% seed-cluster>")
    for ch, D in Q.groupby("channel"):
        cid = D[CFG].astype(str).agg("|".join, axis=1)
        sid = D.variant.astype(str)
        print(f"\n== {ch}: {len(D)} groups, {cid.nunique()} configs, {sid.nunique()} circuit instances (seeds)")
        for col, q in metrics:
            md, lo, hi = med_ci(D[col], rng)
            clo, chi = cluster_ci(D[col], cid, rng)
            slo, shi = cluster_ci(D[col], sid, rng)
            gt1 = int((D[f"{col}_lo"] > 1).sum()) if f"{col}_lo" in D else ""
            lt1 = int((D[f"{col}_hi"] < 1).sum()) if f"{col}_hi" in D else ""
            sm = D.groupby("variant")[col].median()
            ks, ns = int((sm > 1).sum()), len(sm)
            meds.append(dict(q=q, channel=ch, metric=col, groups=len(D), configs=cid.nunique(), seeds=ns, median=md,
                             lo=lo, hi=hi, clo=clo, chi=chi, slo=slo, shi=shi, n_gt1=int((D[col] > 1).sum()),
                             n_ci_gt1=gt1, n_ci_lt1=lt1, seeds_gt1=ks, sign_p=sign_p(ks, ns)))
            print(f"  {q:3s} {col:20s} {md:.3f} [{lo:.3f}, {hi:.3f}] {{{clo:.3f}, {chi:.3f}}} <{slo:.3f}, {shi:.3f}>  "
                  f">1 {int((D[col] > 1).sum())}/{len(D)}" + (f"  CI>1 {gt1} CI<1 {lt1}" if gt1 != "" else "")
                  + f"  seeds>1 {ks}/{ns} sign p {sign_p(ks, ns):.3g}")
    print("\n== POST HOC breakdown (median by L / obs) ==")
    for col in ("W_ST_bf", "W_CV0_bf", "W_STCV_CV0", "W_ST_CV0"):
        print(col)
        print(Q.pivot_table(index="channel", columns="L", values=col, aggfunc="median").round(3))
        print(Q.pivot_table(index="channel", columns="obs", values=col, aggfunc="median").round(3))
    print("\n== per-seed medians (instance level, secondary) ==")
    print(Q.pivot_table(index=["channel", "variant"], values=["W_ST_bf", "W_CV0_bf", "W_STCV_CV0"], aggfunc="median").round(3))

    # secondary: all-observable accounting (per config; one oracle fixed run serving all O_k vs per-target methods)
    rows = []
    for c in C:
        obs = c["obs"]
        V = np.array([c["X"][o].var(axis=1, ddof=1) for o in obs])
        vbf = V.min(axis=1)
        eps2 = vbf / NPROD
        F1 = float(np.min(c["t"] * (V / eps2[:, None]).max(axis=0)))
        cost = {}
        for kd in ("single", "cv0", "stcv"):
            tot = 0.0
            for i, o in enumerate(obs):
                e = c["per"][o][kd]
                v = e["Y"][o].var(ddof=1) if kd == "single" else float(cv_var(e["f"], e["S"])[0])
                tot += e["tau"] * v / eps2[i] + e["bp"] + e["build"]
            cost[kd] = tot
        rows.append(dict(zip(CFG, c["key"]), m=len(obs), F1=F1, W_F1_ST=F1 / cost["single"], W_F1_CV0=F1 / cost["cv0"],
                         W_F1_STCV=F1 / cost["stcv"]))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(R, "b_e13_multi.csv"), index=False)
    print("\n== SECONDARY all-observable accounting (per config; one oracle fixed run for all TN observables vs one "
          "per-target run each, one-time costs paid per target): median [98.75% config bootstrap]")
    for ch, D in M.groupby("channel"):
        for col in ("W_F1_ST", "W_F1_CV0", "W_F1_STCV"):
            md, lo, hi = med_ci(D[col], np.random.default_rng(20261113), 4000)
            meds.append(dict(q="multi", channel=ch, metric=col, groups=int(D.m.sum()), configs=len(D), median=md, lo=lo,
                             hi=hi, n_gt1=int((D[col] > 1).sum())))
            print(f"  {ch:4s} {col:10s} configs {len(D):2d}  {md:.3f} [{lo:.3f}, {hi:.3f}]  >1 {int((D[col] > 1).sum())}/{len(D)}")
    pd.DataFrame(meds).to_csv(os.path.join(R, "b_e13_medians.csv"), index=False)

    print("\n######## DECISIONS (pre-registered; 4 tests, Bonferroni 98.75%: Q1 ad, Q2a ad, Q2b ad, Q2a deph)")
    for ch, col, lab in (("ad", "W_ST_bf", "Q1 "), ("ad", "W_CV0_bf", "Q2a"), ("ad", "W_STCV_CV0", "Q2b"),
                         ("deph", "W_CV0_bf", "Q2a")):
        D = Q[Q.channel == ch]
        if not len(D):
            print(f"  {lab} {ch}: no groups")
            continue
        md, lo, hi = med_ci(D[col], np.random.default_rng(7))
        print(f"  {lab} {ch:4s} {col:11s} median {md:.3f} [{lo:.3f}, {hi:.3f}] groups {len(D)} -> "
              + ("PASS (median > 1, lower > 1)" if md > 1 and lo > 1 else "FAIL"))
    print("  (secondary, same level) " + "; ".join(
        f"{ch} {col} {med_ci(Q[Q.channel == ch][col], np.random.default_rng(7))[0]:.3f} "
        f"[{med_ci(Q[Q.channel == ch][col], np.random.default_rng(7))[1]:.3f}, "
        f"{med_ci(Q[Q.channel == ch][col], np.random.default_rng(7))[2]:.3f}]"
        for ch, col in (("deph", "W_ST_bf"), ("deph", "W_STCV_CV0")) if (Q.channel == ch).any()))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    mode = sys.argv[1]
    if mode == "check":
        check()
    elif mode == "select":
        select(sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "b_e13_sel"))
    elif mode == "scope":
        scope(sys.argv[2])
    else:
        main_analysis(sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "b_e13_main"),
                      sys.argv[3] if len(sys.argv) > 3 else os.path.join(R, "b_e13_raw"))
