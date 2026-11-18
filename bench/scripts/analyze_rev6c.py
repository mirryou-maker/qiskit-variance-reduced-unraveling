"""Analysis of Rev6 C (pre-registered in docs/plan_B_characterization_2026-11-06.md, "Rev6 C/D pre-registration", and
"Rev6 C amendment 1").

Enlarged amplitude-damping fixed baseline: the 5 E13 candidates (E13 fixed runs) + fixed partially-mixed unravelings
(b_rev6c.py, channel ad_mix, phi = pi/4):
  set rev6c  (pre-registered)          : candidates 5, 6, 7  = theta pi/16, pi/8, 3pi/16
  set rev6c2 (amendment 1, after partial data): candidates 8, 9, 10 = tan(theta) = x sqrt(g), x = 0.25, 0.5, 1
all N = 2000 on the 64 E13 ad groups.

  python analyze_rev6c.py              -> bench/results/b_rev6c_summary.csv, b_rev6c_medians.csv (stdout -> b_rev6c_analysis.txt)
  python analyze_rev6c.py --pre_only   -> the pre-registered 8-candidate analysis only (*_preonly.csv), run as soon as
                                          the pre-registered set is complete, before the amendment-1 runs finish

Per group (config k, target o):
  t_c (new c, primary) = wall_c / N x t_jump,E13 / t_anchor   (anchor = jump re-run in the same session as c)
  bf8  = argmin_c Var_c over candidates 0-7 ; bf11 = over 0-10 ; C_bf = Var_bf t_bf (oracle, uncharged)
  Q1-rev6  (pre-registered)            : W_ST_bf8  = C_bf8  / (Var_ST t_ST)
  Q1-rev6b (amended after partial data): W_ST_bf11 = C_bf11 / (Var_ST t_ST)
  t_ST = tau_ST + (bp + build)/N_prod, N_prod = 20000 (E13 decision); PASS iff median > 1 and 98.75% group-bootstrap
  lower bound > 1 (B = 2000, default_rng(7)).
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
OBS4 = A13.OBS4
NB = 500
NEW = (5, 6, 7)
SETS = [("rev6c", "b_rev6c_main", "b_rev6c_raw", (5, 6, 7)), ("rev6c2", "b_rev6c2_main", "b_rev6c2_raw", (8, 9, 10))]


def load_rev6c(maindir, rawdir, new=NEW):
    fs = sorted(glob.glob(os.path.join(maindir, "rev6c_*.csv")))
    if not fs:
        return {}, []
    d = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 0], ignore_index=True)
    out, dups = {}, []
    for k, gk in d.groupby(CFG):
        comp = []
        for wk, g in gk.groupby("worker"):
            kinds = set(zip(g.kind, g.cand))
            raw = os.path.join(rawdir, f"{A13.key_str(*k)}__{wk}_rev6c.npz")
            if kinds >= {("anchor", 0)} | {("fixed", c) for c in new} and os.path.exists(raw):
                comp.append((g.timestamp.max(), wk, g, raw))
        if not comp:
            print(f"{os.path.basename(maindir)}: incomplete config skipped {A13.key_str(*k)}")
            continue
        comp.sort(key=lambda x: x[0])
        if len(comp) > 1:
            dups.append(A13.key_str(*k))
        _, wk, g, raw = comp[0]
        z = np.load(raw)
        g1 = g[g.obs == "x"].set_index(["kind", "cand"])
        out[k] = dict(worker=wk, host=str(g.host.iloc[0]), e13_host=str(g.e13_host.iloc[0]),
                      t_anchor=float(g1.loc[("anchor", 0), "wall_s"]) / float(g1.loc[("anchor", 0), "shots"]),
                      t_new={c: float(g1.loc[("fixed", c), "wall_s"]) / float(g1.loc[("fixed", c), "shots"]) for c in new},
                      labels={c: str(g1.loc[("fixed", c), "scheme"]) for c in new},
                      X={q: np.array([z[f"fixed_{str(g1.loc[('fixed', c), 'scheme'])}_{q}"].astype(np.float64)
                                      for c in new]) for q in OBS4},
                      Xanchor={q: z[f"anchor_jump_{q}"].astype(np.float64) for q in OBS4},
                      idle=dict(gpu_other=";".join(sorted(set(g.gpu_other.astype(str)))),
                                other_py=";".join(sorted(set(g.other_py.astype(str)))),
                                gpu_util_max=float(pd.to_numeric(g.gpu_util, errors="coerce").max()),
                                load1_max=float(pd.to_numeric(g.load1, errors="coerce").max())))
    return out, dups


def summarize(Q, cols, label, meds):
    cid = Q[CFG].astype(str).agg("|".join, axis=1)
    sid = Q.variant.astype(str)
    print(f"\n######## {label}: median [98.75% group bootstrap] {{98.75% config-cluster}} <98.75% seed-cluster>")
    for col in cols:
        md, lo, hi = A13.med_ci(Q[col], np.random.default_rng(7))
        clo, chi = A13.cluster_ci(Q[col], cid, np.random.default_rng(11))
        slo, shi = A13.cluster_ci(Q[col], sid, np.random.default_rng(13))
        sm = Q.groupby("variant")[col].median()
        meds.append(dict(set=label, metric=col, groups=len(Q), configs=cid.nunique(), seeds=sid.nunique(), median=md,
                         lo=lo, hi=hi, clo=clo, chi=chi, slo=slo, shi=shi, n_gt1=int((Q[col] > 1).sum()),
                         seeds_gt1=int((sm > 1).sum()), sign_p=A13.sign_p(int((sm > 1).sum()), len(sm))))
        extra = (f"  CI>1 {int((Q[col + '_lo'] > 1).sum())} CI<1 {int((Q[col + '_hi'] < 1).sum())}"
                 if col + "_lo" in Q else "")
        print(f"  {col:20s} {md:.3f} [{lo:.3f}, {hi:.3f}] {{{clo:.3f}, {chi:.3f}}} <{slo:.3f}, {shi:.3f}>  >1 "
              f"{int((Q[col] > 1).sum())}/{len(Q)}  seeds>1 {int((sm > 1).sum())}/{len(sm)}{extra}")


def decision(Q, col, name, desc):
    md, lo, hi = A13.med_ci(Q[col], np.random.default_rng(7))
    verdict = "PASS" if md > 1 and lo > 1 else "FAIL"
    sub = ("" if verdict == "FAIL" else
           (" -> median < 1.30: a fixed mixed unraveling captures a material part of the gain" if md < 1.30 else
            f" -> median >= 1.30: the E13 conclusion stands against the enlarged baseline ({desc})"))
    print(f"  {name} ad {col} median {md:.3f} [{lo:.3f}, {hi:.3f}] groups {len(Q)} -> {verdict}{sub}")


def main():
    C13, _ = A13.load(os.path.join(R, "b_e13_main"), os.path.join(R, "b_e13_raw"))
    C13 = [c for c in C13 if c["ch"] == "ad"]
    loaded = {}
    sets = [x for x in SETS if "--pre_only" not in sys.argv or x[0] == "rev6c"]
    out_sfx = "_preonly" if "--pre_only" in sys.argv else ""
    for name, md, rd, new in sets:
        loaded[name], dups = load_rev6c(os.path.join(R, md), os.path.join(R, rd), new)
        print(f"set {name} (cands {new}): configs {len(loaded[name])}; duplicate complete runs (first used) {dups}")
    S13 = pd.read_csv(os.path.join(R, "b_e13_summary.csv"))
    have2 = len(loaded.get("rev6c2", {})) > 0
    print(f"Rev6 C analysis: E13 ad configs {len(C13)}")
    rng = np.random.default_rng(2026)
    rows = []
    for c in C13:
        k = c["key"]
        if k not in loaded["rev6c"] or (have2 and k not in loaded["rev6c2"]):
            print(f"  missing rev6c/rev6c2 config {A13.key_str(*k)} -> its groups are excluded")
            continue
        t5 = c["t"]                                          # E13 times of candidates 0-4
        labs, tcal, traw, Xs, anchors = list(c["labels"]), list(t5), list(t5), [c["X"]], {}
        for name, _, _, new in SETS:
            if name not in loaded or k not in loaded[name]:
                continue
            nw = loaded[name][k]
            cal = t5[0] / nw["t_anchor"]
            anchors[name] = nw
            labs += [nw["labels"][j] for j in new]
            tcal += [nw["t_new"][j] * cal for j in new]
            traw += [nw["t_new"][j] for j in new]
            Xs.append(nw["X"])
        tcal, traw = np.array(tcal), np.array(traw)
        ncand = len(labs)
        for o in c["obs"]:
            arr = np.concatenate([X[o] for X in Xs])         # (ncand, N)
            N = arr.shape[1]
            v = arr.var(axis=1, ddof=1)
            s = S13[(S13[CFG + ["obs"]].astype(str) == pd.Series(dict(zip(CFG + ["obs"], [str(x) for x in k + (o,)])))).all(axis=1)]
            assert len(s) == 1, (k, o, len(s))
            s = s.iloc[0]
            i5 = int(np.argmin(v[:5]))
            assert labs[i5] == s.best and np.isclose(v[i5], s.var_bf, rtol=1e-9), (k, o)
            st = c["per"][o]["single"]
            xs = st["Y"][o]
            vs = xs.var(ddof=1)
            ts = {npd: st["tau"] + (st["bp"] + st["build"]) / npd for npd in (20000, 2000)}
            C5 = v[i5] * t5[i5]
            row = dict(zip(CFG + ["obs"], k + (o,)), best5=labs[i5], var_bf5=v[i5], t_bf5=t5[i5], var_st=vs,
                       t_st=ts[20000], t_st_2000=ts[2000], W_ST_bf5=C5 / (vs * ts[20000]),
                       var_cands=";".join(f"{x:.6g}" for x in v), labels=";".join(labs))
            assert np.isclose(row["W_ST_bf5"], s.W_ST_bf, rtol=1e-9), (k, o, row["W_ST_bf5"], s.W_ST_bf)
            for m in (8, 11):
                if m > ncand:
                    continue
                im = int(np.argmin(v[:m]))
                Cm, Cmraw = v[im] * tcal[im], v[im] * traw[im]
                row.update({f"best{m}": labs[im], f"cand{m}": im, f"new_is_best{m}": im >= 5, f"var_bf{m}": v[im],
                            f"t_bf{m}": tcal[im], f"W_ST_bf{m}": Cm / (vs * ts[20000]),
                            f"W_ST_bf{m}_n2000": Cm / (vs * ts[2000]), f"W_ST_bf{m}_rawtime": Cmraw / (vs * ts[20000]),
                            f"R_ST_bf{m}": v[im] / vs, f"W_bf5_bf{m}": C5 / Cm})
                idx = rng.integers(0, N, size=(NB, N))
                stack = np.array([arr[j][idx].var(axis=1, ddof=1) for j in range(m)])
                jb = stack.argmin(axis=0)
                bC = stack.min(axis=0) * tcal[jb]
                bs = xs[rng.integers(0, len(xs), size=(NB, len(xs)))].var(axis=1, ddof=1) * ts[20000]
                row[f"W_ST_bf{m}_lo"], row[f"W_ST_bf{m}_hi"] = np.percentile(bC / bs, [2.5, 97.5])
            mbf = arr[i5].mean()
            for j in range(5, ncand):
                row[f"rv_{j}"] = v[i5] / v[j]
                row[f"z_new{j}"] = (arr[j].mean() - mbf) / np.sqrt(v[j] / N + v[i5] / N)
            for name, nw in anchors.items():
                row[f"anchor_ratio_{name}"] = nw["t_anchor"] / t5[0]
                row[f"host_{name}"] = nw["host"]
                row[f"same_host_{name}"] = nw["host"] == nw["e13_host"]
                row[f"t_new_raw_over_anchor_{name}"] = ";".join(
                    f"{nw['t_new'][j] / nw['t_anchor']:.5f}" for j in sorted(nw["t_new"]))
                for kk, vv in nw["idle"].items():
                    row[f"{kk}_{name}"] = vv
            rows.append(row)
    Q = pd.DataFrame(rows)
    Q.to_csv(os.path.join(R, f"b_rev6c_summary{out_sfx}.csv"), index=False)
    U = Q.drop_duplicates(CFG)
    print(f"groups {len(Q)}, configs {len(U)}, seeds {Q.variant.nunique()}")
    for name, _, _, new in SETS:
        if f"host_{name}" not in Q:
            continue
        print(f"[{name}] hosts {sorted(set(Q[f'host_{name}']))}; same host as E13 in {int(Q[f'same_host_{name}'].sum())}/"
              f"{len(Q)} groups; idle: gpu_other {sorted(set(Q[f'gpu_other_{name}']))}, other_py "
              f"{sorted(set(Q[f'other_py_{name}']))}, gpu_util max {Q[f'gpu_util_max_{name}'].max()} %, load1 max "
              f"{Q[f'load1_max_{name}'].max()}")
        ar = U[f"anchor_ratio_{name}"]
        print(f"[{name}] timing anchor t_anchor / t_jump,E13: median {ar.median():.4f}, range [{ar.min():.4f}, {ar.max():.4f}]")
        tn = np.array([[float(x) for x in s.split(";")] for s in U[f"t_new_raw_over_anchor_{name}"]])
        print(f"[{name}] raw t_new / t_anchor per candidate {new} (median over configs): {np.round(np.median(tn, axis=0), 4)}")
    zc = [c for c in Q.columns if c.startswith("z_new")]
    za = Q[zc].abs().to_numpy()
    print(f"bias |z| new candidates vs E13 best-fixed mean: max {za.max():.2f}, n(>4) {int((za > 4).sum())} of {za.size}")
    print("variance ratio Var_bf5 / Var_c (median over groups, > 1 = candidate c has lower variance than the E13 bf):")
    labs = Q.labels.iloc[0].split(";")
    for j in range(5, len(labs)):
        print(f"   cand {j} {labs[j]:16s} median {Q[f'rv_{j}'].median():.3f}  (> 1 in {int((Q[f'rv_{j}'] > 1).sum())}/{len(Q)})")
    for m in (8, 11):
        if f"best{m}" not in Q:
            continue
        print(f"best fixed over {m} candidates: {Q[f'best{m}'].value_counts().to_dict()}; new candidate best in "
              f"{int(Q[f'new_is_best{m}'].sum())}/{len(Q)} groups (seeds with >= 1 such group "
              f"{Q[Q[f'new_is_best{m}']].variant.nunique()}/{Q.variant.nunique()})")
    meds = []
    summarize(Q, ["W_ST_bf5", "W_ST_bf8", "W_ST_bf8_n2000", "W_ST_bf8_rawtime", "R_ST_bf8", "W_bf5_bf8"],
              "Q1-rev6 (pre-registered, 8 candidates)", meds)
    if "W_ST_bf11" in Q:
        summarize(Q, ["W_ST_bf11", "W_ST_bf11_n2000", "W_ST_bf11_rawtime", "R_ST_bf11", "W_bf5_bf11"],
                  "Q1-rev6b (amended after partial data, 11 candidates)", meds)
    pd.DataFrame(meds).to_csv(os.path.join(R, f"b_rev6c_medians{out_sfx}.csv"), index=False)
    print("\nper-seed medians (instance level):")
    print(Q.groupby("variant")[[c for c in ("W_ST_bf5", "W_ST_bf8", "W_ST_bf11") if c in Q]].median().round(3).T.to_string())
    print("\n######## DECISIONS")
    decision(Q, "W_ST_bf8", "Q1-rev6 (pre-registered)", "phi = pi/4, theta = pi/16, pi/8, 3pi/16")
    if "W_ST_bf11" in Q:
        decision(Q, "W_ST_bf11", "Q1-rev6b (amended after partial data)",
                 "phi = pi/4, six angles incl. tan(theta) = 0.25, 0.5, 1 x sqrt(g)")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
