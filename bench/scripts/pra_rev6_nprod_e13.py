"""POST HOC (referee review 6, major 4): sensitivity of the E13 wall-clock ratios to the production length N_prod over
which the one-time costs (O~ recording bp_s, pilot table build build_s) are amortized.

E13 (analyze_e13.py) charges t_ST = tau_ST + (bp + build)/N_prod and t_CV = tau_CV + bp/N_prod with N_prod = 20000, while
the production runs actually had N = 2000 trajectories. Here every per-group W of the E13 summary is recomputed for
N_prod in {2000, 5000, 10000, 20000, 1e5} from bench/results/b_e13_summary.csv (no new trajectories):
    tau_x = t_x - (one-time cost)/20000   (exact inverse of analyze_e13.py)
    t_x(N_prod) = tau_x + (one-time cost)/N_prod
Variances (oracle best fixed, ST, cross-fitted CV) are unchanged. Intervals, as in the manuscript:
    group bootstrap  : 98.75% percentile, B = 2000, default_rng(7) per (channel, metric, N_prod) (decision-line draw)
    config cluster   : 98.75%, B = 2000, default_rng(11)
    instance (seed)  : 98.75%, B = 2000, default_rng(13) (circuit-instance cluster; the manuscript's "instance-level")
Check: at N_prod = 20000 the recomputed W equal the E13 summary columns (max relative difference printed).

    python pra_rev6_nprod_e13.py   -> bench/results/pra_rev6_nprod_e13.{txt,csv}
CPU only.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
NPROD_E13 = 20000
NPRODS = [2000, 5000, 10000, 20000, 100000]
LO, HI = 0.625, 99.375
CFG = ["channel", "family", "variant", "n", "L", "gt"]
METRICS = [("W_ST_bf", "Q1 static table vs oracle best fixed"),
           ("W_CV0_bf", "Q2a CV on default unraveling vs best fixed"),
           ("W_STCV_CV0", "Q2b ST+CV vs CV0"),
           ("W_CVbf_bf", "CV on best fixed vs best fixed"),
           ("W_STCV_bf", "ST+CV vs best fixed"),
           ("W_ST_CV0", "ST vs CV0")]


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


def recompute(Q, nprod):
    bp, build = Q.st_bp_s, Q.st_build_s
    tau = dict(st=Q.t_st - (bp + build) / NPROD_E13, cv0=Q.t_cv0 - bp / NPROD_E13, cvbf=Q.t_cvbf - bp / NPROD_E13,
               stcv=Q.t_stcv - (bp + build) / NPROD_E13)
    t = dict(st=tau["st"] + (bp + build) / nprod, cv0=tau["cv0"] + bp / nprod, cvbf=tau["cvbf"] + bp / nprod,
             stcv=tau["stcv"] + (bp + build) / nprod)
    Cbf = Q.var_bf * Q.t_bf
    Cst = Q.var_st * t["st"]
    C0 = Q.var_cv0_crossfit * t["cv0"]
    Cb = Q.var_cvbf_crossfit * t["cvbf"]
    Cs = Q.var_stcv_crossfit * t["stcv"]
    return pd.DataFrame(dict(W_ST_bf=Cbf / Cst, W_CV0_bf=Cbf / C0, W_STCV_CV0=C0 / Cs, W_CVbf_bf=Cbf / Cb,
                             W_STCV_bf=Cbf / Cs, W_ST_CV0=C0 / Cst, tST_tbf=t["st"] / Q.t_bf,
                             onetime_frac=(bp + build) / (2000 * Q.t_bf)))


def main():
    Q = pd.read_csv(os.path.join(R, "b_e13_summary.csv"))
    out = []
    W20 = recompute(Q, NPROD_E13)
    dev = max(float(np.max(np.abs(W20[m] / Q[m] - 1))) for m, _ in METRICS)
    lines = ["POST HOC (review 6 M4): E13 wall-clock ratios vs the amortization length N_prod of the one-time costs",
             f"source bench/results/b_e13_summary.csv ({len(Q)} groups); check at N_prod = 20000: max |W/W_E13 - 1| = "
             f"{dev:.2e} over {len(METRICS)} metrics",
             "intervals 98.75%: [group bootstrap, default_rng(7)] {config cluster} <instance (seed) cluster>",
             "one-time cost (bp + build) as a fraction of 2000 t_bf, median: " + "; ".join(
                 f"{ch} {D.st_bp_s.add(D.st_build_s).div(2000 * D.t_bf).median():.3f}" for ch, D in Q.groupby("channel")),
             ""]
    for nprod in NPRODS:
        W = recompute(Q, nprod)
        W[CFG + ["obs"]] = Q[CFG + ["obs"]]
        for ch, D in W.groupby("channel"):
            cid = D[CFG].astype(str).agg("|".join, axis=1)
            sid = D.variant.astype(str)
            for m, desc in METRICS:
                md, lo, hi = med_ci(D[m], np.random.default_rng(7))
                clo, chi = cluster_ci(D[m], cid, np.random.default_rng(11))
                slo, shi = cluster_ci(D[m], sid, np.random.default_rng(13))
                sm = D.groupby("variant")[m].median()
                out.append(dict(N_prod=nprod, channel=ch, metric=m, description=desc, groups=len(D),
                                configs=cid.nunique(), seeds=sid.nunique(), median=md, lo=lo, hi=hi, clo=clo, chi=chi,
                                slo=slo, shi=shi, frac_gt1=float((D[m] > 1).mean()), seeds_gt1=int((sm > 1).sum()),
                                tST_tbf_median=float(D.tST_tbf.median()),
                                pass_rule=bool(md > 1 and lo > 1)))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(R, "pra_rev6_nprod_e13.csv"), index=False)
    for ch in ("ad", "deph"):
        for m, desc in METRICS:
            lines.append(f"[{ch}] {m}  ({desc})")
            for _, r in O[(O.channel == ch) & (O.metric == m)].iterrows():
                lines.append(f"   N_prod {r.N_prod:>6d}: {r['median']:.3f} [{r.lo:.3f}, {r.hi:.3f}] {{{r.clo:.3f}, {r.chi:.3f}}} "
                             f"<{r.slo:.3f}, {r.shi:.3f}>  frac>1 {r.frac_gt1:.3f}  seeds>1 {r.seeds_gt1}/{r.seeds}  "
                             f"median t_ST/t_bf {r.tST_tbf_median:.3f}  (median>1 & lower>1: {r.pass_rule})")
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(R, "pra_rev6_nprod_e13.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
