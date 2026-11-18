"""Rev7 (review 7, minor 6 and M3), POST HOC, existing E13 outputs only (no new simulation).

(1) Heavy tails: per-trajectory excess kurtosis of the oracle best-fixed run and the pilot-table run in the 64 E13
    amplitude-damping groups (and the 30 dephasing groups), and stability of the median W_ST/bf when only the first or
    the second 1000 of the 2000 trajectories of every run are used (oracle re-chosen on the subsample; times and
    one-time costs unchanged, N_prod = 20000).
(2) Var_0 at N = 2000 against the 10^6-trajectory E13-ref reference variance (8 groups).
(3) Structure of the E13 amplitude-damping pilot tables: fraction of table entries per candidate (counting and the
    four homodyne phases), overall and per observable.
Uses analyze_e13.load (the loader of the pre-specified analysis). Run on ubuntu94 (CPU).
Usage: python pra_rev7_tails_tables.py <output .txt>
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("analyze_e13", os.path.join(HERE, "analyze_e13.py"))
A = importlib.util.module_from_spec(sp)
sys.argv_saved = sys.argv
sp.loader.exec_module(A)
out = open(sys.argv[1], "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


def exkurt(x):
    x = np.asarray(x, float)
    d = x - x.mean()
    v = np.mean(d ** 2)
    return np.mean(d ** 4) / v ** 2 - 3.0 if v > 0 else np.nan


C, dups = A.load(os.path.join(A.R, "b_e13_main"), os.path.join(A.R, "b_e13_raw"))
rows = []
tabs = []
for c in C:
    k, labels, t = c["key"], c["labels"], c["t"]
    for o in c["obs"]:
        arr = c["X"][o]                     # (ncand, 2000)
        s = c["per"][o]["single"]
        ys = s["Y"][o]
        ts = s["tau"] + (s["bp"] + s["build"]) / A.NPROD
        rec = dict(channel=k[0], variant=k[2], L=k[4], obs=o)
        for tag, sl in (("all", slice(0, 2000)), ("h1", slice(0, 1000)), ("h2", slice(1000, 2000))):
            vf = arr[:, sl].var(axis=1, ddof=1)
            ib = int(np.argmin(vf))
            rec[f"W_{tag}"] = vf[ib] * t[ib] / (ys[sl].var(ddof=1) * ts)
            if tag == "all":
                rec["k_bf"] = exkurt(arr[ib])
                rec["k_st"] = exkurt(ys)
                rec["k_0"] = exkurt(arr[0])
                rec["var0"] = vf[0]
        rows.append(rec)
        if k[0] == "ad":
            tab = s["tab"]
            for cnd in range(5):
                tabs.append(dict(obs=o, cand=labels[cnd], frac=float(np.mean(tab == cnd)), variant=k[2], L=k[4]))
d = pd.DataFrame(rows)
log("== (1) heavy tails and subsampling (E13 groups TN at 1e7; POST HOC) ==")
for ch in ("ad", "deph"):
    x = d[d.channel == ch]
    log(f"{ch}: groups {len(x)}")
    for col, name in (("k_0", "candidate 0"), ("k_bf", "oracle best fixed"), ("k_st", "pilot table")):
        q = x[col].quantile([0.5, 0.9]).values
        log(f"  excess kurtosis {name}: median {q[0]:.2f}, 90th pct {q[1]:.2f}, max {x[col].max():.2f}")
    log(f"  median W_ST/bf: N=2000 {x.W_all.median():.3f}; first 1000 {x.W_h1.median():.3f}; second 1000 {x.W_h2.median():.3f}")
    log(f"  groups with W>1: N=2000 {(x.W_all > 1).sum()}; first 1000 {(x.W_h1 > 1).sum()}; second 1000 {(x.W_h2 > 1).sum()}")
    r = np.log(x.W_h1 / x.W_h2)
    log(f"  per-group |log(W_h1/W_h2)|: median {np.median(np.abs(r)):.3f}, 90th pct {np.quantile(np.abs(r), 0.9):.3f}")

log("== (2) Var_0 at N = 2000 vs the 10^6-trajectory reference variance (E13-ref groups) ==")
ref = pd.read_csv(os.path.join(A.R, "b_rev6d_summary.csv"))
for _, r in ref.iterrows():
    m = d[(d.channel == r.channel) & (d.variant == r.variant) & (d.L == r.L) & (d.obs == r.obs)]
    if len(m):
        log(f"  {r.channel} {r.variant} L{r.L} {r.obs}: Var0(2000)/Var_ref(1e6) = {m.var0.iloc[0] / r.var_ref:.3f}")
    else:
        log(f"  {r.channel} {r.variant} L{r.L} {r.obs}: not an E13 group")

log("== (3) E13 amplitude-damping pilot tables: median fraction of table entries per candidate ==")
tb = pd.DataFrame(tabs)
piv = tb.groupby(["obs", "cand"]).frac.median().unstack()
log(piv.round(4).to_string())
log("all observables:")
log(tb.groupby("cand").frac.median().round(4).to_string())
log("mean over groups:")
log(tb.groupby("cand").frac.mean().round(4).to_string())
out.close()
