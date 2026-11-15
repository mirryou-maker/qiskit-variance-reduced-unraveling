"""POST HOC (review 3, 2026-11-14): static-table W versus the production length N_prod over which the one-time costs
(PP recording bp_s and pilot/table construction build_s) are amortized.

The pre-specified convention is N_prod = 20000 (as for greedy in E9e). Here also N_prod = 2000 (the production length
actually run) and 1e5. No re-simulation: per-trajectory production time and one-time costs are in the summary CSVs.
  E10 (b_e10_summary.csv; table pilot and ideal): t_run = <t>_t - (<t>_bp_s + <t>_build_s)/20000 (analyze_e10 t_static);
      t_s(N) = t_run + (bp + build)/N; W_bf(N) = var_bf t_f / (<t>_var t_s(N)).
  E11 Q1 (b_e11_summary.csv): t_s(20000) = single_tratio * t_f; same construction.
  E11 Q2 (b_e11_multi.csv, edef A20k, subset all): eps_o^2 = Var_bf(o)/N, so F1(N) = F1 N/20000 and
      S_multi(N) = (S_multi - bp_s - build_s) N/20000 + bp_s + build_s; W_multi(N) = F1(N)/S_multi(N).
Medians: 98.75% percentile bootstrap over groups (B = 2000, default_rng(20261114)) or configurations
(Q2: B = 4000, default_rng(20261113)). Checks: N = 20000 reproduces the stored W values.
Usage (ubuntu94, env sci): python bench/scripts/pra_static_nprod.py > bench/results/pra_static_nprod.txt
Outputs: pra_static_nprod.txt, pra_static_nprod.csv.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench", "results")
LO, HI = 0.625, 99.375
NREF = 20000
NS = (2000, 20000, 100000)
rng = np.random.default_rng(20261114)


def med_ci(x, r, B):
    x = np.asarray(x, float)
    b = [np.median(x[r.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [LO, HI])


rows = []
e10 = pd.read_csv(os.path.join(R, "b_e10_summary.csv"))
e11 = pd.read_csv(os.path.join(R, "b_e11_summary.csv"))
mu = pd.read_csv(os.path.join(R, "b_e11_multi.csv"))
mu = mu[(mu.edef == "A20k") & (mu.subset == "all")]
print("POST HOC: median static-table W_bf vs N_prod (one-time costs amortized over N_prod); 98.75% bootstrap")
for t in ("pilot", "ideal"):
    oh = e10[f"{t}_bp_s"] + e10[f"{t}_build_s"]
    trun = e10[f"{t}_t"] - oh / NREF
    for N in NS:
        W = e10.var_bf * e10.t_f / (e10[f"{t}_var"] * (trun + oh / N))
        if N == NREF:
            assert np.allclose(W, e10[f"{t}_W_bf"], rtol=1e-9)
        for ch in ("ad", "deph"):
            m = e10.channel == ch
            md, lo, hi = med_ci(W[m], rng, 2000)
            tr = float(np.median((trun + oh / N)[m] / e10.t_f[m]))
            rows.append(dict(exp="E10", metric=f"{t}_W_bf", channel=ch, Nprod=N, units=int(m.sum()), median=md, lo=lo,
                             hi=hi, tratio=tr))
            print(f"  E10   {t:5s} {ch:4s} N_prod {N:6d}: groups {int(m.sum()):2d} median W_bf {md:.3f} [{lo:.3f}, {hi:.3f}]"
                  f"  median t_s/t_f {tr:.3f}")
oh = e11.single_bp_s + e11.single_build_s
trun = e11.single_tratio * e11.t_f - oh / NREF
for N in NS:
    W = e11.var_bf * e11.t_f / (e11.single_var * (trun + oh / N))
    if N == NREF:
        assert np.allclose(W, e11.single_W_bf, rtol=1e-9)
    for ch in ("ad", "deph"):
        m = e11.channel == ch
        md, lo, hi = med_ci(W[m], rng, 2000)
        tr = float(np.median((trun + oh / N)[m] / e11.t_f[m]))
        rows.append(dict(exp="E11Q1", metric="single_W_bf", channel=ch, Nprod=N, units=int(m.sum()), median=md, lo=lo,
                         hi=hi, tratio=tr))
        print(f"  E11Q1 pilot {ch:4s} N_prod {N:6d}: groups {int(m.sum()):2d} median W_bf {md:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  median t_s/t_f {tr:.3f}")
oh = mu.bp_s + mu.build_s
for N in NS:
    W = (mu.F1 * N / NREF) / ((mu.Smulti - oh) * N / NREF + oh)
    if N == NREF:
        assert np.allclose(W, mu.W_multi, rtol=1e-9)
    for ch in ("ad", "deph"):
        m = mu.channel == ch
        md, lo, hi = med_ci(W[m], np.random.default_rng(20261113), 4000)
        rows.append(dict(exp="E11Q2", metric="W_multi", channel=ch, Nprod=N, units=int(m.sum()), median=md, lo=lo,
                         hi=hi, tratio=np.nan))
        print(f"  E11Q2 multi {ch:4s} N_prod {N:6d}: configs {int(m.sum()):2d} median W_multi {md:.3f} [{lo:.3f}, {hi:.3f}]")
pd.DataFrame(rows).to_csv(os.path.join(R, "pra_static_nprod.csv"), index=False)
