"""Numbers for the PRA manuscript (paper/pra/main.tex) that no existing printout contains.

Reads ONLY existing result CSVs in bench/results (never writes them). Prints one labelled line per number;
the output is saved as bench/results/pra_numbers.txt and cited in paper/pra/number_sources.md.

Sections
  A5   entangled-circuit std vs projector map (a5_summary_stage2.csv, Aer on iREMB)
  P3   PI / greedy vs best-of(std, proj) at n = 5, 6 (p3_adapt.csv)
  E1   exact-O~ greedy vs best fixed, dephasing (b_ncurve_summary.csv)
  E2   depolarizing / amplitude damping (b_e2_summary.csv)
  E3   TJM unravelings, myopic criterion (b_e3_summary.csv)
  E9b  per-host t_g/t_f (b_e9b_summary.csv)
  E9e  config-cluster bootstrap CIs of median W (b_e9e_summary.csv, b_e9e_summary_v100.csv)  [POST HOC]

Usage: python bench/scripts/pra_numbers.py > bench/results/pra_numbers.txt
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")


def rd(name):
    return pd.read_csv(os.path.join(R, name))


def nondet(d):
    return d[~d.deterministic.astype(str).eq("True")]


def a5():
    d = rd("a5_summary_stage2.csv")
    print("== A5 (a5_summary_stage2.csv): var_ratio = Var_std / Var_proj, time_ratio = T_std / T_proj (time to target SE)")
    print(f"A5 conditions: {len(d)}  families {sorted(d.family.unique())}  obs {sorted(d.obs.unique())}  "
          f"n {sorted(d.n.unique())}  gt {sorted(d['gt'].unique())}  layers {sorted(d.layers.unique())}")
    print(f"A5 proj variance larger (var_ratio < 1): {(d.var_ratio < 1).sum()}")
    print(f"A5 proj variance larger, CI excludes 1 (var_ratio_hi < 1): {(d.var_ratio_hi < 1).sum()}")
    print(f"A5 proj variance smaller (var_ratio > 1): {(d.var_ratio > 1).sum()}; CI excludes 1: {(d.var_ratio_lo > 1).sum()}")
    sm = d[d.var_ratio > 1]
    print(f"A5 conditions with var_ratio > 1 by (family, obs): {sm.groupby(['family', 'obs']).size().to_dict()}")
    print(f"A5 time_ratio < 1 (std faster to target SE in Aer): {(d.time_ratio < 1).sum()} / {len(d)}; "
          f"CI hi < 1: {(d.time_ratio_hi < 1).sum()}; max time_ratio {d.time_ratio.max():.3f}")
    print(f"A5 var_ratio range: {d.var_ratio.min():.3f} .. {d.var_ratio.max():.3f}")
    print(f"A5 std best config counts: {d.groupby(['std_device', 'std_config']).size().to_dict()}")
    print(f"A5 proj best config counts: {d.groupby(['proj_device', 'proj_config']).size().to_dict()}")


def p3():
    d = rd("p3_adapt.csv")
    vmin = 1e-10
    piv = d.pivot_table(index=["family", "variant", "n", "L", "gt", "obs"], columns="scheme",
                        values=["var", "ratio", "ratio_lo", "ratio_hi"], aggfunc="first")
    tot = len(piv)
    keep = piv[("var", "std")] >= vmin
    p = piv[keep]
    print("\n== P3 (p3_adapt.csv): ratio = Var_std / Var_scheme")
    print(f"P3 groups {tot}, degenerate (Var_std < 1e-10) {int((~keep).sum())}, non-degenerate {len(p)}; "
          f"n {sorted(d.n.unique())}, gt {sorted(d['gt'].unique())}")
    for s in ("proj", "greedy", "pi"):
        r = p[("ratio", s)]
        ok = np.isfinite(r) & (r > 0)
        print(f"P3 {s}: groups {int(ok.sum())} better(CI>1) {int((p[('ratio_lo', s)][ok] > 1).sum())} "
              f"worse(CI<1) {int((p[('ratio_hi', s)][ok] < 1).sum())} median ratio {np.median(r[ok]):.3f}")
    best = np.maximum(1.0, p[("ratio", "proj")])
    for s in ("pi", "greedy"):
        rel = p[("ratio", s)] / best
        ok = np.isfinite(rel)
        print(f"P3 {s} / best-of(std, proj): median {np.median(rel[ok]):.3f} over {int(ok.sum())} groups; "
              f"frac < 0.95 {np.mean(rel[ok] < 0.95):.3f}")
    rel = p[("ratio", "pi")] / p[("ratio", "greedy")]
    print(f"P3 PI / greedy variance gain ratio: median {np.median(rel[np.isfinite(rel)]):.3f}")


def e1():
    d = nondet(rd("b_ncurve_summary.csv"))
    print("\n== E1 (b_ncurve_summary.csv), dephasing, candidates {std, proj}")
    print(f"E1 groups total {len(rd('b_ncurve_summary.csv'))}, non-deterministic {len(d)}; max|z| {d.maxabsz.max():.2f}")
    pr = d[d["gt"] <= 0.1]
    for n, g in pr.groupby("n"):
        print(f"E1 primary n={n}: groups {len(g)} median G {g.G.median():.3f}  greedy worse (G_hi<1) {(g.G_hi < 1).sum()}  "
              f"better (G_lo>1) {(g.G_lo > 1).sum()}")
    sec = d[d["gt"] == 0.3]
    print(f"E1 secondary gt=0.3 median G per n: {sec.groupby('n').G.median().round(3).to_dict()}")
    pi = pr[pr.G_pi.notna()]
    print(f"E1 PI n=6 median G_pi {pi.G_pi.median():.3f} over {len(pi)} groups")
    print(f"E1 primary proj vs std (R_proj = Var_std/Var_proj): R_proj_hi < 1 in {(pr.R_proj_hi < 1).sum()} / {len(pr)}, "
          f"R_proj_lo > 1 in {(pr.R_proj_lo > 1).sum()}")


def e2():
    d = nondet(rd("b_e2_summary.csv"))
    a = rd("b_e2_summary.csv")
    print("\n== E2 (b_e2_summary.csv)")
    print(f"E2 groups {len(a)}, deterministic {len(a) - len(d)}; max|z| {d.maxabsz.max():.2f}")
    pr = d[d["gt"] <= 0.1]
    for (ch, n), g in pr.groupby(["channel", "n"]):
        print(f"E2 {ch} n={n}: groups {len(g)} median G {g.G.median():.3f}  median G_vs_std {g.G_vs_std.median():.3f}  "
              f"greedy worse {(g.G_hi < 1).sum()}")


def e3():
    d = nondet(rd("b_e3_summary.csv"))
    a = rd("b_e3_summary.csv")
    print("\n== E3 (b_e3_summary.csv)")
    print(f"E3 groups {len(a)}, deterministic {len(a) - len(d)}; max|z| {d.maxabsz.max():.2f}")
    pr = d[d["gt"] <= 0.1]
    print(f"E3 primary groups (n 6, 8, 10): {len(pr)}")
    print(f"E3 proj worse than std (R_proj_hi < 1): {(pr.R_proj_hi < 1).sum()}; better (R_proj_lo > 1): {(pr.R_proj_lo > 1).sum()}; "
          f"point R_proj < 1: {(pr.R_proj < 1).sum()}, > 1: {(pr.R_proj > 1).sum()}")
    print(f"E3 best fixed counts (primary): {pr.best_fixed.value_counts().to_dict()}")
    for n, g in pr.groupby("n"):
        print(f"E3 n={n}: median G {g.G.median():.3f} M {g.M.median():.3f} T {g['T'].median():.3f} A {g.A.median():.3f}")


def e9b_hosts():
    d = rd("b_e9b_summary.csv")
    print("\n== E9b per host (b_e9b_summary.csv): t_g/t_f medians (POST HOC; hosts ran different groups)")
    for (h, ch), g in d.groupby(["host", "channel"]):
        print(f"E9b {h} {ch}: groups {len(g)} t_f median {g.t_f.median():.4f} s  tratio10 median {g.tratio10.median():.2f}  "
              f"tratio100 median {g.tratio100.median():.2f}")


def boot_cluster(d, col, keys, lo, hi, rng, B=4000):
    x = d[col].to_numpy(float)
    ids = d[keys].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    members = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([members[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return np.median(x), *np.percentile(cb, [lo, hi]), len(x), len(u)


def e9e_cluster():
    print("\n== E9e config-cluster bootstrap of median W (POST HOC robustness check; pre-registered CI is group-level)")
    keys = ["channel", "family", "variant", "n", "L", "gt"]
    rows = []
    for name in ("b_e9e_summary.csv", "b_e9e_summary_v100.csv"):
        d = rd(name)
        rng = np.random.default_rng(20261112)
        for m in (10, 100):
            for ch in ("ad", "deph"):
                g = d[d.channel == ch]
                g = g[np.isfinite(g[f"W{m}"])]
                pt, lo, hi, ng, nc = boot_cluster(g, f"W{m}", keys, 0.625, 99.375, rng)
                tr = g[f"tratio{m}"].median()
                print(f"E9e {name} M={m} {ch}: groups {ng} configs {nc} median W {pt:.3f} cluster 98.75% [{lo:.3f}, {hi:.3f}]  "
                      f"median t_g/t_f {tr:.2f}")
                rows.append(dict(source=name, M=m, channel=ch, groups=ng, configs=nc, W_median=pt,
                                 W_cluster_lo=lo, W_cluster_hi=hi, level="98.75%", tratio_median=tr))
    out = os.path.join(R, "pra_e9e_cluster_ci.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")
    d = rd("b_e9e_summary.csv")
    print(f"E9e u94 t_f median {d.t_f.median():.4f} s; t_f / t_f(E9b) median {(d.t_f / d.t_f_e9b).median():.3f}")


def extra():
    print("\n== extra checks of statements in the plan document (computed after the runs)")
    e3 = nondet(rd("b_e3_summary.csv"))
    e3 = e3[e3["gt"] <= 0.1]
    r = np.minimum(e3["var_fix:std"], e3["var_fix:proj"]) / e3["var_myopic"]
    print(f"E3 myopic vs min(std, proj): median {r.median():.3f}; per n {r.groupby(e3.n).median().round(3).to_dict()}")
    e8 = nondet(rd("b_e8_summary.csv"))
    e8 = e8[e8.M.notna()]
    ex = e8[e8.R_lo > e8.G_exact]
    print(f"E8 rows with R_M CI lower bound > G_exact (same group): {len(ex)} of {len(e8)}; by (channel, M) "
          f"{ex.groupby(['channel', 'M']).size().to_dict()}")
    key = ["family", "variant", "n", "L", "gt", "obs"]
    e2 = nondet(rd("b_e2_summary.csv"))
    e2 = e2[(e2.channel == "ad") & (e2.n == 10) & (e2["gt"] <= 0.1)].copy()
    e7 = nondet(rd("b_e7_summary.csv"))
    e7 = e7[(e7.channel == "ad_phase") & (e7["gt"] <= 0.1)].copy()
    e2["bf"] = e2[["var_fix:jump", "var_fix:homx", "var_fix:homy"]].min(axis=1)
    e7["bf"] = e7[[c for c in e7.columns if c.startswith("var_fix:j") or c.startswith("var_fix:hom")]].min(axis=1)
    m = e2.merge(e7, on=key, suffixes=("_2", "_7"))
    print(f"E2/E7 AD matched groups {len(m)}: median bf var E7/E2 {(m.bf_7 / m.bf_2).median():.3f}; "
          f"median greedy var E2/E7 {(m.var_greedy_2 / m.var_greedy_7).median():.3f}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    extra()
    a5()
    p3()
    e1()
    e2()
    e3()
    e9b_hosts()
    e9e_cluster()
