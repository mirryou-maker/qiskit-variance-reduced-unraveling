"""Numbers for the PRA revision after the independent review (paper/pra/review_independent_2026-11-12.md).

Reads only existing files in bench/results (CSV and E1 raw arrays); never writes them. Outputs:
  bench/results/pra_revision.txt          (stdout, saved by the caller)
  bench/results/pra_e1_cluster_ci.csv     (E1 medians with the degenerate-group exclusion, for Fig. 2a)
  bench/results/pra_w_subsets.csv         (E9b/E9e median W on the cap-1e6 / cap-1e7 TN subsets, POST HOC)

Sections
  C1  |PP_1e7 - PP_1e8| / eps over the groups still TN at cap 1e7
  C3  E1 with the E2 exclusion rule (Var_bf = min(Var_std, Var_proj) < 1e-12)
  M1  per-group Var_greedy / Var_PI at n = 6 from the E1 raw arrays (paired bootstrap, B = 2000)
  C4  TN status of the 98 E9b groups at caps 1e5 / 1e6 / 1e7 and W on the subsets (POST HOC)
  M5  E9e identity-check sets per host; fallback variance vs E9b variance for the 48 re-run greedy runs

Usage (ubuntu94, qsim13): python bench/scripts/pra_revision.py > bench/results/pra_revision.txt
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
EPS = 1e-12
K5 = ["family", "variant", "n", "L", "gt"]
K6 = ["channel"] + K5


def rd(name):
    return pd.read_csv(os.path.join(R, name))


def cluster_median(d, col, keys, lo, hi, rng, B=4000):
    x = d[col].to_numpy(float)
    ids = d[keys].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(cb, [lo, hi]), len(x), len(u)


def c1():
    d = rd("b_e9d_summary.csv")
    d = d[d.n.isin([20, 24]) & d.pp1e8.notna()].copy()
    d["ratio"] = (d.pp10000000 - d.pp1e8).abs() / d.eps
    d["z8"] = (d.pp1e8 - d.traj_mean) / np.sqrt(d.var_std / d.nref)
    tn = d[d.TN10000000.astype(float) == 1]
    sv = d[d.TN10000000.astype(float) == 0]
    print("== C1: |PP_1e7 - PP_1e8| / eps (b_e9d_summary.csv)")
    for lab, g in (("all groups run at 1e8", d), ("still TN at cap 1e7", tn), ("solved at cap 1e7", sv)):
        q = np.percentile(g.ratio, [25, 50, 75])
        print(f"C1 {lab}: groups {len(g)}  median {q[1]:.2f}  IQR [{q[0]:.2f}, {q[2]:.2f}]  min {g.ratio.min():.2f}  "
              f"max|z_1e8 vs traj| {g.z8.abs().max():.2f}  n(|z|>=4) {(g.z8.abs() >= 4).sum()}")


def c3_m1():
    s = rd("b_ncurve_summary.csv")
    s["var_bf"] = np.minimum(s.var_std, s.var_proj)
    deg = s.var_bf < EPS
    print("\n== C3: E1 with the E2 exclusion rule (Var_bf < 1e-12)")
    print(f"C3 groups {len(s)}; excluded {int(deg.sum())}; of these Var_std < 1e-12: {int((s.var_std[deg] < EPS).sum())}, "
          f"Var_greedy < 1e-12: {int((s.var_greedy[deg] < EPS).sum())}; by (family, variant, obs): "
          f"{s[deg].groupby(['family', 'variant', 'obs']).size().to_dict()}")
    print(f"C3 P3-style rule Var_std < 1e-10 selects the same groups: {bool(((s.var_std < 1e-10) == deg).all())}")
    d = s[~deg]
    print(f"C3 max|z| over kept groups: {d.maxabsz.max():.2f}")
    pr = d[d["gt"] <= 0.1]
    rng = np.random.default_rng(20261112)
    rows = []
    for n, g in pr.groupby("n"):
        pt, lo, hi, ng, nc = cluster_median(g, "G", K5, 2.5, 97.5, rng)
        rows.append(dict(quantity=f"G n={n}", n=n, groups=ng, configs=nc, point=pt, cluster_lo=lo, cluster_hi=hi))
        print(f"C3 primary n={n}: groups {ng} configs {nc} median G {pt:.3f} cluster95 [{lo:.3f}, {hi:.3f}]  "
              f"greedy worse (G_hi<1) {(g.G_hi < 1).sum()}  better (G_lo>1) {(g.G_lo > 1).sum()}")
    pi = pr[(pr.n == 6) & pr.G_pi.notna()]
    pt, lo, hi, ng, nc = cluster_median(pi, "G_pi", K5, 2.5, 97.5, rng)
    rows.append(dict(quantity="G_pi n=6", n=6, groups=ng, configs=nc, point=pt, cluster_lo=lo, cluster_hi=hi))
    print(f"C3 PI n=6: groups {ng} median G_pi {pt:.3f} cluster95 [{lo:.3f}, {hi:.3f}]")
    print(f"C3 proj significantly worse than std (R_proj_hi<1): {(pr.R_proj_hi < 1).sum()} / {len(pr)}; "
          f"better (R_proj_lo>1): {(pr.R_proj_lo > 1).sum()}")
    sec = d[d["gt"] == 0.3]
    print(f"C3 secondary gt=0.3 median G per n: {sec.groupby('n').G.median().round(3).to_dict()}")
    pd.DataFrame(rows).to_csv(os.path.join(R, "pra_e1_cluster_ci.csv"), index=False)
    # M1: greedy vs PI per group, n = 6, from the raw arrays (independent seeds -> independent bootstrap)
    raw = os.path.join(R, "b_ncurve_raw")
    rng = np.random.default_rng(7)
    B = 2000
    out = []
    for _, r in pi.iterrows():
        z = np.load(os.path.join(raw, f"{r.family}_{r.variant}_n{r.n}_L{r.L}_gt{r['gt']}.npz"))
        xg, xp = z[f"{r.obs}_greedy"].astype(float), z[f"{r.obs}_pi"].astype(float)
        bg = xg[rng.integers(0, len(xg), (B, len(xg)))].var(axis=1, ddof=1)
        bp = xp[rng.integers(0, len(xp), (B, len(xp)))].var(axis=1, ddof=1)
        rt = bg / np.maximum(bp, 1e-300)
        lo, hi = np.percentile(rt, [2.5, 97.5])
        out.append((xg.var(ddof=1) / xp.var(ddof=1), lo, hi))
    o = np.array(out)
    m = len(o)
    print(f"\n== M1: Var_greedy / Var_PI per group, E1 n=6 primary, {m} groups (bootstrap B={B}, 95%)")
    print(f"M1 median ratio {np.median(o[:, 0]):.3f}; PI significantly better (lo>1): {(o[:, 1] > 1).sum()}; "
          f"greedy significantly better (hi<1): {(o[:, 2] < 1).sum()}")
    alpha = 0.05 / m
    print(f"M1 (note: 95% per-group CIs, no multiplicity correction; expected false positives per side ~ {0.025 * m:.1f})")


def c4():
    e9d = rd("b_e9d_summary.csv")
    e9d = e9d[e9d.n == 20]
    keys = ["channel", "family", "variant", "n", "L", "gt", "obs"]
    st = e9d[keys + ["TN100000", "TN1000000", "TN10000000"]]
    out = []
    print("\n== C4: TN status of the 98 E9b groups (b_e9d_summary.csv) and median W on subsets (POST HOC)")
    for name, tag in (("b_e9b_summary.csv", "E9b"), ("b_e9e_summary.csv", "E9e RTX2060"),
                      ("b_e9e_summary_v100.csv", "E9e V100")):
        s = rd(name).merge(st, on=keys, how="left")
        if tag == "E9b":
            for c in ("TN100000", "TN1000000", "TN10000000"):
                print(f"C4 E9b groups TN at {c[2:]}: {int(s[c].sum())} / {len(s)}  "
                      f"by channel {s.groupby('channel')[c].sum().astype(int).to_dict()}")
        rng = np.random.default_rng(20261112)
        for sub, lab in ((s, "all98"), (s[s.TN1000000 == 1], "TN1e6"), (s[s.TN10000000 == 1], "TN1e7")):
            for m in (10, 100):
                for ch in ("ad", "deph"):
                    g = sub[(sub.channel == ch) & np.isfinite(sub[f"W{m}"])]
                    if len(g) == 0:
                        continue
                    pt, lo, hi, ng, nc = cluster_median(g, f"W{m}", K6, 0.625, 99.375, rng)
                    rr = g[f"R{m}"].median()
                    tr = g[f"tratio{m}"].median()
                    out.append(dict(source=tag, subset=lab, M=m, channel=ch, groups=ng, configs=nc, W_median=pt,
                                    W_cluster_lo=lo, W_cluster_hi=hi, R_median=rr, tratio_median=tr))
                    print(f"C4 {tag:11} {lab:5} M={m:3} {ch:4}: groups {ng:2} configs {nc:2} median W {pt:.3f} "
                          f"cluster98.75 [{lo:.3f}, {hi:.3f}]  median R {rr:.3f}  median t_g/t_f {tr:.2f}")
        if tag == "E9e RTX2060":
            a = s[s.channel == "ad"]
            ci1 = a[a.W100_lo > 1]
            print(f"C4 E9e AD M100 groups with per-group W CI > 1: {len(ci1)}; of these TN at 1e6: "
                  f"{int(ci1.TN1000000.sum())}, at 1e7: {int(ci1.TN10000000.sum())}")
    pd.DataFrame(out).to_csv(os.path.join(R, "pra_w_subsets.csv"), index=False)


def m5():
    print("\n== M5: E9e identity check (first N_t trajectories of each timing run vs E9b raw)")
    keys = ["channel", "family", "variant", "n", "L", "gt", "obs", "M"]
    sets = {}
    for name in ("b_e9e_u94.csv", "b_e9e_v100.csv"):
        d = rd(name)
        print(f"M5 {name}: shots per timing run {sorted(d.shots.unique())}; fixed rows {int((d.kind == 'fixed').sum())} "
              f"with match < 1: {int(((d.kind == 'fixed') & (d.match < 1)).sum())}")
        g = d[d.kind == "greedy"]
        bad = g[g.match < 0.99]
        sets[name] = set(map(tuple, bad[keys].astype(str).values))
        print(f"M5 {name}: greedy rows {len(g)}, match < 0.99: {len(bad)}")
    a, b = sets["b_e9e_u94.csv"], sets["b_e9e_v100.csv"]
    print(f"M5 mismatch sets: common {len(a & b)}; RTX2060 only {sorted(a - b)}; V100 only {sorted(b - a)}")
    fb = pd.concat([rd("b_e9e_fb_u94.csv"), rd("b_e9e_fb_v100.csv")], ignore_index=True)
    e9b = rd("b_e9b_summary.csv")
    rows = []
    for _, r in fb.iterrows():
        q = e9b[(e9b.channel == r.channel) & (e9b.family == r.family) & (e9b.variant == r.variant) &
                (e9b.L == r.L) & (e9b["gt"] == r["gt"]) & (e9b.obs == r.obs)]
        m = int(r.M)
        rows.append(r["var"] / float(q[f"var_g{m}"].iloc[0]))
    x = np.array(rows)
    print(f"M5 fallback runs {len(x)}: Var(new code, N=2000) / Var(E9b, N=2000): median {np.median(x):.3f}, "
          f"IQR [{np.percentile(x, 25):.3f}, {np.percentile(x, 75):.3f}], range [{x.min():.3f}, {x.max():.3f}]")
    print("M5 (both runs use the E9b seed, so they share trajectories up to the first flipped near-tie choice)")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    c1()
    c3_m1()
    c4()
    m5()
