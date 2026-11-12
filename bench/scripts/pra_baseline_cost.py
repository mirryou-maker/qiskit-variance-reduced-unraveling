"""POST HOC cost-inclusive baselines for E9e (adaptive greedy unraveling vs fixed unravelings).

Written 2026-11-12 in response to the independent referee remark that the E9b/E9e baseline "best fixed" is an oracle
(chosen after the runs, at zero cost). This analysis is POST HOC (not pre-registered). The specification below was
written before any output of this script was looked at; it is not to be tuned afterwards. Any later change is recorded
in the CHANGE LOG at the end of this docstring, with the reason.

Inputs (read only; nothing in bench/results is modified)
  b_e9b_u9*.csv, b_e9b_summary.csv, b_e9b_raw/*.npz    E9b variances / raw per-trajectory values (N = 2000 per scheme)
  b_e9e_u94.csv (primary, RTX 2060) / b_e9e_v100.csv   E9e timing runs (optimized implementation)
  b_e9e_fb_*.csv, b_e9e_fb_raw/*.npz                     E9e greedy fallback runs (N = 2000) for match < 0.99
  b_e9e_summary.csv / b_e9e_summary_v100.csv             only to check the reproduction of the paper's W
  b_e9d_summary.csv                                      TN status at the 1e7 Pauli-path cap (subset definition)
Outputs: bench/results/pra_baseline_cost.txt (stdout, saved by the caller), bench/results/pra_baseline_cost.csv (summary)
  and bench/results/pra_baseline_cost_pergroup.csv (per-group W, RTX 2060).

Per-group quantities (group = channel, family, variant, n, L, gt, obs; 98 groups: AD 62, dephasing 36)
  Candidates c (AD: jump, hom0.000, hom0.785, hom1.571, hom2.356; dephasing: std, proj, analog0.05/0.1/0.2/0.3/0.785).
  x_c   raw per-trajectory values (E9b, N = 2000), V_c = var(x_c, ddof=1) from the raw arrays (float64).
  t_c   per-trajectory time of candidate c = E9e fixed wall / shots (same host; timing noise ignored, times known).
  Greedy (M = 10, 100): x_g = E9b greedy raw array, or the E9e fallback array when the E9e greedy run matched the E9b
        trajectories in < 99 % (same rule as analyze_e9e.py); V_g = var(x_g, ddof=1);
        tau_g = E9e greedy wall / shots (per trajectory, without back-propagation); bp = bp_s (one-time cost per
        circuit, observable and M: record_after + group precompute, measured once per run in b_e9e.py).
  Oracle best fixed bf = argmin_c V_c (as the paper). Target error eps^2 = V_bf / N_prod, N_prod in {20000, 100000}
        (PRIMARY: N_prod = 20000, the production size the paper amortizes bp over).
  Greedy cost at eps^2 (honest bp convention): C_g = n_g * tau_g + bp, n_g = V_g / eps^2 (bp paid once).
  Greedy cost, paper bp convention:            C_g' = n_g * (tau_g + bp / 20000)   (bp amortized per trajectory
        as if 20000 trajectories were run, regardless of n_g). Reported for comparability only.
  W = C_fixed / C_g (> 1: greedy cheaper at equal error).

Models
  M1   oracle (paper): W = V_bf t_bf / (V_g (tau_g + bp/20000)). Variances here from the E9b CSV exactly as
       analyze_e9e.py, so the medians must reproduce b_e9e_summary.csv (AD M10 0.670, AD M100 1.010, deph M10 0.584,
       deph M100 0.426). Also recomputed with raw-array variances (difference reported).
  M1b  oracle by variance, honest bp: C_fixed = N_prod t_bf.
  M1c  oracle by cost, honest bp: bf_c = argmin_c V_c t_c, C_fixed = V_bfc / eps^2 * t_bfc.
  M2   default unraveling (candidate 0 in src/noise/traj_kraus.py, "the Aer default": AD = jump, dephasing = std),
       honest bp: C_fixed = V_def / eps^2 * t_def.
  M3   pilot-then-commit, honest bp. Np in {100, 300, 1000, 2000}. Selection criterion s in {var, cost (= var * t)}.
       One replicate r: for every candidate an independent random permutation of its 2000 trajectories; the first Np
       are the pilot, the remaining 2000 - Np the hold-out. c* = argmin over pilot var (s = var) or pilot var * t_c
       (s = cost). The true variance of c* is estimated by the hold-out variance (out of sample, unbiased).
       Cost_r = Np * sum_{c != c*} t_c + max(Np, V_hold(c*) / eps^2) * t_c*   (the Np pilot trajectories of c* count
       toward the production estimate; the other pilots are wasted). C_fixed = mean over R = 500 replicates
       (expected cost; selection error simulated by resampling). Seed: default_rng(20261112 + group index).
       Np = 2000 is the whole array: no out-of-sample error can be simulated. It is reported as an IN-SAMPLE limit:
       c* = full-sample argmin (s = var: = bf; s = cost: = bf_c) with its full-sample variance and no selection error,
       i.e. it is optimistic for the fixed baseline (lower bound on its cost given a 2000-trajectory pilot).
       Np = 1000 is the largest pilot with genuine out-of-sample evaluation (hold-out 1000 trajectories).
  MR   referee's model, reproduced for checking (paper bp convention, pilots not reused, no selection error):
       C_fixed = Np sum_c t_c + N_prod t_bf, C_g = V_g / eps^2 * (tau_g + bp/20000).

Statistics, per (timing host, model, N_prod, Np, criterion, channel, M, subset)
  median W over groups; group bootstrap CI of the median (resample groups with replacement, B = 2000, percentiles
  0.625 / 99.375 = 98.75 %, as pre-registered for E9e); configuration-cluster bootstrap CI (cluster = channel, family,
  variant, n, L, gt, i.e. the 4 observables of a configuration move together; B = 4000, same percentiles, as
  pra_revision.py); number of groups with W > 1. These CIs reflect between-group variation only (within-group
  estimation noise of the variances is not propagated), as in the pre-registered rule. "pass" = median > 1 and
  group-bootstrap lower bound > 1 (the pre-registered rule, applied post hoc); "pass_cluster" = same with the cluster
  lower bound.
Subsets: all 98 groups; TN1e7 = groups whose E9d TN status at the 1e7 cap is 1 (b_e9d_summary.csv, n = 20; 19 groups,
  as pra_revision.py); complement (79 groups).
Hosts: RTX 2060 (b_e9e_u94.csv, primary); V100 (b_e9e_v100.csv, secondary, same models).

Known limitations stated in advance: timing noise ignored; t_c from 296-trajectory timing runs; a fixed run yields all
4 observables at once while greedy is run per observable (ignored here, as in the paper; it favours fixed in practice);
candidates' pilots are independent runs (no common random numbers); the M choice (10 vs 100) itself is not charged.

CHANGE LOG
  (none)
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = ["channel", "family", "variant", "n", "L", "gt"]
MS = [10, 100]
NCAND = {"deph": 7, "ad": 5}
DEFAULT = {"ad": "jump", "deph": "std"}
NPRODS = [20000, 100000]
NPS = [100, 300, 1000, 2000]
NREP = 500
LO, HI = 0.625, 99.375


def e9b_table():
    parts = []
    for j, f in enumerate(sorted(p for p in os.listdir(R) if p.startswith("b_e9b_u9") and p.endswith(".csv"))):
        d = pd.read_csv(os.path.join(R, f)).drop_duplicates(KEYS + ["kind", "scheme", "M"], keep="last")
        d["_src"] = j
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def load_groups(timing_csv):
    """Per-group data, selection of rows / fallback exactly as analyze_e9e.py."""
    old_i = e9b_table().set_index(KEYS).sort_index()
    new = pd.read_csv(os.path.join(R, timing_csv)).drop_duplicates(KEYS + ["kind", "scheme", "M"], keep="last")
    fb = {}
    for p in os.listdir(R):
        if p.startswith("b_e9e_fb_") and p.endswith(".csv"):
            for _, q in pd.read_csv(os.path.join(R, p)).iterrows():
                fb[(q["channel"], q["family"], q["variant"], int(q["n"]), int(q["L"]), float(q["gt"]), q["obs"],
                    int(q["M"]))] = q
    G = []
    for k, gn in new.groupby(KEYS):
        ch, fam, var, n, L, gt, o = k
        fx_n = gn[gn.kind == "fixed"]
        gr_n = gn[gn.kind == "greedy"].set_index("M")
        if len(fx_n) != NCAND[ch] or not all(float(m) in set(gr_n.index.astype(float)) for m in MS):
            continue
        go = old_i.loc[k].reset_index()
        fx_o = go[go.kind == "fixed"].drop_duplicates("scheme")
        gr_o = go[go.kind == "greedy"].drop_duplicates("M").set_index("M")
        raw = np.load(os.path.join(R, "b_e9b_raw", f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}.npz"))
        vf_csv = dict(zip(fx_o.scheme, fx_o["var"]))
        tf = dict(zip(fx_n.scheme, fx_n.wall_s / fx_n.shots))
        cands = list(vf_csv)
        g = dict(key=k, cands=cands, t=np.array([tf[s] for s in cands]), vcsv=np.array([vf_csv[s] for s in cands]),
                 x=np.array([raw[f"fixed_{s}_{o}"].astype(np.float64) for s in cands]))
        match_min = min(fx_n.match.min(), gr_n.match.min())
        fb_used = [m for m in MS if float(gr_n.loc[float(m), "match"]) < 0.99] if match_min < 0.99 else []
        for m in MS:
            rn = gr_n.loc[float(m)]
            fk = (ch, fam, var, int(n), int(L), float(gt), o, m)
            if m in fb_used and fk in fb:
                vg_csv = float(fb[fk]["var"])
                xg = np.load(os.path.join(R, "b_e9e_fb_raw", f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}_M{m}_{o}.npz"))["x"]
            else:
                if m in fb_used:
                    print("WARNING missing fallback", fk)
                vg_csv = float(gr_o.loc[float(m), "var"] if float(m) in gr_o.index else gr_o.loc[m, "var"])
                xg = raw[f"greedy_M{m}_{o}"]
            g[f"vg_csv{m}"] = vg_csv
            g[f"vg{m}"] = np.asarray(xg, np.float64).var(ddof=1)
            g[f"tau{m}"] = float(rn.wall_s) / float(rn.shots)
            g[f"bp{m}"] = float(rn.bp_s)
        g["v"] = g["x"].var(axis=1, ddof=1)
        G.append(g)
    return G


def pilot_stats(g, Np, rng):
    """Pilot / hold-out variances for R = NREP replicates (independent permutation per candidate); None if in-sample."""
    x = g["x"]
    C, N = x.shape
    if Np >= N:
        return None
    perm = np.argsort(rng.random((C, NREP, N)), axis=2)
    xs = np.take_along_axis(np.broadcast_to(x[:, None, :], (C, NREP, N)), perm, axis=2)
    return xs[:, :, :Np].var(axis=2, ddof=1), xs[:, :, Np:].var(axis=2, ddof=1)


def pilot_cost(g, Np, eps2, crit, st):
    """Expected fixed cost for pilot-then-commit (see docstring, M3)."""
    t = g["t"]
    if st is None:  # in-sample limit (Np = whole array)
        score = g["v"] * (t if crit == "cost" else 1.0)
        c = int(np.argmin(score))
        return Np * (t.sum() - t[c]) + max(Np, g["v"][c] / eps2) * t[c], np.array([c])
    vp, vh = st
    score = vp * (t[:, None] if crit == "cost" else 1.0)
    cs = score.argmin(axis=0)
    vh_c = vh[cs, np.arange(vp.shape[1])]
    cost = Np * (t.sum() - t[cs]) + np.maximum(Np, vh_c / eps2) * t[cs]
    return cost.mean(), cs


def per_group_W(G):
    rows = []
    for gi, g in enumerate(G):
        k = g["key"]
        ch = k[0]
        cands, t, v = g["cands"], g["t"], g["v"]
        ib = int(np.argmin(g["vcsv"]))      # paper's bf (CSV variances; identical choice to raw expected)
        ibr = int(np.argmin(v))
        ibc = int(np.argmin(v * t))
        idf = cands.index(DEFAULT[ch])
        base = dict(zip(KEYS, k), best=cands[ib], best_raw=cands[ibr], best_cost=cands[ibc])
        rng = np.random.default_rng(20261112 + gi)
        pilot = {}
        for Np in NPS:
            st = pilot_stats(g, Np, rng)
            for Nprod in NPRODS:
                eps2 = v[ibr] / Nprod
                for crit in ("var", "cost"):
                    pilot[(Nprod, Np, crit)] = pilot_cost(g, Np, eps2, crit, st)
        for m in MS:
            vg, tau, bp = g[f"vg{m}"], g[f"tau{m}"], g[f"bp{m}"]
            out = []
            out.append(("M1_oracle_paper", 20000, 0, "-", "paper",
                        g["vcsv"][ib] * t[ib] / (g[f"vg_csv{m}"] * (tau + bp / 20000))))
            out.append(("M1_oracle_paper_rawvar", 20000, 0, "-", "paper", v[ibr] * t[ibr] / (vg * (tau + bp / 20000))))
            for Nprod in NPRODS:
                eps2 = v[ibr] / Nprod
                ng = vg / eps2
                for bpc, Cg in (("honest", ng * tau + bp), ("paper", ng * (tau + bp / 20000))):
                    out.append(("M1b_oracle_var", Nprod, 0, "-", bpc, Nprod * t[ibr] / Cg))
                    out.append(("M1c_oracle_cost", Nprod, 0, "-", bpc, v[ibc] / eps2 * t[ibc] / Cg))
                    out.append(("M2_default", Nprod, 0, "-", bpc, v[idf] / eps2 * t[idf] / Cg))
                    for Np in NPS:
                        for crit in ("var", "cost"):
                            out.append(("M3_pilot", Nprod, Np, crit, bpc, pilot[(Nprod, Np, crit)][0] / Cg))
                Cg_ref = ng * (tau + bp / 20000)
                for Np in NPS:
                    out.append(("MR_referee", Nprod, Np, "-", "paper", (Np * t.sum() + Nprod * t[ibr]) / Cg_ref))
            for mod, Nprod, Np, crit, bpc, W in out:
                rows.append(dict(base, M=m, model=mod, Nprod=Nprod, Np=Np, crit=crit, bp_conv=bpc, W=W))
        # diagnostics: how often the pilot picks bf (Np=300 / 1000, var criterion, Nprod 20000)
        for Np in (100, 300, 1000):
            cs = pilot[(20000, Np, "var")][1]
            rows.append(dict(base, M=0, model=f"diag_pick_bf_Np{Np}", Nprod=20000, Np=Np, crit="var", bp_conv="-",
                             W=float(np.mean(cs == ibr))))
    return pd.DataFrame(rows)


def med_ci(x, rng, B=2000):
    x = np.asarray(x)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.median(x), *np.percentile(b, [LO, HI])


def cluster_ci(d, rng, B=4000):
    x = d.W.to_numpy(float)
    ids = d[CFG].astype(str).agg("|".join, axis=1).to_numpy()
    u, inv = np.unique(ids, return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return (*np.percentile(cb, [LO, HI]), len(u))


def summarize(P, host, tn):
    P = P.merge(tn, on=KEYS, how="left")
    out = []
    subs = {"all98": P, "TN1e7": P[P.TN10000000 == 1], "nonTN1e7": P[P.TN10000000 != 1]}
    for sub, S in subs.items():
        for (mod, Nprod, Np, crit, bpc, ch, m), d in S[S.M > 0].groupby(
                ["model", "Nprod", "Np", "crit", "bp_conv", "channel", "M"]):
            rng = np.random.default_rng(7)
            md, lo, hi = med_ci(d.W, rng)
            clo, chi, ncl = cluster_ci(d, np.random.default_rng(20261112))
            out.append(dict(host=host, subset=sub, model=mod, Nprod=Nprod, Np=Np, crit=crit, bp_conv=bpc, channel=ch,
                            M=m, groups=len(d), configs=ncl, W_median=md, W_lo=lo, W_hi=hi, W_cl_lo=clo, W_cl_hi=chi,
                            n_W_gt1=int((d.W > 1).sum()), passed=bool(md > 1 and lo > 1),
                            passed_cluster=bool(md > 1 and clo > 1)))
    return pd.DataFrame(out)


def show(T, title, sel):
    print(f"\n== {title}")
    d = T[sel(T)]
    for _, r in d.iterrows():
        tag = f"{r.model}" + (f" Np={r.Np} sel={r.crit}" if r.Np else "") + f" Nprod={r.Nprod} bp={r.bp_conv}"
        print(f"  {r.subset:8s} {tag:52s} {r.channel:4s} M={r.M:3d}  med W {r.W_median:6.3f}  "
              f"grp98.75 [{r.W_lo:.3f}, {r.W_hi:.3f}]  clus98.75 [{r.W_cl_lo:.3f}, {r.W_cl_hi:.3f}]  "
              f"W>1 {r.n_W_gt1:2d}/{r.groups:2d}{'  PASS' if r.passed else ''}{' (cluster pass)' if r.passed_cluster else ''}")


def main():
    pd.set_option("display.width", 220)
    e9d = pd.read_csv(os.path.join(R, "b_e9d_summary.csv"))
    tn = e9d[e9d.n == 20][KEYS + ["TN10000000"]].drop_duplicates(KEYS)
    tabs = []
    for host, csv, summ in (("RTX2060", "b_e9e_u94.csv", "b_e9e_summary.csv"),
                            ("V100", "b_e9e_v100.csv", "b_e9e_summary_v100.csv")):
        G = load_groups(csv)
        P = per_group_W(G)
        S0 = pd.read_csv(os.path.join(R, summ)).set_index(KEYS)
        print(f"\n######## host {host}: groups {len(G)} (by channel "
              f"{pd.Series([g['key'][0] for g in G]).value_counts().to_dict()})")
        vr = np.concatenate([np.abs(g["v"] / g["vcsv"] - 1) for g in G])
        print(f"max |V_raw / V_csv - 1| over candidates: {vr.max():.2e};  bf(raw) != bf(csv): "
              f"{int(sum(np.argmin(g['v']) != np.argmin(g['vcsv']) for g in G))}")
        p1 = P[P.model == "M1_oracle_paper"].set_index(KEYS)
        for m in MS:
            a = p1[p1.M == m].W
            b = S0.loc[a.index, f"W{m}"]
            print(f"reproduction M1 vs {summ} W{m}: max rel diff {np.max(np.abs(a / b - 1)):.2e}")
        dg = P[P.model.str.startswith("diag")]
        print("fraction of resamples where the var-pilot picks the full-sample bf (median over groups):",
              dg.groupby(["channel", "model"]).W.median().round(3).to_dict())
        print("bf by var == default:", P[(P.model == "M2_default") & (P.M == 10) & (P.Nprod == 20000) &
                                          (P.bp_conv == "honest")].assign(eq=lambda d: d.best == d.channel.map(DEFAULT))
              .groupby("channel").eq.mean().round(3).to_dict())
        T = summarize(P, host, tn)
        tabs.append(T)
        main_sel = lambda T: ((T.model.isin(["M1_oracle_paper", "M1_oracle_paper_rawvar"])) |
                              ((T.bp_conv == "honest") & (T.Nprod == 20000) &
                               (T.model.isin(["M1b_oracle_var", "M1c_oracle_cost", "M2_default"]) |
                                ((T.model == "M3_pilot") & (T.crit == "cost")))))
        for sub in ("all98", "TN1e7", "nonTN1e7"):
            show(T, f"{host} {sub}: main models (Nprod 20000, honest bp; pilot selection by var*t)",
                 lambda T, s=sub: main_sel(T) & (T.subset == s))
        show(T, f"{host} all98: pilot selection by var only (Nprod 20000, honest bp)",
             lambda T: (T.subset == "all98") & (T.model == "M3_pilot") & (T.crit == "var") & (T.Nprod == 20000) &
                       (T.bp_conv == "honest"))
        show(T, f"{host} all98: Nprod 100000 (honest bp)",
             lambda T: (T.subset == "all98") & (T.Nprod == 100000) & (T.bp_conv == "honest") &
                       (T.model.isin(["M1b_oracle_var", "M2_default"]) | ((T.model == "M3_pilot") & (T.crit == "cost"))))
        show(T, f"{host} all98: paper bp convention (Nprod 20000) for attribution",
             lambda T: (T.subset == "all98") & (T.Nprod == 20000) & (T.bp_conv == "paper") &
                       (T.model.isin(["M1b_oracle_var", "M2_default", "MR_referee"]) |
                        ((T.model == "M3_pilot") & (T.crit == "cost"))))
        show(T, f"{host} all98: referee model reproduction (Nprod 100000)",
             lambda T: (T.subset == "all98") & (T.model == "MR_referee") & (T.Nprod == 100000))
        if host == "RTX2060":
            P.to_csv(os.path.join(R, "pra_baseline_cost_pergroup.csv"), index=False)
    pd.concat(tabs, ignore_index=True).to_csv(os.path.join(R, "pra_baseline_cost.csv"), index=False)


if __name__ == "__main__":
    main()
