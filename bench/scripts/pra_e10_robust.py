"""POST HOC robustness analyses of the E10 static per-location table result (multi-observable accounting and
cost-inclusive fixed baselines).

Written 2026-11-13, after the pre-registered E10 decision (docs/plan_B_characterization_2026-11-06.md, "E10 result":
ad pilot table median W_bf 1.374 [1.207, 1.572]). Both analyses below are POST HOC (not pre-registered). This
specification was written before any output of this script was looked at; it is not to be tuned afterwards. Any later
change is recorded in the CHANGE LOG at the end of this docstring, with the reason.

Inputs (read only; nothing in bench/results is modified)
  b_e10_u94.csv            E10 ledger: fixed re-timing rows (wall_s / shots = t_c, one run gives all 4 observables) and
                           static rows (wall_s, shots = 2000, bp_s, build_s) per (config, target observable, table).
  b_e10_raw/<cfg>_<table>_<o>.npz   static production arrays: x (target o) and all_<q> for ALL 4 observables q of the
                           SAME static run (so a table built for o can be evaluated on the other observables).
  b_e9b_raw/<cfg>.npz      fixed candidate arrays fixed_<label>_<q> (N = 2000) for all 4 observables (E10 identity check
                           passed with match 1.000, so these are the variances of the E10-timed fixed runs).
  b_e9d_summary.csv        TN status at the 1e7 Pauli-path cap (n = 20; TN10000000 == 1 -> TN@1e7 subset, 19 groups, as
                           pra_revision.py C4 / pra_w_subsets.csv).
  b_e10_summary.csv        only to check that the reproduced single-observable W equals analyze_e10.py.
  bench/scripts/pra_baseline_cost.py   imported for its pilot-then-commit functions (pilot_stats, pilot_cost), so the
                           fixed-baseline model is identical to the E9e cost-inclusive analysis.
Outputs: bench/results/pra_e10_robust.txt (stdout, saved by the caller), bench/results/pra_e10_robust.csv (long format:
  level = "summary" rows (medians and CIs) and level = "unit" rows (per config for analysis 1, per group for analysis 2)).

Notation. Group = (channel, family, variant, n, L, gt, obs), 98 groups (ad 62, deph 36). Config k = (channel, family,
  variant, n, L, gt), 37 configs (ad 26, deph 11). O_k = the observables of config k that are E10 groups (|O_k| = 1..4).
  Candidates c (ad 5, deph 7; c = 0 = jump / std = default). V_c(o) = var(fixed_<c>_<o>, ddof = 1) from E9b raw (float64).
  t_c = E10 fixed wall / N_t (seconds per trajectory, all observables at once). bf(o) = argmin_c V_c(o) (oracle by
  variance, as the paper and analyze_e10.py). Table types T in {pilot, ideal}, analysed separately.
  Static run of table built for o* (target): tau(o*) = wall / 2000 per trajectory, ovh(o*) = bp_s + build_s (one-time:
  back-propagation + group precompute + table construction; for the pilot table build_s is the measured wall time of the
  74 candidate-0 pilot trajectories with the V_c evaluation, i.e. the pilot is charged in full and not reused).
  Vs(o*, o) = var(all_<o>) of that static run (float64, ddof = 1).
  Cost conventions: "honest" = N tau + ovh (one-time costs paid once); "paper" = N (tau + ovh / 20000) (amortized as in
  analyze_e10.py t_static). Fixed runs have no one-time cost.

ANALYSIS 1 — multi-observable accounting (per config k, per table type T)
  Target error eps_o^2 per observable o in O_k; required trajectories of a run with per-trajectory variance V: V / eps_o^2.
  Error definitions:
    A20k  (primary)  eps_o^2 = V_bf(o)(o) / 20000: the oracle best fixed for that observable needs exactly 20000.
    A100k            same with 100000 (larger production; overheads matter less).
    Brel             common relative error: eps_o = r_k |mu_o|, mu_o = mean of the bf(o) fixed array (E9b). Observables with
                     |mu_o| <= 4 sqrt(V_bf(o) / 2000) (mean not resolved) are excluded from O_k under Brel (count
                     reported); configs left with no observable are dropped. Scale: r_k^2 = max_{o in O_k}
                     [V_bf(o)(o) / mu_o^2] / 20000, i.e. the relatively hardest observable needs 20000 under its own
                     oracle fixed scheme. (Under A the per-observable weights are equalized at the oracle; under Brel
                     the observable with the largest relative variance dominates.)
  Fixed strategies (all oracle: chosen with full knowledge of the N = 2000 variances, zero selection cost):
    F1 single  = min_c t_c max_{o in O_k} V_c(o) / eps_o^2      (one fixed run serves all observables; the requested
                 "fixed" baseline)
    F2 perobs  = sum_{o} t_bf(o) V_bf(o)(o) / eps_o^2          (per-observable oracle, a separate run per observable,
                 as requested; under A20k = sum_o 20000 t_bf(o))
    F3 port    = min over assignments a: O_k -> candidates of sum_{c used} t_c max_{o: a(o) = c} V_c(o) / eps_o^2
                 (best fixed portfolio, runs shared; F3 <= min(F1, F2): the cheapest fixed oracle)
  Static strategies:
    S1 perobs  = sum_{o} [tau(o) Vs(o, o) / eps_o^2 + ovh(o)]  (one table per observable, each run used only for its
                 target; the requested "static" cost). S1p = same in the paper convention.
    S2 shared  = min_{o*} [tau(o*) max_{o} Vs(o*, o) / eps_o^2 + ovh(o*)]  (one table run serves all observables;
                 oracle choice of the target o*)
    S3 port    = min over assignments a: O_k -> tables {o*} of sum_{o* used} [tau(o*) max_{o: a(o) = o*} Vs(o*, o) /
                 eps_o^2 + ovh(o*)]   (best static portfolio; S3 <= min(S1, S2); oracle)
  Ratios (> 1: static cheaper): W_multi = F1 / S1 (PRIMARY), and F2/S1, F3/S1, F1/S1p, F1/S2, F1/S3, F3/S3.
  Also W_single = per-config median over o of the single-observable honest W (20000 t_bf / (N_s tau + ovh)) for reference.
  Statistics: per (error definition, table type, channel, subset, ratio) median over configs with a config bootstrap
  (resample configs, B = 4000, percentiles 0.625 / 99.375 = 98.75 %, default_rng(20261113)); number of configs with
  ratio > 1. "pass" = median > 1 and lower bound > 1. Only between-config variation is in the CI (within-config variance
  estimation noise not propagated, as in the E9e/E10 rules).
  Subsets: all37 = all configs, all of O_k. TN1e7r (primary TN subset) = configs with >= 1 TN@1e7 observable, O_k
  restricted to the TN@1e7 observables (an observable that the 1e7 Pauli-path run solves would not be simulated by
  trajectories). TN1e7a = the same configs with all of O_k.
  Cross-observable transfer (configs with |O_k| >= 2, every ordered pair (o*, o'), o' != o*, per table type): does a
  table built for o* help o'? R_cross_bf = V_bf(o')(o') / Vs(o*, o'); R_cross_def = V_0(o') / Vs(o*, o');
  own_vs_cross = Vs(o*, o') / Vs(o', o') (> 1: the own table is better for o'); W_cross_bf = V_bf t_bf / (Vs(o*, o')
  tau(o*)) (no one-time cost). Medians over pairs with a config-cluster bootstrap 98.75 % CI (B = 4000), fraction of pairs
  > 1. Bias check: |z| of the cross mean vs the bf(o') fixed mean (max and count > 4).

ANALYSIS 2 — cost-inclusive fixed baselines for the E10 static result (per group, per table type)
  Same models as pra_baseline_cost.py with the greedy rule replaced by the static table; t_c from E10 (same session as
  the static runs). eps^2 = V_bf / N_prod, N_prod in {20000 (primary), 100000}. n_s = Vs(o, o) / eps^2.
  Static cost: honest C_s = n_s tau + bp_s + build_s (pilot and build charged once, in full); paper C_s' = n_s (tau +
  (bp_s + build_s) / 20000) (= analyze_e10.py convention).
  Fixed models (W = C_fixed / C_s):
    M1_oracle_paper   V_bf t_bf / (Vs t_static): must reproduce b_e10_summary.csv <table>_W_bf (check printed).
    M1b_oracle_var    C = N_prod t_bf.          M1c_oracle_cost  bf_c = argmin V_c t_c, C = V_bfc / eps^2 t_bfc.
    M2_default        C = V_0 / eps^2 t_0 (candidate 0: jump / std).
    M3_pilot          pilot-then-commit, Np in {100, 300, 1000, 2000}, selection by var and by cost (var * t), pilot of
                      the committed candidate reused, R = 500 resampling replicates, hold-out variance as the true
                      variance; Np = 2000 = in-sample limit (pra_baseline_cost.pilot_stats / pilot_cost, unchanged).
                      Replicate seed default_rng(20261112 + gi), gi = index of the group in sorted KEYS order.
    MR_referee        Np sum_c t_c + N_prod t_bf (paper convention only).
  Statistics as pra_baseline_cost.py: per (subset, model, N_prod, Np, crit, convention, channel, table) median W over
  groups, group bootstrap 98.75 % (B = 2000, default_rng(7)), config-cluster bootstrap 98.75 % (B = 4000,
  default_rng(20261112)), n(W > 1), pass / pass_cluster. Subsets: all98, TN1e7 (19 groups), nonTN1e7.
  Diagnostic: fraction of replicates where the var-pilot picks the full-sample bf (Np 100 / 300 / 1000).

Known limitations stated in advance: all fixed baselines in analysis 1 are oracles (favour fixed); S2 / S3 choose the
target table post hoc (favour static) and are reported as bounds, not as the primary; timing noise ignored (t from one
296-trajectory run per candidate and one 2000-trajectory static run); multi-observable pilot-then-commit is not modelled;
the 74 static pilot trajectories (candidate 0) are not credited to any fixed baseline.

Usage (ubuntu94): cd ~/work/qiskit-co && . ~/activate_sci.sh && nice -n 10 python bench/scripts/pra_e10_robust.py \
  > bench/results/pra_e10_robust.txt

CHANGE LOG
  (none)
"""
import importlib.util
import itertools
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
KEYS = ["channel", "family", "variant", "n", "L", "gt", "obs"]
CFG = ["channel", "family", "variant", "n", "L", "gt"]
TABLES = ["pilot", "ideal"]
NCAND = {"deph": 7, "ad": 5}
LO, HI = 0.625, 99.375
NPROD_REF = 20000

_spec = importlib.util.spec_from_file_location("pbc", os.path.join(os.path.dirname(__file__), "pra_baseline_cost.py"))
PBC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PBC)


def key_str(ch, fam, var, n, L, gt):
    return f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"


def med_ci(x, rng, B):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [LO, HI])


def cluster_ci(x, ids, rng, B=4000):
    x = np.asarray(x, float)
    u, inv = np.unique(np.asarray(ids), return_inverse=True)
    mem = [np.nonzero(inv == j)[0] for j in range(len(u))]
    cb = [np.median(x[np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])]) for _ in range(B)]
    return (*np.percentile(cb, [LO, HI]), len(u))


# ----------------------------------------------------------------------------------------------------------- loading
def load():
    d = pd.read_csv(os.path.join(R, "b_e10_u94.csv")).drop_duplicates(KEYS + ["kind", "scheme"], keep="last")
    e9d = pd.read_csv(os.path.join(R, "b_e9d_summary.csv"))
    e9d = e9d[e9d.n == 20][KEYS + ["TN10000000"]].drop_duplicates(KEYS)
    tn = {tuple(r[:-1]): int(r[-1] == 1) for r in e9d.itertuples(index=False)}
    C = []
    for k, g in d.groupby(CFG):
        ch = k[0]
        st = g[g.kind == "static"]
        obs = sorted(set(st.obs))
        fx = g[g.kind == "fixed"]
        labels = list(fx[fx.obs == obs[0]].scheme)
        assert len(labels) == NCAND[ch] and labels[0] in ("std", "jump"), (k, labels)
        tvec = np.array([float(fx[(fx.obs == obs[0]) & (fx.scheme == s)].wall_s.iloc[0]) /
                         float(fx[(fx.obs == obs[0]) & (fx.scheme == s)].shots.iloc[0]) for s in labels])
        for o in obs:                                     # same fixed run for every observable
            f2 = fx[fx.obs == o].set_index("scheme")
            assert np.allclose([float(f2.loc[s, "wall_s"]) / float(f2.loc[s, "shots"]) for s in labels], tvec)
        ks = key_str(*k)
        raw = np.load(os.path.join(R, "b_e9b_raw", ks + ".npz"))
        X = {o: np.array([raw[f"fixed_{s}_{o}"].astype(np.float64) for s in labels]) for o in obs}
        S = {}
        for t in TABLES:
            for o in obs:
                r = st[(st.obs == o) & (st.scheme == t)]
                assert len(r) == 1, (k, o, t)
                r = r.iloc[0]
                z = np.load(os.path.join(R, "b_e10_raw", f"{ks}_{t}_{o}.npz"))
                assert np.array_equal(z["x"], z[f"all_{o}"])
                S[(t, o)] = dict(tau=float(r.wall_s) / float(r.shots), ovh=float(r.bp_s) + float(r.build_s),
                                 bp=float(r.bp_s), build=float(r.build_s), shots=int(r.shots),
                                 Y={q: z[f"all_{q}"].astype(np.float64) for q in obs})
        C.append(dict(key=k, ch=ch, obs=obs, labels=labels, t=tvec, X=X, S=S,
                      tn={o: tn.get(k + (o,), 0) for o in obs}))
    return C


# --------------------------------------------------------------------------------------------------------- analysis 1
def best_assign(n_obs, n_opt, cost_of_group):
    """min over assignments obs -> options of sum over used options of cost_of_group(option, [obs indices])."""
    best = np.inf
    for a in itertools.product(range(n_opt), repeat=n_obs):
        tot = 0.0
        for c in set(a):
            tot += cost_of_group(c, [i for i in range(n_obs) if a[i] == c])
            if tot >= best:
                break
        best = min(best, tot)
    return best


def a1_config(c, t, obs, edef):
    """Costs for one config, table type t, observable set obs, error definition edef. None if obs empty."""
    V = np.array([c["X"][o].var(axis=1, ddof=1) for o in obs])     # (m, C)
    mu = np.array([c["X"][o][np.argmin(V[i])].mean() for i, o in enumerate(obs)])
    vbf = V.min(axis=1)
    ibf = V.argmin(axis=1)
    if edef in ("A20k", "A100k"):
        eps2 = vbf / (20000 if edef == "A20k" else 100000)
        keep = np.ones(len(obs), bool)
    else:
        keep = np.abs(mu) > 4 * np.sqrt(vbf / 2000)
        if not keep.any():
            return None, int((~keep).sum())
        r2 = np.max(vbf[keep] / mu[keep] ** 2) / 20000
        eps2 = r2 * mu ** 2
    obs = [o for o, kk in zip(obs, keep) if kk]
    V, vbf, ibf, eps2 = V[keep], vbf[keep], ibf[keep], eps2[keep]
    m = len(obs)
    tf = c["t"]
    need = V / eps2[:, None]                                       # (m, C) trajectories
    F1 = float(np.min(tf * need.max(axis=0)))
    F2 = float(np.sum(tf[ibf] * vbf / eps2))
    F3 = best_assign(m, len(tf), lambda cc, ii: tf[cc] * need[ii, cc].max())
    S = [c["S"][(t, o)] for o in obs]
    Vs = np.array([[S[j]["Y"][o].var(ddof=1) for o in obs] for j in range(m)])   # Vs[target j, observable i]
    needs = Vs / eps2[None, :]
    tau = np.array([s["tau"] for s in S])
    ovh = np.array([s["ovh"] for s in S])
    S1 = float(np.sum(tau * np.diag(needs) + ovh))
    S1p = float(np.sum(np.diag(needs) * (tau + ovh / 20000)))
    S2 = float(np.min(tau * needs.max(axis=1) + ovh))
    S3 = best_assign(m, m, lambda j, ii: tau[j] * needs[j, ii].max() + ovh[j])
    Wsing = np.median([vbf[i] / eps2[i] * tf[ibf[i]] / (needs[i, i] * tau[i] + ovh[i]) for i in range(m)])
    out = dict(m=m, F1=F1, F2=F2, F3=F3, S1=S1, S1p=S1p, S2=S2, S3=S3,
               W_multi=F1 / S1, W_F2_S1=F2 / S1, W_F3_S1=F3 / S1, W_F1_S1p=F1 / S1p, W_F1_S2=F1 / S2,
               W_F1_S3=F1 / S3, W_F3_S3=F3 / S3, W_single_med=float(Wsing))
    return out, int((~keep).sum())


RATIOS = ["W_multi", "W_F2_S1", "W_F3_S1", "W_F1_S1p", "W_F1_S2", "W_F1_S3", "W_F3_S3", "W_single_med"]


def analysis1(C):
    print("\n################ ANALYSIS 1 — multi-observable accounting (POST HOC)")
    ms = pd.Series([len(c["obs"]) for c in C])
    print(f"configs {len(C)} (by channel {pd.Series([c['ch'] for c in C]).value_counts().to_dict()}); |O_k| "
          f"distribution {ms.value_counts().sort_index().to_dict()}; groups {int(ms.sum())}")
    units, summ = [], []
    for edef in ("A20k", "A100k", "Brel"):
        for t in TABLES:
            for sub in ("all37", "TN1e7r", "TN1e7a"):
                rows, nexcl = [], 0
                for c in C:
                    obs = c["obs"]
                    if sub != "all37":
                        if not any(c["tn"].values()):
                            continue
                        if sub == "TN1e7r":
                            obs = [o for o in obs if c["tn"][o]]
                    r, ex = a1_config(c, t, obs, edef)
                    nexcl += ex
                    if r is None:
                        continue
                    rows.append(dict(zip(CFG, c["key"]), **r))
                D = pd.DataFrame(rows)
                for r in rows:
                    units.append(dict(level="unit", analysis="A1", edef=edef, table=t, subset=sub, **r))
                print(f"\n== A1 eps={edef} table={t} subset={sub}: configs {len(D)}"
                      + (f" (observables excluded as unresolved means: {nexcl})" if edef == "Brel" else ""))
                for ch, d in D.groupby("channel"):
                    print(f"  {ch:4s} configs {len(d):2d}  obs {int(d.m.sum()):2d}")
                    for col in RATIOS:
                        md, lo, hi = med_ci(d[col], np.random.default_rng(20261113), 4000)
                        ok = md > 1 and lo > 1
                        summ.append(dict(level="summary", analysis="A1", edef=edef, table=t, subset=sub, channel=ch,
                                         ratio=col, units=len(d), W_median=md, W_lo=lo, W_hi=hi,
                                         n_W_gt1=int((d[col] > 1).sum()), passed=ok))
                        print(f"      {col:13s} median {md:6.3f}  cfg98.75 [{lo:.3f}, {hi:.3f}]  >1 "
                              f"{int((d[col] > 1).sum()):2d}/{len(d):2d}{'  PASS' if ok else ''}")
    return units, summ


def cross(C):
    print("\n################ ANALYSIS 1b — does a table built for o* help the other observables o' of its config?")
    rows = []
    for c in C:
        obs = c["obs"]
        if len(obs) < 2:
            continue
        for t in TABLES:
            for os_ in obs:
                s = c["S"][(t, os_)]
                for op in obs:
                    if op == os_:
                        continue
                    x = s["Y"][op]
                    vs = x.var(ddof=1)
                    vs_own = c["S"][(t, op)]["Y"][op].var(ddof=1)
                    V = c["X"][op].var(axis=1, ddof=1)
                    ib = int(np.argmin(V))
                    xb = c["X"][op][ib]
                    z = (x.mean() - xb.mean()) / np.sqrt(vs / len(x) + V[ib] / len(xb))
                    rows.append(dict(zip(CFG, c["key"]), table=t, target=os_, obs=op, R_cross_bf=V[ib] / vs,
                                     R_cross_def=V[0] / vs, own_vs_cross=vs / vs_own,
                                     W_cross_bf=V[ib] * c["t"][ib] / (vs * s["tau"]), z=z,
                                     tn_obs=c["tn"][op]))
    D = pd.DataFrame(rows)
    D["cfg"] = D[CFG].astype(str).agg("|".join, axis=1)
    summ = []
    print(f"pairs {len(D) // len(TABLES)} per table type, configs {D.cfg.nunique()}")
    for (t, ch), d in D.groupby(["table", "channel"]):
        print(f"  table {t:5s} {ch:4s} pairs {len(d):3d} configs {d.cfg.nunique():2d}  max|z| {d.z.abs().max():.2f} "
              f"n(|z|>4) {int((d.z.abs() > 4).sum())}")
        for col in ("R_cross_bf", "W_cross_bf", "R_cross_def", "own_vs_cross"):
            md = float(d[col].median())
            lo, hi, _ = cluster_ci(d[col], d.cfg, np.random.default_rng(20261113))
            summ.append(dict(level="summary", analysis="A1cross", table=t, subset="all", channel=ch, ratio=col,
                             units=len(d), W_median=md, W_lo=lo, W_hi=hi, n_W_gt1=int((d[col] > 1).sum()),
                             passed=bool(md > 1 and lo > 1)))
            print(f"      {col:13s} median {md:6.3f}  cluster98.75 [{lo:.3f}, {hi:.3f}]  >1 {int((d[col] > 1).sum())}/{len(d)}")
    units = [dict(level="unit", analysis="A1cross", **r) for r in D.drop(columns="cfg").to_dict("records")]
    return units, summ


# --------------------------------------------------------------------------------------------------------- analysis 2
def analysis2(C):
    print("\n################ ANALYSIS 2 — cost-inclusive fixed baselines vs the E10 static table (POST HOC)")
    G = []
    for c in C:
        for o in c["obs"]:
            G.append((c["key"] + (o,), c, o))
    G.sort(key=lambda g: tuple(str(v) for v in g[0]))  # sorted KEYS order (string sort, deterministic)
    ref = pd.read_csv(os.path.join(R, "b_e10_summary.csv")).set_index(KEYS)
    rows, diag, repro = [], [], []
    for gi, (k, c, o) in enumerate(G):
        ch = k[0]
        t, x = c["t"], c["X"][o]
        v = x.var(axis=1, ddof=1)
        g = dict(x=x, t=t, v=v)
        ibr, ibc = int(np.argmin(v)), int(np.argmin(v * t))
        rng = np.random.default_rng(20261112 + gi)
        pil = {}
        for Np in PBC.NPS:
            stt = PBC.pilot_stats(g, Np, rng)
            for Nprod in PBC.NPRODS:
                for crit in ("var", "cost"):
                    pil[(Nprod, Np, crit)] = PBC.pilot_cost(g, Np, v[ibr] / Nprod, crit, stt)
        for Np in (100, 300, 1000):
            diag.append(dict(channel=ch, Np=Np, f=float(np.mean(pil[(20000, Np, "var")][1] == ibr))))
        base = dict(zip(KEYS, k), best=c["labels"][ibr], best_cost=c["labels"][ibc], gi=gi)
        for tb in TABLES:
            s = c["S"][(tb, o)]
            vs = s["Y"][o].var(ddof=1)
            tstat = s["tau"] + s["ovh"] / 20000
            W1 = v[ibr] * t[ibr] / (vs * tstat)
            repro.append(W1 / float(ref.loc[k, f"{tb}_W_bf"]) - 1)
            out = [("M1_oracle_paper", 20000, 0, "-", "paper", W1)]
            for Nprod in PBC.NPRODS:
                eps2 = v[ibr] / Nprod
                ns = vs / eps2
                for conv, Cs in (("honest", ns * s["tau"] + s["ovh"]), ("paper", ns * tstat)):
                    out.append(("M1b_oracle_var", Nprod, 0, "-", conv, Nprod * t[ibr] / Cs))
                    out.append(("M1c_oracle_cost", Nprod, 0, "-", conv, v[ibc] / eps2 * t[ibc] / Cs))
                    out.append(("M2_default", Nprod, 0, "-", conv, v[0] / eps2 * t[0] / Cs))
                    for Np in PBC.NPS:
                        for crit in ("var", "cost"):
                            out.append(("M3_pilot", Nprod, Np, crit, conv, pil[(Nprod, Np, crit)][0] / Cs))
                for Np in PBC.NPS:
                    out.append(("MR_referee", Nprod, Np, "-", "paper", (Np * t.sum() + Nprod * t[ibr]) / (ns * tstat)))
            for mod, Nprod, Np, crit, conv, W in out:
                rows.append(dict(base, table=tb, model=mod, Nprod=Nprod, Np=Np, crit=crit, bp_conv=conv, W=W))
    P = pd.DataFrame(rows)
    print(f"groups {len(G)}; reproduction of b_e10_summary <table>_W_bf by M1_oracle_paper: max |rel diff| "
          f"{np.max(np.abs(repro)):.2e}")
    dg = pd.DataFrame(diag)
    print("fraction of replicates where the var-pilot picks the full-sample bf (median over groups):",
          dg.groupby(["channel", "Np"]).f.median().round(3).to_dict())
    tnmap = {g[0]: g[1]["tn"][g[2]] for g in G}
    P["TN1e7"] = [tnmap[tuple(r)] for r in P[KEYS].itertuples(index=False)]
    summ = []
    subs = {"all98": P, "TN1e7": P[P.TN1e7 == 1], "nonTN1e7": P[P.TN1e7 != 1]}
    for sub, S in subs.items():
        for (tb, mod, Nprod, Np, crit, conv, ch), d in S.groupby(["table", "model", "Nprod", "Np", "crit", "bp_conv",
                                                                  "channel"]):
            md, lo, hi = med_ci(d.W, np.random.default_rng(7), 2000)
            clo, chi, ncl = cluster_ci(d.W, d[CFG].astype(str).agg("|".join, axis=1), np.random.default_rng(20261112))
            summ.append(dict(level="summary", analysis="A2", subset=sub, table=tb, model=mod, Nprod=Nprod, Np=Np,
                             crit=crit, bp_conv=conv, channel=ch, units=len(d), configs=ncl, W_median=md, W_lo=lo,
                             W_hi=hi, W_cl_lo=clo, W_cl_hi=chi, n_W_gt1=int((d.W > 1).sum()),
                             passed=bool(md > 1 and lo > 1), passed_cluster=bool(md > 1 and clo > 1)))
    T = pd.DataFrame(summ)

    def show(title, sel):
        print(f"\n== {title}")
        for _, r in T[sel].iterrows():
            tag = f"{r.model}" + (f" Np={r.Np} sel={r.crit}" if r.Np else "") + f" Nprod={r.Nprod} bp={r.bp_conv}"
            print(f"  {r.subset:8s} {r.table:5s} {tag:48s} {r.channel:4s} med W {r.W_median:6.3f}  grp98.75 "
                  f"[{r.W_lo:.3f}, {r.W_hi:.3f}]  clus98.75 [{r.W_cl_lo:.3f}, {r.W_cl_hi:.3f}]  W>1 "
                  f"{r.n_W_gt1:2d}/{r.units:2d}{'  PASS' if r.passed else ''}{' (cluster pass)' if r.passed_cluster else ''}")

    main = (T.model == "M1_oracle_paper") | ((T.bp_conv == "honest") & (T.Nprod == 20000) & (
        T.model.isin(["M1b_oracle_var", "M1c_oracle_cost", "M2_default"]) | ((T.model == "M3_pilot"))))
    for sub in subs:
        show(f"{sub}: main models (Nprod 20000, honest one-time costs; M1_oracle_paper = analyze_e10 W_bf)",
             main & (T.subset == sub))
    show("all98 + TN1e7: Nprod 100000 (honest)", T.subset.isin(["all98", "TN1e7"]) & (T.Nprod == 100000) &
         (T.bp_conv == "honest") & (T.model.isin(["M1b_oracle_var", "M2_default"]) |
                                    ((T.model == "M3_pilot") & (T.crit == "cost"))))
    show("all98: paper convention (Nprod 20000) incl. referee model", (T.subset == "all98") & (T.Nprod == 20000) &
         (T.bp_conv == "paper") & (T.model.isin(["M1b_oracle_var", "M2_default", "MR_referee"]) |
                                    ((T.model == "M3_pilot") & (T.crit == "cost"))))
    units = [dict(level="unit", analysis="A2", units=1, **r) for r in P.to_dict("records")]
    return units, summ


def main():
    pd.set_option("display.width", 220)
    C = load()
    u1, s1 = analysis1(C)
    u1c, s1c = cross(C)
    u2, s2 = analysis2(C)
    out = pd.DataFrame(s1 + s1c + s2 + u1 + u1c + u2)
    out.to_csv(os.path.join(R, "pra_e10_robust.csv"), index=False)
    print(f"\nwrote {os.path.join(R, 'pra_e10_robust.csv')} ({len(out)} rows)")


if __name__ == "__main__":
    main()
