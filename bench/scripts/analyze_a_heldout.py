"""Plan A step 3 analysis: held-out n=20/24 time-to-target-SE, F2 decision.

Inputs: bench/results/a_heldout_ours.csv, a_heldout_aer.csv, per-trajectory raw npy files.
T(eps) = setup_s + Var * t_per_shot / eps^2  (setup only for ours_table). Aer best = the faster
(smaller T) of configs batched / batched_maxq per row. Ratios R = T_ref / T_cand (>1: cand faster).
95% CI by bootstrap over trajectories (variance resampled; t_per_shot and setup held fixed).
Rows with Var_std < 1e-12 (deterministic estimator) are excluded from ratios.
Output: bench/results/a_heldout_summary.csv and a printed table.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
EPS = (1e-3, 3e-4, 1e-4)
NB = 2000


def bvar(x, rng):
    return np.array([x[rng.integers(0, len(x), len(x))].var(ddof=1) for _ in range(NB)])


def main():
    o = pd.read_csv(os.path.join(R, "a_heldout_ours.csv"))
    a = pd.read_csv(os.path.join(R, "a_heldout_aer.csv"))
    rng = np.random.default_rng(11)
    rows = []
    for (fam, n, gt, obs), g in o.groupby(["family", "n", "gt", "obs"]):
        s = g[g.method == "ours_std"].iloc[0]
        t = g[g.method == "ours_table"].iloc[0]
        ag = a[(a["family"] == fam) & (a["n"] == n) & (a["gt"] == gt) & (a["obs"] == obs)
               & (a["method"] == "std")]
        if s["var"] < 1e-12 or len(ag) == 0:
            rows.append(dict(family=fam, n=n, gt=gt, obs=obs, note="deterministic or no Aer row"))
            continue
        xs = np.load(os.path.join(ROOT, s.raw_file))
        xt = np.load(os.path.join(ROOT, t.raw_file))
        vs_b, vt_b = bvar(xs, rng), bvar(xt, rng)
        # Aer: pick best config by point estimate at each eps, bootstrap its variance
        aer = {}
        for _, ar in ag.iterrows():
            xa = np.load(os.path.join(ROOT, ar.raw_file))
            aer[ar.config] = (ar, xa, bvar(xa, rng))
        # consistency of means (same circuit): z of ours_std vs each Aer config
        zs = {c: (s["mean"] - v[0]["mean"]) / np.hypot(s.se, v[0].se) for c, v in aer.items()}
        rec = dict(family=fam, n=n, gt=gt, obs=obs, var_ratio=s["var"] / t["var"],
                   tps_std_ms=s.t_per_shot_s * 1e3, tps_tab_ms=t.t_per_shot_s * 1e3,
                   setup_s=t.setup_s, frac_table=t.frac_table,
                   z_mean_vs_aer=max(abs(z) for z in zs.values()), note="")
        lo, hi = np.percentile(vs_b / vt_b, [2.5, 97.5])
        rec.update(var_ratio_lo=lo, var_ratio_hi=hi)
        for c, (ar, xa, vb) in aer.items():
            rec[f"aer_{c}_tps_ms"] = ar.t_per_shot_s * 1e3
        for eps in EPS:
            Ts = vs_b * s.t_per_shot_s / eps ** 2
            Tt = t.setup_s + vt_b * t.t_per_shot_s / eps ** 2
            best = min(aer, key=lambda c: aer[c][0]["var"] * aer[c][0].t_per_shot_s)
            ar, xa, vb = aer[best]
            Ta = vb * ar.t_per_shot_s / eps ** 2
            e = f"{eps:g}"
            rec[f"aer_best_{e}"] = best
            for name, num, den in (("pol", Ts, Tt), ("aer_vs_tab", Ta, Tt), ("aer_vs_std", Ta, Ts)):
                pt = {"pol": (s["var"] * s.t_per_shot_s / eps ** 2)
                      / (t.setup_s + t["var"] * t.t_per_shot_s / eps ** 2),
                      "aer_vs_tab": (ar["var"] * ar.t_per_shot_s / eps ** 2)
                      / (t.setup_s + t["var"] * t.t_per_shot_s / eps ** 2),
                      "aer_vs_std": (ar["var"] * ar.t_per_shot_s) / (s["var"] * s.t_per_shot_s)}[name]
                l, h = np.percentile(num / den, [2.5, 97.5])
                rec[f"{name}_{e}"] = pt
                rec[f"{name}_{e}_lo"] = l
                rec[f"{name}_{e}_hi"] = h
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "a_heldout_summary.csv"), index=False)
    d = df[df.note == ""]
    pd.set_option("display.width", 250)
    cols = ["family", "n", "gt", "obs", "var_ratio", "tps_std_ms", "tps_tab_ms", "setup_s",
            "pol_0.001", "pol_0.001_lo", "pol_0.001_hi", "pol_0.0001",
            "aer_vs_std_0.001", "aer_vs_tab_0.001", "aer_vs_tab_0.001_lo", "aer_best_0.001",
            "z_mean_vs_aer"]
    print(d[cols].round(3).to_string(index=False))
    print("\nexcluded:", df[df.note != ""][["family", "n", "gt", "obs", "note"]].to_string(index=False))
    for e in EPS:
        e = f"{e:g}"
        print(f"\neps={e}: policy gain (ours_std/ours_table) median {d[f'pol_{e}'].median():.3f}, "
              f"CI_lo>1 in {(d[f'pol_{e}_lo'] > 1).sum()}/{len(d)}, CI_hi<1 in {(d[f'pol_{e}_hi'] < 1).sum()}")
        print(f"  F2 (Aer best / ours+policy): min CI_hi {d[f'aer_vs_tab_{e}_hi'].min():.2f}, "
              f"rows with CI_hi<1: {(d[f'aer_vs_tab_{e}_hi'] < 1).sum()}/{len(d)}")
    print(f"\nimplementation-only (Aer best / ours_std): median {d['aer_vs_std_0.001'].median():.2f}, "
          f"range {d['aer_vs_std_0.001'].min():.2f}-{d['aer_vs_std_0.001'].max():.2f}")


if __name__ == "__main__":
    main()
