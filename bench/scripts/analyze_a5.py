"""Summarise A5 runs: variance ratio and time-to-target ratio std/proj with bootstrap CIs.

For every (family, obs, layers, n, gt) group:
  * var_ratio      = Var_std / Var_proj  (>1: projector has lower per-shot variance)
  * best configs   : for each method the (device, config) row with the smallest
                     time_to_se_s (wall/N * Var / eps^2)
  * time_ratio     = T_std(best) / T_proj(best)  (>1: projector reaches the target SE sooner)
  * bias           : (mean - exact)/SE for each method when an exact reference exists
95 % CIs from a percentile bootstrap of the per-shot raw values (variance only; the
per-shot wall time is treated as fixed).

Usage: python bench/scripts/analyze_a5.py bench/results/a5_aer_walltime.csv [more.csv] \
          --out bench/results/a5_summary.csv
"""
import argparse
import csv
import os

import numpy as np

B = 1000
RNG = np.random.default_rng(20261104)


def boot_var(x):
    n = len(x)
    out = np.empty(B)
    for b in range(B):
        out[b] = x[RNG.integers(0, n, n)].var(ddof=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".", help="base dir that raw_file paths are relative to")
    args = ap.parse_args()
    rows = []
    for f in args.csv:
        rows += list(csv.DictReader(open(f, newline="")))
    groups = {}
    for r in rows:
        k = (r["family"], r["obs"], int(r["layers"]), int(r["n"]), float(r["gt"]))
        groups.setdefault(k, []).append(r)
    out = []
    for k in sorted(groups):
        g = groups[k]
        best = {}
        for m in ("std", "proj"):
            cand = [r for r in g if r["method"] == m]
            if cand:
                best[m] = min(cand, key=lambda r: float(r["time_to_se_s"]))
        if len(best) < 2:
            continue
        rec = dict(zip(("family", "obs", "layers", "n", "gt"), k))
        bv = {}
        for m, r in best.items():
            rec[f"{m}_device"] = r["device"]
            rec[f"{m}_config"] = r["config"]
            rec[f"{m}_shots"] = r["shots"]
            rec[f"{m}_var"] = float(r["var"])
            rec[f"{m}_t_per_shot_s"] = float(r["t_per_shot_s"])
            rec[f"{m}_time_to_se_s"] = float(r["time_to_se_s"])
            rec[f"{m}_bso"] = r["batched_shots_optimization"]
            if r["exact_mean"] not in ("", None):
                rec[f"{m}_bias_sigma"] = (float(r["mean"]) - float(r["exact_mean"])) / float(r["se"])
            raw = os.path.join(args.root, r["raw_file"]) if r.get("raw_file") else ""
            if raw and os.path.exists(raw):
                bv[m] = boot_var(np.load(raw))
        vr = rec["std_var"] / rec["proj_var"]
        tr = rec["std_time_to_se_s"] / rec["proj_time_to_se_s"]
        rec["var_ratio"] = vr
        rec["time_ratio"] = tr
        if len(bv) == 2:
            r_b = bv["std"] / bv["proj"]
            c = rec["std_t_per_shot_s"] / rec["proj_t_per_shot_s"]
            rec["var_ratio_lo"], rec["var_ratio_hi"] = np.percentile(r_b, [2.5, 97.5])
            rec["time_ratio_lo"], rec["time_ratio_hi"] = np.percentile(r_b * c, [2.5, 97.5])
        out.append(rec)
    fields = []
    for r in out:
        for f in r:
            if f not in fields:
                fields.append(f)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    for r in out:
        lo, hi = r.get("var_ratio_lo", np.nan), r.get("var_ratio_hi", np.nan)
        tlo, thi = r.get("time_ratio_lo", np.nan), r.get("time_ratio_hi", np.nan)
        print(f"{r['family']:9} {r['obs']:2} L={r['layers']} n={r['n']:2} gt={r['gt']:<5} "
              f"var_ratio={r['var_ratio']:7.3f} [{lo:6.3f},{hi:6.3f}]  "
              f"time_ratio={r['time_ratio']:7.3f} [{tlo:6.3f},{thi:6.3f}]  "
              f"std={r['std_device']}/{r['std_config']} proj={r['proj_device']}/{r['proj_config']}")


if __name__ == "__main__":
    main()
