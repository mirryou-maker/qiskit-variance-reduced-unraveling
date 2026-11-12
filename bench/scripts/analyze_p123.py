"""Summaries for P1/P2 (p12_scan.csv) and P3 (p3_adapt.csv, p3_fixed.csv).

Usage: analyze_p123.py [--p12 CSV ...] [--adapt CSV ...] [--fixed CSV ...]
Prints plain-text tables; numbers quoted in reports must come from this output.
"""
import argparse
import collections
import csv
import math

import numpy as np

VMIN = 1e-10  # variances below this are numerically zero (deterministic estimator)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load(paths):
    rows = []
    for p in paths or []:
        rows += list(csv.DictReader(open(p)))
    return rows


def p12(rows):
    print(f"== P1/P2: {len(rows)} rows (config x observable)")
    zs = [abs(f(r[k])) for r in rows for k, v in (("z_std", "var_std"), ("z_proj", "var_proj"))
          if math.isfinite(f(r[k])) and f(r[v]) >= VMIN]
    print(f"bias (var >= {VMIN:g} only): {len(zs)} tests, max|z|={max(zs):.2f}, "
          f"n(|z|>4)={sum(z > 4 for z in zs)}, n(|z|>3)={sum(z > 3 for z in zs)}")
    # deterministic estimators: check they sit on the exact mean
    dev = [abs(f(r[m]) - f(r["exact"])) for r in rows for m, v in (("mean_std", "var_std"),
           ("mean_proj", "var_proj")) if f(r[v]) < VMIN]
    print(f"deterministic estimators (var < {VMIN:g}): {len(dev)}, max|mean-exact|="
          f"{max(dev, default=math.nan):.2e}")
    nstd0 = [r for r in rows if f(r["var_std"]) < VMIN and f(r["var_proj"]) >= VMIN]
    print(f"std deterministic but proj not: {len(nstd0)} rows; max var_proj there = "
          f"{max((f(r['var_proj']) for r in nstd0), default=math.nan):.3g}")
    # G1: sign accuracy where the measured ratio is significantly != 1
    for key in ("R_traj", "R_ideal"):
        tot = ok = 0
        lr, lp = [], []
        for r in rows:
            lo, hi, R, P = f(r["ratio_lo"]), f(r["ratio_hi"]), f(r["ratio"]), f(r[key])
            if not (math.isfinite(lo) and math.isfinite(P) and math.isfinite(R)) or R <= 0 or P <= 0:
                continue
            if lo > 1 or hi < 1:
                tot += 1
                ok += (R > 1) == (P > 1)
                lr.append(math.log(R))
                lp.append(math.log(P))
        rho = np.corrcoef(np.argsort(np.argsort(lr)), np.argsort(np.argsort(lp)))[0, 1] if lr else math.nan
        print(f"G1 {key}: sign agreement {ok}/{tot} = {ok / max(tot, 1):.3f}; Spearman(log) = {rho:.3f}")
    # degenerate rows
    inf = sum(1 for r in rows if f(r["ratio"]) == math.inf)
    zero = sum(1 for r in rows if f(r["ratio"]) == 0)
    nan = sum(1 for r in rows if not math.isfinite(f(r["ratio"])) and f(r["ratio"]) != math.inf)
    print(f"degenerate: ratio=inf (std>0, proj var 0) {inf}; ratio=0 (std var 0) {zero}; undefined {nan}")
    # advantage map
    print("\nadvantage map: per (family, variant, obs): #better (CI>1) / #worse (CI<1) / #n.s. / "
          "median ratio ; realistic gt<=0.1 in brackets")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["family"], r["variant"], r["n"], r["obs"])].append(r)
    for k in sorted(g):
        rr = g[k]

        def cnt(sel):
            b = sum(1 for r in sel if f(r["ratio_lo"]) > 1 or f(r["ratio"]) == math.inf)
            wv = sum(1 for r in sel if f(r["ratio_hi"]) < 1 or f(r["ratio"]) == 0)
            return b, wv, len(sel) - b - wv
        b, wv, ns = cnt(rr)
        rb, rw, rn = cnt([r for r in rr if f(r["gt"]) <= 0.1])
        med = np.nanmedian([f(r["ratio"]) for r in rr if math.isfinite(f(r["ratio"]))] or [math.nan])
        print(f"  {k[0]:9} {k[1]:8} n={k[2]:2} {k[3]:4}: {b:2}/{wv:2}/{ns:2}  med={med:7.3f}   "
              f"[{rb}/{rw}/{rn}]")
    # H2: ising x, ratio vs gt for each h, L
    print("\nH2 ising obs=x ratio by (h, L) over gt:")
    gts = sorted({f(r["gt"]) for r in rows})
    print("  h      L  " + " ".join(f"{g_:>7}" for g_ in gts))
    tab = {(r["variant"], int(r["L"]), f(r["gt"])): f(r["ratio"]) for r in rows
           if r["family"] == "ising" and r["obs"] == "x" and r["n"] == "8"}
    for h in sorted({k[0] for k in tab}, key=lambda s: float(s[1:])):
        for L in sorted({k[1] for k in tab}):
            print(f"  {h:6} {L:2}  " + " ".join(f"{tab.get((h, L, g_), math.nan):7.2f}" for g_ in gts))





def adapt(rows):
    print(f"\n== P3 adapt: {len(rows)} rows")
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["family"], r["variant"], r["n"], r["L"], r["gt"], r["obs"])][r["scheme"]] = r
    degen = [k for k, d in by.items() if "std" in d and f(d["std"]["var"]) < VMIN]
    print(f"degenerate groups (Var_std < {VMIN:g}; excluded): {len(degen)} "
          f"-> PI var max there = {max((f(by[k]['pi']['var']) for k in degen if 'pi' in by[k]), default=math.nan):.2e}, "
          f"proj var max there = {max((f(by[k]['proj']['var']) for k in degen if 'proj' in by[k]), default=math.nan):.2e}")
    for k in degen:
        del by[k]
    zs = [abs(f(d[s]["z"])) for d in by.values() for s in d
          if s != "std" and f(d[s]["var"]) >= VMIN and math.isfinite(f(d[s]["z"]))]
    print(f"bias (non-std schemes, non-degenerate): {len(zs)} tests, max|z|={max(zs):.2f}, "
          f"n(|z|>3)={sum(z > 3 for z in zs)}, n(|z|>4)={sum(z > 4 for z in zs)}")
    print(f"non-degenerate groups: {len(by)}")
    for sch in ("proj", "greedy", "pi"):
        rr = [d[sch] for d in by.values() if sch in d and math.isfinite(f(d[sch]["ratio"]))
              and f(d[sch]["ratio"]) > 0]
        better = sum(f(r["ratio_lo"]) > 1 for r in rr)
        worse = sum(f(r["ratio_hi"]) < 1 for r in rr)
        rat = np.array([f(r["ratio"]) for r in rr])
        print(f"  {sch:6}: n={len(rr)} better(CI>1)={better} worse(CI<1)={worse} "
              f"median={np.median(rat):.3f} min={rat.min():.3f} max={rat.max():.3f} "
              f"geo-mean={np.exp(np.mean(np.log(rat))):.3f}")
    # PI vs best-of fixed (std, proj)
    gains, rel = [], []
    for d in by.values():
        if all(s in d for s in ("std", "proj", "pi")):
            rp, ri = f(d["proj"]["ratio"]), f(d["pi"]["ratio"])
            if math.isfinite(rp) and math.isfinite(ri):
                best = max(1.0, rp)
                rel.append(ri / best)
                gains.append(ri)
    rel = np.array(rel)
    print(f"  PI / best-of(std, proj): median={np.median(rel):.3f}, frac>1.05={np.mean(rel > 1.05):.3f}, "
          f"frac<0.95={np.mean(rel < 0.95):.3f}")
    print("  per (family, variant, obs): PI median ratio [gt<=0.1 median]  | greedy median | proj median")
    g = collections.defaultdict(list)
    for k, d in by.items():
        g[(k[0], k[1], k[5])].append((f(k[4]), d))
    for k in sorted(g):
        def med(s, sel=lambda gt: True):
            v = [f(d[s]["ratio"]) for gt, d in g[k] if s in d and sel(gt) and math.isfinite(f(d[s]["ratio"]))]
            return np.median(v) if v else math.nan
        print(f"    {k[0]:9} {k[1]:8} {k[2]:3}: PI {med('pi'):6.3f} [{med('pi', lambda gt: gt <= 0.1):6.3f}]"
              f" | gr {med('greedy'):6.3f} | proj {med('proj'):6.3f}")


def fixed(rows):
    print(f"\n== P3 fixed: {len(rows)} rows")
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["family"], r["variant"], r["gt"], r["obs"])].append(r)
    print("  (family variant gt obs): best scheme/param ratio | lam=1 ratio")
    for k in sorted(by):
        rr = [r for r in by[k] if math.isfinite(f(r["ratio"]))]
        if not rr:
            continue
        best = max(rr, key=lambda r: f(r["ratio"]))
        l1 = [r for r in rr if r["scheme"] == "lam" and f(r["param"]) == 1.0]
        print(f"    {k[0]:9} {k[1]:6} {k[2]:5} {k[3]:4}: {best['scheme']}={best['param']:<8} "
              f"{f(best['ratio']):6.3f} [{f(best['ratio_lo']):.3f},{f(best['ratio_hi']):.3f}] | "
              f"lam1 {f(l1[0]['ratio']) if l1 else math.nan:6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p12", nargs="*")
    ap.add_argument("--adapt", nargs="*")
    ap.add_argument("--fixed", nargs="*")
    a = ap.parse_args()
    if a.p12:
        p12(load(a.p12))
    if a.adapt:
        adapt(load(a.adapt))
    if a.fixed:
        fixed(load(a.fixed))
