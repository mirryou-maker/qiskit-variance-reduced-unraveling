"""P3: adaptive / optimised unravelings with the reference sampler.

--part fixed  (n=8): global lam in {0, .25, .5, .75, 1} and weak-measurement theta in
               {pi/8, pi/4, 3pi/8}; every observable measured from the same trajectories.
--part adapt  (n=5 default, --n): per observable, schemes
               std, proj, greedy (local increment rule A^2 > V_B), PI (policy improvement
               with exact std continuation; theorem: Var_PI <= Var_std).
All schemes are unbiased (choices depend only on the past); checked against the exact mean.
Ratio columns are Var_std / Var_scheme (>1 better than std), bootstrap 95% CI.
Ledger-style CSV (rows already present are skipped).
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import traj_ref as T  # noqa: E402


def build(fam, var, n, L, gt):
    if fam == "ising":
        return T.ising_ops(n, L, gt, h=float(var[1:]))
    return T.brickwork_ops(n, L, gt, clifford=(var == "clifford"))


def boot(xs, xq, B=500, seed=2):
    rng = np.random.default_rng(seed)
    r = [xs[rng.integers(0, len(xs), len(xs))].var(ddof=1)
         / max(xq[rng.integers(0, len(xq), len(xq))].var(ddof=1), 1e-300) for _ in range(B)]
    return np.percentile(r, [2.5, 97.5])


FIELDS = ["part", "family", "variant", "n", "L", "gt", "obs", "scheme", "param", "shots",
          "exact", "mean", "z", "var", "var_std", "ratio", "ratio_lo", "ratio_hi",
          "frac_proj", "seed", "wall_s", "timestamp"]


def seed_of(*k):
    return int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["fixed", "adapt"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, f"bench/results/p3_{a.part}.csv")
    done = set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            done.add((r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]),
                      r["obs"], r["scheme"], r["param"]))
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    def emit(rows):
        ts = dt.datetime.now().isoformat(timespec="seconds")
        for r in rows:
            r["timestamp"] = ts
            w.writerow(r)
        f.flush()

    if a.part == "fixed":
        n = a.n or 8
        obs = list(T.OBS_ALL)
        schemes = [("lam", x) for x in (0.0, 0.25, 0.5, 0.75, 1.0)] + \
                  [("weak", x) for x in (np.pi / 8, np.pi / 4, 3 * np.pi / 8)]
        for fam, var in (("brickwork", "random"), ("ising", "h0.7")):
            for gt in (0.01, 0.03, 0.1, 0.3):
                L = 8
                ops = build(fam, var, n, L, gt)
                exact = {o: T.exact_mean(ops, n, o) for o in obs}
                base = None
                for sch in schemes:
                    key_param = f"{sch[1]:.6g}"
                    if all((fam, var, n, L, gt, o, sch[0], key_param) in done for o in obs) \
                            and sch != ("lam", 0.0):
                        continue
                    t0 = time.time()
                    sd = seed_of(fam, var, n, L, gt, sch)
                    res = T.run(ops, n, obs, a.shots, sch, seed=sd)
                    if sch == ("lam", 0.0):
                        base = res
                    rows = []
                    for o in obs:
                        x = res[o]
                        xs = base[o]
                        v, vs = x.var(ddof=1), xs.var(ddof=1)
                        lo, hi = boot(xs, x) if v > 1e-14 else (np.nan, np.nan)
                        rows.append(dict(part="fixed", family=fam, variant=var, n=n, L=L, gt=gt,
                                         obs=o, scheme=sch[0], param=key_param, shots=a.shots,
                                         exact=exact[o], mean=x.mean(),
                                         z=(x.mean() - exact[o]) / np.sqrt(max(v, 1e-300) / len(x)),
                                         var=v, var_std=vs, ratio=vs / v if v > 1e-14 else np.nan,
                                         ratio_lo=lo, ratio_hi=hi, frac_proj="", seed=sd,
                                         wall_s=round(time.time() - t0, 2)))
                    if (fam, var, n, L, gt, obs[0], sch[0], key_param) not in done:
                        emit(rows)
                    print(f"fixed {fam:9} {var:6} gt={gt:<5} {sch[0]:4} {sch[1]:.3f} "
                          + " ".join(f"{r['obs']}={r['ratio']:.2f}" for r in rows), flush=True)
    else:
        n = a.n or 5
        variants = [("brickwork", "random"), ("brickwork", "clifford"), ("ising", "h0"),
                    ("ising", "h0.7"), ("ising", "h1.4"), ("ising", "h3")]
        obs_list = ("x", "zz", "xc", "zc")
        Ls = (2, 4, 8) if n <= 5 else (2, 4)
        for fam, var in variants:
            for L in Ls:
                for gt in (0.01, 0.03, 0.1, 0.3):
                    ops = build(fam, var, n, L, gt)
                    t0 = time.time()
                    sd = seed_of(fam, var, n, L, gt)
                    base = T.run(ops, n, list(obs_list), a.shots, ("lam", 0.0), seed=sd)
                    prj = T.run(ops, n, list(obs_list), a.shots, ("lam", 1.0), seed=sd + 1)
                    for o in obs_list:
                        if (fam, var, n, L, gt, o, "pi", "") in done:
                            continue
                        ex = T.exact_mean(ops, n, o)
                        gr = T.run(ops, n, [o], a.shots, ("greedy", None), seed=sd + 2,
                                   obs_for_policy=o)
                        pi_x, pi_f = T.run_pi(ops, n, o, a.shots, seed=sd + 3)
                        xs = base[o]
                        vs = xs.var(ddof=1)
                        rows = []
                        for sch, x, fr in (("std", xs, 0.0), ("proj", prj[o], 1.0),
                                           ("greedy", gr[o], float(np.mean(gr["_frac_proj"]))),
                                           ("pi", pi_x, float(np.mean(pi_f)))):
                            v = x.var(ddof=1)
                            ok = v > 1e-14 and vs > 1e-14
                            lo, hi = boot(xs, x) if ok and sch != "std" else (np.nan, np.nan)
                            rows.append(dict(part="adapt", family=fam, variant=var, n=n, L=L,
                                             gt=gt, obs=o, scheme=sch, param="", shots=a.shots,
                                             exact=ex, mean=x.mean(),
                                             z=(x.mean() - ex) / np.sqrt(max(v, 1e-300) / len(x)),
                                             var=v, var_std=vs,
                                             ratio=(vs / v if v > 1e-14 else
                                                    (np.inf if vs > 1e-14 else np.nan)),
                                             ratio_lo=lo, ratio_hi=hi, frac_proj=fr, seed=sd,
                                             wall_s=round(time.time() - t0, 2)))
                        emit(rows)
                        print(f"adapt n={n} {fam:9} {var:8} L={L} gt={gt:<5} {o:3} "
                              + " ".join(f"{r['scheme']}={r['ratio']:.2f}" for r in rows[1:])
                              + f" fracPI={rows[3]['frac_proj']:.2f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
