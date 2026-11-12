"""P1 + P2: mechanism check (H1) and advantage-region map, reference sampler (src/noise/traj_ref.py).

For each configuration (circuit family/variant, n, L, gt) and EVERY observable in OBS_ALL:
  * std run (lam=0) with exact martingale decomposition -> Var_std, and the H1 cross-measure
    predictor of Var_proj (projector increment formula on std-distributed states);
  * projector run (lam=1) -> Var_proj;
  * exact mean (dense backward pass) for bias checks; noiseless-trajectory predictor R_ideal.
Ratio = Var_std / Var_proj (>1: projector better) with percentile bootstrap 95% CI.
Ledger-style: rows already in --out are skipped. Raw per-trajectory values -> --raw-dir (npz).

Usage: p12_scan.py --grid main|n10 [--shots 20000] [--out ...] [--raw-dir ...]
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

GTS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LS = (1, 2, 4, 8, 16)
HS = (0.0, 0.35, 0.7, 1.4, 3.0)


def grid(name):
    if name == "main":
        n = 8
        for L in LS:
            for gt in GTS:
                for h in HS:
                    yield ("ising", f"h{h:g}", n, L, gt)
                yield ("brickwork", "random", n, L, gt)
                yield ("brickwork", "clifford", n, L, gt)
    elif name == "n10":
        n = 10
        for L in (4, 8):
            for gt in GTS:
                yield ("ising", "h0.7", n, L, gt)
                yield ("brickwork", "random", n, L, gt)
    else:
        raise ValueError(name)


def build(fam, var, n, L, gt):
    if fam == "ising":
        return T.ising_ops(n, L, gt, h=float(var[1:]))
    return T.brickwork_ops(n, L, gt, clifford=(var == "clifford"))


def boot_ratio(xs, xp_, B=500, seed=1):
    rng = np.random.default_rng(seed)
    N1, N2 = len(xs), len(xp_)
    r = np.empty(B)
    for b in range(B):
        a = xs[rng.integers(0, N1, N1)]
        c = xp_[rng.integers(0, N2, N2)]
        r[b] = a.var(ddof=1) / max(c.var(ddof=1), 1e-300)
    return np.percentile(r, [2.5, 97.5])


FIELDS = ["family", "variant", "n", "L", "gt", "obs", "shots", "exact", "mean_std", "mean_proj",
          "z_std", "z_proj", "var_std", "var_proj", "ratio", "ratio_lo", "ratio_hi",
          "dec_std", "pred_proj", "R_traj", "R_ideal", "A2_sum", "VB_sum", "seed", "wall_s",
          "timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/p12_scan.csv"))
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "bench/results/p12_raw"))
    a = ap.parse_args()
    os.makedirs(a.raw_dir, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"])))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    # n=10 dense back-propagated observables cost 8 MB x events x obs of GPU memory;
    # keep that grid to four observables so it fits next to the concurrent P3 job.
    obs = list(T.OBS_ALL) if a.grid == "main" else ["x", "zz", "xc", "zc"]
    for cfg in grid(a.grid):
        if cfg in done:
            continue
        fam, var, n, L, gt = cfg
        seed = int(hashlib.md5(repr(cfg).encode()).hexdigest()[:8], 16)
        t0 = time.time()
        ops = build(*cfg)
        dec = T.decompose_multi(ops, n, obs, a.shots, lam=0.0, seed=seed)
        pr = T.run(ops, n, obs, a.shots, ("lam", 1.0), seed=seed + 1)
        raw = {}
        rows = []
        for o in obs:
            xs, xq = dec[o]["est"], pr[o]
            raw[f"{o}_std"] = xs.astype(np.float32)
            raw[f"{o}_proj"] = xq.astype(np.float32)
            ex = T.exact_mean(ops, n, o)
            vs, vq = xs.var(ddof=1), xq.var(ddof=1)
            if max(vs, vq) < 1e-14:  # both estimators deterministic: ratio undefined
                vs = vq = 0.0
            lo, hi = boot_ratio(xs, xq) if max(vs, vq) > 0 else (np.nan, np.nan)
            ds, dq = dec[o]["std"].sum(), dec[o]["proj"].sum()
            si, qi = T.ideal_predictor(ops, n, o)
            rows.append(dict(
                family=fam, variant=var, n=n, L=L, gt=gt, obs=o, shots=a.shots, exact=ex,
                mean_std=xs.mean(), mean_proj=xq.mean(),
                z_std=(xs.mean() - ex) / np.sqrt(max(vs, 1e-300) / len(xs)),
                z_proj=(xq.mean() - ex) / np.sqrt(max(vq, 1e-300) / len(xq)),
                var_std=vs, var_proj=vq,
                ratio=(vs / vq if vq > 0 else (np.inf if vs > 0 else np.nan)),
                ratio_lo=lo, ratio_hi=hi,
                dec_std=ds, pred_proj=dq, R_traj=ds / max(dq, 1e-300),
                R_ideal=si / max(qi, 1e-300) if qi > 0 else np.nan,
                A2_sum=dec[o]["A2"].sum(), VB_sum=dec[o]["VB"].sum(), seed=seed))
        wall = time.time() - t0
        ts = dt.datetime.now().isoformat(timespec="seconds")
        for r in rows:
            r["wall_s"] = round(wall, 2)
            r["timestamp"] = ts
            w.writerow(r)
        f.flush()
        np.savez_compressed(os.path.join(a.raw_dir, f"{fam}_{var}_n{n}_L{L}_gt{gt:g}.npz"), **raw)
        rx = {r["obs"]: r["ratio"] for r in rows}
        print(f"{fam:9} {var:8} n={n} L={L:2} gt={gt:<5} wall={wall:6.1f}s  ratio "
              + " ".join(f"{o}={rx[o]:.2f}" for o in ("x", "zz", "z", "xc") if o in rx), flush=True)
    f.close()


if __name__ == "__main__":
    main()
