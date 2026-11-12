"""P0-3 validation of src/noise/traj_ref.py.

Checks, for brickwork/ising x X/ZZ at n=8, L=8, gt in {0.01, 0.03, 0.1, 0.3}:
  1. mean of std and projector (lam=0/1) agrees with the dense exact mean (|z| < 4);
  2. variance agrees with the Aer stage-2 measurement (a5_entangled.csv, CPU plain rows)
     within the combined sampling error of the two variances;
  3. martingale identity: sum of per-event mean increment variances == sample variance.
Writes bench/results/p0_validate_ref.csv.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.noise import traj_ref as T  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
aer = {}
for r in csv.DictReader(open(os.path.join(ROOT, "bench/results/a5_entangled.csv"))):
    if r["device"] == "CPU" and int(r["n"]) == 8:
        aer[(r["family"], r["obs"], float(r["gt"]), r["method"])] = (float(r["var"]), int(r["shots"]))

n, L, shots = 8, 8, 40000
rows = []
for fam in ("brickwork", "ising"):
    for gt in (0.01, 0.03, 0.1, 0.3):
        ops = T.brickwork_ops(n, L, gt) if fam == "brickwork" else T.ising_ops(n, L, gt)
        for obs in ("x", "zz"):
            ex = T.exact_mean(ops, n, obs)
            for meth, lam in (("std", 0.0), ("proj", 1.0)):
                r = T.run(ops, n, [obs], shots, ("lam", lam), seed=101, decompose=True)
                x = r[obs]
                mean, var = x.mean(), x.var(ddof=1)
                z = (mean - ex) / np.sqrt(var / shots)
                dec = r["_dec"]["used"].sum()
                av, an = aer.get((fam, obs, gt, meth), (np.nan, 0))
                # sampling sd of a variance estimate ~ var*sqrt(2/(N-1)) (Gaussian approx;
                # kurtosis-corrected value from the data below)
                m4 = np.mean((x - mean) ** 4)
                sd_v = np.sqrt((m4 - var ** 2) / shots)
                sd_a = sd_v / var * av * np.sqrt(shots / max(an, 1))
                zv = (var - av) / np.sqrt(sd_v ** 2 + sd_a ** 2)
                row = dict(family=fam, obs=obs, gt=gt, method=meth, shots=shots, exact=ex,
                           mean=mean, z_mean=z, var=var, var_aer=av, shots_aer=an, z_var=zv,
                           dec_sum=dec, dec_rel_err=(dec - var) / var)
                rows.append(row)
                print(f"{fam:9} {obs:2} gt={gt:<5} {meth:4} mean={mean:+.5f} ex={ex:+.5f} z={z:+.2f} "
                      f"var={var:.5f} aer={av:.5f} zv={zv:+.2f} dec/var-1={(dec - var) / var:+.3f}",
                      flush=True)
out = os.path.join(ROOT, "bench/results/p0_validate_ref.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("max|z_mean|", max(abs(r["z_mean"]) for r in rows), "max|z_var|",
      max(abs(r["z_var"]) for r in rows), "max|dec_rel|", max(abs(r["dec_rel_err"]) for r in rows))
