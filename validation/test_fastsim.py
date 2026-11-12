"""Validate src/noise/fastsim.py against src/noise/traj_ref.py.

1. Noiseless (gt=0): every trajectory equals the reference statevector value (|diff| < 1e-10),
   n = 8 (one group) and n = 13 (three groups), both circuit families, all observables.
2. Noisy, std and a static random table: mean vs exact (n=8) and vs reference sampler (n=12),
   variance ratio fastsim/reference within sampling error.
Usage: python validation/test_fastsim.py   (prints PASS/FAIL lines; exit code 1 on failure)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.noise import fastsim as F  # noqa: E402
from src.noise import traj_ref as T  # noqa: E402

OBS = ("x", "zz", "xc", "zc")
fail = 0


def check(cond, msg):
    global fail
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    fail += 0 if cond else 1


def ref_ops(fam, n, L, gt, h=0.7):
    return T.ising_ops(n, L, gt, h=h) if fam == "ising" else T.brickwork_ops(n, L, gt)


def fs_st(fam, n, L, gt, h=0.7):
    return F.ising_circuit(n, L, gt, h=h) if fam == "ising" else F.brickwork_circuit(n, L, gt)


# 1. noiseless
for n in (8, 13):
    sim = F.FastSim(n)
    for fam in ("ising", "brickwork"):
        ops, st = ref_ops(fam, n, 3, 0.0), fs_st(fam, n, 3, 0.0)
        for o in OBS:
            ref = T.run(ops, n, [o], 1, ("lam", 0.0), seed=0)[o][0]
            x = sim.run(st, o, 4, seed=1)
            check(np.max(np.abs(x - ref)) < 1e-10, f"noiseless n={n} {fam} {o}: max|diff|={np.max(np.abs(x - ref)):.2e}")

# 2. noisy
rng = np.random.default_rng(5)
for n, L, gt in ((8, 4, 0.1), (12, 3, 0.1)):
    sim = F.FastSim(n)
    for fam in ("ising", "brickwork"):
        ops, st = ref_ops(fam, n, L, gt), fs_st(fam, n, L, gt)
        ne = F.n_events(st, n)
        for tname, tab in (("std", np.zeros(ne)), ("rand", (rng.random(ne) < 0.5).astype(float)),
                           ("proj", np.ones(ne))):
            shots = 20000 if n == 8 else 8000
            for o in OBS:
                x = sim.run(st, o, shots, table=tab, seed=11)
                r = T.run(ops, n, [o], shots, ("table", tab), seed=12)[o]
                if n == 8:
                    ex = T.exact_mean(ops, n, o)
                    z = (x.mean() - ex) / np.sqrt(max(x.var(ddof=1), 1e-300) / len(x))
                else:
                    z = (x.mean() - r.mean()) / np.sqrt(max(x.var(ddof=1) / len(x) + r.var(ddof=1) / len(r), 1e-300))
                vx, vr = x.var(ddof=1), r.var(ddof=1)
                if max(vx, vr) < 1e-14:
                    check(abs(x.mean() - r.mean()) < 1e-8, f"n={n} {fam} {tname} {o}: deterministic, means agree")
                    continue
                # var ratio sampling sd ~ sqrt(2/N * kurtosis-ish); accept 5 sd with sd from bootstrap
                bs = np.random.default_rng(0)
                rr = [x[bs.integers(0, len(x), len(x))].var(ddof=1) / max(r[bs.integers(0, len(r), len(r))].var(ddof=1), 1e-300)
                      for _ in range(200)]
                lo, hi = np.percentile(rr, [0.5, 99.5])
                check(abs(z) < 4.0 and lo < 1.0 < hi,
                      f"n={n} {fam} {tname} {o}: z={z:+.2f} var ratio={vx / max(vr, 1e-300):.3f} [{lo:.3f},{hi:.3f}]")
print("TOTAL FAIL:", fail)
sys.exit(1 if fail else 0)
