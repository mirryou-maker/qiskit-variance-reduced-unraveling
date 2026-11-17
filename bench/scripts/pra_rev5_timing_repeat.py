"""POST HOC (review 5, minor 12): timing scatter from repeated timings that already exist (no new runs).

The fixed-candidate timing runs (N_t = 296 trajectories, same seeds, same chunk, same v2+opt2 code) were repeated in
different sessions on the same GPU model:
  RTX 2060 (ubuntu94): E9e (b_e9e_u94.csv, 2026-11-11/12) vs E10 (b_e10_u94.csv, 2026-11-12/13)
  V100 (iREMB-E-147):  E9e secondary (b_e9e_v100.csv) vs E10-V100 (b_e10_v100.csv) vs E12 (b_e12_v100.csv)
One fixed run serves all observables of a configuration, so runs are de-duplicated on (configuration, candidate).
Reported: per-run ratio t_2/t_1 (median, IQR, 2.5-97.5% range) and the SD of log(t_2/t_1). If the two timings of a
run scatter independently with log-SD s, SD(log ratio) = s * sqrt(2); a per-group W = Var_f t_f / (Var_s t_s) then has a
timing log-SD of about s * sqrt(2) = SD(log ratio) (t_f and t_s timed separately). Reads CSVs only.
Usage: python pra_rev5_timing_repeat.py <project root>
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..")
R = os.path.join(ROOT, "bench", "results")
KEY = ["channel", "family", "variant", "L", "gt", "scheme", "seed"]


def fixed(f):
    d = pd.read_csv(os.path.join(R, f))
    d = d[d.kind == "fixed"]
    assert (d.shots == 296).all(), f
    t = d.groupby(KEY).wall_s.agg(["first", "nunique", "size"]).reset_index()
    return t.rename(columns={"first": "wall_s"}), d.gpu_other.astype(str).unique()


def compare(f1, f2, label):
    a, ia = fixed(f1)
    b, ib = fixed(f2)
    m = a.merge(b, on=KEY, suffixes=("_1", "_2"))
    r = m.wall_s_2 / m.wall_s_1
    lr = np.log(r)
    q = np.percentile(r, [2.5, 25, 50, 75, 97.5])
    print(f"{label}: {f1} vs {f2}")
    print(f"  unique fixed runs paired {len(m)} (of {len(a)} / {len(b)}); gpu_other {list(ia)} / {list(ib)}")
    print(f"  ratio t2/t1: median {q[2]:.4f}  IQR [{q[1]:.4f}, {q[3]:.4f}]  2.5-97.5% [{q[0]:.4f}, {q[4]:.4f}]"
          f"  min {r.min():.4f} max {r.max():.4f}")
    print(f"  SD(log ratio) {lr.std(ddof=1):.4f}  -> per-run timing log-SD s = {lr.std(ddof=1) / np.sqrt(2):.4f}")
    print(f"  |ratio - 1| > 0.05 in {(np.abs(r - 1) > 0.05).sum()} of {len(r)} runs")
    return r


print(__doc__.splitlines()[0])
compare("b_e9e_u94.csv", "b_e10_u94.csv", "RTX 2060, E9e vs E10")
compare("b_e9e_v100.csv", "b_e10_v100.csv", "V100, E9e vs E10-V100")
compare("b_e10_v100.csv", "b_e12_v100.csv", "V100, E10-V100 vs E12")
