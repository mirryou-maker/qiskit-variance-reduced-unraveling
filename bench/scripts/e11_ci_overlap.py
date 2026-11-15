"""POST HOC (E11 deviation (d), reproduced for review 3, 2026-11-14): which E11 timed runs overlapped a CI job on ubuntu94,
and does excluding the affected configurations change the amplitude-damping Q1 / Q2 results?

CI jobs: GitHub Actions self-hosted runners on ubuntu94 (~/actions-runner-*/_diag/Worker_*.log). One Worker log per job;
  job interval = [first, last] log time stamp (UTC in the log, converted to KST = UTC + 9 h, the zone of the E11 ledger).
Timed runs: rows of b_e11_u94.csv, one per (configuration, kind, scheme), and per target observable for the single-target
  tables (a fixed or multi run gives all observables at once and has one ledger row per TN observable).
  The ledger time stamp is written right after the run; run window = [timestamp - wall_s - build_s - 1 s, timestamp]
  (the 1 s covers the whole-second rounding of the time stamp; build_s = 0 for fixed runs).
  The runner's idle check (before every run) waits while a Runner.Worker process exists, so a run can only overlap a CI
  job that STARTED during the run; both "CI start inside the window" and "any intersection" are counted.
Per-trajectory time check: for every overlapped run, wall/shots is compared with the same quantity of the non-overlapped
  runs of the same configuration that have the same per-trajectory work (fixed: same candidate is unique, so the
  comparison is with candidate 0 for static runs, and static / multi runs with each other).
Sensitivity (POST HOC): drop every configuration with an overlapped run; Q1 ad median single_W_bf with the decision-line
  bootstrap of analyze_e11 (default_rng(7), B = 2000, 98.75%); Q2 ad median W_multi (A20k/all) with default_rng(20261113),
  B = 4000, 98.75%.
Usage (ubuntu94, env sci): python bench/scripts/e11_ci_overlap.py > bench/results/e11_ci_overlap.txt
Outputs: bench/results/e11_ci_overlap.txt (stdout), e11_ci_overlap.csv (one row per timed run).
"""
import datetime as dt
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench", "results")
CFG = ["channel", "family", "variant", "n", "L", "gt"]
LO, HI = 0.625, 99.375
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)Z ")


def med_ci(x, rng, B):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), *np.percentile(b, [LO, HI])


jobs = []
for f in sorted(glob.glob(os.path.expanduser("~/actions-runner-*/_diag/Worker_*.log"))):
    first = last = None
    with open(f, errors="replace") as fh:
        for line in fh:
            m = TS.match(line)
            if m:
                t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") + dt.timedelta(hours=9)
                first = first or t
                last = t
    if first:
        jobs.append((os.path.basename(os.path.dirname(os.path.dirname(f))), os.path.basename(f), first, last))
J = pd.DataFrame(jobs, columns=["runner", "log", "start", "end"])

d = pd.read_csv(os.path.join(R, "b_e11_u94.csv"))
d["runkey"] = np.where(d.kind == "single", d.obs.astype(str), "")      # fixed and multi rows: one run gives all obs
runs = d.drop_duplicates(CFG + ["kind", "scheme", "runkey"], keep="last").copy()
runs["build_s"] = pd.to_numeric(runs.build_s, errors="coerce").fillna(0.0)
runs["t_end"] = pd.to_datetime(runs.timestamp)
runs["t_start"] = runs.t_end - pd.to_timedelta(runs.wall_s + runs.build_s + 1.0, unit="s")
lo_t, hi_t = runs.t_start.min(), runs.t_end.max()
Jw = J[(J.end >= lo_t) & (J.start <= hi_t)]
print(f"E11 main-run ledger: {len(runs)} timed runs in {runs[CFG].drop_duplicates().shape[0]} configurations, "
      f"{lo_t} .. {hi_t} (KST)")
print(f"CI Worker logs found: {len(J)}; CI jobs intersecting the main-run period: {len(Jw)}")
for r in Jw.itertuples():
    print(f"  {r.runner:28s} {r.log}  {r.start} .. {r.end}")
st, inter, which = [], [], []
for r in runs.itertuples():
    s_in = Jw[(Jw.start >= r.t_start) & (Jw.start <= r.t_end)]
    a_in = Jw[(Jw.start <= r.t_end) & (Jw.end >= r.t_start)]
    st.append(len(s_in) > 0)
    inter.append(len(a_in) > 0)
    which.append(";".join(a_in.log))
runs["ci_start_inside"], runs["ci_intersect"], runs["ci_logs"] = st, inter, which
runs["tau"] = runs.wall_s / runs.shots
runs["cfgkey"] = runs[CFG].astype(str).agg("|".join, axis=1)
ov = runs[runs.ci_start_inside]
ovc = ov[CFG].drop_duplicates()
print(f"\ntimed runs with a CI job starting during the run: {len(ov)} of {len(runs)}, in {len(ovc)} configurations; "
      f"runs intersecting a CI job at all: {int(runs.ci_intersect.sum())}")
for slack in (5,):     # diagnostic: window widened by 5 s on both sides (covers the runner's 2 s pause before sampling)
    hit = [((Jw.start <= r.t_end + pd.Timedelta(seconds=slack)) & (Jw.end >= r.t_start - pd.Timedelta(seconds=slack))).any()
           for r in runs.itertuples()]
    hs = runs[hit]
    print(f"  diagnostic, window widened by {slack} s: runs intersecting a CI job {len(hs)}, in "
          f"{hs[CFG].drop_duplicates().shape[0]} configurations (same configurations: "
          f"{set(map(tuple, hs[CFG].drop_duplicates().astype(str).to_numpy())) == set(map(tuple, ovc.astype(str).to_numpy()))})")
for r in ov.itertuples():
    same = runs[(runs.cfgkey == r.cfgkey) & ~runs.ci_start_inside]
    if r.kind == "fixed":
        refs = same[(same.kind != "fixed")]
    else:
        refs = same[(same.kind != "fixed") | (same.scheme.isin(["std", "jump"]))]
    rel = r.tau / refs.tau.median() - 1 if len(refs) else np.nan
    print(f"  {r.channel} {r.family} {r.variant} L{r.L} gt{r.gt} {r.kind:6s} {str(r.scheme):9s} obs={r.obs:3s} "
          f"window {r.t_start} .. {r.t_end}  tau {r.tau:.5f} s, vs same-config reference runs {rel:+.4%}")
runs.drop(columns=["t_start", "t_end"]).to_csv(os.path.join(R, "e11_ci_overlap.csv"), index=False)

S = pd.read_csv(os.path.join(R, "b_e11_summary.csv"))
M = pd.read_csv(os.path.join(R, "b_e11_multi.csv"))
M = M[(M.edef == "A20k") & (M.subset == "all")]
bad = set(map(tuple, ovc.astype(str).to_numpy()))
keep = lambda D: D[[tuple(x) not in bad for x in D[CFG].astype(str).to_numpy()]]
print("\nPOST HOC sensitivity, configurations with an overlapped run excluded:")
for ch in ("ad", "deph"):
    q1 = keep(S[S.channel == ch])
    q2 = keep(M[M.channel == ch])
    a = med_ci(q1.single_W_bf, np.random.default_rng(7), 2000)
    b = med_ci(q2.W_multi, np.random.default_rng(20261113), 4000)
    print(f"  {ch:4s} Q1 median single_W_bf {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] ({len(q1)} groups, "
          f"{q1[CFG].drop_duplicates().shape[0]} configs);  Q2 median W_multi {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] "
          f"({len(q2)} configs)")
