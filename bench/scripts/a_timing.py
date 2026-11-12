"""Plan A step 3: wall-clock of the native sampler (std vs static continuation table) at n=20/24.

Held-out runs: the policy hyper-parameters (--abs-frac, --reps) are frozen from the n=8 tuning set
(bench/scripts/a_tune.py) BEFORE this script is run. Aer std (batched_shots_gpu) is timed in the
same job with bench_a5_aer_walltime.py (separate CSV) on the same GPU.

Per (family, n, gt, obs): idle state; table construction time (setup_s); ours_std and ours_table
runs with a wall-clock budget each; per-shot time, variance, time-to-target-SE
    T(eps) = setup_s + Var * t_per_shot / eps^2       (setup_s = 0 for ours_std)
Raw per-trajectory values -> --raw-dir (npz) for later bootstrap. Ledger-style CSV.
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import cupy as cp  # noqa: E402

from src.noise import fastsim as F  # noqa: E402

FIELDS = ["family", "variant", "n", "L", "gt", "obs", "method", "shots", "wall_s", "t_per_shot_s",
          "mean", "var", "se", "setup_s", "frac_table", "proj_events_per_shot", "target_se",
          "time_to_se_s", "reps", "abs_frac", "gpu_name", "gpu_util_pct_before",
          "gpu_mem_mib_before", "loadavg1_before", "host", "timestamp", "raw_file"]


def idle_state():
    util = mem = ""
    try:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        util, mem = [s.strip() for s in out.split(",")[:2]]
    except Exception:  # noqa: BLE001
        pass
    return util, mem, f"{os.getloadavg()[0]:.2f}"


def timed_run(sim, st, obs, table, budget_s, min_shots, max_shots, seed):
    """Pilot one batch for the per-shot time, then run N shots (timed, synchronised)."""
    pilot = max(1, min(sim.B, min_shots))
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    sim.run(st, obs, pilot, table=table, seed=seed + 999)
    cp.cuda.Device().synchronize()
    tps = (time.perf_counter() - t0) / pilot
    N = int(np.clip(budget_s / max(tps, 1e-9), min_shots, max_shots))
    N = max(sim.B, (N // sim.B) * sim.B)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    x, stats = sim.run(st, obs, N, table=table, seed=seed, return_stats=True)
    cp.cuda.Device().synchronize()
    wall = time.perf_counter() - t0
    return x, wall, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["ising", "brickwork"])
    ap.add_argument("--h", type=float, default=0.7)
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--gt", type=float, nargs="+", required=True)
    ap.add_argument("--obs", nargs="+", default=["x", "zz", "xc", "zc"])
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--abs-frac", type=float, required=True)
    ap.add_argument("--budget-s", type=float, default=60.0)
    ap.add_argument("--min-shots", type=int, default=1000)
    ap.add_argument("--max-shots", type=int, default=200000)
    ap.add_argument("--target-se", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/a_timing.csv"))
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "bench/results/a_timing_raw"))
    a = ap.parse_args()
    os.makedirs(a.raw_dir, exist_ok=True)
    host = socket.gethostname()
    gpu = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    var = f"h{a.h:g}" if a.family == "ising" else "random"
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["family"], r["variant"], r["n"], r["L"], r["gt"], r["obs"], r["method"]))
    new = not os.path.exists(a.out)
    fo = open(a.out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for n in a.n:
        sim = F.FastSim(n)
        for gt in a.gt:
            st = (F.ising_circuit(n, a.L, gt, h=a.h) if a.family == "ising"
                  else F.brickwork_circuit(n, a.L, gt))
            sim.run(st, "x", sim.B, seed=1)  # warm-up / kernel compilation
            sim.run(st, "zz", sim.B, table=np.ones(F.n_events(st, n)), seed=2)
            for o in a.obs:
                base = (a.family, var, str(n), str(a.L), str(gt), o)
                if base + ("ours_table",) in done:
                    continue
                sd = int(hashlib.md5(repr(base).encode()).hexdigest()[:8], 16)
                rows = []
                # table construction (setup, counted in time-to-SE)
                util, mem, la = idle_state()
                cp.cuda.Device().synchronize()
                t0 = time.perf_counter()
                tab, s_ev, q_ev = sim.continuation_table(st, o, reps=a.reps, seed=sd + 7,
                                                         abs_frac=a.abs_frac)
                cp.cuda.Device().synchronize()
                setup = time.perf_counter() - t0
                for method, table, su in (("ours_std", None, 0.0), ("ours_table", tab, setup)):
                    if base + (method,) in done:
                        continue
                    util, mem, la = idle_state()
                    x, wall, stats = timed_run(sim, st, o, table, a.budget_s, a.min_shots,
                                               a.max_shots, sd + (0 if table is None else 1))
                    v = float(x.var(ddof=1))
                    tps = wall / len(x)
                    raw = os.path.join(a.raw_dir, f"{'_'.join(base)}_{method}.npy")
                    np.save(raw, x.astype(np.float64))
                    rows.append(dict(
                        family=a.family, variant=var, n=n, L=a.L, gt=gt, obs=o, method=method,
                        shots=len(x), wall_s=round(wall, 4), t_per_shot_s=tps, mean=float(x.mean()),
                        var=v, se=float(np.sqrt(v / len(x))), setup_s=round(su, 4),
                        frac_table=(float(tab.mean()) if table is not None else 0.0),
                        proj_events_per_shot=stats["proj_events_per_shot"], target_se=a.target_se,
                        time_to_se_s=su + v * tps / a.target_se ** 2, reps=a.reps,
                        abs_frac=a.abs_frac, gpu_name=gpu, gpu_util_pct_before=util,
                        gpu_mem_mib_before=mem, loadavg1_before=la, host=host,
                        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                        raw_file=os.path.relpath(raw, ROOT)))
                for r in rows:
                    w.writerow(r)
                fo.flush()
                rs = {r["method"]: r for r in rows}
                if len(rs) == 2:
                    s_, t_ = rs["ours_std"], rs["ours_table"]
                    print(f"{a.family} {var} n={n} gt={gt} {o}: t/shot std={s_['t_per_shot_s']*1e3:.2f}ms "
                          f"table={t_['t_per_shot_s']*1e3:.2f}ms var std={s_['var']:.3e} table={t_['var']:.3e} "
                          f"setup={setup:.1f}s T_se std={s_['time_to_se_s']:.1f}s table={t_['time_to_se_s']:.1f}s",
                          flush=True)
    fo.close()


if __name__ == "__main__":
    main()
