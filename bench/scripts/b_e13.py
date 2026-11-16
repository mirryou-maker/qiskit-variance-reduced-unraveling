"""E13 main runner (docs/plan_B_characterization_2026-11-06.md, "E13 pre-registration"): static table and martingale
control variate on FRESH groups that are still TN at the Pauli-propagation cap 1e7 (bench/results/b_e13_groups.csv).

Per config k (all runs of a config by ONE worker = one GPU of one node, in one session, in this order):
  1. kind=fixed : every candidate (ad: ad_phase 5, deph: deph_sweep 7), N = 2000, all 4 observables in one run (opt2),
                  seeds seed_of(tag-fixed, k, c). Variances and t_f(c) = wall / N from these runs.
  2. per TN observable o (O_k = the config's E13 groups):
     bp_s   : sparse_policy.record_after(o, M = 100) + group precompute (counted in full for every method using O~).
     single : E10 pilot table (static_table.build_table, 74 candidate-0 pilot trajectories, seed tag-pilot; build_s),
              production N = 2000 (traj_kraus.run table, opt2 + v2), seed tag-single.                         [ST]
     cv0    : cv_martingale.run_cv fixed candidate 0 (std / jump), N = 2000, seed tag-cv0.                   [CV0]
     cvbf   : run_cv fixed c* = argmin_c Var_c(o) from step 1 (oracle), seed tag-cvbf; if c* = 0 no run
              (the CV0 run is used, row scheme "cv0").                                                        [CVbf]
     stcv   : run_cv with the ST table (same table as `single`), N = 2000, seed tag-stcv.                    [ST+CV]
Before EVERY timed run: wait until idle (b_e10.py pbs rule: no foreign process on this GPU, no other own python
process) and record gpu_util, gpu_mem, gpu_other, other_py, top_cpu, load1 in the row.
Work sharing: configs are claimed (claim files on the shared scratch, heartbeat, stale after 20 min) in the order of
--order (scope list). The worker name contains the PBS job id; a config is valid only for the worker named in its
.done marker (a config interrupted by a dead job is re-run in full by another worker; partial rows are ignored).
Raw arrays: <raw>/<key>__<worker>_{fixed,single_<o>,cv0_<o>,cvbf_<o>,stcv_<o>}.npz.
"""
import argparse
import csv
import datetime as dt
import importlib.util
import os
import socket
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import cv_martingale as CV  # noqa: E402
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import static_table as ST  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402


def _load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


E = _load("b_e9b")
B10 = _load("b_e10")
SEL = _load("b_e13_select")
B10.IDLE_MODE = "pbs"

M_POL = 100
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "scheme", "M", "value", "var", "shots",
          "wall_s", "bp_s", "build_s", "pilot_shots", "nonstd", "cand", "seed", "pilot_seed", "chunk", "host", "gpu",
          "worker", "jobid", "gpu_util", "gpu_mem", "gpu_other", "other_py", "runner", "top_cpu", "load1", "impl",
          "timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e13_groups.csv"))
    ap.add_argument("--order", default=None, help="scope list: one config key per line (lines starting # ignored)")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--pilot", type=int, default=74)
    ap.add_argument("--mem_gb", type=float, default=5.0, help="sets the chunk (37 at n = 20, as E9b-E12)")
    ap.add_argument("--seed_tag", default="e13")
    ap.add_argument("--workdir", default=os.path.join(ROOT, "bench/results/b_e13_main"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_e13_raw"))
    ap.add_argument("--claims", default=os.path.join(ROOT, "e13_claims_main"))
    ap.add_argument("--max_hours", type=float, default=1e9, help="stop claiming new configs after this many hours")
    a = ap.parse_args()
    S.IMPL = "v2"
    cfg = E.configs(a.groups)
    if a.order:
        names = [l.strip() for l in open(a.order) if l.strip() and not l.startswith("#")]
    else:
        names = [SEL.key_str(k) for k in SEL.hash_order(list(cfg), "e13-order")]
    bykey = {SEL.key_str(k): k for k in cfg}
    todo = [bykey[s] for s in names]
    os.makedirs(a.raw, exist_ok=True)
    os.makedirs(a.workdir, exist_ok=True)
    host = socket.gethostname()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    jobid = os.environ.get("PBS_JOBID", "nojob").split(".")[0]
    worker = f"{host}-g{dev}-{jobid}"
    gname = SEL.gpu_name()
    cl = SEL.Claims(a.claims, worker)
    led = os.path.join(a.workdir, f"main_{worker}.csv")
    new = not os.path.exists(led)
    f = open(led, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
        f.flush()
    print(f"E13 main: worker {worker} on {gname}; {len(todo)} configs in order; N {a.N}, pilot {a.pilot}, M {M_POL}, "
          f"seed tag {a.seed_tag}; start idle {B10.idle_state()}", flush=True)

    # warm-up (kernel compilation outside every timed region)
    ops0 = K.build("brickwork", "s7", 20, 4, 0.01, "ad_phase")
    rec0, _, _ = S.record_after(ops0, 20, "zz", M_POL)
    tab0, _ = ST.build_table(ops0, 20, rec0, "pilot", shots=4, seed=0, chunk=4)
    K.run(ops0, 20, list(E.OBS), 4, ("table", tab0), seed=0, chunk=4, fast=True, opt2=True)
    K.run(ops0, 20, list(E.OBS), 4, ("fixed", 1), seed=0, chunk=4, fast=True, opt2=True)
    CV.run_cv(ops0, 20, list(E.OBS), 4, rec0, ("fixed", 1), seed=0, chunk=4)
    CV.run_cv(ops0, 20, list(E.OBS), 4, rec0, ("table", tab0), seed=0, chunk=4)
    P.sync()
    T0 = time.time()

    def emit(rows):
        ts = dt.datetime.now().isoformat(timespec="seconds")
        for rr in rows:
            rr["timestamp"] = ts
            w.writerow(rr)
        f.flush()

    def timed(fn):
        P.sync()
        t1 = time.perf_counter()
        r = fn()
        P.sync()
        return r, time.perf_counter() - t1

    for k in todo:
        if time.time() - T0 > a.max_hours * 3600:
            print("max_hours reached, not claiming more configs", flush=True)
            break
        cl.release_stale()
        key = SEL.key_str(k)
        if not cl.try_claim(f"C__{key}"):
            continue
        ch, fam, var, n, L, gt = k
        tn_obs = cfg[k]
        chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (8 * 16 * 2 ** n))))
        ops = K.build(fam, var, n, L, gt, E.CHMAP[ch])
        labels = next(op[2].labels for op in ops if op[0] == "ch")
        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, gpu=gname, worker=worker,
                    jobid=jobid, chunk=chunk, impl="v2+opt2")
        pre = os.path.join(a.raw, f"{key}__{worker}")
        t0 = time.time()
        # 1. fixed candidates
        fraw, vfix = {}, {}
        for c, lab in enumerate(labels):
            b = dict(base, **B10.wait_idle())
            sd = E.seed_of(a.seed_tag + "-fixed", *k, c)
            r, wt = timed(lambda: K.run(ops, n, list(E.OBS), a.N, ("fixed", c), seed=sd, chunk=chunk, fast=True,
                                        opt2=True))
            for q in E.OBS:
                fraw[f"fixed_{lab}_{q}"] = np.asarray(r[q], np.float32)
            vfix[c] = {q: float(np.var(np.asarray(r[q], np.float64), ddof=1)) for q in E.OBS}
            emit([dict(b, obs=o, kind="fixed", scheme=lab, cand=c, M="", value=float(np.mean(r[o])),
                       var=vfix[c][o], shots=a.N, seed=sd, wall_s=round(wt, 4)) for o in tn_obs])
        np.savez(pre + "_fixed.npz", **fraw)
        # 2. per TN observable
        for o in tn_obs:
            (rec, _, _), bp = timed(lambda: S.record_after(ops, n, o, M_POL))
            _, bp2 = timed(lambda: [S._groups(ev, S.GMAX) for ev in rec])
            bp += bp2
            # ST (E10 pilot table)
            b = dict(base, **B10.wait_idle())
            psd = E.seed_of(a.seed_tag + "-pilot", *k, o)
            (tab, Vm), bs = timed(lambda: ST.build_table(ops, n, rec, "pilot", shots=a.pilot, seed=psd, chunk=chunk))
            sd = E.seed_of(a.seed_tag + "-single", *k, o)
            r, wt = timed(lambda: K.run(ops, n, list(E.OBS), a.N, ("table", tab), seed=sd, chunk=chunk, fast=True,
                                        opt2=True))
            x = np.asarray(r[o], np.float64)
            np.savez(pre + f"_single_{o}.npz", x=x.astype(np.float32), tab=tab.astype(np.int8), Vm=Vm.astype(np.float32),
                     **{f"all_{q}": np.asarray(r[q], np.float32) for q in E.OBS})
            emit([dict(b, obs=o, kind="single", scheme="pilot", M=M_POL, value=float(x.mean()), var=float(x.var(ddof=1)),
                       shots=len(x), wall_s=round(wt, 4), bp_s=round(bp, 4), build_s=round(bs, 4),
                       pilot_shots=a.pilot, nonstd=round(float(np.mean(tab != 0)), 5), seed=sd, pilot_seed=psd)])
            # CV runs
            cbf = int(np.argmin([vfix[c][o] for c in range(len(labels))]))
            plan = [("cv0", ("fixed", 0), 0), ("cvbf", ("fixed", cbf), cbf), ("stcv", ("table", tab), -1)]
            for kind, sc, cand in plan:
                if kind == "cvbf" and cbf == 0:
                    emit([dict(base, obs=o, kind="cvbf", scheme="cv0", cand=0, M=M_POL, bp_s=round(bp, 4))])
                    continue
                b = dict(base, **B10.wait_idle())
                sd = E.seed_of(a.seed_tag + "-" + kind, *k, o)
                r, wt = timed(lambda: CV.run_cv(ops, n, list(E.OBS), a.N, rec, sc, seed=sd, chunk=chunk, opt2=True))
                x = np.asarray(r[o], np.float64)
                Y, beta = CV.cv_estimates(x, r["_S"])
                np.savez(pre + f"_{kind}_{o}.npz", f=x, S=r["_S"], **{f"all_{q}": np.asarray(r[q], np.float32)
                                                                      for q in E.OBS})
                emit([dict(b, obs=o, kind=kind, scheme=("table" if kind == "stcv" else labels[cand]), cand=cand,
                           M=M_POL, value=float(Y["crossfit"].mean()), var=float(Y["crossfit"].var(ddof=1)),
                           shots=len(x), wall_s=round(wt, 4), bp_s=round(bp, 4),
                           build_s=round(bs, 4) if kind == "stcv" else "", pilot_shots=a.pilot if kind == "stcv" else "",
                           seed=sd)])
        cl.finish(f"C__{key}")
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} obs={','.join(tn_obs)} "
              f"({time.time() - t0:.0f}s) other={E.gpu_other()}", flush=True)
    f.close()
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} worker {worker} done", flush=True)


if __name__ == "__main__":
    main()
