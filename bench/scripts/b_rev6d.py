"""Rev6 D runner (referee review 6, major 2; pre-registered in docs/plan_B_characterization_2026-11-06.md,
"Rev6 C/D pre-registration"): independent high-accuracy references for E13 groups.

  traj : long trajectory reference with the DEFAULT unraveling (candidate 0: photon-count jump for ad, std Pauli jump
         for deph), the E13 channel sets (ad_phase / deph_sweep), opt2, chunk 37, all 4 observables. Work is split into
         blocks of --block trajectories (seed seed_of("rev6d-ref", k, block)); blocks are claimed via claim files, so any
         number of GPU workers can share a config and a resubmitted job skips finished blocks. Not a timing experiment.
         Raw: <raw>/<key>__blk<b>.npz (per-trajectory float64 values of all 4 observables).
  pp   : Pauli propagation at caps > 1e8 (pauli_prop.propagate_big, the E9d / E13 stage-C implementation) for the
         config's observables, in cap order; ledger rows ok / oom; resubmission skips finished (obs, cap).

  python b_rev6d.py traj --keys K1,K2 --blocks 100 --block 10000
  python b_rev6d.py pp --keys K1 --caps 300000000,1000000000
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
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise.traj_ref import GPU, xp  # noqa: E402


def _load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


E = _load("b_e9b")
SEL = _load("b_e13_select")
FIELDS = ["kind", "channel", "family", "variant", "n", "L", "gt", "obs", "block", "M", "value", "sumsq", "shots", "peak",
          "status", "wall_s", "seed", "host", "gpu", "worker", "jobid", "gpu_other", "timestamp"]


def parse_key(s, cfg):
    return {SEL.key_str(k): k for k in cfg}[s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["traj", "pp"])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e13_groups.csv"))
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--block", type=int, default=10000)
    ap.add_argument("--caps", default="300000000,1000000000")
    ap.add_argument("--chunk_big", type=int, default=5_000_000)
    ap.add_argument("--mem_gb", type=float, default=5.0)
    ap.add_argument("--workdir", default=os.path.join(ROOT, "bench/results/b_rev6d_main"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_rev6d_raw"))
    ap.add_argument("--claims", default=os.path.join(ROOT, "rev6d_claims"))
    a = ap.parse_args()
    cfg = E.configs(a.groups)
    keys = [parse_key(s, cfg) for s in a.keys.split(",")]
    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(a.raw, exist_ok=True)
    host = socket.gethostname()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    jobid = os.environ.get("PBS_JOBID", "nojob").split(".")[0]
    worker = f"{host}-g{dev}-{jobid}"
    led = os.path.join(a.workdir, f"rev6d_{a.mode}_{worker}.csv")
    new = not os.path.exists(led)
    f = open(led, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
        f.flush()
    base0 = dict(host=host, gpu=SEL.gpu_name(), worker=worker, jobid=jobid)
    print(f"rev6d {a.mode}: worker {worker} on {base0['gpu']}; keys {[SEL.key_str(k) for k in keys]}", flush=True)

    def emit(row):
        row["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
        w.writerow(row)
        f.flush()

    if a.mode == "traj":
        cl = SEL.Claims(a.claims, worker)
        for k in keys:
            ch, fam, var, n, L, gt = k
            ops = K.build(fam, var, n, L, gt, E.CHMAP[ch])
            chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (8 * 16 * 2 ** n))))
            key = SEL.key_str(k)
            for b in range(a.blocks):
                cl.release_stale()
                name = f"T__{key}__{b}"
                if not cl.try_claim(name):
                    continue
                sd = E.seed_of("rev6d-ref", *k, b)
                t1 = time.perf_counter()
                r = K.run(ops, n, list(E.OBS), a.block, ("fixed", 0), seed=sd, chunk=chunk, fast=True, opt2=True)
                P.sync()
                wt = time.perf_counter() - t1
                arr = {q: np.asarray(r[q], np.float64) for q in E.OBS}
                np.savez(os.path.join(a.raw, f"{key}__blk{b:04d}.npz"), **arr)
                for q in E.OBS:
                    emit(dict(base0, kind="traj", channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, obs=q, block=b,
                              value=float(arr[q].sum()), sumsq=float((arr[q] ** 2).sum()), shots=a.block, seed=sd,
                              status="ok", wall_s=round(wt, 2), gpu_other=E.gpu_other()))
                cl.finish(name)
                print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key} block {b} ({wt:.0f}s)", flush=True)
    else:
        caps = [int(float(c)) for c in a.caps.split(",")]
        prev = []
        import glob
        for fn in glob.glob(os.path.join(a.workdir, "rev6d_pp_*.csv")):
            if os.path.getsize(fn) > 0:
                prev.append(pd.read_csv(fn))
        prev = pd.concat(prev) if prev else pd.DataFrame(columns=FIELDS)
        for m in caps:
            for k in keys:
                ch, fam, var, n, L, gt = k
                ops = K.build(fam, var, n, L, gt, E.CHMAP[ch])
                tables = {}
                for o in cfg[k]:
                    done = prev[(prev.variant == var) & (prev.channel == ch) & (prev.L == L) & (prev.obs == o)
                                & (prev.M.astype(float) == float(m))]
                    if len(done):
                        continue
                    if GPU:
                        xp.get_default_memory_pool().free_all_blocks()
                    P.sync()
                    t1 = time.perf_counter()
                    st = {}
                    try:
                        v, peak = P.propagate_big(ops, n, o, m, tables, chunk=a.chunk_big, nb=64, stats=st)
                        P.sync()
                        status = "ok"
                    except Exception as e:  # noqa: BLE001
                        if not ("OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()
                                or isinstance(e, MemoryError)):
                            raise
                        v, peak, status = "", "", "oom"
                    wt = time.perf_counter() - t1
                    emit(dict(base0, kind="pp", channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, obs=o, M=m,
                              value=v, peak=peak, status=status, wall_s=round(wt, 1), gpu_other=E.gpu_other()))
                    print(f"{dt.datetime.now().isoformat(timespec='seconds')} PP {SEL.key_str(k)} {o} M={m:.0e} "
                          f"{status} {v} ({wt:.0f}s)", flush=True)
    f.close()
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} worker {worker} done", flush=True)


if __name__ == "__main__":
    main()
