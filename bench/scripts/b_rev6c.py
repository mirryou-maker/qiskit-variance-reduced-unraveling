"""Rev6 C runner (referee review 6, major 1; pre-registered in docs/plan_B_characterization_2026-11-06.md,
"Rev6 C/D pre-registration"): fixed partially-mixed amplitude-damping unravelings on the 28 E13 ad configs.

Channel "ad_mix" (src/noise/traj_kraus.amp_damp_mix): indices 0-4 = the E13 ad_phase candidates (unchanged), indices
5, 6, 7 = ad_mix_set(theta, phi = pi/4) for theta = pi/16, pi/8, 3pi/16 (unitary mixing of {K0, K1} interpolating
between photon counting (theta = 0) and the hom0.785 homodyne set (theta = pi/4)). Amendment 1 (--cands 8,9,10,
--seed_tag rev6c2): theta = arctan(x sqrt(g)), x = 0.25, 0.5, 1.

Per config k (one worker = one GPU of one node, the SAME node that ran config k in E13 unless --any_host), in order:
  anchor : candidate 0 (jump) re-timed, N = 2000, seed seed_of(tag-anchor, k, 0)      (timing anchor to the E13 session)
  new c  : candidates 5, 6, 7, N = 2000, all 4 observables, opt2, chunk 37, seed seed_of(tag-fixed, k, c)
Identical run call to E13 step 1 (K.run(..., ("fixed", c), fast=True, opt2=True)). Before every timed run the worker
waits until idle (b_e10 --idle pbs rule) and records the idle state in the row.
Ledger: <workdir>/rev6c_<worker>.csv (one row per (run, observable), all 4 observables); raw arrays
<raw>/<key>__<worker>_rev6c.npz. Claims per config (claim files); finished configs are skipped on resubmission.
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


def _load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


E = _load("b_e9b")
B10 = _load("b_e10")
SEL = _load("b_e13_select")
B10.IDLE_MODE = "pbs"
NEW = (5, 6, 7)
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "scheme", "cand", "value", "var", "shots",
          "wall_s", "seed", "chunk", "host", "e13_host", "gpu", "worker", "jobid", "gpu_util", "gpu_mem", "gpu_other",
          "other_py", "runner", "top_cpu", "load1", "timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e13_groups.csv"))
    ap.add_argument("--order", default=os.path.join(ROOT, "bench/results/b_e13_scope.txt"))
    ap.add_argument("--e13_summary", default=os.path.join(ROOT, "bench/results/b_e13_summary.csv"))
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--mem_gb", type=float, default=5.0)
    ap.add_argument("--seed_tag", default="rev6c")
    ap.add_argument("--workdir", default=os.path.join(ROOT, "bench/results/b_rev6c_main"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_rev6c_raw"))
    ap.add_argument("--claims", default=os.path.join(ROOT, "rev6c_claims"))
    ap.add_argument("--any_host", action="store_true", help="claim configs regardless of their E13 host")
    ap.add_argument("--only", default=None, help="comma-separated config keys (smoke)")
    ap.add_argument("--cands", default="5,6,7", help="new candidate indices (amendment 1: 8,9,10)")
    a = ap.parse_args()
    NEWC = tuple(int(c) for c in a.cands.split(","))
    cfg = E.configs(a.groups)
    names = [l.strip() for l in open(a.order) if l.strip() and not l.startswith("#")]
    bykey = {SEL.key_str(k): k for k in cfg}
    todo = [bykey[s] for s in names if bykey[s][0] == "ad"]
    if a.only:
        todo = [bykey[s] for s in a.only.split(",")]
    S13 = pd.read_csv(a.e13_summary)
    S13["key"] = [SEL.key_str((r.channel, r.family, r.variant, int(r.n), int(r.L), float(r.gt))) for r in S13.itertuples()]
    e13host = S13.drop_duplicates("key").set_index("key").host.to_dict()
    os.makedirs(a.raw, exist_ok=True)
    os.makedirs(a.workdir, exist_ok=True)
    host = socket.gethostname()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    jobid = os.environ.get("PBS_JOBID", "nojob").split(".")[0]
    worker = f"{host}-g{dev}-{jobid}"
    gname = SEL.gpu_name()
    cl = SEL.Claims(a.claims, worker)
    led = os.path.join(a.workdir, f"rev6c_{worker}.csv")
    new = not os.path.exists(led)
    f = open(led, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
        f.flush()
    mine = [k for k in todo if a.any_host or e13host.get(SEL.key_str(k)) == host]
    print(f"rev6c: worker {worker} on {gname}; {len(mine)} of {len(todo)} ad configs for this host; cands {NEWC}; N {a.N}; "
          f"seed tag {a.seed_tag}; start idle {B10.idle_state()}", flush=True)

    ops0 = K.build("brickwork", "s7", 20, 4, 0.01, "ad_mix")              # warm-up (kernel compilation)
    for c in (0,) + NEWC:
        K.run(ops0, 20, list(E.OBS), 4, ("fixed", c), seed=0, chunk=4, fast=True, opt2=True)
    P.sync()

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

    for k in mine:
        cl.release_stale()
        key = SEL.key_str(k)
        if not cl.try_claim(f"C__{key}"):
            continue
        ch, fam, var, n, L, gt = k
        chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (8 * 16 * 2 ** n))))
        ops = K.build(fam, var, n, L, gt, "ad_mix")
        labels = next(op[2].labels for op in ops if op[0] == "ch")
        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, e13_host=e13host.get(key, ""),
                    gpu=gname, worker=worker, jobid=jobid, chunk=chunk)
        raw = {}
        t0 = time.time()
        for kind, c in [("anchor", 0)] + [("fixed", c) for c in NEWC]:
            b = dict(base, **B10.wait_idle())
            sd = E.seed_of(f"{a.seed_tag}-{kind}", *k, c)
            r, wt = timed(lambda: K.run(ops, n, list(E.OBS), a.N, ("fixed", c), seed=sd, chunk=chunk, fast=True,
                                        opt2=True))
            for q in E.OBS:
                raw[f"{kind}_{labels[c]}_{q}"] = np.asarray(r[q], np.float32)
            emit([dict(b, obs=q, kind=kind, scheme=labels[c], cand=c, value=float(np.mean(r[q])),
                       var=float(np.var(np.asarray(r[q], np.float64), ddof=1)), shots=a.N, seed=sd,
                       wall_s=round(wt, 4)) for q in E.OBS])
        np.savez(os.path.join(a.raw, f"{key}__{worker}_rev6c.npz"), **raw)
        cl.finish(f"C__{key}")
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} ({time.time() - t0:.0f}s) "
              f"other={E.gpu_other()}", flush=True)
    f.close()
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} worker {worker} done", flush=True)


if __name__ == "__main__":
    main()
