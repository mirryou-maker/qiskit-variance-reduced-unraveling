"""E13 group selection (docs/plan_B_characterization_2026-11-06.md, "E13 pre-registration"): which FRESH groups are
still 'needs trajectories' (TN) at the Pauli-propagation cap 1e7 (E9a -> E9c -> E9d rule chain, bench/scripts/e13_rules.py).

Candidate ensemble (fixed in the pre-registration): brickwork seeds s11..s24 (ad) and s11..s17 (deph), n = 20,
gamma*t = 0.01, L in {16, 32}, observables x, zz, xc, zc; channels ad (ad_phase candidates) / deph (deph_sweep).
Per config:
  stage A : PP (pauli_prop.propagate) at M 1e2..1e6 for the 4 observables + std/jump reference N_ref = 1000 trajectories
            (traj_kraus.run fixed 0, fast path, seed seed_of("e13-sel", cfg)); E9a classification per observable.
Per group (config, observable) with E9a TN (non-deterministic):
  stage B : PP at M 3e6, 1e7, 3e7 (pauli_prop.propagate_big, same truncation semantics as propagate); cap-3e6 rule.
  stage C : if TN(3e6): PP at M 1e8 (propagate_big, as E9d); cap-1e7 rule (analyze_e13.py select).
Work is shared by several workers (PBS jobs on different nodes / GPUs) through claim files in --claims on the shared
scratch file system (O_CREAT | O_EXCL, atomic on Lustre). A claim is kept alive by a heartbeat thread (mtime touched
every 60 s); claims older than 20 min without a .done marker are released by any worker (dead job). Every worker
writes its own ledger CSV (<workdir>/sel_<worker>.csv). Not a timing experiment (PP times are recorded only for
information; the E9a clause t_PP(M*) > t_traj20k uses the same-worker times as E9a).
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import importlib.util
import os
import socket
import sys
import threading
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise.traj_ref import GPU, xp  # noqa: E402
import e13_rules as RU  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze_e11", os.path.join(os.path.dirname(__file__), "analyze_e11.py"))
A11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A11)
_spec = importlib.util.spec_from_file_location("b_e9b", os.path.join(os.path.dirname(__file__), "b_e9b.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

OBS = ("x", "zz", "xc", "zc")
SEEDS = {"ad": [f"s{s}" for s in range(11, 25)], "deph": [f"s{s}" for s in range(11, 18)]}
LS, GTS, N_Q = (16, 32), (0.01,), 20
NREF = 1000
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "M", "value", "var", "shots", "peak", "uniq",
          "wall_s", "status", "impl", "seed", "host", "gpu", "worker", "jobid", "gpu_other", "load1", "timestamp"]
STALE_S = 1200


def ensemble():
    return [(ch, "brickwork", v, N_Q, L, gt) for ch in ("ad", "deph") for v in SEEDS[ch] for L in LS for gt in GTS]


def key_str(k):
    ch, fam, var, n, L, gt = k
    return f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"


def parse_key(s):
    ch, fam, var, n, L, gt = s.split("_")
    return (ch, fam, var, int(n[1:]), int(L[1:]), float(gt[2:]))


def seed_of(*k):
    return int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)


def hash_order(keys, tag="e13-sel-order"):
    return sorted(keys, key=lambda k: hashlib.md5(repr((tag, key_str(k))).encode()).hexdigest())


def gpu_name():
    if not GPU:
        return "cpu"
    try:
        return xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)["name"].decode()
    except Exception:  # noqa: BLE001
        return "gpu"


class Claims:
    def __init__(self, d, worker):
        self.d, self.worker = d, worker
        os.makedirs(d, exist_ok=True)
        self.live = set()
        self.lock = threading.Lock()
        t = threading.Thread(target=self._beat, daemon=True)
        t.start()

    def _beat(self):
        while True:
            time.sleep(60)
            with self.lock:
                for p in list(self.live):
                    try:
                        os.utime(p)
                    except OSError:
                        pass

    def path(self, name):
        return os.path.join(self.d, name)

    def done(self, name):
        return os.path.exists(self.path(name) + ".done")

    def try_claim(self, name):
        if self.done(name):
            return False
        p = self.path(name)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, f"{self.worker} {os.environ.get('PBS_JOBID', '')} {dt.datetime.now().isoformat()}\n".encode())
        os.close(fd)
        with self.lock:
            self.live.add(p)
        return True

    def finish(self, name):
        p = self.path(name)
        with open(p + ".done", "w") as fh:
            fh.write(f"{self.worker} {os.environ.get('PBS_JOBID', '')} {dt.datetime.now().isoformat()}\n")
        with self.lock:
            self.live.discard(p)

    def release_stale(self):
        n = 0
        for p in glob.glob(os.path.join(self.d, "*")):
            if p.endswith(".done") or os.path.exists(p + ".done") or p in self.live:
                continue
            try:
                if time.time() - os.path.getmtime(p) > STALE_S:
                    os.remove(p)
                    n += 1
            except OSError:
                pass
        return n


def read_all(workdir):
    fs = sorted(glob.glob(os.path.join(workdir, "sel_*.csv")))
    if not fs:
        return pd.DataFrame(columns=FIELDS)
    d = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 0], ignore_index=True)
    return d


def stage_a_summary(d, k):
    """E9a summary rows (analyze_e11.classify) for config k from the ledger rows."""
    sub = d[(d.channel == k[0]) & (d.family == k[1]) & (d.variant == k[2]) & (d.n == k[3]) & (d.L == k[4])
            & np.isclose(d["gt"], k[5]) & d.kind.isin(["pp", "traj"])]
    sub = sub[(sub.kind == "traj") | sub.M.astype(float).isin([float(m) for m in RU.MS_A])].copy()
    sub["M"] = sub.M.astype(float)
    return A11.classify(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(ROOT, "bench/results/b_e13_sel"))
    ap.add_argument("--claims", default=os.path.join(ROOT, "e13_claims_sel"))
    ap.add_argument("--worker", default=None)
    ap.add_argument("--only", nargs="+", default=None, help="explicit config keys (smoke)")
    ap.add_argument("--only_obs", default=None, help="smoke: restrict stage B/C to this observable")
    ap.add_argument("--mem_gb", type=float, default=5.0)
    ap.add_argument("--chunk_big", type=int, default=5_000_000)
    ap.add_argument("--max_hours", type=float, default=1e9, help="stop claiming new work after this many hours")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    host, gname = socket.gethostname(), gpu_name()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    worker = a.worker or f"{host}-g{dev}"
    jobid = os.environ.get("PBS_JOBID", "")
    cl = Claims(a.claims, worker)
    led = os.path.join(a.workdir, f"sel_{worker}.csv")
    new = not os.path.exists(led)
    f = open(led, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
        f.flush()
    T0 = time.time()

    def emit(rows):
        ts = dt.datetime.now().isoformat(timespec="seconds")
        for rr in rows:
            rr["timestamp"] = ts
            w.writerow(rr)
        f.flush()

    def base(k):
        ch, fam, var, n, L, gt = k
        return dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, gpu=gname, worker=worker,
                    jobid=jobid, gpu_other=E.gpu_other(), load1=E.load1())

    keys = [parse_key(s) for s in a.only] if a.only else ensemble()
    keys = hash_order(keys)
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} worker {worker} on {gname}: {len(keys)} configs", flush=True)
    chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (6 * 16 * 2 ** N_Q))))

    # ---------------------------------------------------------------- stage A (config level)
    for k in keys:
        if time.time() - T0 > a.max_hours * 3600:
            break
        name = f"A__{key_str(k)}"
        if not cl.try_claim(name):
            continue
        ch, fam, var, n, L, gt = k
        ops = K.build(fam, var, n, L, gt, ch)          # E9a used the base channels ("ad", "deph") for selection
        b = base(k)
        rows, tables = [], {}
        for o in OBS:
            for m in RU.MS_A:
                P.sync()
                t1 = time.perf_counter()
                v, peak = P.propagate(ops, n, o, m, tables)
                P.sync()
                rows.append(dict(b, obs=o, kind="pp", M=m, value=v, peak=peak, status="ok",
                                 wall_s=round(time.perf_counter() - t1, 4)))
        sd = seed_of("e13-sel", *k)
        P.sync()
        t1 = time.perf_counter()
        r = K.run(ops, n, list(OBS), NREF, ("fixed", 0), seed=sd, chunk=chunk, fast=True)
        P.sync()
        wt = time.perf_counter() - t1
        for o in OBS:
            x = np.asarray(r[o], np.float64)
            rows.append(dict(b, obs=o, kind="traj", M="", value=float(x.mean()), var=float(x.var(ddof=1)),
                             shots=len(x), seed=sd, status="ok", wall_s=round(wt, 3)))
        emit(rows)
        cl.finish(name)
        if GPU:
            xp.get_default_memory_pool().free_all_blocks()
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} A {key_str(k)} ({wt:.0f}s traj)", flush=True)

    # ---------------------------------------------------------------- stages B and C (group level)
    while True:
        if time.time() - T0 > a.max_hours * 3600:
            print("max_hours reached, not claiming more work", flush=True)
            break
        cl.release_stale()
        d = read_all(a.workdir)
        pending_a = [k for k in keys if not cl.done(f"A__{key_str(k)}")]
        tasks = []
        for k in keys:
            if not cl.done(f"A__{key_str(k)}"):
                continue
            Sa = stage_a_summary(d, k)
            for _, r in Sa.iterrows():
                if r.deterministic or not r.TN or (a.only_obs and r.obs != a.only_obs):
                    continue
                tasks.append((k, r.obs))
        tasks.sort(key=lambda t: -t[0][4])                  # deep circuits first (load balance)
        todo = [t for t in tasks if not cl.done(f"G__{key_str(t[0])}__{t[1]}")]
        if not todo and not pending_a:
            break
        got = None
        for t in todo:
            if cl.try_claim(f"G__{key_str(t[0])}__{t[1]}"):
                got = t
                break
        if got is None:
            if not pending_a:
                print(f"{dt.datetime.now().isoformat(timespec='seconds')} nothing claimable "
                      f"({len(todo)} groups claimed by other workers)", flush=True)
                break
            time.sleep(60)
            continue
        k, o = got
        ch, fam, var, n, L, gt = k
        ops = K.build(fam, var, n, L, gt, ch)
        b = base(k)
        prev = d[(d.channel == ch) & (d.family == fam) & (d.variant == var) & (d.n == n) & (d.L == L) & np.isclose(d["gt"], gt) & (d.obs == o)
                 & (d.kind == "pp") & (d.status == "ok")]
        have = {int(float(m)): float(v) for m, v in zip(prev.M, prev.value)}
        tables, t0 = {}, time.time()

        def run_big(m):
            if m in have:
                return have[m]
            if GPU:
                xp.get_default_memory_pool().free_all_blocks()
            P.sync()
            t1 = time.perf_counter()
            st = {}

            def is_oom(e):
                return ("OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()
                        or isinstance(e, MemoryError))
            try:
                impl = "big"
                if m < RU.M_C:                   # deviation 1: plain propagate first (as E9c), propagate_big on OOM
                    try:
                        impl = "plain"
                        v, peak = P.propagate(ops, n, o, m, tables)
                        P.sync()
                    except Exception as e:  # noqa: BLE001
                        if not is_oom(e):
                            raise
                        if GPU:
                            xp.get_default_memory_pool().free_all_blocks()
                        impl = "big"
                if impl == "big":
                    v, peak = P.propagate_big(ops, n, o, m, tables, chunk=a.chunk_big, nb=64, stats=st)
                    P.sync()
                row = dict(b, obs=o, kind="pp", M=m, value=v, peak=peak, uniq=st.get("peak_unique", ""),
                           status="ok", impl=impl, wall_s=round(time.perf_counter() - t1, 3))
            except Exception as e:  # noqa: BLE001
                if not is_oom(e):
                    raise
                v = np.nan
                row = dict(b, obs=o, kind="pp", M=m, value="", status="oom",
                           wall_s=round(time.perf_counter() - t1, 3))
            emit([row])
            have[m] = v
            return v

        for m in RU.MS_B:
            run_big(m)
        Sa = stage_a_summary(d, k)
        Sg = Sa[Sa.obs == o].copy()
        for m in RU.MS_B:
            Sg[f"pp{m}"] = have.get(m, np.nan)
        Sg = RU.caps(Sg)
        tn3 = bool(Sg.TN3000000.iloc[0] == 1)
        if tn3:
            run_big(RU.M_C)
        cl.finish(f"G__{key_str(k)}__{o}")
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} G {key_str(k)} {o} TN3e6={tn3} "
              f"({time.time() - t0:.0f}s)", flush=True)
    f.close()
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} worker {worker} done", flush=True)


if __name__ == "__main__":
    main()
