"""E11 (docs/plan_B_characterization_2026-11-06.md, "E11 pre-registration"): confirmatory static tables on FRESH groups
(n = 20, groups from b_e11_select.py + analyze_e11.py select -> bench/results/b_e11_groups.csv).

Per config k = (channel, family, variant, n, L, gt), all on ubuntu94 in one session, in this order:
  1. kind=fixed  : every candidate, N = 2000 trajectories, all 4 observables in one run (opt2), fresh seeds.
                   t_f(c) = wall / N; variances from these runs. Raw arrays of all 4 observables.
  2. per TN observable o: record_after(o, M = 100) + group precompute (bp_s);
     kind=single scheme=pilot : E10 pilot table (static_table.build_table, 74 candidate-0 pilot trajectories, own seed;
                                build_s), production N = 2000 (new seed).
  3. kind=multi scheme=multi : multi-target table over the config's TN observables O_k
                               (static_table_multi.build_table_multi, one 74-trajectory pilot, own seed; build_s),
                               bp_s = sum of the bp_s of step 2 (all O~ of O_k), ONE production run N = 2000 (new seed);
                               one row per o in O_k with the variance of o from that run.
Before EVERY timed run: wait until no other python job from ~/work, no CI Runner.Worker and no other GPU process runs
(the runner itself and its ancestors excluded), then record gpu_util, gpu_mem, gpu_other, other_py, runner, top_cpu, load1.
Ledger CSV (resumable): rows already present are skipped; raw arrays in --raw (<cfg>_fixed.npz, <cfg>_single_<o>.npz,
<cfg>_multi.npz). Configs are run in the order of --order (scope list from analyze_e11.py scope), else hash order.
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import os
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import static_table as ST  # noqa: E402
from src.noise import static_table_multi as SM  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402

_spec = importlib.util.spec_from_file_location("b_e9b", os.path.join(os.path.dirname(__file__), "b_e9b.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

M_POL = 100
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "scheme", "M", "value", "var", "shots",
          "wall_s", "bp_s", "build_s", "pilot_shots", "nobs", "v0hat", "nonstd", "seed", "pilot_seed", "chunk", "host",
          "gpu_util", "gpu_mem", "gpu_other", "other_py", "runner", "top_cpu", "load1", "impl", "timestamp"]


def key_str(k):
    ch, fam, var, n, L, gt = k
    return f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"


def hash_order(keys, tag="e11-order"):
    return sorted(keys, key=lambda k: hashlib.md5(repr((tag, key_str(k))).encode()).hexdigest())


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        return f"err:{e}"


def _ancestors():
    anc, p_ = set(), os.getpid()
    while p_ > 1:
        anc.add(str(p_))
        try:
            p_ = int(open(f"/proc/{p_}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return anc


def idle_state():
    """Same rule as the E10 ubuntu94 runner (b_e10.py, default mode)."""
    anc = _ancestors()
    q = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").strip().split(",")
    py = []
    for line in sh("ps -eo pid,args").splitlines()[1:]:
        p = line.strip().split(None, 1)
        if len(p) == 2 and p[0] not in anc and "python" in p[1]:
            try:
                cwd = os.readlink(f"/proc/{p[0]}/cwd")
                comm = open(f"/proc/{p[0]}/comm").read().strip()
            except OSError:
                continue
            if not comm.startswith("python"):
                continue
            if "/work" in p[1] or "/work" in cwd:
                py.append(p[0])
    runner = sum(1 for l in sh("ps -eo args").splitlines() if "Runner.Worker" in l)
    top = sh("top -b -n1 -o %CPU | sed -n 8,10p").split("\n")
    top = ";".join("/".join(t.split()[i] for i in (0, 8, 11)) for t in top if len(t.split()) >= 12)
    return dict(gpu_util=q[0].strip() if q else "", gpu_mem=q[1].strip() if len(q) > 1 else "",
                gpu_other=E.gpu_other(), other_py=";".join(py) or "none", runner=runner, top_cpu=top, load1=E.load1())


def wait_idle(log_every=600):
    t0, last = time.time(), -1e9
    while True:
        time.sleep(2)                    # let nvidia-smi's utilization window drop our own previous run
        s = idle_state()
        if s["other_py"] == "none" and s["runner"] == 0 and s["gpu_other"] == "none":
            return s
        if time.time() - last > log_every:
            print(f"{dt.datetime.now().isoformat(timespec='seconds')} waiting for idle ({time.time() - t0:.0f}s): {s}",
                  flush=True)
            last = time.time()
        time.sleep(28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e11_groups.csv"))
    ap.add_argument("--order", default=None, help="file with one config key per line (scope list), run in that order")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--pilot", type=int, default=74)
    ap.add_argument("--mem_gb", type=float, default=5.0, help="sets the chunk (37 at n = 20, as E9b / E10)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="config key substring (smoke)")
    ap.add_argument("--seed_tag", default="e11", help="smoke runs use another tag so their seeds differ")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e11_u94.csv"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_e11_raw"))
    a = ap.parse_args()
    S.IMPL = "v2"
    cfg = E.configs(a.groups)
    if a.order:
        names = [l.strip() for l in open(a.order) if l.strip() and not l.startswith("#")]
        bykey = {key_str(k): k for k in cfg}
        todo = [bykey[s] for s in names]
    else:
        todo = hash_order(list(cfg))
    if a.only:
        todo = [k for k in todo if a.only in key_str(k)]
    todo = todo[:a.limit]
    os.makedirs(a.raw, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]),
                      r["kind"], r["scheme"], r["obs"]))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    host = socket.gethostname()
    print(f"E11: {len(todo)} configs, N {a.N}, pilot {a.pilot}, M {M_POL}, seed tag {a.seed_tag}; start idle "
          f"{idle_state()}", flush=True)

    # warm-up (kernel compilation outside every timed region)
    ops0 = K.build("brickwork", "s7", 20, 4, 0.01, "ad_phase")
    rec0, _, _ = S.record_after(ops0, 20, "zz", M_POL)
    rec1, _, _ = S.record_after(ops0, 20, "x", M_POL)
    ST.build_table(ops0, 20, rec0, "pilot", shots=4, seed=0, chunk=4)
    SM.build_table_multi(ops0, 20, {"zz": rec0, "x": rec1}, shots=4, seed=0, chunk=4)
    K.run(ops0, 20, list(E.OBS), 4, ("table", np.zeros(len(rec0), int)), seed=0, chunk=4, fast=True, opt2=True)
    K.run(ops0, 20, list(E.OBS), 4, ("fixed", 1), seed=0, chunk=4, fast=True, opt2=True)
    P.sync()

    def emit(rows):
        ts = dt.datetime.now().isoformat(timespec="seconds")
        for rr in rows:
            rr["timestamp"] = ts
            w.writerow(rr)
        f.flush()

    for k in todo:
        ch, fam, var, n, L, gt = k
        tn_obs = cfg[k]
        chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (8 * 16 * 2 ** n))))
        key = key_str(k)
        ops = K.build(fam, var, n, L, gt, E.CHMAP[ch])
        labels = next(op[2].labels for op in ops if op[0] == "ch")
        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, chunk=chunk, impl="v2+opt2")
        t0 = time.time()
        # 1. fixed candidates, N = 2000, all observables in one run
        fxf = os.path.join(a.raw, f"{key}_fixed.npz")
        fraw = dict(np.load(fxf)) if os.path.exists(fxf) else {}
        for c, lab in enumerate(labels):
            if all(k + ("fixed", lab, o) in done for o in tn_obs) and f"fixed_{lab}_x" in fraw:
                continue
            b = dict(base, **wait_idle())
            sd = E.seed_of(a.seed_tag + "-fixed", *k, c)
            P.sync()
            t1 = time.perf_counter()
            r = K.run(ops, n, list(E.OBS), a.N, ("fixed", c), seed=sd, chunk=chunk, fast=True, opt2=True)
            P.sync()
            wt = time.perf_counter() - t1
            for q in E.OBS:
                fraw[f"fixed_{lab}_{q}"] = np.asarray(r[q], np.float32)
            np.savez(fxf, **fraw)
            emit([dict(b, obs=o, kind="fixed", scheme=lab, M="", value=float(np.mean(r[o])),
                       var=float(np.var(np.asarray(r[o], np.float64), ddof=1)), shots=a.N, seed=sd,
                       wall_s=round(wt, 4)) for o in tn_obs])
        # 2. single-target E10 pilot tables, one per TN observable
        recs, bps = {}, {}
        for o in tn_obs:
            P.sync()
            t1 = time.perf_counter()
            rec, _, _ = S.record_after(ops, n, o, M_POL)
            for ev in rec:
                S._groups(ev, S.GMAX)
            P.sync()
            recs[o], bps[o] = rec, time.perf_counter() - t1
            if k + ("single", "pilot", o) in done:
                continue
            b = dict(base, **wait_idle())
            psd = E.seed_of(a.seed_tag + "-pilot", *k, o)
            P.sync()
            t1 = time.perf_counter()
            tab, Vm = ST.build_table(ops, n, rec, "pilot", shots=a.pilot, seed=psd, chunk=chunk)
            P.sync()
            bs = time.perf_counter() - t1
            sd = E.seed_of(a.seed_tag + "-single", *k, o)
            t1 = time.perf_counter()
            r = K.run(ops, n, list(E.OBS), a.N, ("table", tab), seed=sd, chunk=chunk, fast=True, opt2=True)
            P.sync()
            wt = time.perf_counter() - t1
            x = np.asarray(r[o], np.float64)
            np.savez(os.path.join(a.raw, f"{key}_single_{o}.npz"), x=x.astype(np.float32), tab=tab.astype(np.int8),
                     Vm=Vm.astype(np.float32), **{f"all_{q}": np.asarray(r[q], np.float32) for q in E.OBS})
            emit([dict(b, obs=o, kind="single", scheme="pilot", M=M_POL, value=float(x.mean()),
                       var=float(x.var(ddof=1)), shots=len(x), wall_s=round(wt, 4), bp_s=round(bps[o], 4),
                       build_s=round(bs, 4), pilot_shots=a.pilot, nobs=1, nonstd=round(float(np.mean(tab != 0)), 5),
                       seed=sd, pilot_seed=psd)])
        # 3. multi-target table over O_k, one production run
        if not all(k + ("multi", "multi", o) in done for o in tn_obs):
            b = dict(base, **wait_idle())
            psd = E.seed_of(a.seed_tag + "-mpilot", *k)
            P.sync()
            t1 = time.perf_counter()
            tab, Vm, V0 = SM.build_table_multi(ops, n, {o: recs[o] for o in tn_obs}, shots=a.pilot, seed=psd,
                                               chunk=chunk)
            P.sync()
            bs = time.perf_counter() - t1
            sd = E.seed_of(a.seed_tag + "-multi", *k)
            t1 = time.perf_counter()
            r = K.run(ops, n, list(E.OBS), a.N, ("table", tab), seed=sd, chunk=chunk, fast=True, opt2=True)
            P.sync()
            wt = time.perf_counter() - t1
            np.savez(os.path.join(a.raw, f"{key}_multi.npz"), tab=tab.astype(np.int8), obs=np.array(tn_obs),
                     **{f"Vm_{o}": Vm[o].astype(np.float32) for o in tn_obs},
                     **{f"v0hat_{o}": np.float64(V0[o]) for o in tn_obs},
                     **{f"all_{q}": np.asarray(r[q], np.float32) for q in E.OBS})
            bp = sum(bps.values())
            emit([dict(b, obs=o, kind="multi", scheme="multi", M=M_POL, value=float(np.mean(r[o])),
                       var=float(np.var(np.asarray(r[o], np.float64), ddof=1)), shots=a.N, wall_s=round(wt, 4),
                       bp_s=round(bp, 4), build_s=round(bs, 4), pilot_shots=a.pilot, nobs=len(tn_obs),
                       v0hat=f"{V0[o]:.6g}", nonstd=round(float(np.mean(tab != 0)), 5), seed=sd, pilot_seed=psd)
                  for o in tn_obs])
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} obs={','.join(tn_obs)} "
              f"({time.time() - t0:.0f}s) other={E.gpu_other()}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
