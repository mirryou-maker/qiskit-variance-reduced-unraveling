"""E10 (docs/plan_B_characterization_2026-11-06.md, "E10 pre-registration"): STATIC per-location unraveling table at n = 20
on the 98 E9b/E9e groups. A static table fixes one candidate per channel event for all trajectories before production, so
it has no per-trajectory selection cost (src/noise/static_table.py; traj_kraus.run(scheme=("table", tab))).

Per config (channel, family, variant, L, gt), all on one host, in this order:
  1. kind=fixed : every candidate re-timed with N_t = 296 trajectories (E9b seed and chunk, opt2), all 4 observables in
                  one run; identity check vs the E9b raw arrays (match / maxdiff) -> variance stays the E9b N = 2000 one.
  2. per TN observable o:  record_after(o, M = 100) + group precompute (bp_s, shared by both tables, counted in each);
     kind=static scheme=pilot : table from 74 std/jump pilot trajectories (own seed; build_s), production N = 2000
     kind=static scheme=ideal : table from the noiseless trajectory (build_s), production N = 2000
     production seeds are new (independent of pilot and of E9b). Raw per-trajectory arrays + table in --raw.
Before EVERY timed run: wait until no other python job from ~/work and no CI Runner.Worker runs and no other GPU process,
then record the idle state (gpu_util, gpu_mem, gpu_other, other_py, runner, top_cpu, load1) in the row.
Ledger CSV (resumable): rows already present are skipped.
"""
import argparse
import csv
import datetime as dt
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
from src.noise import traj_kraus as K  # noqa: E402

_spec = importlib.util.spec_from_file_location("b_e9b", os.path.join(os.path.dirname(__file__), "b_e9b.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

M_POL = 100
TABLES = ("pilot", "ideal")
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "scheme", "M", "value", "var", "shots",
          "wall_s", "bp_s", "build_s", "pilot_shots", "nonstd", "match", "maxdiff", "seed", "chunk", "host",
          "gpu_util", "gpu_mem", "gpu_other", "other_py", "runner", "top_cpu", "load1", "impl", "timestamp"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        return f"err:{e}"


def idle_state():
    me = os.getpid()
    anc, p_ = set(), me                                       # self and ancestors (the launching shell) are not "other"
    while p_ > 1:
        anc.add(str(p_))
        try:
            p_ = int(open(f"/proc/{p_}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
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
    t0 = time.time()
    last = -1e9
    while True:
        time.sleep(2)                    # let nvidia-smi's utilization window drop our own previous run
        s = idle_state()
        if s["other_py"] == "none" and s["runner"] == 0 and s["gpu_other"] == "none":
            return s
        if time.time() - last > log_every:
            print(f"{dt.datetime.now().isoformat(timespec='seconds')} waiting for idle "
                  f"({time.time() - t0:.0f}s): {s}", flush=True)
            last = time.time()
        time.sleep(28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e9b_groups.csv"))
    ap.add_argument("--e9b_raw", default=os.path.join(ROOT, "bench/results/b_e9b_raw_all"))
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--Nt", type=int, default=296)
    ap.add_argument("--pilot", type=int, default=74)
    ap.add_argument("--tables", nargs="+", default=list(TABLES))
    ap.add_argument("--mem_gb", type=float, default=5.0, help="must equal E9b (sets the chunk)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="config key substring (smoke)")
    ap.add_argument("--seed_tag", default="e10", help="smoke runs use another tag so their seeds differ")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e10_u94.csv"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_e10_raw"))
    a = ap.parse_args()
    S.IMPL = "v2"
    cfg = E.configs(a.groups)
    todo = E.partition(cfg, 0, 1, 2000, [10, 100])
    if a.only:
        todo = [k for k in todo if a.only in "_".join(map(str, k))]
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
    print(f"E10: {len(todo)} configs, N {a.N}, N_t {a.Nt}, pilot {a.pilot}, tables {a.tables}; start idle "
          f"{idle_state()}", flush=True)

    # warm-up (kernel compilation outside every timed region)
    ops0 = K.build("brickwork", "s7", 20, 4, 0.01, "ad_phase")
    rec0, _, _ = S.record_after(ops0, 20, "zz", M_POL)
    ST.build_table(ops0, 20, rec0, "pilot", shots=4, seed=0, chunk=4)
    ST.build_table(ops0, 20, rec0, "ideal")
    K.run(ops0, 20, list(E.OBS), 4, ("table", np.zeros(len(rec0), int)), seed=0, chunk=4, fast=True, opt2=True)
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
        key = f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"
        raw = np.load(os.path.join(a.e9b_raw, key + ".npz"))
        ops = K.build(fam, var, n, L, gt, E.CHMAP[ch])
        labels = next(op[2].labels for op in ops if op[0] == "ch")
        base = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, chunk=chunk, impl="v2+opt2")
        t0 = time.time()
        for c, lab in enumerate(labels):
            if all(k + ("fixed", lab, o) in done for o in tn_obs):
                continue
            b = dict(base, **wait_idle())
            sd = E.seed_of("e9b-fixed", *k, c)
            P.sync()
            t1 = time.perf_counter()
            r = K.run(ops, n, list(E.OBS), a.Nt, ("fixed", c), seed=sd, chunk=chunk, fast=True, opt2=True)
            P.sync()
            wt = time.perf_counter() - t1
            rows = []
            for o in tn_obs:
                x = np.asarray(r[o], np.float64)
                ref = raw[f"fixed_{lab}_{o}"].astype(np.float64)[:len(x)]
                d = np.abs(x - ref)
                rows.append(dict(b, obs=o, kind="fixed", scheme=lab, M="", shots=a.Nt, seed=sd, wall_s=round(wt, 4),
                                 match=round(float(np.mean(d <= 1e-5 * np.maximum(1.0, np.abs(ref)))), 5),
                                 maxdiff=f"{d.max():.3g}"))
            emit(rows)
        for o in tn_obs:
            todo_t = [t for t in a.tables if k + ("static", t, o) not in done]
            if not todo_t:
                continue
            P.sync()
            t1 = time.perf_counter()
            rec, _, _ = S.record_after(ops, n, o, M_POL)
            for ev in rec:
                S._groups(ev, S.GMAX)
            P.sync()
            bp = time.perf_counter() - t1
            for t in todo_t:
                b = dict(base, **wait_idle())
                psd = E.seed_of(a.seed_tag + "-pilot", *k, o)
                P.sync()
                t1 = time.perf_counter()
                tab, Vm = ST.build_table(ops, n, rec, t, shots=a.pilot, seed=psd, chunk=chunk)
                P.sync()
                bs = time.perf_counter() - t1
                sd = E.seed_of(a.seed_tag + "-static", t, *k, o)
                t1 = time.perf_counter()
                r = K.run(ops, n, list(E.OBS), a.N, ("table", tab), seed=sd, chunk=chunk, fast=True, opt2=True)
                P.sync()
                wt = time.perf_counter() - t1
                x = np.asarray(r[o], np.float64)
                np.savez(os.path.join(a.raw, f"{key}_{t}_{o}.npz"), x=x.astype(np.float32), tab=tab.astype(np.int8),
                         Vm=Vm.astype(np.float32), **{f"all_{q}": np.asarray(r[q], np.float32) for q in E.OBS})
                emit([dict(b, obs=o, kind="static", scheme=t, M=M_POL, value=float(x.mean()), var=float(x.var(ddof=1)),
                           shots=len(x), wall_s=round(wt, 4), bp_s=round(bp, 4), build_s=round(bs, 4),
                           pilot_shots=a.pilot if t == "pilot" else 0, nonstd=round(float(np.mean(tab != 0)), 5),
                           seed=sd)])
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} obs={','.join(tn_obs)} "
              f"({time.time() - t0:.0f}s) other={E.gpu_other()}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
