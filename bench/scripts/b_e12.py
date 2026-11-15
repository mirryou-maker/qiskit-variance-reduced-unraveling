"""E12 (docs/plan_B_characterization_2026-11-06.md, "E12 pre-registration"): sensitivity of the static PILOT table (E10) to
the pilot size and to the truncation M of O~, on the 98 E9b/E9e groups at n = 20. A sensitivity study, not a decision test.

Settings (pilot trajectories P, truncation M):  (37, 100), (74, 100), (296, 100), (74, 10), (74, 1000).
Per config (channel, family, variant, L, gt), all in one session on one GPU, in this order:
  1. kind=fixed : every candidate re-timed with N_t = 296 (E9b seed and chunk, opt2), identity check vs the E9b raw arrays
                  (exactly as b_e10.py; the variance stays the E9b N = 2000 one).
  2. per TN observable o, per M in (10, 100, 1000): record_after(o, M) + group precompute (bp_s, counted in full for every
     setting that uses this M); per pilot size P with (P, M) in the settings:
       kind=static scheme=p<P>_M<M> : table from P std/jump pilot trajectories (build_s), production N = 2000.
     The pilot seed depends only on (config, o), not on (P, M): with chunk 37 the 37-pilot is the first chunk of the
     74-pilot and that is the first two chunks of the 296-pilot (same RNG stream, same chunk sizes), and the three M
     values at P = 74 see the same pilot trajectories (the pilot evolution does not depend on M). Production seeds are new
     for every (setting, config, o).
Idle checks, ledger and raw files as b_e10.py (--idle pbs on iREMB).
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
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import static_table as ST  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402

_spec = importlib.util.spec_from_file_location("b_e10", os.path.join(os.path.dirname(__file__), "b_e10.py"))
B10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B10)
E = B10.E

SETTINGS = ((37, 100), (74, 100), (296, 100), (74, 10), (74, 1000))
FIELDS = B10.FIELDS


def label(p, m):
    return f"p{p}_M{m}"


def by_M(settings):
    """{M: [P, ...]} in first-appearance order of M (ascending M for the default settings)."""
    out = {}
    for p, m in sorted(settings, key=lambda s: (s[1], s[0])):
        out.setdefault(m, []).append(p)
    return out


def pilot_seed(tag, k, o):
    return E.seed_of(tag + "-pilot", *k, o)


def prod_seed(tag, p, m, k, o):
    return E.seed_of(tag + "-static", p, m, *k, o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=os.path.join(ROOT, "bench/results/b_e9b_groups.csv"))
    ap.add_argument("--e9b_raw", default=os.path.join(ROOT, "bench/results/b_e9b_raw_all"))
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--Nt", type=int, default=296)
    ap.add_argument("--settings", default=",".join(f"{p}:{m}" for p, m in SETTINGS), help="P:M,P:M,...")
    ap.add_argument("--mem_gb", type=float, default=5.0, help="must equal E9b (sets the chunk)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="config key substring (smoke)")
    ap.add_argument("--subset", default=None, help="csv of configs (channel,family,variant,n,L,gt) to keep")
    ap.add_argument("--seed_tag", default="e12", help="smoke runs use another tag so their seeds differ")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e12_v100.csv"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_e12_raw"))
    ap.add_argument("--idle", choices=["u94", "pbs"], default="pbs")
    a = ap.parse_args()
    B10.IDLE_MODE = a.idle
    S.IMPL = "v2"
    settings = [tuple(int(x) for x in s.split(":")) for s in a.settings.split(",")]
    groupsM = by_M(settings)
    cfg = E.configs(a.groups)
    todo = E.partition(cfg, 0, 1, 2000, [10, 100])
    if a.only:
        todo = [k for k in todo if a.only in "_".join(map(str, k))]
    if a.subset:
        keep = {(r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]))
                for r in csv.DictReader(open(a.subset))}
        todo = [k for k in todo if k in keep]
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
    print(f"E12: {len(todo)} configs, N {a.N}, N_t {a.Nt}, settings {settings}; start idle {B10.idle_state()}", flush=True)

    ops0 = K.build("brickwork", "s7", 20, 4, 0.01, "ad_phase")          # warm-up outside every timed region
    for m in groupsM:
        rec0, _, _ = S.record_after(ops0, 20, "zz", m)
        ST.build_table(ops0, 20, rec0, "pilot", shots=4, seed=0, chunk=4)
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
            b = dict(base, **B10.wait_idle())
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
            for m, ps in groupsM.items():
                todo_p = [p for p in ps if k + ("static", label(p, m), o) not in done]
                if not todo_p:
                    continue
                P.sync()
                t1 = time.perf_counter()
                rec, _, _ = S.record_after(ops, n, o, m)
                for ev in rec:
                    S._groups(ev, S.GMAX)
                P.sync()
                bp = time.perf_counter() - t1
                for p in todo_p:
                    b = dict(base, **B10.wait_idle())
                    P.sync()
                    t1 = time.perf_counter()
                    tab, Vm = ST.build_table(ops, n, rec, "pilot", shots=p, seed=pilot_seed(a.seed_tag, k, o), chunk=chunk)
                    P.sync()
                    bs = time.perf_counter() - t1
                    sd = prod_seed(a.seed_tag, p, m, k, o)
                    t1 = time.perf_counter()
                    r = K.run(ops, n, list(E.OBS), a.N, ("table", tab), seed=sd, chunk=chunk, fast=True, opt2=True)
                    P.sync()
                    wt = time.perf_counter() - t1
                    x = np.asarray(r[o], np.float64)
                    np.savez(os.path.join(a.raw, f"{key}_{label(p, m)}_{o}.npz"), x=x.astype(np.float32),
                             tab=tab.astype(np.int8), Vm=Vm.astype(np.float32),
                             **{f"all_{q}": np.asarray(r[q], np.float32) for q in E.OBS})
                    emit([dict(b, obs=o, kind="static", scheme=label(p, m), M=m, value=float(x.mean()),
                               var=float(x.var(ddof=1)), shots=len(x), wall_s=round(wt, 4), bp_s=round(bp, 4),
                               build_s=round(bs, 4), pilot_shots=p, nonstd=round(float(np.mean(tab != 0)), 5), seed=sd)])
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} obs={','.join(tn_obs)} "
              f"({time.time() - t0:.0f}s) other={E.gpu_other()}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
