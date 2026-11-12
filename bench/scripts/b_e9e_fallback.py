"""E9e fallback (pre-registered in docs/plan_B_characterization_2026-11-06.md, "E9e pre-registration", Variance):
a greedy run whose E9e timing run matched < 99% of the E9b trajectories is re-run in full (N = 2000) with the new code
(sparse_policy.IMPL = "v2", traj_kraus.run(opt2=True)), same E9b seed and chunk, and its variance is taken from that run.
Fixed runs all matched (100%), so they are not re-run; times stay the E9e timing-run times.

Run list: greedy rows of the E9e primary CSV (b_e9e_u94.csv) with match < 0.99. Split over hosts with --part/--nparts
and --weights (relative host speed), by the u94 timing-run wall. Ledger CSV (resumable) + one raw npz per run.
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
from src.noise import traj_kraus as K  # noqa: E402

_spec = importlib.util.spec_from_file_location("b_e9b", os.path.join(os.path.dirname(__file__), "b_e9b.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "kind", "scheme", "M", "value", "var", "shots",
          "wall_s", "bp_s", "nonstd", "match_e9b", "seed", "chunk", "host", "gpu_other", "load1", "impl", "timestamp"]


def runs(e9e_csv):
    out = []
    for r in csv.DictReader(open(e9e_csv)):
        if r["kind"] == "greedy" and float(r["match"]) < 0.99:
            out.append(((r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"])),
                        r["obs"], int(float(r["M"])), float(r["wall_s"])))
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def split(rs, part, wts):
    load = [0.0] * len(wts)
    out = [[] for _ in wts]
    for r in sorted(rs, key=lambda t: -t[3]):
        j = min(range(len(wts)), key=lambda j: (load[j] + r[3]) / wts[j])
        load[j] += r[3]
        out[j].append(r)
    return sorted(out[part], key=lambda t: (t[0], t[1], t[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e9e", default=os.path.join(ROOT, "bench/results/b_e9e_u94.csv"))
    ap.add_argument("--e9b_raw", default=os.path.join(ROOT, "bench/results/b_e9b_raw_all"))
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--mem_gb", type=float, default=5.0, help="must equal E9b (sets the chunk)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e9e_fb.csv"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "bench/results/b_e9e_fb_raw"))
    a = ap.parse_args()
    S.IMPL = "v2"
    todo = split(runs(a.e9e), a.part, a.weights)
    os.makedirs(a.raw, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]), float(r["gt"]), r["obs"],
                      int(float(r["M"]))))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    host = socket.gethostname()
    print(f"E9e fallback part {a.part} weights {a.weights}: {len(todo)} runs", flush=True)
    for k, o, m, _ in todo:
        if k + (o, m) in done:
            continue
        ch, fam, var, n, L, gt = k
        chunk = int(max(1, min(4096, a.mem_gb * 1e9 / (8 * 16 * 2 ** n))))
        key = f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"
        ops = K.build(fam, var, n, L, gt, E.CHMAP[ch])
        b = dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, host=host, chunk=chunk, impl="v2+opt2",
                 gpu_other=E.gpu_other(), load1=E.load1())
        sd = E.seed_of("e9b-greedy", *k, o, m)
        P.sync()
        t1 = time.perf_counter()
        rec, _, _ = S.record_after(ops, n, o, m)
        for ev in rec:
            S._groups(ev, S.GMAX)
        P.sync()
        bp = time.perf_counter() - t1
        t1 = time.perf_counter()
        r = K.run(ops, n, list(E.OBS), a.N, ("greedy", None), seed=sd, chunk=chunk, after=rec, fast=True, opt2=True)
        P.sync()
        wt = time.perf_counter() - t1
        x = np.asarray(r[o], np.float64)
        ref = np.load(os.path.join(a.e9b_raw, key + ".npz"))[f"greedy_M{m}_{o}"].astype(np.float64)
        mt = float(np.mean(np.abs(x - ref) <= 1e-5 * np.maximum(1.0, np.abs(ref))))
        np.savez(os.path.join(a.raw, f"{key}_M{m}_{o}.npz"), x=x.astype(np.float32))
        w.writerow(dict(b, obs=o, kind="greedy", scheme="greedy", M=m, value=float(x.mean()), var=float(x.var(ddof=1)),
                        shots=len(x), wall_s=round(wt, 3), bp_s=round(bp, 3),
                        nonstd=round(float(1 - r["_choice"][:, 0].mean()), 5), match_e9b=round(mt, 5), seed=sd,
                        timestamp=dt.datetime.now().isoformat(timespec="seconds")))
        f.flush()
        print(f"{dt.datetime.now().isoformat(timespec='seconds')} {key:40} {o} M{m} ({wt:.0f}s) match {mt:.3f} "
              f"other={E.gpu_other()}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
