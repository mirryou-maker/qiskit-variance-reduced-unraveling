"""E8 (docs/plan_B_characterization_2026-11-06.md): greedy with a truncated-Pauli O~_t. Variance only.

Dense emulation: tests the information in M Pauli terms, not the cost of a sparse rule.
Schemes: greedyM<M> for every M; with --baseline also every fixed candidate and exact greedy
(n = 10 reuses the E7 runs instead). Ledger CSV; raw float32 estimates in bench/results/b_e8_raw.
pp_mean = <0|O~_0^M|0>, the truncated-propagation answer alone.
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import socket
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import traj_kraus as K  # noqa: E402
from src.noise import traj_ref as T  # noqa: E402
from src.noise.traj_ref import xp  # noqa: E402

VARIANTS = [("brickwork", "s7"), ("brickwork", "s8"), ("brickwork", "s9"),
            ("ising", "h0.7"), ("ising", "h1.4")]
OBS = ("x", "zz", "xc", "zc")
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "scheme", "M", "shots", "exact", "mean",
          "z", "var", "pp_mean", "seed", "wall_s", "host", "timestamp"]


def seed_of(*k):
    return int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", nargs="+", default=["deph_sweep", "ad_phase"])
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--gt", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--M", type=int, nargs="+", default=[1, 10, 100, 1000, 10000])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e8.csv"))
    ap.add_argument("--rawname", default="b_e8_raw")
    a = ap.parse_args()
    rawdir = os.path.join(ROOT, "bench/results", a.rawname)
    os.makedirs(rawdir, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["channel"], r["family"], r["variant"], int(r["n"]), int(r["L"]),
                      float(r["gt"]), r["obs"], r["scheme"]))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    host = socket.gethostname()

    for ch in a.channel:
        labels = K.CHANNELS[ch](0.1).labels
        mschemes = [f"greedyM{m}" for m in a.M]
        schemes = mschemes + ([f"fix:{l}" for l in labels] + ["greedy"] if a.baseline else [])
        for n in a.n:
            for fam, var in VARIANTS:
                for L in a.L:
                    for gt in a.gt:
                        if all((ch, fam, var, n, L, gt, o, s) in done for o in OBS for s in schemes):
                            continue
                        key = f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"
                        ops = K.build(fam, var, n, L, gt, ch)
                        sd = seed_of("e8", ch, fam, var, n, L, gt)
                        t0 = time.time()
                        exact = {o: K.exact_mean(ops, n, o) for o in OBS}
                        raw, rows = {}, []

                        def row(o, s, x, m, pp, seed, wall):
                            v = float(x.var(ddof=1))
                            raw[f"{o}_{s}"] = x.astype(np.float32)
                            return dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, obs=o, scheme=s,
                                        M=m, shots=len(x), exact=exact[o], mean=float(x.mean()),
                                        z=(float(x.mean()) - exact[o]) / np.sqrt(max(v, 1e-300) / len(x)),
                                        var=v, pp_mean=pp, seed=seed, wall_s=round(wall, 2), host=host)

                        if a.baseline:
                            for c, l in enumerate(labels):
                                t1 = time.time()
                                r = K.run(ops, n, list(OBS), a.shots, ("fixed", c), seed=sd + c, chunk=a.chunk)
                                for o in OBS:
                                    rows.append(row(o, f"fix:{l}", r[o], "", "", sd + c, time.time() - t1))
                            for o in OBS:
                                t1 = time.time()
                                r = K.run(ops, n, [o], a.shots, ("greedy", None), seed=sd + 100, chunk=a.chunk,
                                          obs_for_policy=o)
                                rows.append(row(o, "greedy", r[o], "", "", sd + 100, time.time() - t1))
                        psi0 = xp.zeros(2 ** n, complex)
                        psi0[0] = 1.0
                        for j, m in enumerate(a.M):
                            for o in OBS:
                                t1 = time.time()
                                after, O0 = K.backprop_truncated(ops, n, T.obs_matrix(o, n), m)
                                pp = float(xp.real(xp.vdot(psi0, O0 @ psi0)))
                                r = K.run(ops, n, [o], a.shots, ("greedy", None), seed=sd + 200 + j, chunk=a.chunk,
                                          obs_for_policy=o, after=after)
                                del after
                                rows.append(row(o, f"greedyM{m}", r[o], m, pp, sd + 200 + j, time.time() - t1))
                        np.savez_compressed(os.path.join(rawdir, key + ".npz"), **raw)
                        ts = dt.datetime.now().isoformat(timespec="seconds")
                        for rr in rows:
                            rr["timestamp"] = ts
                            w.writerow(rr)
                        f.flush()
                        print(f"{ts} {key:38} ({time.time() - t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
