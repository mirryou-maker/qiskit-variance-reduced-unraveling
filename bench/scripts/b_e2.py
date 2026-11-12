"""E2 (docs/plan_B_characterization_2026-11-06.md): adaptive vs best-fixed for depolarizing and
amplitude damping. Variance only (no timing claims). Ledger CSV; raw float32 estimates in
bench/results/b_e2_raw/<channel>_<circuit>.npz (keys '<obs>_<scheme>', plus '<obs>_choice').
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

VARIANTS = [("brickwork", "s7"), ("brickwork", "s8"), ("brickwork", "s9"),
            ("ising", "h0.7"), ("ising", "h1.4")]
OBS = ("x", "zz", "xc", "zc")
FIELDS = ["channel", "family", "variant", "n", "L", "gt", "obs", "scheme", "shots", "exact", "mean",
          "z", "var", "frac_std", "seed", "wall_s", "host", "timestamp"]


def seed_of(*k):
    return int(hashlib.md5(repr(k).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", nargs="+", default=["depol", "ad"])
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8, 10])
    ap.add_argument("--L", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--gt", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench/results/b_e2.csv"))
    ap.add_argument("--rawname", default="b_e2_raw", help="raw directory under bench/results (E7: b_e7_raw)")
    ap.add_argument("--families", nargs="+", default=None, help="restrict circuit families (E7b: ising)")
    a = ap.parse_args()
    variants = [v for v in VARIANTS if a.families is None or v[0] in a.families]
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
        schemes = [f"fix:{l}" for l in labels] + ["greedy"]
        for n in a.n:
            for fam, var in variants:
                for L in a.L:
                    for gt in a.gt:
                        if all((ch, fam, var, n, L, gt, o, s) in done for o in OBS for s in schemes):
                            continue
                        key = f"{ch}_{fam}_{var}_n{n}_L{L}_gt{gt}"
                        ops = K.build(fam, var, n, L, gt, ch)
                        sd = seed_of(ch, fam, var, n, L, gt)
                        t0 = time.time()
                        exact = {o: K.exact_mean(ops, n, o) for o in OBS}
                        raw, rows = {}, []

                        def row(o, s, x, fs, seed, wall):
                            v = float(x.var(ddof=1))
                            raw[f"{o}_{s}"] = x.astype(np.float32)
                            return dict(channel=ch, family=fam, variant=var, n=n, L=L, gt=gt, obs=o,
                                        scheme=s, shots=len(x), exact=exact[o], mean=float(x.mean()),
                                        z=(float(x.mean()) - exact[o]) / np.sqrt(max(v, 1e-300) / len(x)),
                                        var=v, frac_std=fs, seed=seed, wall_s=round(wall, 2), host=host)

                        for c, l in enumerate(labels):
                            t1 = time.time()
                            r = K.run(ops, n, list(OBS), a.shots, ("fixed", c), seed=sd + c, chunk=a.chunk)
                            for o in OBS:
                                rows.append(row(o, f"fix:{l}", r[o], float(c == 0), sd + c, time.time() - t1))
                        gs = sd + 100
                        for o in OBS:
                            t2 = time.time()
                            r = K.run(ops, n, [o], a.shots, ("greedy", None), seed=gs, chunk=a.chunk,
                                      obs_for_policy=o)
                            raw[f"{o}_choice"] = r["_choice"].astype(np.float32)
                            rows.append(row(o, "greedy", r[o], float(np.mean(r["_choice"][:, 0])), gs,
                                            time.time() - t2))
                        np.savez_compressed(os.path.join(rawdir, key + ".npz"), **raw)
                        ts = dt.datetime.now().isoformat(timespec="seconds")
                        for rr in rows:
                            rr["timestamp"] = ts
                            w.writerow(rr)
                        f.flush()
                        vs = {(rr["obs"], rr["scheme"]): rr["var"] for rr in rows}
                        msg = " ".join(
                            f"{o}:G={min(vs[(o, s)] for s in schemes[:-1]) / max(vs[(o, 'greedy')], 1e-300):.2f}"
                            for o in OBS)
                        print(f"{ts} {key:38} {msg} ({time.time() - t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
