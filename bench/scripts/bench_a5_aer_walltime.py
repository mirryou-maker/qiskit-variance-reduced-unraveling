"""A5: wall-clock of standard vs projector dephasing unravelings inside Qiskit-Aer.

Both methods are run through the same Aer statevector simulator and the same
estimator (per-shot ``save_expectation_value``), so the comparison isolates the
unraveling. For each (family, n, gt, method, config) row we record:

  shots N, wall-clock of sim.run().result() (warm, after a warm-up run),
  Aer-reported time_taken, the per-shot mean / variance of the estimator,
  and the derived time-to-target  T(eps) = (wall / N) * Var / eps^2.

Whether Aer kept its batched-shots GPU optimisation is read back from the
result metadata (``batched_shots_optimization``) and stored with the row —
mid-circuit measurement + if_test could switch it off.

Rows are appended to the CSV as they finish; rows already present are skipped
(ledger-style resume). The idle state of the GPU/CPU before each row is stored.

Example:
  python bench/scripts/bench_a5_aer_walltime.py --family product \
      --n 8 12 16 --gt 0.1 1.5 --out bench/results/a5_aer_walltime.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "validation"))
import aer_unravel_circuits as auc  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

FIELDS = [
    "family", "obs", "layers", "n", "gt", "method", "config", "device", "shots", "seed",
    "wall_s", "aer_time_taken_s", "mean", "var", "se", "exact_mean", "exact_var",
    "t_per_shot_s", "time_to_se_s", "target_se",
    "batched_shots_optimization", "parallel_shots", "num_qubits_sim", "n_events",
    "gpu_util_pct_before", "gpu_mem_mib_before", "loadavg1_before", "host",
    "qiskit", "qiskit_aer", "timestamp", "raw_file",
]

CONFIGS = {
    # Aer defaults plus GPU shot batching (max 16 qubits by default in Aer 0.15)
    "batched": dict(batched_shots_gpu=True),
    "batched_maxq": dict(batched_shots_gpu=True, batched_shots_gpu_max_qubits=30),
    "plain": dict(batched_shots_gpu=False),
}


def idle_state():
    util, mem = "", ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        rows = [r.split(", ") for r in out.strip().splitlines()]
        for r in rows:
            if vis == "" or r[0] == vis:
                util, mem = r[1], r[2]
                break
    except Exception:
        pass
    try:
        la = os.getloadavg()[0]
    except Exception:
        la = ""
    return util, mem, la


_EXACT_CACHE = {}


def build(family, n, gt, method, args):
    if family == "product":
        qc = auc.product_circuit(n, gt, method)
        ex = auc.product_exact(n, gt)
        return qc, ex["mean"], ex["var_" + method]
    if family in auc.FAMILIES:
        qc, _ = auc.FAMILIES[family](n, args.layers, gt, method, seed=args.circ_seed, obs=args.obs)
        ex = ""
        if n <= args.exact_max_n:
            k = (n, gt)
            if k not in _EXACT_CACHE:
                t0 = time.perf_counter()
                _EXACT_CACHE[k] = auc.family_exact(family, n, args.layers, gt, seed=args.circ_seed,
                                                   obs=args.obs, device=args.device)
                print(f"exact DM n={n} gt={gt}: {_EXACT_CACHE[k]:.8f} "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            ex = _EXACT_CACHE[k]
        return qc, ex, ""
    raise ValueError(family)


def run_once(sim, qc, shots, seed):
    t0 = time.perf_counter()
    r = sim.run(qc, shots=shots, seed_simulator=seed).result()
    wall = time.perf_counter() - t0
    if not r.success:
        raise RuntimeError(r.status)
    v = np.asarray(r.data(0)["ev"], dtype=float)
    md = r.results[0].metadata
    return wall, float(r.time_taken), v, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="product", choices=["product", "brickwork", "ising"])
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--gt", type=float, nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["std", "proj"])
    ap.add_argument("--configs", nargs="+", default=["batched"])
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--budget-s", type=float, default=60.0,
                    help="target wall seconds per row (sets N from a pilot run)")
    ap.add_argument("--min-shots", type=int, default=2000)
    ap.add_argument("--max-shots", type=int, default=200000)
    ap.add_argument("--target-se", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--circ-seed", type=int, default=7)
    ap.add_argument("--obs", default="x", choices=["x", "zz", "xc", "zc"])
    ap.add_argument("--exact-max-n", type=int, default=12,
                    help="brickwork: compute the density_matrix reference for n <= this")
    ap.add_argument("--raw-dir", default="",
                    help="if set, save per-shot estimator values (float64 .npy) per row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import qiskit
    import qiskit_aer
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["family"], row["obs"], int(row["layers"]), int(row["n"]),
                          float(row["gt"]), row["method"], row["config"], row["device"]))
    new = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if new:
        w.writeheader()
        fout.flush()

    for cfg in args.configs:
        for n in args.n:
            for gt in args.gt:
                for m in args.methods:
                    obs_tag = args.obs if args.family != "product" else "x"
                    lay_tag = args.layers if args.family != "product" else 1
                    key = (args.family, obs_tag, lay_tag, n, float(gt), m, cfg, args.device)
                    if key in done:
                        print("skip", key, flush=True)
                        continue
                    if (m == "proj" and args.device == "GPU"
                            and CONFIGS[cfg].get("batched_shots_gpu")):
                        # Aer 0.15.1: batched_shots_gpu executor rejects the if_test control
                        # flow ("QubitVector::State::invalid instruction 'jump'"); see
                        # validation/probe_proj_variants.py
                        print("skip (unsupported: if_test under batched_shots_gpu)", key, flush=True)
                        continue
                    qc, ex_mean, ex_var = build(args.family, n, gt, m, args)
                    sim = AerSimulator(method="statevector", device=args.device, **CONFIGS[cfg])
                    # warm-up (CUDA context, allocations) and pilot for shots choice
                    run_once(sim, qc, 8 if n >= 26 else 16, args.seed + 1)
                    pilot = 16 if n >= 26 else (64 if n >= 22 else 256)
                    pw, _, _, _ = run_once(sim, qc, pilot, args.seed + 2)
                    tps = pw / pilot
                    shots = int(np.clip(args.budget_s / max(tps, 1e-9), args.min_shots, args.max_shots))
                    util, mem, la = idle_state()
                    wall, tt, v, md = run_once(sim, qc, shots, args.seed)
                    var = float(v.var(ddof=1))
                    raw_file = ""
                    if args.raw_dir:
                        os.makedirs(args.raw_dir, exist_ok=True)
                        raw_file = os.path.join(
                            args.raw_dir, f"{args.family}_{obs_tag}_L{lay_tag}_n{n}_gt{gt:g}_{m}_{cfg}_{args.device}.npy")
                        np.save(raw_file, v)
                    row = dict(
                        family=args.family, obs=obs_tag, layers=lay_tag, raw_file=raw_file, n=n, gt=gt, method=m, config=cfg, device=args.device,
                        shots=shots, seed=args.seed, wall_s=f"{wall:.6f}", aer_time_taken_s=f"{tt:.6f}",
                        mean=f"{v.mean():.8f}", var=f"{var:.8e}", se=f"{np.sqrt(var / shots):.3e}",
                        exact_mean=ex_mean, exact_var=ex_var,
                        t_per_shot_s=f"{wall / shots:.6e}",
                        time_to_se_s=f"{wall / shots * var / args.target_se ** 2:.6e}",
                        target_se=args.target_se,
                        batched_shots_optimization=md.get("batched_shots_optimization", ""),
                        parallel_shots=md.get("parallel_shots", ""),
                        num_qubits_sim=qc.num_qubits,
                        n_events=sum(1 for ins in qc.data if ins.operation.name in ("quantum_channel", "if_else")),
                        gpu_util_pct_before=util, gpu_mem_mib_before=mem, loadavg1_before=la,
                        host=os.uname().nodename if hasattr(os, "uname") else "",
                        qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    w.writerow(row)
                    fout.flush()
                    print(json.dumps({k: row[k] for k in ("n", "gt", "method", "config", "shots", "wall_s",
                                                           "mean", "var", "exact_var", "time_to_se_s",
                                                           "batched_shots_optimization")}), flush=True)
    fout.close()


if __name__ == "__main__":
    main()
