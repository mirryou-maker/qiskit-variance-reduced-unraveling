"""A4 head-to-head: variance-reduced GPU trajectory vs Aer batched_shots_gpu.

Same physical model both engines can represent: per-qubit Z-dephasing over time gt on
|+>^n, observable mean_i <X_i> (exact value e^{-2 gt}). Metric: number of trajectories/shots
AND wall-clock to reach a target standard error, comparing:
  (1) Aer  batched_shots_gpu  (standard trajectory + single-shot X-basis measurement)
  (2) ours standard unraveling (expectation per trajectory)
  (3) ours projector unraveling (variance-reduced expectation per trajectory)

(2 vs 1): benefit of expectation-per-trajectory. (3 vs 2): benefit of projector variance reduction.
Wall-clock is reported honestly: Aer is optimized C++/CUDA; our engine is a Python prototype.
The variance reduction is a statistical property that transfers to Aer's C++ path once integrated
(the upstream-PR argument).

Run:  python -u /mnt/d/Claude-Code-R/Qiskit-CO/validation/test_a4_vs_aer.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trajectory.gpu_trajectory import GPUTrajectorySim  # noqa: E402

import cupy as cp  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, pauli_error  # noqa: E402

SEED = 20260719
TARGET_SE = 0.01
THETA0 = 0.30
N_QUBITS = 10
GT = 1.5                                   # strong-noise regime (projector favored)
P = (1.0 - np.exp(-2.0 * GT)) / 2.0        # dephasing time gt -> per-qubit Z-error prob


def aer_shot_values(n, p, shots):
    """Run Aer batched_shots_gpu; return per-shot averaged X value (mean_i (1-2 b_i))."""
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.h(q)          # |+>
    for q in range(n):
        qc.id(q)         # noise carrier
    for q in range(n):
        qc.h(q)          # rotate to X basis
    qc.measure(range(n), range(n))
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(pauli_error([("Z", p), ("I", 1 - p)]), ["id"])
    sim = AerSimulator(method="statevector", device="GPU", noise_model=nm,
                       batched_shots_gpu=True)
    t0 = time.time()
    res = sim.run(qc, shots=shots, memory=True, seed_simulator=SEED).result()
    dt = time.time() - t0
    mem = res.get_memory()                 # list of bitstrings, little-endian string
    # bit '0' -> +1, '1' -> -1 ; average over qubits per shot
    vals = np.array([np.mean([1.0 - 2.0 * int(c) for c in bs]) for bs in mem])
    return vals, dt


def ours_traj_values(n, gt, unravel, N):
    sim = GPUTrajectorySim(n, backend="cupy")
    rng = np.random.default_rng(SEED + 5)
    vals = np.empty(N)
    t0 = time.time()
    for t in range(N):
        sim.reset()
        for q in range(n):
            sim.apply_1q(sim.g["H"], q)
        for q in range(n):
            if unravel == "standard":
                if rng.poisson(gt) % 2 == 1:
                    sim.apply_1q(sim.g["Z"], q)
            elif unravel == "projector":
                if rng.random() < (1.0 - np.exp(-2.0 * gt)):
                    plus = rng.random() < sim.prob_plus("Z", q)
                    sim.apply_projector("Z", q, plus)
        vals[t] = sum(sim.expval_1q(sim.g["X"], q) for q in range(n)) / n
    return vals, time.time() - t0


def summarize(name, vals, wall):
    mean = vals.mean()
    std = vals.std(ddof=1)
    n_to_target = (std / TARGET_SE) ** 2
    thr = len(vals) / wall
    print(f"  {name:>22}: mean={mean:+.4f}  per-sample std={std:.4f}  "
          f"N_to_SE<=0.01={n_to_target:>7.0f}  ({len(vals)} in {wall:.1f}s, {thr:.0f}/s)")
    return mean, n_to_target, wall / len(vals)


def main():
    print(f"A4 head-to-head: n={N_QUBITS}, gt={GT}, Z-error p={P:.4f}, "
          f"exact mean=e^(-2gt)={np.exp(-2*GT):.4f}, target SE={TARGET_SE}\n")

    va, ta = aer_shot_values(N_QUBITS, P, 40000)
    vs, ts = ours_traj_values(N_QUBITS, GT, "standard", 8000)
    vp, tp = ours_traj_values(N_QUBITS, GT, "projector", 8000)

    m_a, N_a, c_a = summarize("Aer batched_shots_gpu", va, ta)
    m_s, N_s, c_s = summarize("ours standard", vs, ts)
    m_p, N_p, c_p = summarize("ours projector", vp, tp)

    print("\n  --- mean cross-check (all should ~ e^-2gt = %.4f) ---" % np.exp(-2 * GT))
    print(f"    Aer={m_a:+.4f}  ours_std={m_s:+.4f}  ours_proj={m_p:+.4f}")

    print("\n  --- trajectories/shots to target SE ---")
    print(f"    Aer                : {N_a:>7.0f}")
    print(f"    ours standard      : {N_s:>7.0f}  ({N_a/N_s:.1f}x fewer than Aer)")
    print(f"    ours projector     : {N_p:>7.0f}  ({N_a/N_p:.1f}x fewer than Aer, "
          f"{N_s/N_p:.1f}x fewer than ours-standard)")

    print("\n  --- wall-clock to target SE (honest: Aer=C++/CUDA, ours=Python prototype) ---")
    wa, ws, wp = N_a * c_a, N_s * c_s, N_p * c_p
    print(f"    Aer                : {wa*1e3:>8.1f} ms")
    print(f"    ours standard      : {ws*1e3:>8.1f} ms")
    print(f"    ours projector     : {wp*1e3:>8.1f} ms  "
          f"(projector needs {N_s/N_p:.1f}x fewer traj but {c_p/c_s:.1f}x cost/traj)")
    print("\n  NOTE: variance reduction is engine-independent; porting projector unraveling into")
    print("        Aer's C++ batched path would carry the ~{:.0f}x trajectory saving to wall-clock."
          .format(N_a / N_p))


if __name__ == "__main__":
    main()
