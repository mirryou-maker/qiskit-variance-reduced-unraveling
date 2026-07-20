"""A1 validation: GPU trajectory ensemble vs Aer density_matrix (exact).

Builds a small noisy circuit (depolarizing noise on 1-qubit gates only; CX ideal),
computes the exact reference density matrix with Aer's `density_matrix` method, then
reconstructs rho from N GPU trajectories and checks that the trace distance
D(rho_N, rho_ref) -> 0 as N grows (~ 1/sqrt(N) Monte-Carlo convergence).

Also checks endianness alignment against Qiskit at param=0 (pure state).

Run inside WSL venv:  python /mnt/d/Claude-Code-R/Qiskit-CO/validation/test_a1_tracedistance.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trajectory.gpu_trajectory import (  # noqa: E402
    GPUTrajectorySim, gate_lib, ry, depolarizing_1q_probs, sample_pauli_noise,
)

import cupy as cp  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector, DensityMatrix  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: E402

SEED = 20260719
N_QUBITS = 6
PARAM = 0.05                       # depolarizing strength on 1q gates
NOISY_1Q = ["h", "ry"]
CHECKPOINTS = [200, 1000, 5000, 20000, 60000]

# Circuit definition shared by both engines: list of (name, qubits, params)
# GHZ-ish + rotations for non-trivial mixed state.
def build_ops(n):
    ops = [("h", [0], [])]
    for i in range(n - 1):
        ops.append(("cx", [i, i + 1], []))
    for i in range(n):
        ops.append(("ry", [i], [0.3 * (i + 1)]))
    ops.append(("cx", [0, n - 1], []))
    return ops


def qiskit_circuit(n, ops):
    qc = QuantumCircuit(n)
    for name, qs, ps in ops:
        getattr(qc, name)(*(ps + qs))
    return qc


def aer_reference_rho(n, ops, param):
    qc = qiskit_circuit(n, ops)
    qc.save_density_matrix()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(param, 1), NOISY_1Q)
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    rho = sim.run(qc, shots=1, seed_simulator=SEED).result().data(0)["density_matrix"]
    return np.asarray(DensityMatrix(rho).data)


def trace_distance(rho_a, rho_b):
    d = rho_a - rho_b
    d = 0.5 * (d + d.conj().T)  # hermitize numerical noise
    ev = cp.linalg.eigvalsh(d)
    return float(0.5 * cp.sum(cp.abs(ev)))


def run_trajectories(n, ops, param, n_traj, checkpoints):
    sim = GPUTrajectorySim(n, backend="cupy")
    rng = np.random.default_rng(SEED)
    probs = depolarizing_1q_probs(param)
    dim = 2 ** n
    rho = cp.zeros((dim, dim), dtype=cp.complex128)
    snaps = {}
    cps = sorted(checkpoints)
    for t in range(1, n_traj + 1):
        sim.reset()
        for name, qs, ps in ops:
            if name == "cx":
                sim.apply_cx(qs[0], qs[1])
            elif name == "h":
                sim.apply_1q(sim.g["H"], qs[0])
                sample_pauli_noise(sim, qs[0], probs, rng)
            elif name == "ry":
                sim.apply_1q(ry(sim.xp, ps[0]), qs[0])
                sample_pauli_noise(sim, qs[0], probs, rng)
        psi = sim.statevector()
        rho += cp.outer(psi, cp.conj(psi))
        if t in cps:
            snaps[t] = (rho / t).copy()
    return snaps


def main():
    ops = build_ops(N_QUBITS)

    # --- endianness sanity at param=0 (pure state) ---
    sim0 = GPUTrajectorySim(N_QUBITS, backend="cupy")
    for name, qs, ps in ops:
        if name == "cx":
            sim0.apply_cx(qs[0], qs[1])
        elif name == "h":
            sim0.apply_1q(sim0.g["H"], qs[0])
        elif name == "ry":
            sim0.apply_1q(ry(sim0.xp, ps[0]), qs[0])
    mine = cp.asnumpy(sim0.statevector())
    ref_sv = np.asarray(Statevector(qiskit_circuit(N_QUBITS, ops)))
    fid = abs(np.vdot(ref_sv, mine)) ** 2
    print(f"[endianness/ideal] fidelity vs Qiskit Statevector = 1 - {1-fid:.2e}  "
          f"{'PASS' if fid >= 1 - 1e-10 else 'FAIL'}")

    # --- exact reference density matrix (Aer) ---
    rho_ref = cp.asarray(aer_reference_rho(N_QUBITS, ops, PARAM))
    print(f"[reference] Aer density_matrix built: {rho_ref.shape}, "
          f"purity Tr(rho^2) = {float(cp.real(cp.trace(rho_ref @ rho_ref))):.4f}")

    # --- trajectory convergence ---
    t0 = time.time()
    snaps = run_trajectories(N_QUBITS, ops, PARAM, max(CHECKPOINTS), CHECKPOINTS)
    dt = time.time() - t0
    print(f"[trajectories] {max(CHECKPOINTS)} trajectories in {dt:.1f}s "
          f"({max(CHECKPOINTS)/dt:.0f} traj/s)")

    print("\n  N_traj    trace_distance   x sqrt(N) (should be ~const)")
    prev = None
    for N in sorted(snaps):
        td = trace_distance(snaps[N], rho_ref)
        print(f"  {N:>7d}   {td:>12.5f}     {td*np.sqrt(N):>8.3f}")
        prev = td
    # convergence assertion
    final = trace_distance(snaps[max(CHECKPOINTS)], rho_ref)
    ok = final < 0.02
    print(f"\n[A1 RESULT] final trace distance @N={max(CHECKPOINTS)} = {final:.5f} "
          f"-> {'PASS (converging to exact rho)' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
