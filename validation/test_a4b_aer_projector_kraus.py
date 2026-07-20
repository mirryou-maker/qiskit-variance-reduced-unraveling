"""A4b: projector unraveling INSIDE Aer via a custom Kraus decomposition (no C++ fork).

Key idea: the single-qubit dephasing channel E(rho)=(1-p) rho + p Z rho Z has the usual
Kraus set {sqrt(1-p) I, sqrt(p) Z} (-> Aer samples I/Z = STANDARD unraveling, since both are
unitary). The SAME channel also admits the projector decomposition
    {sqrt(1-2p) I, sqrt(2p) Pi+, sqrt(2p) Pi-},  Pi+/-=diag(1,0)/diag(0,1),
which is trace preserving ((1-2p)I + 2p(Pi+ + Pi-) = I) and reproduces the identical map
(off-diagonal factor 1-2p). Because Pi+/- are NON-unitary, Aer keeps it as a genuine Kraus
error and samples one operator with prob ||K_i psi||^2 (quantum jump) -> collapse -> absorbing
window -> the SAME variance reduction we built by hand in A2-A4, but now running on Aer's own
(C++/CUDA) trajectory engine.

If Aer reproduces the ~20x trajectory reduction here, the "upstream integration" is essentially
free: ship projector unraveling as a NoiseModel transformation, no fork, and the wall-clock win is
realized on Aer's optimized engine rather than projected.

Run:  python -u /mnt/d/Claude-Code-R/Qiskit-CO/validation/test_a4b_aer_projector_kraus.py [cpu|gpu]
"""
import sys
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, kraus_error

SEED = 20260719
TARGET_SE = 0.01
N_QUBITS = 10
GT = 1.5
P = (1.0 - np.exp(-2.0 * GT)) / 2.0

I2 = np.eye(2, dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PIp = np.array([[1, 0], [0, 0]], dtype=complex)
PIm = np.array([[0, 0], [0, 1]], dtype=complex)


def standard_kraus(p):
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z2]


def projector_kraus(p):
    return [np.sqrt(1 - 2 * p) * I2, np.sqrt(2 * p) * PIp, np.sqrt(2 * p) * PIm]


def run(kind, kraus, device, shots):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    for q in range(N_QUBITS):
        qc.h(q)          # |+>
    for q in range(N_QUBITS):
        qc.id(q)         # noise carrier
    for q in range(N_QUBITS):
        qc.h(q)          # X-basis readout
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(kraus_error(kraus), ["id"])
    sim = AerSimulator(method="statevector", device=device, noise_model=nm)
    t0 = time.time()
    res = sim.run(qc, shots=shots, memory=True, seed_simulator=SEED).result()
    dt = time.time() - t0
    vals = np.array([np.mean([1.0 - 2.0 * int(c) for c in bs]) for bs in res.get_memory()])
    std = vals.std(ddof=1)
    n_to_target = (std / TARGET_SE) ** 2
    print(f"  {kind:>20} [{device}]: mean={vals.mean():+.4f}  std={std:.4f}  "
          f"N_to_SE<=0.01={n_to_target:>7.0f}  ({shots} shots in {dt:.2f}s, {shots/dt:.0f}/s)")
    return vals.mean(), n_to_target, dt / shots


def main():
    device = "GPU" if (len(sys.argv) > 1 and sys.argv[1] == "gpu") else "CPU"
    shots = 40000
    print(f"A4b: projector unraveling via Aer Kraus decomposition — device={device}")
    print(f"n={N_QUBITS}, gt={GT}, p={P:.4f}, exact mean e^-2gt={np.exp(-2*GT):.4f}, "
          f"target SE={TARGET_SE}\n")
    m_s, N_s, c_s = run("standard {I,Z}", standard_kraus(P), device, shots)
    m_p, N_p, c_p = run("projector {I,Pi+,Pi-}", projector_kraus(P), device, shots)

    print("\n  --- result ---")
    print(f"    means: standard={m_s:+.4f}  projector={m_p:+.4f}  (exact {np.exp(-2*GT):+.4f})")
    print(f"    N_to_target: standard={N_s:.0f}  projector={N_p:.0f}  "
          f"-> {N_s/N_p:.1f}x fewer shots via projector Kraus, ON AER's OWN ENGINE")
    print(f"    wall-clock/shot: standard={c_s*1e6:.1f}us  projector={c_p*1e6:.1f}us")
    net = (N_s * c_s) / (N_p * c_p)
    print(f"    wall-clock to target: standard/projector = {net:.1f}x "
          f"(projector faster overall)" if net > 1 else
          f"    wall-clock to target ratio = {net:.2f}")


if __name__ == "__main__":
    main()
