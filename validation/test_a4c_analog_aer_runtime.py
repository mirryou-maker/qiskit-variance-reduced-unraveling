import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, kraus_error

SEED = 20260719
TARGET_SE = 0.01
N_QUBITS = 10
GT = 1.5
GAMMA_EFF = GT  # effective gamma*t budget to match on dephasing strength
Z = np.diag([1, -1]).astype(complex)
I2 = np.eye(2, dtype=complex)

# match total dephasing probability p (same physical channel as projector/standard tests)
P = (1.0 - np.exp(-2.0 * GT)) / 2.0

def standard_kraus(p):
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]

def analog_kraus(p, theta):
    # two-point rotation e^{+-i theta Z}; weight w s.t. w*sin^2(theta) matches target "error strength"
    # choose w=0.5 each branch (symmetric), theta picked so that the *mean* decay matches p regime
    Up = np.cos(theta) * I2 + 1j * np.sin(theta) * Z
    Um = np.cos(theta) * I2 - 1j * np.sin(theta) * Z
    return [np.sqrt(0.5) * Up, np.sqrt(0.5) * Um]

def run(kind, kraus, shots):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    for q in range(N_QUBITS): qc.h(q)
    for q in range(N_QUBITS): qc.id(q)
    for q in range(N_QUBITS): qc.h(q)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(kraus_error(kraus), ["id"])
    sim = AerSimulator(method="statevector", device="CPU", noise_model=nm)
    t0 = time.time()
    res = sim.run(qc, shots=shots, memory=True, seed_simulator=SEED).result()
    dt = time.time() - t0
    vals = np.array([np.mean([1.0 - 2.0 * int(c) for c in bs]) for bs in res.get_memory()])
    std = vals.std(ddof=1)
    n_to_target = (std / TARGET_SE) ** 2
    print(f"  {kind:>20}: mean={vals.mean():+.4f}  std={std:.4f}  N_to_SE<=0.01={n_to_target:>7.0f}")
    return vals.mean(), n_to_target

print(f"n={N_QUBITS}, target p={P:.4f}, exact mean=e^-2gt={np.exp(-2*GT):.4f}\n")
run("standard {I,Z}", standard_kraus(P), 40000)
# analog: theta chosen so single-shot decoherence matches p to leading order: p ~ sin^2(theta)
theta = np.arcsin(np.sqrt(P))
run(f"analog theta={theta:.3f}", analog_kraus(P, theta), 40000)
