import sys, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

SEED = 20260719
GT = 1.5
P = (1.0 - np.exp(-2.0 * GT)) / 2.0
SHOTS = 20000
n = int(sys.argv[1])

qc = QuantumCircuit(n, n)
for q in range(n): qc.h(q)
for q in range(n): qc.id(q)
for q in range(n): qc.h(q)
qc.measure(range(n), range(n))
nm = NoiseModel()
nm.add_all_qubit_quantum_error(pauli_error([("Z", P), ("I", 1 - P)]), ["id"])
sim = AerSimulator(method="statevector", device="GPU", noise_model=nm, batched_shots_gpu=True)
t0 = time.time()
res = sim.run(qc, shots=SHOTS, memory=True, seed_simulator=SEED).result()
dt = time.time() - t0
mem = res.get_memory()
vals = np.array([np.mean([1.0 - 2.0 * int(c) for c in bs]) for bs in mem])
mean = vals.mean(); std = vals.std(ddof=1)
n_aer = (std/0.01)**2
print(f"n={n} mean={mean:.4f} std={std:.4f} N_aer={n_aer:.0f} dt={dt:.1f}s")
