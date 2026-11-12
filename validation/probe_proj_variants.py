"""Probe which public-API encodings of the projector unraveling (a) give the projector
variance and (b) survive Aer's batched_shots_gpu executor.

Variants on |+>^n, one dephasing per qubit, observable mean X (gt=1.5):
  std      pauli_error
  if_test  ancilla coin + `with if_test` mid-circuit measure
  c_if     ancilla coin + legacy measure(...).c_if(coin, 1)
  mix      QuantumError([(I, 1-2p), (Kraus{P0,P1}, 2p)]) appended directly
"""
import sys
import warnings

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IGate
from qiskit.quantum_info import Kraus, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import QuantumError, pauli_error

warnings.filterwarnings("ignore", category=DeprecationWarning)
GT = 1.5
P = 0.5 * (1 - np.exp(-2 * GT))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)


def make(variant):
    q = QuantumRegister(n, "q")
    if variant in ("if_test", "c_if"):
        a = QuantumRegister(1, "a")
        coin = ClassicalRegister(1, "coin")
        junk = ClassicalRegister(1, "junk")
        qc = QuantumCircuit(q, a, coin, junk)
    else:
        qc = QuantumCircuit(q)
    qc.h(q)
    for i in range(n):
        if variant == "std":
            qc.append(pauli_error([("Z", P), ("I", 1 - P)]), [q[i]])
        elif variant == "mix":
            err = QuantumError([(IGate(), 1 - 2 * P), (Kraus([P0, P1]), 2 * P)])
            qc.append(err, [q[i]])
        else:
            qc.reset(a[0])
            qc.ry(2 * np.arcsin(np.sqrt(2 * P)), a[0])
            qc.measure(a[0], coin[0])
            if variant == "if_test":
                with qc.if_test((coin[0], 1)):
                    qc.measure(q[i], junk[0])
            else:
                qc.measure(q[i], junk[0]).c_if(coin, 1)
    obs = SparsePauliOp.from_sparse_list([("X", [i], 1 / n) for i in range(n)], num_qubits=n)
    if qc.num_qubits > n:
        obs = obs.expand(SparsePauliOp("I"))
    qc.save_expectation_value(obs, list(range(qc.num_qubits)), pershot=True, label="ev")
    return qc


ex_std = (1 - np.exp(-4 * GT)) / n
ex_proj = np.exp(-2 * GT) * (1 - np.exp(-2 * GT)) / n
print(f"n={n} exact var std={ex_std:.5f} proj={ex_proj:.5f} mean={np.exp(-2*GT):.4f}")
for dev, batched in [("CPU", False), ("GPU", False), ("GPU", True)]:
    for v in ["std", "if_test", "c_if", "mix"]:
        sim = AerSimulator(method="statevector", device=dev, batched_shots_gpu=batched)
        try:
            r = sim.run(make(v), shots=20000, seed_simulator=3).result()
            if not r.success:
                raise RuntimeError(r.status)
            x = np.asarray(r.data(0)["ev"], float)
            md = r.results[0].metadata
            print(f"{dev} batched={batched!s:5} {v:8} mean={x.mean():.4f} var={x.var(ddof=1):.5f} "
                  f"bso={md.get('batched_shots_optimization')} t={r.time_taken:.3f}s")
        except Exception as e:  # noqa: BLE001
            print(f"{dev} batched={batched!s:5} {v:8} FAILED: {str(e)[:110]}")
