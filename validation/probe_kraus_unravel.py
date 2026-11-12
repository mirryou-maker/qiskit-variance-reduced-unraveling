"""P0-2: can a projector-type unraveling be fed to Aer WITHOUT an ancilla?

Dephasing E(rho) = (1-p) rho + p Z rho Z, written for lam in [0, 1] as
    a rho + b Z rho Z + c (P0 rho P0 + P1 rho P1),
    a = 1 - p - lam p,  b = (1 - lam) p,  c = 2 lam p,
so lam = 0 is the standard Pauli unraveling and lam = 1 the projector unraveling.
Kraus set {sqrt(a) I, sqrt(b) Z, sqrt(c) P0, sqrt(c) P1}.

Product |+>^n, observable mean X: per-qubit trajectory value is +1 / -1 / 0 with
probabilities a / b / c, so Var(mean X) = (1 - 2 lam p - (1 - 2p)^2) / n.

Encodings tried for each lam:
  qerr     QuantumError(Kraus(...)) appended to the circuit
  qerr_rot same Kraus set mixed by a random 4x4 unitary that keeps the operators
           diagonal (a different, still projector-type, unraveling is NOT expected;
           mixing changes the unraveling, so this row only tests whether Aer keeps
           user Kraus at all: its expected variance is computed numerically)
  nm       NoiseModel with the Kraus error attached to an 'id'-like custom gate
If Aer honours the Kraus set, the measured variance follows the lam formula.
If Aer canonicalises, every lam gives the lam = 0 variance.
"""
import sys
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError

warnings.filterwarnings("ignore", category=DeprecationWarning)

GT = 1.5
P = 0.5 * (1 - np.exp(-2 * GT))
I2 = np.eye(2, dtype=complex)
Z2 = np.diag([1.0, -1.0]).astype(complex)
P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)


def kraus_set(lam, p=P):
    a, b, c = 1 - p - lam * p, (1 - lam) * p, 2 * lam * p
    ops = [np.sqrt(a) * I2, np.sqrt(b) * Z2, np.sqrt(c) * P0, np.sqrt(c) * P1]
    return [k for k in ops if np.linalg.norm(k) > 1e-14]


def mix(ops, seed=11):
    """K'_i = sum_j U_ij K_j with a random unitary U (same channel, different unraveling)."""
    rng = np.random.default_rng(seed)
    m = len(ops)
    x = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    u, _ = np.linalg.qr(x)
    return [sum(u[i, j] * ops[j] for j in range(m)) for i in range(m)]


def exact_var_single(ops):
    """Per-qubit Var of <X> along trajectories starting from |+>, Born sampling of ops."""
    psi = np.array([1, 1], complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], complex)
    m1 = m2 = 0.0
    for k in ops:
        phi = k @ psi
        w = np.vdot(phi, phi).real
        if w < 1e-15:
            continue
        x = (np.vdot(phi, X @ phi) / w).real
        m1 += w * x
        m2 += w * x * x
    return m2 - m1 * m1


def circuit(n, ops):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    err = QuantumError(Kraus(ops))
    for i in range(n):
        qc.append(err, [i])
    obs = SparsePauliOp.from_sparse_list([("X", [i], 1 / n) for i in range(n)], num_qubits=n)
    qc.save_expectation_value(obs, list(range(n)), pershot=True, label="ev")
    return qc


def circuit_nm(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n):
        qc.id(i)
    obs = SparsePauliOp.from_sparse_list([("X", [i], 1 / n) for i in range(n)], num_qubits=n)
    qc.save_expectation_value(obs, list(range(n)), pershot=True, label="ev")
    return qc


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    devices = sys.argv[3].split(",") if len(sys.argv) > 3 else ["CPU", "GPU"]
    print(f"n={n} gt={GT} p={P:.5f} shots={shots}")
    for lam in (0.0, 0.5, 1.0):
        ops = kraus_set(lam)
        rot = mix(ops)
        expect = {"qerr": exact_var_single(ops) / n, "qerr_rot": exact_var_single(rot) / n,
                  "nm": exact_var_single(ops) / n}
        for dev in devices:
            for bat in ([False, True] if dev == "GPU" else [False]):
                for enc in ("qerr", "qerr_rot", "nm"):
                    try:
                        if enc == "nm":
                            nm = NoiseModel(basis_gates=["id", "h"])
                            nm.add_all_qubit_quantum_error(QuantumError(Kraus(ops)), ["id"])
                            sim = AerSimulator(method="statevector", device=dev, noise_model=nm,
                                               batched_shots_gpu=bat)
                            qc = circuit_nm(n)
                        else:
                            sim = AerSimulator(method="statevector", device=dev, batched_shots_gpu=bat)
                            qc = circuit(n, ops if enc == "qerr" else rot)
                        r = sim.run(qc, shots=shots, seed_simulator=5).result()
                        if not r.success:
                            raise RuntimeError(r.status)
                        x = np.asarray(r.data(0)["ev"], float)
                        md = r.results[0].metadata
                        print(f"lam={lam:.1f} {dev} bat={bat!s:5} {enc:8} mean={x.mean():.4f} "
                              f"var={x.var(ddof=1):.5f} expect={expect[enc]:.5f} "
                              f"std_var={exact_var_single(kraus_set(0.0)) / n:.5f} "
                              f"uniq={len(np.unique(np.round(x, 8)))} "
                              f"bso={md.get('batched_shots_optimization')} t={r.time_taken:.3f}s")
                    except Exception as e:  # noqa: BLE001
                        print(f"lam={lam:.1f} {dev} bat={bat!s:5} {enc:8} FAILED: {str(e)[:120]}")


if __name__ == "__main__":
    main()
