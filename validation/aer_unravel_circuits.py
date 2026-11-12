"""Circuit builders for dephasing unravelings expressed with the PUBLIC Qiskit-Aer API.

Two unravelings of the single-qubit dephasing channel
    E(rho) = (1 - p) rho + p Z rho Z,     p = (1 - e^{-2 gamma t}) / 2

* "std"  (standard Pauli unraveling): the channel is appended as an Aer
  ``pauli_error`` instruction; each shot applies Z with probability p.
  This is how a user writes dephasing in Aer today.
* "proj" (projector unraveling): rewrite E = (1 - 2p) rho + 2p (P0 rho P0 + P1 rho P1).
  Each shot flips a coin with probability 2p (an ancilla rotated by
  RY(2 asin sqrt(2p)) and measured) and, conditioned on the coin, performs a
  mid-circuit Z-basis measurement of the data qubit (Born-rule collapse).
  Only public instructions are used: reset, ry, measure, ``if_test``.

Aer does NOT honour a projector Kraus set passed to ``kraus_error``: it is
canonicalised to the Pauli form and the per-shot variance is identical to "std"
(see ``validation/round3_opus_review/aer_pershot.py``). The mid-circuit
measurement construction is the route that actually changes the unraveling.

A single ancilla qubit and two classical bits are reused for every dephasing
event (the coin bit is read immediately after it is written).
"""
from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import pauli_error

METHODS = ("std", "proj")


def dephasing_p(gt: float) -> float:
    """Pauli-Z probability of a dephasing channel of strength gamma*t."""
    return 0.5 * (1.0 - np.exp(-2.0 * gt))


class UnravelBuilder:
    """Holds the registers and appends dephasing events for one method."""

    def __init__(self, n: int, method: str):
        if method not in METHODS:
            raise ValueError(method)
        self.n = n
        self.method = method
        self.q = QuantumRegister(n, "q")
        if method == "proj":
            self.a = QuantumRegister(1, "a")
            self.coin = ClassicalRegister(1, "coin")
            self.junk = ClassicalRegister(1, "junk")
            self.qc = QuantumCircuit(self.q, self.a, self.coin, self.junk)
        else:
            self.a = None
            self.qc = QuantumCircuit(self.q)
        self.n_events = 0

    @property
    def num_qubits(self) -> int:
        return self.qc.num_qubits

    def dephase(self, i: int, p: float) -> None:
        """Append one dephasing channel of Pauli probability p on data qubit i."""
        if p <= 0.0:
            return
        if p > 0.5 + 1e-15:
            raise ValueError("p must be <= 1/2")
        self.n_events += 1
        qc = self.qc
        if self.method == "std":
            qc.append(pauli_error([("Z", p), ("I", 1.0 - p)]), [self.q[i]])
            return
        q2 = min(2.0 * p, 1.0)
        a = self.a[0]
        qc.reset(a)
        qc.ry(2.0 * np.arcsin(np.sqrt(q2)), a)
        qc.measure(a, self.coin[0])
        with qc.if_test((self.coin[0], 1)):
            qc.measure(self.q[i], self.junk[0])

    def pad_obs(self, obs: SparsePauliOp) -> SparsePauliOp:
        """Extend an n-qubit observable with identity on the ancilla (if any)."""
        if self.a is None:
            return obs
        # ancilla is the last (highest-index) qubit; A.expand(B) = B (x) A puts B on top
        return obs.expand(SparsePauliOp("I"))


def mean_x_obs(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list([("X", [i], 1.0 / n) for i in range(n)], num_qubits=n)


def xc_obs(n: int) -> SparsePauliOp:
    """X on the centre qubit n // 2 (same logical qubit as src/noise/traj_ref.py)."""
    return SparsePauliOp.from_sparse_list([("X", [n // 2], 1.0)], num_qubits=n)


def zc_obs(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list([("Z", [n // 2], 1.0)], num_qubits=n)


def product_circuit(n: int, gt: float, method: str):
    """|+>^n, one dephasing channel per qubit, observable mean_i X_i.

    Closed forms (per-shot variance of the mean-X estimator):
        mean      e^{-2gt}
        Var_std   (1 - e^{-4gt}) / n
        Var_proj  e^{-2gt} (1 - e^{-2gt}) / n
    """
    b = UnravelBuilder(n, method)
    for i in range(n):
        b.qc.h(b.q[i])
    p = dephasing_p(gt)
    for i in range(n):
        b.dephase(i, p)
    obs = b.pad_obs(mean_x_obs(n))
    b.qc.save_expectation_value(obs, list(range(b.num_qubits)), pershot=True, label="ev")
    return b.qc


def zz_obs(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list(
        [("ZZ", [i, i + 1], 1.0 / (n - 1)) for i in range(n - 1)], num_qubits=n)


OBS_BUILDERS = {"x": mean_x_obs, "zz": zz_obs, "xc": xc_obs, "zc": zc_obs}


def brickwork_circuit(n: int, layers: int, gt_layer: float, method: str, seed: int = 7,
                      obs: str = "x", pershot: bool = True):
    """Entangling benchmark without a closed form.

    Start in |0>^n. Each layer: random RZ(a) RX(b) on every qubit (angles fixed by
    ``seed`` and identical for every method), CZ on a brickwork of neighbouring pairs
    (even pairs on even layers, odd pairs on odd layers), then one dephasing channel
    of strength ``gt_layer`` on every qubit. A final random single-qubit layer is
    applied before measuring ``obs`` ("x": mean_i X_i, "zz": mean_i Z_i Z_{i+1}).
    """
    rng = np.random.default_rng(seed)
    b = UnravelBuilder(n, method)
    qc = b.qc
    p = dephasing_p(gt_layer)
    for layer in range(layers):
        ang = rng.uniform(0, 2 * np.pi, size=(n, 2))
        for i in range(n):
            qc.rz(ang[i, 0], b.q[i])
            qc.rx(ang[i, 1], b.q[i])
        for i in range(layer % 2, n - 1, 2):
            qc.cz(b.q[i], b.q[i + 1])
        for i in range(n):
            b.dephase(i, p)
    ang = rng.uniform(0, 2 * np.pi, size=(n, 2))
    for i in range(n):
        qc.rz(ang[i, 0], b.q[i])
        qc.rx(ang[i, 1], b.q[i])
    o = OBS_BUILDERS[obs](n)
    qc.save_expectation_value(b.pad_obs(o), list(range(b.num_qubits)), pershot=pershot, label="ev")
    return qc, {"layers": layers, "gt_total": layers * gt_layer, "n_events": b.n_events}


def ising_circuit(n: int, layers: int, gt_layer: float, method: str, seed: int = 7,
                  obs: str = "x", pershot: bool = True, J: float = 1.0, h: float = 0.7,
                  dt: float = 0.25):
    """Trotterised transverse-field Ising chain H = J sum Z_i Z_{i+1} + h sum X_i.

    Start in |+>^n; each Trotter step applies RZZ(2 J dt) on every bond, RX(2 h dt) on
    every qubit, then one dephasing channel of strength ``gt_layer`` on every qubit.
    ``seed`` is unused (kept for a uniform signature).
    """
    b = UnravelBuilder(n, method)
    qc = b.qc
    p = dephasing_p(gt_layer)
    qc.h(b.q)
    for _ in range(layers):
        for i in range(0, n - 1, 2):
            qc.rzz(2 * J * dt, b.q[i], b.q[i + 1])
        for i in range(1, n - 1, 2):
            qc.rzz(2 * J * dt, b.q[i], b.q[i + 1])
        for i in range(n):
            qc.rx(2 * h * dt, b.q[i])
        for i in range(n):
            b.dephase(i, p)
    o = OBS_BUILDERS[obs](n)
    qc.save_expectation_value(b.pad_obs(o), list(range(b.num_qubits)), pershot=pershot, label="ev")
    return qc, {"layers": layers, "gt_total": layers * gt_layer, "n_events": b.n_events}


FAMILIES = {"brickwork": brickwork_circuit, "ising": ising_circuit}


def family_exact(family: str, n: int, layers: int, gt_layer: float, seed: int = 7,
                 obs: str = "x", device: str = "CPU") -> float:
    """Exact <obs> via Aer density_matrix (pauli_error channels applied exactly)."""
    from qiskit_aer import AerSimulator
    qc, _ = FAMILIES[family](n, layers, gt_layer, "std", seed=seed, obs=obs, pershot=False)
    sim = AerSimulator(method="density_matrix", device=device)
    return float(sim.run(qc, shots=1).result().data(0)["ev"])


def brickwork_exact(n: int, layers: int, gt_layer: float, seed: int = 7, obs: str = "x",
                    device: str = "CPU") -> float:
    """Exact <obs> via Aer density_matrix (pauli_error channels applied exactly)."""
    from qiskit_aer import AerSimulator
    qc, _ = brickwork_circuit(n, layers, gt_layer, "std", seed=seed, obs=obs, pershot=False)
    sim = AerSimulator(method="density_matrix", device=device)
    return float(sim.run(qc, shots=1).result().data(0)["ev"])


def product_exact(n: int, gt: float) -> dict:
    e2 = np.exp(-2.0 * gt)
    return {"mean": e2, "var_std": (1.0 - e2 * e2) / n, "var_proj": e2 * (1.0 - e2) / n}
