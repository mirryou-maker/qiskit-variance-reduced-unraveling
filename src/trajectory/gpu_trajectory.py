"""GPU dense-statevector trajectory engine (Track A, milestone A1).

A minimal quantum-jump (MCWF) trajectory simulator on a dense statevector held on
the GPU via cupy. This A1 version implements the STANDARD Pauli unraveling only;
the variance-reduced projector/analog unravelings (A2) plug into `sample_pauli_noise`.

State layout: an n-qubit statevector is stored as a rank-n tensor of shape (2,)*n,
axis i == qubit i. `statevector()` returns a little-endian flat vector matching
Qiskit/Aer conventions (qubit 0 = least significant bit).
"""
from __future__ import annotations
import numpy as np

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:  # pragma: no cover
    _cp = None
    _HAS_CUPY = False


def get_xp(backend: str = "cupy"):
    if backend == "cupy":
        if not _HAS_CUPY:
            raise RuntimeError("cupy not available")
        return _cp
    return np


# --- gate matrices (built per-backend) -------------------------------------
def gate_lib(xp):
    s2 = 1.0 / np.sqrt(2.0)
    return {
        "I": xp.asarray([[1, 0], [0, 1]], dtype=xp.complex128),
        "X": xp.asarray([[0, 1], [1, 0]], dtype=xp.complex128),
        "Y": xp.asarray([[0, -1j], [1j, 0]], dtype=xp.complex128),
        "Z": xp.asarray([[1, 0], [0, -1]], dtype=xp.complex128),
        "H": xp.asarray([[s2, s2], [s2, -s2]], dtype=xp.complex128),
    }


def ry(xp, theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return xp.asarray([[c, -s], [s, c]], dtype=xp.complex128)


def rx(xp, theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return xp.asarray([[c, -1j * s], [-1j * s, c]], dtype=xp.complex128)


class GPUTrajectorySim:
    def __init__(self, n: int, backend: str = "cupy"):
        self.n = n
        self.xp = get_xp(backend)
        self.g = gate_lib(self.xp)
        self.reset()

    def reset(self):
        xp = self.xp
        psi = xp.zeros((2,) * self.n, dtype=xp.complex128)
        psi[(0,) * self.n] = 1.0
        self.psi = psi

    def apply_1q(self, U, q: int):
        xp = self.xp
        self.psi = xp.moveaxis(xp.tensordot(U, self.psi, axes=([1], [q])), 0, q)

    def apply_2q(self, U4, q0: int, q1: int):
        """Apply a 4x4 gate (basis order |q0 q1>) to qubits (q0, q1)."""
        xp = self.xp
        U = U4.reshape(2, 2, 2, 2)  # (out0,out1,in0,in1)
        psi = xp.tensordot(U, self.psi, axes=([2, 3], [q0, q1]))
        self.psi = xp.moveaxis(psi, [0, 1], [q0, q1])

    def apply_cx(self, c: int, t: int):
        xp = self.xp
        CX = xp.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=xp.complex128,
        )
        self.apply_2q(CX, c, t)

    # --- observables & unravelings (A2) ------------------------------------
    def expval_1q(self, O, q: int) -> float:
        """Real part of <psi| O_q |psi> for a 1-qubit Hermitian O."""
        xp = self.xp
        Opsi = xp.moveaxis(xp.tensordot(O, self.psi, axes=([1], [q])), 0, q)
        return float(xp.real(xp.sum(xp.conj(self.psi) * Opsi)))

    def apply_projector(self, Pname: str, q: int, plus: bool):
        """Collapse onto the +/- eigenspace of Pauli P via Pi_+/- = (I +/- P)/2."""
        xp = self.xp
        P = self.g[Pname]
        Ppsi = xp.moveaxis(xp.tensordot(P, self.psi, axes=([1], [q])), 0, q)
        psi = 0.5 * (self.psi + Ppsi) if plus else 0.5 * (self.psi - Ppsi)
        nrm = xp.sqrt(xp.sum(xp.abs(psi) ** 2))
        self.psi = psi / nrm

    def prob_plus(self, Pname: str, q: int) -> float:
        """Probability of the + outcome of a projective P measurement: (1+<P>)/2."""
        return 0.5 * (1.0 + self.expval_1q(self.g[Pname], q))

    def apply_analog_kick(self, Pname: str, q: int, theta: float):
        """Apply exp(i*theta*P) = cos(theta) I + i sin(theta) P (P^2 = I)."""
        c, s = np.cos(theta), np.sin(theta)
        U = c * self.g["I"] + 1j * s * self.g[Pname]
        self.apply_1q(U, q)

    def statevector(self):
        """Return little-endian (qubit0 = LSB) flat statevector."""
        xp = self.xp
        return xp.transpose(self.psi, axes=tuple(range(self.n)[::-1])).reshape(-1)


def depolarizing_1q_probs(param: float):
    """Match qiskit depolarizing_error(param, 1): p_I, p_X, p_Y, p_Z."""
    return np.array([1.0 - 0.75 * param, param / 4, param / 4, param / 4])


def sample_pauli_noise(sim: GPUTrajectorySim, q: int, probs, rng):
    """Standard unraveling: sample one Pauli from `probs` over (I,X,Y,Z) and apply."""
    k = rng.choice(4, p=probs)
    if k == 0:
        return  # identity
    sim.apply_1q(sim.g[("I", "X", "Y", "Z")[k]], q)
