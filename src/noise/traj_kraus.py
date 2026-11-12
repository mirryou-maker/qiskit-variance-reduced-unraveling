"""Trajectory sampler for general single-qubit channels with several candidate unravelings (E2).

Extends ``traj_ref`` (diagonal-Kraus dephasing only) to channels with non-diagonal Kraus operators
(depolarizing, amplitude damping). Variance studies only; no timing claims.

Model
-----
Circuits are traj_ref op lists in which every dephasing op ("dp", i, p) is replaced by
    ("ch", i, chan)
where ``chan`` is a :class:`Channel`: a stack of candidate unravelings ``cands`` of shape (C, K, 2, 2)
(zero-padded to a common K). Every candidate realises the SAME channel  E(rho) = sum_k K rho K^dag;
candidate 0 is the "std" unraveling (Pauli jumps / photon-count jumps, the Aer default).

Greedy rule
-----------
Let O~ be the observable back-propagated through the channel-averaged remaining circuit to just
after the event. With u_b the part of psi with qubit i = b,
    N[b', b]        = <u_b'|u_b>
    R[a', b', a, b] = <a' u_b'| O~ |a u_b>
the outcome weights and martingale numerators of candidate c are
    w_ck = sum conj(K_{a b'}) K_{a b} N[b', b],     num_ck = sum conj(K_{a' b'}) K_{a b} R[a', b', a, b]
and the increment variance is  V_c = sum_k (num_ck - w_ck m)^2 / w_ck  with m = sum_k num_ck
(identical for every candidate, = <psi|E^dag(O~)|psi>). Greedy picks argmin_c V_c per trajectory
(std unless strictly better). Unbiased for every observable because the choice depends only on the past.
For dephasing with candidates {lam=0, lam=1} V_c reproduces traj_ref's  4p(1-p)A^2  and
2p(1-2p)A^2 + 2p V_B  (tested in validation/test_traj_kraus.py).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import traj_ref as T
from .traj_ref import GPU, I2, XM, YM, ZM, xp

# ----------------------------------------------------------------------------- channels


@dataclass
class Channel:
    name: str
    cands: np.ndarray            # (C, K, 2, 2) complex, zero-padded
    labels: list = field(default_factory=list)

    def __post_init__(self):
        for c in range(self.cands.shape[0]):
            s = sum(K.conj().T @ K for K in self.cands[c])
            assert np.allclose(s, I2, atol=1e-12), (self.name, c)

    def superop_check(self, rho):
        """Channel output of every candidate (for tests)."""
        return [sum(K @ rho @ K.conj().T for K in self.cands[c]) for c in range(self.cands.shape[0])]


def _pad(sets):
    K = max(len(s) for s in sets)
    out = np.zeros((len(sets), K, 2, 2), complex)
    for c, s in enumerate(sets):
        for k, M in enumerate(s):
            out[c, k] = M
    return out


def _projs(P):
    return (I2 + P) / 2, (I2 - P) / 2


def depolarizing(gt):
    """E = (1-p) rho + p I/2, p = 1 - exp(-gt); candidates: lam in {0,1}^3 over axes (X, Y, Z)."""
    p = 1.0 - np.exp(-gt)
    r = p / 4.0
    sets, labels = [], []
    for lam in itertools.product((0, 1), repeat=3):
        a = 1.0 - sum((1 + l) * r for l in lam)
        assert a >= -1e-15
        s = [np.sqrt(max(a, 0.0)) * I2]
        for l, P in zip(lam, (XM, YM, ZM)):
            if l == 0:
                s.append(np.sqrt(r) * P)
            else:
                pp, pm = _projs(P)
                s += [np.sqrt(2 * r) * pp, np.sqrt(2 * r) * pm]
        sets.append(s)
        labels.append("".join(map(str, lam)))
    return Channel("depol", _pad(sets), labels)


def amp_damp(gt):
    """AD with g = 1 - exp(-gt); candidates: jump, x-homodyne, y-homodyne."""
    g = 1.0 - np.exp(-gt)
    K0 = np.diag([1.0, np.sqrt(1 - g)]).astype(complex)
    K1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], complex)
    s2 = np.sqrt(0.5)
    sets = [[K0, K1],
            [s2 * (K0 + K1), s2 * (K0 - K1)],
            [s2 * (K0 + 1j * K1), s2 * (K0 - 1j * K1)]]
    return Channel("ad", _pad(sets), ["jump", "homx", "homy"])


def dephasing(gt):
    """Dephasing as a Channel (test only): candidates lam=0 (std) and lam=1 (projector)."""
    p = T.dephasing_p(gt)
    sets = []
    for lam in (0.0, 1.0):
        d = T.kraus_lam(p, lam)
        sets.append([np.diag(row) for row in d])
    return Channel("deph", _pad(sets), ["std", "proj"])


def _poisson_kicks(gt, theta, tail=1e-15):
    """Weights P(m) of the net kick count m for TJM's analog unraveling of dephasing:
    k ~ Poisson(mu), mu = gt / sin^2(theta), each kick exp(+/- i theta Z) with prob 1/2."""
    from math import exp
    mu = float(gt / np.sin(theta) ** 2)
    probs = {}
    k, pk = 0, exp(-mu)
    binom = np.array([1.0])                     # Binomial(k, 1/2) pmf, updated by convolution (no overflow)
    acc = 0.0
    while True:
        for j in range(k + 1):
            m = 2 * j - k
            probs[m] = probs.get(m, 0.0) + pk * binom[j]
        acc += pk
        if 1.0 - acc < tail and k > mu:
            break
        k += 1
        pk = pk * mu / k
        binom = 0.5 * (np.append(binom, 0.0) + np.append(0.0, binom))
    s = sum(probs.values())
    return {m: v / s for m, v in probs.items() if v > 1e-18}


def dephasing_tjm(gt, thetas=(0.3, np.pi / 4)):
    """Dephasing with candidates: 0 std, 1 projector (= TJM projector), 2.. TJM analog(theta).
    Analog: per layer, net Z-rotation exp(i m theta Z) with Poisson-kick weights; channel equals
    dephasing with 1 - 2p = exp(-2 gt) exactly (characteristic function of the kick process)."""
    p = T.dephasing_p(gt)
    sets = [[np.diag(row) for row in T.kraus_lam(p, lam)] for lam in (0.0, 1.0)]
    labels = ["std", "proj"]
    for th in thetas:
        w = _poisson_kicks(gt, th)
        sets.append([np.sqrt(pm) * np.diag([np.exp(1j * m * th), np.exp(-1j * m * th)])
                     for m, pm in sorted(w.items())])
        labels.append(f"analog{th:.3f}")
    return Channel("deph_tjm", _pad(sets), labels)


def dephasing_sweep(gt):
    """E7: dephasing with candidates std, projector and a wider analog angle sweep."""
    c = dephasing_tjm(gt, thetas=(0.05, 0.1, 0.2, 0.3, np.pi / 4))
    return Channel("deph_sweep", c.cands, c.labels)


def dephasing_small(gt):
    """E7b: dephasing with std, projector and small analog angles (theta -> 0 convergence on Ising)."""
    c = dephasing_tjm(gt, thetas=(0.0125, 0.025, 0.05))
    return Channel("deph_small", c.cands, c.labels)


def amp_damp_phase(gt,phis=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    """E7: AD with candidates jump and homodyne (K0 +/- e^{i phi} K1)/sqrt2 at several phases."""
    g = 1.0 - np.exp(-gt)
    K0 = np.diag([1.0, np.sqrt(1 - g)]).astype(complex)
    K1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], complex)
    s2 = np.sqrt(0.5)
    sets = [[K0, K1]] + [[s2 * (K0 + np.exp(1j * f) * K1), s2 * (K0 - np.exp(1j * f) * K1)] for f in phis]
    return Channel("ad_phase", _pad(sets), ["jump"] + [f"hom{f:.3f}" for f in phis])


CHANNELS = {"depol": depolarizing, "ad": amp_damp, "deph": dephasing, "deph_tjm": dephasing_tjm,
            "deph_sweep": dephasing_sweep, "ad_phase": amp_damp_phase, "deph_small": dephasing_small}


def with_channel(ops, chan):
    """Replace every ("dp", i, p) of a traj_ref circuit by ("ch", i, chan)."""
    return [("ch", op[1], chan) if op[0] == "dp" else op for op in ops]


def build(fam, var, n, L, gt, chname):
    if fam == "ising":
        ops = T.ising_ops(n, L, gt, h=float(var[1:]))
    else:
        ops = T.brickwork_ops(n, L, gt, seed=int(var[1:]))
    return with_channel(ops, CHANNELS[chname](gt))


# ----------------------------------------------------------------------------- backward pass

def backprop_local(ops, n, O, host=True):
    """Return (after, O~_0): O~ just AFTER each channel event (forward order), complex64,
    host-stored if host and GPU. Channel action on O~: sum_k K^dag O~ K (candidate 0)."""
    D = 2 ** n
    Ot = xp.asarray(O, dtype=complex)
    mats = {}
    after = []
    for op in reversed(ops):
        kind, i, M = op
        if kind == "ch":
            a = Ot.astype(xp.complex64)
            after.append(xp.asnumpy(a) if (host and GPU) else a)
            new = xp.zeros_like(Ot)
            for K in M.cands[0]:
                if not np.any(K):
                    continue
                new += T._gate_conj_local(Ot, ("g1", i, None), n, xp.asarray(K))
            Ot = new
        else:
            if id(M) not in mats:
                mats[id(M)] = xp.asarray(M)
            Ot = T._gate_conj_local(Ot, op, n, mats[id(M)])
    after.reverse()
    return after, Ot


def _axis_slots(T, q, n):
    """Index tuples of the (row, col) = (a, b) slices of qubit q in a [2]*(2n) tensor."""
    def s(a, b):
        idx = [slice(None)] * (2 * n)
        idx[q], idx[n + q] = a, b
        return tuple(idx)
    return s


def to_pauli(O, n):
    """Coefficients c_P of O = sum c_P P as a [2]*(2n) tensor; per-qubit slots (r,c) = 00:I 01:X 10:Y 11:Z."""
    T_ = xp.array(O, dtype=complex).reshape((2,) * (2 * n))
    for q in range(n):
        s = _axis_slots(T_, q, n)
        a00, a01, a10, a11 = T_[s(0, 0)].copy(), T_[s(0, 1)].copy(), T_[s(1, 0)].copy(), T_[s(1, 1)].copy()
        T_[s(0, 0)] = (a00 + a11) / 2
        T_[s(1, 1)] = (a00 - a11) / 2
        T_[s(0, 1)] = (a01 + a10) / 2
        T_[s(1, 0)] = 1j * (a01 - a10) / 2
    return T_


def from_pauli(c, n):
    """Inverse of :func:`to_pauli`."""
    T_ = c.copy()
    for q in range(n):
        s = _axis_slots(T_, q, n)
        i, x, y, z = T_[s(0, 0)].copy(), T_[s(0, 1)].copy(), T_[s(1, 0)].copy(), T_[s(1, 1)].copy()
        T_[s(0, 0)] = i + z
        T_[s(1, 1)] = i - z
        T_[s(0, 1)] = x - 1j * y
        T_[s(1, 0)] = x + 1j * y
    D = 2 ** n
    return T_.reshape(D, D)


def truncate_pauli(O, n, M):
    """Keep the identity and the M largest-magnitude non-identity Pauli coefficients of O."""
    c = to_pauli(O, n).reshape(-1)
    if M + 1 >= c.size:
        return xp.asarray(O, dtype=complex)
    mag = xp.abs(c)
    mag[0] = 0.0                                    # identity is always kept, not counted in M
    idx = xp.argsort(mag)[-M:]
    keep = xp.zeros_like(c)
    keep[idx] = c[idx]
    keep[0] = c[0]
    return from_pauli(keep.reshape((2,) * (2 * n)), n)


def backprop_truncated(ops, n, O, M, host=True):
    """As :func:`backprop_local`, truncating O~ to M Pauli terms at every channel event (E8).
    Dense emulation: tests the information in M terms, not the cost of a sparse implementation."""
    Ot = xp.asarray(O, dtype=complex)
    mats = {}
    after = []
    for op in reversed(ops):
        kind, i, X = op
        if kind == "ch":
            Ot = truncate_pauli(Ot, n, M)
            a = Ot.astype(xp.complex64)
            after.append(xp.asnumpy(a) if (host and GPU) else a)
            new = xp.zeros_like(Ot)
            for K in X.cands[0]:
                if not np.any(K):
                    continue
                new += T._gate_conj_local(Ot, ("g1", i, None), n, xp.asarray(K))
            Ot = new
        else:
            if id(X) not in mats:
                mats[id(X)] = xp.asarray(X)
            Ot = T._gate_conj_local(Ot, op, n, mats[id(X)])
    after.reverse()
    return after, Ot


def exact_mean(ops, n, obs_name):
    _, O0 = backprop_local(ops, n, T.obs_matrix(obs_name, n))
    psi0 = xp.zeros(2 ** n, complex)
    psi0[0] = 1.0
    return float(xp.real(xp.vdot(psi0, O0 @ psi0)))


def dm_means_fast(ops, n, obs_names):
    """Exact <O> by forward density-matrix evolution, rho stored as a 2n-qubit vector vec(rho)[row * D + col]
    (row qubit q at position q, column qubit q at position n + q), with the fast elementwise kernels.
    U rho U^dag = (U at q) (conj U at n + q); channel = sum_k (K at q)(conj K at n + q). For E9a at n = 12."""
    n2 = 2 * n
    rho = xp.zeros((1, 4 ** n), complex)
    rho[0, 0] = 1.0
    mats = {}

    keep = []                                      # keep matrices alive so their ids stay unique

    def put(M, conj):
        k = ("dm", id(M), conj)
        if k not in mats:
            A = xp.asarray(np.conj(M) if conj else M)
            keep.append(A)
            mats[k] = A
            mats[id(A)] = A
        return mats[k]

    for op in ops:
        kind, i, M = op
        if kind == "ch":
            new = None
            for K in M.cands[0]:
                if not np.any(K):
                    continue
                t = rho.copy()
                A, Ac = xp.asarray(K), xp.asarray(np.conj(K))
                t = apply_gate_fast(t, ("g1", i, A), n2, {id(A): A})
                t = apply_gate_fast(t, ("g1", n + i, Ac), n2, {id(Ac): Ac})
                new = t if new is None else new + t
            rho = new
        else:
            A, Ac = put(M, False), put(M, True)
            rho = apply_gate_fast(rho, (kind, i, A), n2, mats)
            rho = apply_gate_fast(rho, (kind, n + i, Ac), n2, mats)
    D = 2 ** n
    R2 = rho.reshape(D, D)
    diag = xp.real(xp.diagonal(R2)).copy()
    bits = xp.arange(D)
    out = {}
    for o in obs_names:
        tot = 0.0
        for coef, term in T.obs_terms(o, n):
            if all(np.allclose(m, ZM) for m in term.values()):     # diagonal: weight the populations
                s = xp.ones(D)
                for q in term:
                    s = s * (1 - 2 * ((bits >> (n - 1 - q)) & 1))
                tot += coef * float(xp.sum(diag * s))
            else:                                                  # general: Tr(P rho) = sum_j <j|P rho|j>
                t = rho.copy()
                for q, m in term.items():
                    A = xp.asarray(m)
                    t = apply_gate_fast(t, ("g1", q, A), n2, {id(A): A})
                tot += coef * float(xp.real(xp.trace(t.reshape(D, D))))
        out[o] = tot
    return out


def dm_mean(ops, n, obs_name):
    """Dense density-matrix reference (numpy, n <= 6)."""
    def full(op):
        U = T.full_gate(op, n)
        return U.get() if GPU else U

    D = 2 ** n
    rho = np.zeros((D, D), complex)
    rho[0, 0] = 1.0
    for op in ops:
        kind, i, M = op
        if kind == "ch":
            new = np.zeros_like(rho)
            for K in M.cands[0]:
                U = full(("g1", i, K))
                new += U @ rho @ U.conj().T
            rho = new
        else:
            U = full(op)
            rho = U @ rho @ U.conj().T
    return float(np.real(np.trace(T.obs_matrix(obs_name, n) @ rho)))


# ----------------------------------------------------------------------------- kernels

def _NR(psi, Ot, i, n, need_R=True):
    """N (B,2,2) and R (B,2,2,2,2) [a', b', a, b] for qubit i (see module docstring)."""
    B, D = psi.shape
    lo, hi = 2 ** i, 2 ** (n - i - 1)
    v = psi.reshape(B, lo, 2, hi)
    u = [v[:, :, b, :] for b in (0, 1)]                       # (B, lo, hi)
    N = xp.empty((B, 2, 2), complex)
    for bp in (0, 1):
        for b in (0, 1):
            N[:, bp, b] = xp.sum(xp.conj(u[bp]) * u[b], axis=(1, 2))
    if not need_R:
        return N, None
    dt = Ot.dtype
    O4 = Ot.reshape(D, lo, 2, hi)
    Oa = [xp.ascontiguousarray(O4[:, :, a, :].reshape(D, lo * hi)) for a in (0, 1)]
    uf = [x.reshape(B, lo * hi).astype(dt) for x in u]
    R = xp.empty((B, 2, 2, 2, 2), complex)
    for a in (0, 1):
        for b in (0, 1):
            y = (uf[b] @ Oa[a].T).reshape(B, lo, 2, hi)       # O~ |a u_b>
            for ap in (0, 1):
                for bp in (0, 1):
                    R[:, ap, bp, a, b] = xp.sum(xp.conj(uf[bp].reshape(B, lo, hi)) * y[:, :, ap, :],
                                                axis=(1, 2)).astype(complex)
    return N, R


def increment_vars(N, R, cands_x):
    """V (B, C): increment variance of every candidate; m (B,): common mean."""
    Kc = xp.conj(cands_x)
    w = xp.real(xp.einsum("ckaq,ckab,xqb->xck", Kc, cands_x, N))
    num = xp.real(xp.einsum("ckpq,ckab,xpqab->xck", Kc, cands_x, R))
    m = num[:, 0, :].sum(axis=1)
    dev = num - w * m[:, None, None]
    V = xp.where(w > 1e-14, dev ** 2 / xp.maximum(w, 1e-300), 0.0).sum(axis=2)
    return V, m, num


_K = {}


def _kern():
    """Single-pass cupy kernels for the fast path (built lazily; GPU only)."""
    if not _K:
        _K["mix"] = xp.ElementwiseKernel(
            "complex128 a, complex128 b, complex128 c, complex128 d", "complex128 v0, complex128 v1",
            "complex<double> t0 = v0, t1 = v1; v0 = a * t0 + b * t1; v1 = c * t0 + d * t1;", "qco_mix2")
        _K["abs2"] = xp.ReductionKernel("complex128 x", "float64 y", "x.real() * x.real() + x.imag() * x.imag()",
                                        "a + b", "y = a", "0", "qco_abs2")
        _K["cross"] = xp.ReductionKernel("complex128 x, complex128 y", "complex128 z", "conj(x) * y",
                                         "a + b", "z = a", "0", "qco_cross")
    return _K


def _mix2(v, A):
    """In-place 2x2 action on axis 2 of v (B, lo, 2, hi). A: (2, 2) shared or (B, 2, 2) per trajectory."""
    v0, v1 = v[:, :, 0, :], v[:, :, 1, :]
    if A.ndim == 3:
        A = A[:, :, :, None, None]                 # (B, 2, 2, 1, 1) broadcast over (lo, hi)
        a = [[A[:, r, c] for c in (0, 1)] for r in (0, 1)]
    else:
        a = [[A[r, c] for c in (0, 1)] for r in (0, 1)]
    if GPU:
        _kern()["mix"](a[0][0], a[0][1], a[1][0], a[1][1], v0, v1)
        return v
    t0, t1 = v0.copy(), v1.copy()
    v[:, :, 0, :] = a[0][0] * t0 + a[0][1] * t1
    v[:, :, 1, :] = a[1][0] * t0 + a[1][1] * t1
    return v


def _N_fast(psi, i, n):
    B = psi.shape[0]
    v = psi.reshape(B, 2 ** i, 2, 2 ** (n - i - 1))
    u0, u1 = v[:, :, 0, :], v[:, :, 1, :]
    N = xp.empty((B, 2, 2), complex)
    if GPU:
        k = _kern()
        N[:, 0, 0] = k["abs2"](u0, axis=(1, 2))
        N[:, 1, 1] = k["abs2"](u1, axis=(1, 2))
        N[:, 0, 1] = k["cross"](u0, u1, axis=(1, 2))
    else:
        N[:, 0, 0] = np.sum(np.abs(u0) ** 2, axis=(1, 2))
        N[:, 1, 1] = np.sum(np.abs(u1) ** 2, axis=(1, 2))
        N[:, 0, 1] = np.sum(np.conj(u0) * u1, axis=(1, 2))
    N[:, 1, 0] = xp.conj(N[:, 0, 1])
    return N


_N3_SRC = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void qco_n3(const complex<double>* psi, int D, int p, int nrest, double* part,
                                  complex<float>* ps, int write_ps) {
    // grid (nb, B), block 256: part[(t * nb + blk) * 4 + {0: |u0|^2, 1: |u1|^2, 2: Re u0^H u1, 3: Im u0^H u1}]
    // write_ps: also store psi as complex64 into ps (input of the sparse policy kernel) in the same pass
    const int t = blockIdx.y, nb = gridDim.x;
    const complex<double>* v = psi + (size_t)t * D;
    complex<float>* w = ps + (size_t)t * D;
    const unsigned int hb = 1u << p, lowm = hb - 1u;
    double s0 = 0, s1 = 0, cr = 0, ci = 0;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < nrest; r += nb * blockDim.x) {
        const unsigned int x = ((r & ~lowm) << 1) | (r & lowm);
        const complex<double> a = v[x], b = v[x | hb];
        if (write_ps) { w[x] = complex<float>(a); w[x | hb] = complex<float>(b); }
        s0 += a.real() * a.real() + a.imag() * a.imag();
        s1 += b.real() * b.real() + b.imag() * b.imag();
        const complex<double> c = conj(a) * b;
        cr += c.real(); ci += c.imag();
    }
    __shared__ double sh[4][256];
    sh[0][threadIdx.x] = s0; sh[1][threadIdx.x] = s1; sh[2][threadIdx.x] = cr; sh[3][threadIdx.x] = ci;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w)
            for (int k = 0; k < 4; ++k) sh[k][threadIdx.x] += sh[k][threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x < 4) part[((size_t)t * nb + blockIdx.x) * 4 + threadIdx.x] = sh[threadIdx.x][0];
}
"""


def _N_fused(psi, i, n, nb=64, ps=None):
    """Same N as :func:`_N_fast` from ONE pass over psi (E9e, opt2); GPU only.
    ps: optional complex64 (B, D) buffer that receives psi cast to complex64 in the same pass."""
    if "n3" not in _K:
        _kern()
        _K["n3"] = xp.RawKernel(_N3_SRC, "qco_n3")
    B, D = psi.shape
    nb = max(1, min(nb, D // 2 // 256))
    part = xp.empty((B, nb, 4), float)
    wp = ps if ps is not None else xp.empty((1,), xp.complex64)
    _K["n3"]((nb, B), (256,), (psi, np.int32(D), np.int32(n - 1 - i), np.int32(D // 2), part, wp,
                               np.int32(ps is not None)))
    s = part.sum(axis=1)
    N = xp.empty((B, 2, 2), complex)
    N[:, 0, 0] = s[:, 0]
    N[:, 1, 1] = s[:, 1]
    N[:, 0, 1] = s[:, 2] + 1j * s[:, 3]
    N[:, 1, 0] = xp.conj(N[:, 0, 1])
    return N


def _abs2(u):
    return _kern()["abs2"](u, axis=(1, 2)) if GPU else np.sum(np.abs(u) ** 2, axis=(1, 2))


def _cross(u, v):
    return _kern()["cross"](u, v, axis=(1, 2)) if GPU else np.sum(np.conj(u) * v, axis=(1, 2))


def _x1(psi, q, n):
    v = psi.reshape(psi.shape[0], 2 ** q, 2, 2 ** (n - q - 1))
    return 2.0 * xp.real(_cross(v[:, :, 0, :], v[:, :, 1, :]))


def _z1(psi, q, n):
    v = psi.reshape(psi.shape[0], 2 ** q, 2, 2 ** (n - q - 1))
    return _abs2(v[:, :, 0, :]) - _abs2(v[:, :, 1, :])


def _zz(psi, q, n):
    v = psi.reshape(psi.shape[0], 2 ** q, 4, 2 ** (n - q - 2))
    s = [_abs2(v[:, :, r, :]) for r in range(4)]
    return s[0] - s[1] - s[2] + s[3]


def expect_fast(psi, name, n):
    """Per-trajectory <O> for x, zz, xc, zc (same definitions as traj_ref.obs_terms), one reduction per term."""
    c = n // 2
    if name == "x":
        return sum(_x1(psi, q, n) for q in range(n)) / n
    if name == "zz":
        return sum(_zz(psi, q, n) for q in range(n - 1)) / (n - 1)
    if name == "xc":
        return _x1(psi, c, n)
    if name == "zc":
        return _z1(psi, c, n)
    return None


def apply_gate_fast(psi, op, n, mats):
    """Same result as traj_ref.apply_gate, elementwise (cupy einsum is slow for this pattern)."""
    kind, i, M = op
    B = psi.shape[0]
    if kind == "g1":                                # in place: run() owns psi and reassigns it
        return _mix2(psi.reshape(B, 2 ** i, 2, 2 ** (n - i - 1)), mats[id(M)]).reshape(psi.shape)
    v = psi.reshape(B, 2 ** i, 4, 2 ** (n - i - 2))
    Mh = mats.setdefault(("host", id(M)), xp.asnumpy(mats[id(M)]) if GPU else mats[id(M)])
    if np.count_nonzero(Mh - np.diag(np.diag(Mh))) == 0:     # diagonal (CZ, RZZ): in-place scaling
        for r in range(4):
            if Mh[r, r] != 1:
                v[:, :, r, :] *= Mh[r, r]
        return psi
    out = xp.zeros_like(v)
    for r in range(4):
        for c in range(4):
            if Mh[r, c] != 0:
                out[:, :, r, :] += Mh[r, c] * v[:, :, c, :]
    return out.reshape(psi.shape)


def apply_kraus_fast(psi, i, n, Ks, rng, N=None):
    """Same as :func:`apply_kraus` (same RNG draws), elementwise final update. N: precomputed (opt2)."""
    B, D = psi.shape
    if N is None:
        N = _N_fast(psi, i, n)
    w = xp.real(xp.einsum("xkaq,xkab,xqb->xk", xp.conj(Ks), Ks, N))
    w = xp.maximum(w, 0.0)
    cum = xp.cumsum(w, axis=1)
    uu = rng.random(B) * cum[:, -1]
    k = xp.minimum(xp.sum(cum < uu[:, None], axis=1), Ks.shape[1] - 1)
    idx = xp.arange(B)
    Kk = Ks[idx, k] / xp.sqrt(w[idx, k])[:, None, None]
    return _mix2(psi.reshape(B, 2 ** i, 2, 2 ** (n - i - 1)), Kk).reshape(B, D)   # in place: psi is run()'s own


def apply_kraus(psi, i, n, Ks, rng):
    """Ks: (B, K, 2, 2) per-trajectory Kraus set. Born-samples one outcome per trajectory."""
    B, D = psi.shape
    N, _ = _NR(psi, None, i, n, need_R=False)
    w = xp.real(xp.einsum("xkaq,xkab,xqb->xk", xp.conj(Ks), Ks, N))
    w = xp.maximum(w, 0.0)
    cum = xp.cumsum(w, axis=1)
    uu = rng.random(B) * cum[:, -1]
    k = xp.minimum(xp.sum(cum < uu[:, None], axis=1), Ks.shape[1] - 1)
    idx = xp.arange(B)
    Kk = Ks[idx, k] / xp.sqrt(w[idx, k])[:, None, None]
    lo, hi = 2 ** i, 2 ** (n - i - 1)
    v = psi.reshape(B, lo, 2, hi)
    return xp.einsum("xab,xibj->xiaj", Kk, v).reshape(B, D)


# ----------------------------------------------------------------------------- driver

def run(ops, n, obs_names, shots, scheme=("fixed", 0), seed=0, chunk=4096, obs_for_policy=None,
        greedy_cands=None, myopic=False, after=None, fast=False, opt2=False):
    """scheme: ("fixed", c) candidate c at every event | ("greedy", None) per-trajectory argmin V_c |
    ("table", tab) static per-location table: candidate tab[e] at channel event e (forward order), the same for every
    trajectory (E10). A table with every entry = c draws the same random numbers as ("fixed", c), so it reproduces it.

    greedy_cands: restrict greedy to these candidate indices (must include 0 = std; default all).
    myopic: use the BARE observable O at every event instead of the back-propagated O~_t
    (Cao-He-Li-type criterion: minimise the instantaneous variance growth of <O>).
    after: use this O~ list for greedy (e.g. from backprop_truncated) instead of the exact one; entries may be
    sparse_policy.SparseEvent (Pauli-sum O~, E9b).
    fast: elementwise gate / Kraus kernels (E9a); same RNG draws, results equal up to rounding.
    opt2 (E9e, GPU, implies fast): one-pass N kernel for BOTH fixed and greedy, N computed once per event and
    shared by the policy and the Kraus step, choice counts kept on the GPU (no per-event host sync). The sparse
    policy kernel is chosen separately by sparse_policy.IMPL. Same RNG draws; results equal up to rounding.
    Returns dict obs -> per-trajectory estimates; greedy also '_choice' (n_ev, C) fractions.
    """
    rng = xp.random.default_rng(seed)
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "ch"}
    chans = {id(op[2]): xp.asarray(op[2].cands) for op in ops if op[0] == "ch"}
    terms = {o: T.obs_terms(o, n) for o in obs_names}
    greedy = scheme[0] == "greedy"
    n_ev = sum(1 for op in ops if op[0] == "ch")
    if scheme[0] == "table":
        tab = [int(c) for c in np.asarray(scheme[1]).reshape(-1)]
        assert len(tab) == n_ev, (len(tab), n_ev)
    else:
        tab = None
    if greedy and after is None:
        Opol = T.obs_matrix(obs_for_policy or obs_names[0], n)
        if myopic:
            Ox = xp.asarray(Opol.astype(np.complex64))
            after = [Ox] * n_ev
        else:
            after, _ = backprop_local(ops, n, Opol)
    C = max(op[2].cands.shape[0] for op in ops if op[0] == "ch")
    gsel = None if greedy_cands is None else xp.asarray(np.array(greedy_cands))
    if gsel is not None:
        assert int(greedy_cands[0]) == 0
    choice = np.zeros((n_ev, C))
    opt2 = bool(opt2 and GPU)
    fast = fast or opt2
    choice_d = xp.zeros((n_ev, C)) if opt2 else None
    out = {o: [] for o in obs_names}
    gate_f = apply_gate_fast if fast else T.apply_gate
    kraus_f = apply_kraus_fast if fast else apply_kraus
    done = 0
    while done < shots:
        B = min(chunk, shots - done)
        psi = xp.zeros((B, 2 ** n), complex)
        psi[:, 0] = 1.0
        ps64 = None
        ev = 0
        for op in ops:
            if op[0] != "ch":
                psi = gate_f(psi, op, n, mats)
                continue
            i, cx = op[1], chans[id(op[2])]
            sparse_ev = greedy and hasattr(after[ev], "xm")
            if opt2:
                if sparse_ev and ps64 is None:
                    ps64 = xp.empty(psi.shape, xp.complex64)
                Nev = _N_fused(psi, i, n, ps=ps64 if sparse_ev else None)
            else:
                Nev = None
            if greedy:
                if sparse_ev:                            # sparse_policy.SparseEvent (E9b): Pauli-sum O~
                    from . import sparse_policy as SP
                    if opt2 and SP.IMPL == "v2":
                        N, R = SP.NR_sparse(psi, after[ev], n, Nev, ps=ps64)
                    else:
                        N, R = SP.NR_sparse(psi, after[ev], n, Nev if opt2 else _N_fast(psi, i, n))
                else:
                    N, R = _NR(psi, xp.asarray(after[ev]), i, n)
                cs = cx if gsel is None else cx[gsel]
                V, _, _ = increment_vars(N, R, cs)
                best = xp.argmin(V, axis=1)
                vb = V[xp.arange(B), best]
                keep = vb >= V[:, 0] - (1e-4 * V[:, 0] + 1e-15)   # tie tolerance -> std
                c = xp.where(keep, 0, best)
                if gsel is not None:
                    c = gsel[c]
                if opt2:
                    choice_d[ev, :cx.shape[0]] += xp.bincount(c, minlength=cx.shape[0])[:cx.shape[0]]
                else:
                    for j in range(cx.shape[0]):
                        choice[ev, j] += float(xp.sum(c == j))
                Ks = cx[c]
            else:
                c_ev = scheme[1] if tab is None else tab[ev]
                Ks = xp.broadcast_to(cx[c_ev][None], (B,) + cx.shape[1:])
            psi = apply_kraus_fast(psi, i, n, Ks, rng, N=Nev) if opt2 else kraus_f(psi, i, n, Ks, rng)
            ev += 1
        for o in obs_names:
            e = expect_fast(psi, o, n) if fast else None
            out[o].append(T.expect_terms(psi, terms[o], n) if e is None else e)
        done += B
    res = {o: np.concatenate([xp.asnumpy(a) if GPU else a for a in out[o]]) for o in obs_names}
    if greedy:
        if opt2:
            choice = xp.asnumpy(choice_d)
        res["_choice"] = choice / shots
    return res
