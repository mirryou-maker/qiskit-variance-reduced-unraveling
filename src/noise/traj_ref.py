"""Reference trajectory sampler for general diagonal-Kraus unravelings of dephasing.

Purpose: variance studies (P1-P3 of docs/research_plan_unraveling_advantage_2026-11-05.md),
independent of Qiskit-Aer's restrictions. Not a performance code; no timing claims.

Model
-----
A circuit is a list of ops
    ("g1", i, M)      2x2 unitary on qubit i
    ("g2", i, M)      4x4 unitary on qubits (i, i+1), basis |q_i q_{i+1}>
    ("dp", i, p)      dephasing E(rho) = (1-p) rho + p Z_i rho Z_i
Each dephasing event is unravelled by a set of DIAGONAL Kraus operators
D_k = diag(d_k0, d_k1) with sum_k |D_k|^2 = I whose channel equals E. Families:
    lam   (std <-> projector)  {sqrt(a) I, sqrt(b) Z, sqrt(c) P0, sqrt(c) P1},
          a = 1 - p - lam p, b = (1 - lam) p, c = 2 lam p     (lam=0 std, lam=1 projector)
    weak  K1 = cos(t) sqrt(1-p) I + sin(t) sqrt(p) Z,
          K2 = -sin(t) sqrt(1-p) I + cos(t) sqrt(p) Z         (t=0 std; weak measurement)
Outcome k is drawn with Born probability ||D_k psi||^2 and psi -> D_k psi / norm.

Exact variance decomposition (martingale identity)
--------------------------------------------------
Let O~_t be the observable back-propagated (Heisenberg picture, through the
channel-AVERAGED remaining circuit) to just after event t. Along a trajectory
M_t = <psi_t| O~_t |psi_t> is a martingale for every unraveling of the same
channels, M_0 = exact mean, M_T = per-trajectory estimator. Hence
    Var(estimator) = sum_t E[ Var_k( M_{t} | psi_{t-1} ) ]        (orthogonal increments)
For one dephasing event on qubit i with A = <O~_anti>, C = <O~_comm>
(O~_anti/comm = (O~ -/+ Z_i O~ Z_i)/2) and V_B = Born variance of the
post-measurement values of O~_comm, the local increment variances are
    std        4 p (1-p) A^2
    projector  2 p (1-2p) A^2 + 2 p V_B
    lam        linear interpolation  (1-lam)*std + lam*projector
so projector beats std at this event iff A^2 > V_B.
"""
from __future__ import annotations

import numpy as np

try:  # GPU if available, numpy otherwise
    import cupy as xp  # type: ignore
    GPU = True
except Exception:  # noqa: BLE001
    xp = np
    GPU = False

I2 = np.eye(2, dtype=complex)
ZM = np.diag([1.0, -1.0]).astype(complex)
XM = np.array([[0, 1], [1, 0]], complex)
YM = np.array([[0, -1j], [1j, 0]], complex)
HM = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)


def rz(a):
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def rx(b):
    c, s = np.cos(b / 2), np.sin(b / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def rzz(t):
    e, f = np.exp(-0.5j * t), np.exp(0.5j * t)
    return np.diag([e, f, f, e])


CZ = np.diag([1, 1, 1, -1]).astype(complex)


def dephasing_p(gt):
    return 0.5 * (1.0 - np.exp(-2.0 * gt))


# ----------------------------------------------------------------------------- circuits
# Builders reproduce validation/aer_unravel_circuits.py gate-for-gate (same rng draws).

def brickwork_ops(n, layers, gt_layer, seed=7, clifford=False):
    rng = np.random.default_rng(seed)
    p = dephasing_p(gt_layer)
    ops = []

    def ang():
        a = rng.uniform(0, 2 * np.pi, size=(n, 2))
        if clifford:
            a = np.round(a / (np.pi / 2)) * (np.pi / 2)
        return a

    for layer in range(layers):
        a = ang()
        for i in range(n):
            ops.append(("g1", i, rx(a[i, 1]) @ rz(a[i, 0])))
        for i in range(layer % 2, n - 1, 2):
            ops.append(("g2", i, CZ))
        for i in range(n):
            ops.append(("dp", i, p))
    a = ang()
    for i in range(n):
        ops.append(("g1", i, rx(a[i, 1]) @ rz(a[i, 0])))
    return ops


def ising_ops(n, layers, gt_layer, J=1.0, h=0.7, dt=0.25, init="plus"):
    p = dephasing_p(gt_layer)
    ops = []
    if init == "plus":
        for i in range(n):
            ops.append(("g1", i, HM))
    for _ in range(layers):
        for i in range(0, n - 1, 2):
            ops.append(("g2", i, rzz(2 * J * dt)))
        for i in range(1, n - 1, 2):
            ops.append(("g2", i, rzz(2 * J * dt)))
        for i in range(n):
            ops.append(("g1", i, rx(2 * h * dt)))
        for i in range(n):
            ops.append(("dp", i, p))
    return ops


def product_ops(n, gt):
    p = dephasing_p(gt)
    return [("g1", i, HM) for i in range(n)] + [("dp", i, p) for i in range(n)]


# ----------------------------------------------------------------------------- observables
# Each observable is a list of (coef, {qubit: 2x2 matrix}) Pauli-like terms.

def obs_terms(name, n):
    c = n // 2
    if name == "x":
        return [(1.0 / n, {i: XM}) for i in range(n)]
    if name == "z":
        return [(1.0 / n, {i: ZM}) for i in range(n)]
    if name == "zz":
        return [(1.0 / (n - 1), {i: ZM, i + 1: ZM}) for i in range(n - 1)]
    if name == "xx":
        return [(1.0 / (n - 1), {i: XM, i + 1: XM}) for i in range(n - 1)]
    if name == "xc":
        return [(1.0, {c: XM})]
    if name == "zc":
        return [(1.0, {c: ZM})]
    if name == "yc":
        return [(1.0, {c: YM})]
    if name == "xpar":
        return [(1.0, {i: XM for i in range(n)})]
    raise ValueError(name)


OBS_ALL = ("x", "z", "zz", "xx", "xc", "zc", "yc", "xpar")


def obs_matrix(name, n):
    """Dense 2^n x 2^n matrix (qubit 0 = most significant), numpy."""
    D = 2 ** n
    out = np.zeros((D, D), complex)
    for coef, term in obs_terms(name, n):
        m = np.array([[1.0]], complex)
        for q in range(n):
            m = np.kron(m, term.get(q, I2))
        out += coef * m
    return out


# ----------------------------------------------------------------------------- kernels

def _v(psi, i, n, width=2):
    B = psi.shape[0]
    return psi.reshape(B, 2 ** i, width, 2 ** (n - i - (1 if width == 2 else 2)))


def apply_gate(psi, op, n, mats):
    kind, i, M = op
    width = 2 if kind == "g1" else 4
    v = _v(psi, i, n, width)
    return xp.einsum("ab,xibj->xiaj", mats[id(M)], v).reshape(psi.shape)


def expect_terms(psi, terms, n):
    """Per-trajectory <psi|O|psi> for O given as Pauli-like terms (exact)."""
    B = psi.shape[0]
    tot = xp.zeros(B)
    for coef, term in terms:
        phi = psi
        for q, m in term.items():
            v = _v(phi, q, n)
            phi = xp.einsum("ab,xibj->xiaj", xp.asarray(m), v).reshape(psi.shape)
        tot += coef * xp.real(xp.sum(xp.conj(psi) * phi, axis=1))
    return tot


def kraus_lam(p, lam):
    a, b, c = 1 - p - lam * p, (1 - lam) * p, 2 * lam * p
    return np.array([[np.sqrt(a), np.sqrt(a)], [np.sqrt(b), -np.sqrt(b)],
                     [np.sqrt(c), 0.0], [0.0, np.sqrt(c)]], complex)


def kraus_weak(p, t):
    s1, s2 = np.sqrt(1 - p), np.sqrt(p)
    k1 = np.cos(t) * s1 * np.array([1, 1]) + np.sin(t) * s2 * np.array([1, -1])
    k2 = -np.sin(t) * s1 * np.array([1, 1]) + np.cos(t) * s2 * np.array([1, -1])
    return np.array([k1, k2], complex)


def apply_noise(psi, i, n, d, rng):
    """d: (K, 2) or (B, K, 2) diagonal Kraus entries. Born-samples one per trajectory."""
    B = psi.shape[0]
    v = _v(psi, i, n)
    pi0 = xp.sum(xp.abs(v[:, :, 0, :]) ** 2, axis=(1, 2))
    pi1 = xp.sum(xp.abs(v[:, :, 1, :]) ** 2, axis=(1, 2))
    if d.ndim == 2:
        d = xp.broadcast_to(d[None], (B,) + d.shape)
    pr = xp.abs(d[:, :, 0]) ** 2 * pi0[:, None] + xp.abs(d[:, :, 1]) ** 2 * pi1[:, None]
    cum = xp.cumsum(pr, axis=1)
    u = rng.random(B) * cum[:, -1]
    k = xp.minimum(xp.sum(cum < u[:, None], axis=1), d.shape[1] - 1)
    idx = xp.arange(B)
    norm = xp.sqrt(pr[idx, k])
    f0 = d[idx, k, 0] / norm
    f1 = d[idx, k, 1] / norm
    v = v.copy()
    v[:, :, 0, :] *= f0[:, None, None]
    v[:, :, 1, :] *= f1[:, None, None]
    return v.reshape(psi.shape)


# ----------------------------------------------------------------------------- backward pass

def backprop_observables(ops, n, O):
    """Return list (one per 'dp' op, in forward order) of O~ just AFTER that event, plus O~_0.

    O~ evolves backwards: gate U -> U^dag O~ U; dephasing -> (1-p) O~ + p Z O~ Z.
    Dense; intended for n <= 10. Stored as xp arrays (complex128).
    """
    D = 2 ** n
    Ot = xp.asarray(O)
    after = []
    zdiag = {}
    for q in range(n):
        bits = (np.arange(D) >> (n - 1 - q)) & 1
        zdiag[q] = xp.asarray(1.0 - 2.0 * bits)
    for op in reversed(ops):
        kind, i, M = op
        if kind == "dp":
            # stored in single precision: only used for diagnostics / decision rules
            # (FP64 GEMM is ~64x slower on consumer GPUs); the trajectories stay FP64
            after.append(Ot.astype(xp.complex64))
            z = zdiag[i]
            Ot = (1 - M) * Ot + M * (z[:, None] * Ot * z[None, :])
        else:
            U = full_gate(op, n)
            Ot = U.conj().T @ Ot @ U
    after.reverse()
    return after, Ot


def _gate_conj_local(Ot, op, n, M):
    """Return U^dag Ot U for a 1- or 2-qubit gate without forming the 2^n x 2^n gate."""
    kind, i, _ = op
    k = 1 if kind == "g1" else 2
    w = 2 ** k
    D = Ot.shape[0]
    lo, hi = 2 ** i, 2 ** (n - i - k)
    Md = M.conj().T
    v = Ot.reshape(lo, w, hi, D)
    Ot = xp.einsum("ab,ibjc->iajc", Md, v).reshape(D, D)  # U^dag O (rows)
    v = Ot.reshape(D, lo, w, hi)
    return xp.einsum("xibj,ba->xiaj", v, M).reshape(D, D)  # (.) U (columns)


def backprop_observables_local(ops, n, O, host=True):
    """Same as backprop_observables but O(D^2) per gate (local application).

    Returns (after, O~_0). ``after`` entries are complex64; numpy (host) if host=True so that
    n = 12 with ~100 events fits (128 MB each). The running O~ is complex128 on the device.
    """
    D = 2 ** n
    Ot = xp.asarray(O, dtype=complex)
    mats = {}
    zdiag = {}
    for q in range(n):
        bits = (np.arange(D) >> (n - 1 - q)) & 1
        zdiag[q] = xp.asarray(1.0 - 2.0 * bits)
    after = []
    for op in reversed(ops):
        kind, i, M = op
        if kind == "dp":
            a = Ot.astype(xp.complex64)
            after.append(xp.asnumpy(a) if (host and GPU) else a)
            z = zdiag[i]
            Ot = (1 - M) * Ot + M * (z[:, None] * Ot * z[None, :])
        else:
            if id(M) not in mats:
                mats[id(M)] = xp.asarray(M)
            Ot = _gate_conj_local(Ot, op, n, mats[id(M)])
    after.reverse()
    return after, Ot


def full_gate(op, n):
    kind, i, M = op
    if kind == "g1":
        mats = [I2] * n
        mats = mats[:i] + [M] + mats[i + 1:]
        U = np.array([[1.0]], complex)
        for m in mats:
            U = np.kron(U, m)
    else:
        U = np.kron(np.kron(np.eye(2 ** i), M), np.eye(2 ** (n - i - 2)))
    return xp.asarray(U)


def local_AV(psi, Ot, i, n):
    """A = <O~_anti>, C = <O~_comm>, V_B (Born variance of comm values) per trajectory."""
    B = psi.shape[0]
    psi = psi.astype(Ot.dtype)
    v = _v(psi, i, n)
    p0 = xp.zeros_like(v)
    p0[:, :, 0, :] = v[:, :, 0, :]
    p0 = p0.reshape(psi.shape)
    p1 = psi - p0
    Op = psi @ Ot.T
    Op0 = p0 @ Ot.T
    Op1 = Op - Op0
    m00 = xp.real(xp.sum(xp.conj(p0) * Op0, axis=1)).astype(xp.float64)
    m11 = xp.real(xp.sum(xp.conj(p1) * Op1, axis=1)).astype(xp.float64)
    m01 = xp.sum(xp.conj(p0) * Op1, axis=1).astype(xp.complex128)
    m = m00 + m11 + 2 * xp.real(m01)
    mz = m00 + m11 - 2 * xp.real(m01)
    A = 0.5 * (m - mz)
    C = 0.5 * (m + mz)
    pi0 = xp.real(xp.sum(xp.abs(p0) ** 2, axis=1)).astype(xp.float64)
    pi1 = 1.0 - pi0
    s0 = xp.where(pi0 > 1e-14, m00 / xp.maximum(pi0, 1e-300), 0.0)
    s1 = xp.where(pi1 > 1e-14, m11 / xp.maximum(pi1, 1e-300), 0.0)
    VB = pi0 * s0 ** 2 + pi1 * s1 ** 2 - C ** 2
    _ = B
    return A, C, xp.maximum(VB, 0.0)


# ----------------------------------------------------------------------------- driver

def decompose_multi(ops, n, obs_names, shots, lam=0.0, seed=0, chunk=4096):
    """Sample lam-unraveled trajectories once; for EVERY observable return
    (a) per-trajectory estimates and (b) the per-event mean local increment variances
    of std and projector evaluated on these trajectories.

    With lam=0 the 'std' sum is an unbiased estimate of Var_std (martingale identity) and
    the 'proj' sum is a *cross-measure predictor* of Var_proj (projector formula on
    std-distributed states): this is the H1 predictor R_traj = sum std / sum proj.
    """
    rng = xp.random.default_rng(seed)
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "dp"}
    terms = {o: obs_terms(o, n) for o in obs_names}
    after = {o: backprop_observables(ops, n, obs_matrix(o, n))[0] for o in obs_names}
    n_ev = sum(1 for op in ops if op[0] == "dp")
    dsum = {o: {"std": np.zeros(n_ev), "proj": np.zeros(n_ev), "A2": np.zeros(n_ev),
                "VB": np.zeros(n_ev)} for o in obs_names}
    est = {o: [] for o in obs_names}
    done = 0
    d_used = None
    while done < shots:
        B = min(chunk, shots - done)
        psi = xp.zeros((B, 2 ** n), complex)
        psi[:, 0] = 1.0
        ev = 0
        for op in ops:
            if op[0] != "dp":
                psi = apply_gate(psi, op, n, mats)
                continue
            i, p = op[1], op[2]
            for o in obs_names:
                A, C, VB = local_AV(psi, after[o][ev], i, n)
                dsum[o]["std"][ev] += float(xp.sum(4 * p * (1 - p) * A ** 2))
                dsum[o]["proj"][ev] += float(xp.sum(2 * p * (1 - 2 * p) * A ** 2 + 2 * p * VB))
                dsum[o]["A2"][ev] += float(xp.sum(A ** 2))
                dsum[o]["VB"][ev] += float(xp.sum(VB))
            d_used = xp.asarray(kraus_lam(p, lam))
            psi = apply_noise(psi, i, n, d_used, rng)
            ev += 1
        for o in obs_names:
            est[o].append(expect_terms(psi, terms[o], n))
        done += B
    res = {}
    for o in obs_names:
        x = np.concatenate([xp.asnumpy(a) if GPU else a for a in est[o]])
        res[o] = {"est": x, **{k: v / shots for k, v in dsum[o].items()}}
    return res


def ideal_predictor(ops, n, obs_name):
    """H1 predictor on the single NOISELESS trajectory (no sampling): returns
    (sum std increments, sum proj increments). Cheap proxy of R_traj."""
    after, _ = backprop_observables(ops, n, obs_matrix(obs_name, n))
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "dp"}
    psi = xp.zeros((1, 2 ** n), complex)
    psi[:, 0] = 1.0
    s = q = 0.0
    ev = 0
    for op in ops:
        if op[0] != "dp":
            psi = apply_gate(psi, op, n, mats)
            continue
        i, p = op[1], op[2]
        A, C, VB = local_AV(psi, after[ev], i, n)
        s += float(4 * p * (1 - p) * A[0] ** 2)
        q += float(2 * p * (1 - 2 * p) * A[0] ** 2 + 2 * p * VB[0])
        ev += 1
    return s, q


def ideal_events(ops, n, obs_name):
    """Per-event (std, proj) local increment variances on the noiseless trajectory."""
    after, _ = backprop_observables(ops, n, obs_matrix(obs_name, n))
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "dp"}
    psi = xp.zeros((1, 2 ** n), complex)
    psi[:, 0] = 1.0
    s, q = [], []
    ev = 0
    for op in ops:
        if op[0] != "dp":
            psi = apply_gate(psi, op, n, mats)
            continue
        i, p = op[1], op[2]
        A, C, VB = local_AV(psi, after[ev], i, n)
        s.append(float(4 * p * (1 - p) * A[0] ** 2))
        q.append(float(2 * p * (1 - 2 * p) * A[0] ** 2 + 2 * p * VB[0]))
        ev += 1
    return np.array(s), np.array(q)


def expect_cross(pa, pb, terms, n):
    """Per-row <pa|O|pb> (complex) for O given as Pauli-like terms."""
    tot = xp.zeros(pa.shape[0], dtype=complex)
    for coef, term in terms:
        phi = pb
        for q, m in term.items():
            v = _v(phi, q, n)
            phi = xp.einsum("ab,xibj->xiaj", xp.asarray(m), v).reshape(pb.shape)
        tot += coef * xp.sum(xp.conj(pa) * phi, axis=1)
    return tot


def continuation_events(ops, n, obs_name, reps=8, seed=0, start=None):
    """Matrix-free per-event (std, proj) increment variances on the NOISELESS trajectory.

    Scalable replacement of ideal_events (no 4^n back-propagated observable). At event e on
    qubit i the state psi is split into P0 psi, P1 psi; both are continued through the rest
    of the circuit with the SAME std Pauli-jump realisation (std Kraus probabilities do not
    depend on the state, so E_jumps <P_a psi| U_r^dag O U_r |P_b psi> = <P_a psi|O~|P_b psi>
    exactly, i.e. the channel-averaged back-propagated observable). ``reps`` realisations.
    A = 2 Re m01, C = m00 + m11, V_B = pi0 s0^2 + pi1 s1^2 - C^2 (s_k = m_kk / pi_k).
    Cost: O(#events x #remaining gates x 2 reps) statevector gate applications.
    """
    rng = np.random.default_rng(seed)
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "dp"}
    terms = obs_terms(obs_name, n)
    zmat = xp.asarray(ZM)
    psi = xp.zeros((1, 2 ** n), complex)
    psi[:, 0] = 1.0
    s_out, q_out = [], []
    for k, op in enumerate(ops):
        if op[0] != "dp":
            psi = apply_gate(psi, op, n, mats)
            continue
        i, p = op[1], op[2]
        v = _v(psi, i, n)
        p0 = xp.zeros_like(v)
        p0[:, :, 0, :] = v[:, :, 0, :]
        p0 = p0.reshape(psi.shape)
        p1 = psi - p0
        pi0 = float(xp.sum(xp.abs(p0) ** 2))
        pi1 = 1.0 - pi0
        phi = xp.concatenate([xp.repeat(p0, reps, 0), xp.repeat(p1, reps, 0)])
        for op2 in ops[k + 1:]:
            if op2[0] != "dp":
                phi = apply_gate(phi, op2, n, mats)
                continue
            jump = rng.random(reps) < op2[2]
            if jump.any():
                rows = np.concatenate([np.nonzero(jump)[0], reps + np.nonzero(jump)[0]])
                sub = phi[rows]
                vv = _v(sub, op2[1], n)
                sub = xp.einsum("ab,xibj->xiaj", zmat, vv).reshape(sub.shape)
                phi[rows] = sub
        a0, a1 = phi[:reps], phi[reps:]
        m00 = float(xp.mean(xp.real(expect_cross(a0, a0, terms, n))))
        m11 = float(xp.mean(xp.real(expect_cross(a1, a1, terms, n))))
        m01 = complex(xp.mean(expect_cross(a0, a1, terms, n)))
        A = 2.0 * m01.real
        C = m00 + m11
        s0 = m00 / pi0 if pi0 > 1e-14 else 0.0
        s1 = m11 / pi1 if pi1 > 1e-14 else 0.0
        VB = max(pi0 * s0 ** 2 + pi1 * s1 ** 2 - C ** 2, 0.0)
        s_out.append(4 * p * (1 - p) * A ** 2)
        q_out.append(2 * p * (1 - 2 * p) * A ** 2 + 2 * p * VB)
    return np.array(s_out), np.array(q_out)


def static_table(ops, n, obs_name, pilot_shots, rounds=1, seed=0, start=None, tol=1e-3):
    """Static per-event std/projector table by (iterated) pilot runs.

    Round r samples ``pilot_shots`` trajectories under the current table and sets
    table[e] = 1 (projector) iff mean proj increment < (1 - tol) * mean std increment at e,
    both evaluated on the current table's own trajectory distribution.
    Returns (table, list of per-round tables). Unbiased for any table (static choice).
    """
    n_ev = sum(1 for op in ops if op[0] == "dp")
    tab = np.zeros(n_ev) if start is None else np.asarray(start, float).copy()
    hist = []
    for r in range(rounds):
        res = run(ops, n, [obs_name], pilot_shots, ("table", tab), seed=seed + 1000 * r,
                  decompose=True, obs_for_policy=obs_name)
        dec = res["_dec"]
        tab = (dec["proj"] < (1 - tol) * dec["std"]).astype(float)
        hist.append(tab.copy())
    return tab, hist


def backprop_two_copy(ops, n, O, dtype=complex, host=False):
    """Two-copy std back-propagation for the policy-improvement (PI) scheme.

    Under the std (mixed-Pauli) unraveling E[ rho (x) rho ] evolves LINEARLY:
        gate:  (U (x) U) . (U (x) U)^dag,   dephasing: (1-p) id + p (Z_i (x) Z_i) . (Z_i (x) Z_i).
    Back-propagating W = O (x) O gives, for each event t, W_t such that
        E_std[ M_T^2 | psi just after event t ] = <psi psi| W_t |psi psi>.
    Returns list of W_t (forward event order). Dense 4^n x 4^n: n <= 6.
    """
    D = 2 ** n
    W = xp.asarray(np.kron(O, O).astype(dtype))
    zd = {}
    for q in range(n):
        bits = (np.arange(D) >> (n - 1 - q)) & 1
        z = 1.0 - 2.0 * bits
        zd[q] = xp.asarray(np.kron(z, z).astype(dtype))
    after = []
    for op in reversed(ops):
        kind, i, M = op
        if kind == "dp":
            after.append(xp.asnumpy(W) if (host and GPU) else W.copy())
            z = zd[i]
            pf = float(M)  # python float: keeps W's dtype (np.float64 would promote c64->c128)
            W = (1 - pf) * W + pf * (z[:, None] * W * z[None, :])
        else:
            U = full_gate(op, n)
            UU = xp.kron(U, U).astype(dtype)
            W = UU.conj().T @ W @ UU
    after.reverse()
    return after


def static_pi_table(ops, n, obs_name, pilot_shots=0, seed=0, rel_tol=1e-5, abs_frac=1e-3,
                    dtype=np.complex64):
    """STATIC table from the PI criterion (local cost + exact std continuation), dense n <= 6.

    gain_e = E[e_std - e_proj] at event e, with e_a = E_a[Q_t(next)] as in run_pi, averaged over
    the noiseless trajectory (pilot_shots=0) or over std pilot trajectories. table[e] = 1 iff
    gain_e > rel_tol * |E e_std| and gain_e > abs_frac * max_e |E e_std - Q(before)| (negligible
    gains keep std). Returns (table, gain).
    """
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "dp"}
    Ws = backprop_two_copy(ops, n, obs_matrix(obs_name, n), dtype=dtype)
    n_ev = len(Ws)
    B = max(1, pilot_shots)
    rng = xp.random.default_rng(seed)
    psi = xp.zeros((B, 2 ** n), complex)
    psi[:, 0] = 1.0
    gs, ps_, base = np.zeros(n_ev), np.zeros(n_ev), np.zeros(n_ev)
    ev = 0
    for op in ops:
        if op[0] != "dp":
            psi = apply_gate(psi, op, n, mats)
            continue
        i, p = op[1], op[2]
        W = Ws[ev]
        v = _v(psi, i, n)
        p0 = xp.zeros_like(v)
        p0[:, :, 0, :] = v[:, :, 0, :]
        p0 = p0.reshape(psi.shape)
        p1 = psi - p0
        pi0 = xp.real(xp.sum(xp.abs(p0) ** 2, axis=1))
        pi1 = 1.0 - pi0
        q_id = _q2(psi, W)
        q_z = _q2(p0 - p1, W)
        q0 = xp.where(pi0 > 1e-14, _q2(p0, W) / xp.maximum(pi0, 1e-300), 0.0)
        q1 = xp.where(pi1 > 1e-14, _q2(p1, W) / xp.maximum(pi1, 1e-300), 0.0)
        e_std = (1 - p) * q_id + p * q_z
        e_proj = (1 - 2 * p) * q_id + 2 * p * (q0 + q1)
        gs[ev] = float(xp.mean(e_std))
        ps_[ev] = float(xp.mean(e_proj))
        base[ev] = float(xp.mean(q_id))
        if pilot_shots:
            psi = apply_noise(psi, i, n, xp.asarray(kraus_lam(p, 0.0)), rng)
        ev += 1
    gain = gs - ps_
    scale = np.max(np.abs(gs - base)) if n_ev else 0.0
    tab = ((gain > rel_tol * np.abs(gs)) & (gain > abs_frac * scale)).astype(float)
    return tab, gain


def _q2(phi, W):
    """<phi phi|W|phi phi> per trajectory, phi: (B, D)."""
    B, D = phi.shape
    pp = (phi[:, :, None] * phi[:, None, :]).reshape(B, D * D).astype(W.dtype)
    return xp.real(xp.sum(xp.conj(pp) * (pp @ W.T), axis=1))


def run_pi(ops, n, obs_name, shots, seed=0, chunk=2048, dtype=np.complex64, host=False):
    """Policy-improvement adaptive unraveling (per event, per trajectory: std or projector).

    Action a at event t is chosen to minimise E_a[ Q_t(next state) ], Q_t from
    backprop_two_copy, i.e. local increment variance + std continuation value.
    By the policy-improvement theorem the resulting scheme has Var <= Var_std
    (exactly, for the observable it is tuned to). Unbiased for any observable.
    Returns (estimates, fraction of events that chose the projector, per event).
    """
    rng = xp.random.default_rng(seed)
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "dp"}
    Ws = backprop_two_copy(ops, n, obs_matrix(obs_name, n), dtype=dtype, host=host)
    terms = obs_terms(obs_name, n)
    n_ev = len(Ws)
    frac = np.zeros(n_ev)
    est = []
    done = 0
    while done < shots:
        B = min(chunk, shots - done)
        psi = xp.zeros((B, 2 ** n), complex)
        psi[:, 0] = 1.0
        ev = 0
        for op in ops:
            if op[0] != "dp":
                psi = apply_gate(psi, op, n, mats)
                continue
            i, p = op[1], op[2]
            W = xp.asarray(Ws[ev])
            v = _v(psi, i, n)
            p0 = xp.zeros_like(v)
            p0[:, :, 0, :] = v[:, :, 0, :]
            p0 = p0.reshape(psi.shape)
            p1 = psi - p0
            pi0 = xp.real(xp.sum(xp.abs(p0) ** 2, axis=1))
            pi1 = 1.0 - pi0
            q_id = _q2(psi, W)
            q_z = _q2(p0 - p1, W)
            # Q(P_s psi / sqrt(pi_s)) * pi_s = Q(P_s psi) / pi_s  (Q is quartic)
            q0 = xp.where(pi0 > 1e-14, _q2(p0, W) / xp.maximum(pi0, 1e-300), 0.0)
            q1 = xp.where(pi1 > 1e-14, _q2(p1, W) / xp.maximum(pi1, 1e-300), 0.0)
            e_std = (1 - p) * q_id + p * q_z
            e_proj = (1 - 2 * p) * q_id + 2 * p * (q0 + q1)
            # strict improvement with a tie tolerance: Q is FP32, and exact ties (e.g. a qubit
            # already in a Z eigenstate, or symmetry-protected zero variance) must keep std
            choose = e_proj < e_std - (1e-4 * xp.abs(e_std) + 1e-12)
            frac[ev] += float(xp.sum(choose))
            ds, dp_ = xp.asarray(kraus_lam(p, 0.0)), xp.asarray(kraus_lam(p, 1.0))
            d = xp.where(choose[:, None, None], dp_[None], ds[None])
            psi = apply_noise(psi, i, n, d, rng)
            ev += 1
        est.append(expect_terms(psi, terms, n))
        done += B
    x = np.concatenate([xp.asnumpy(a) if GPU else a for a in est])
    return x, frac / shots


def run(ops, n, obs_names, shots, scheme=("lam", 0.0), seed=0, chunk=4096,
        decompose=False, obs_for_policy=None, local_bp=False):
    """Sample trajectories; return dict obs -> np.array(shots) of per-trajectory estimates.

    scheme: ("lam", lam) | ("weak", theta) | ("greedy", None) | ("table", lam_per_event)
      table: a STATIC per-event lam (e.g. 0/1 std/projector), same for every trajectory.
      greedy: per event and per trajectory pick std or projector, whichever has the
      smaller local increment variance for observable ``obs_for_policy`` (needs the
      dense backward pass, n <= 10). Unbiased: the choice depends only on the past.
    decompose=True also returns per-event mean increment variances for obs_names[0]
    (std and projector formulas evaluated on the sampled trajectories) under key '_dec'.
    """
    rng = xp.random.default_rng(seed)
    mats = {}
    for op in ops:
        if op[0] != "dp":
            mats[id(op[2])] = xp.asarray(op[2])
    terms = {o: obs_terms(o, n) for o in obs_names}
    need_bp = decompose or scheme[0] == "greedy"
    if need_bp:
        pol = obs_for_policy or obs_names[0]
        if local_bp:
            after, _ = backprop_observables_local(ops, n, obs_matrix(pol, n))
        else:
            after, _ = backprop_observables(ops, n, obs_matrix(pol, n))
    out = {o: [] for o in obs_names}
    dec_std, dec_proj, dec_used, frac_proj = None, None, None, None
    n_ev = sum(1 for op in ops if op[0] == "dp")
    if decompose:
        dec_std = np.zeros(n_ev)
        dec_proj = np.zeros(n_ev)
        dec_used = np.zeros(n_ev)
    if scheme[0] == "greedy":
        frac_proj = np.zeros(n_ev)
    done = 0
    while done < shots:
        B = min(chunk, shots - done)
        psi = xp.zeros((B, 2 ** n), complex)
        psi[:, 0] = 1.0
        ev = 0
        for op in ops:
            if op[0] != "dp":
                psi = apply_gate(psi, op, n, mats)
                continue
            i, p = op[1], op[2]
            if need_bp:
                A, C, VB = local_AV(psi, xp.asarray(after[ev]), i, n)
                vs = 4 * p * (1 - p) * A ** 2
                vp = 2 * p * (1 - 2 * p) * A ** 2 + 2 * p * VB
            if scheme[0] in ("lam", "table"):
                lam_ev = scheme[1] if scheme[0] == "lam" else float(scheme[1][ev])
                d = xp.asarray(kraus_lam(p, lam_ev))
                if decompose:
                    used = (1 - lam_ev) * vs + lam_ev * vp
            elif scheme[0] == "weak":
                d = xp.asarray(kraus_weak(p, scheme[1]))
                if decompose:
                    used = xp.full(B, np.nan)
            elif scheme[0] == "greedy":
                choose = vp < vs - (1e-4 * vs + 1e-15)  # tie tolerance (see run_pi)
                ds, dp_ = xp.asarray(kraus_lam(p, 0.0)), xp.asarray(kraus_lam(p, 1.0))
                d = xp.where(choose[:, None, None], dp_[None], ds[None])
                frac_proj[ev] += float(xp.sum(choose))
                if decompose:
                    used = xp.where(choose, vp, vs)
            else:
                raise ValueError(scheme)
            if decompose:
                dec_std[ev] += float(xp.sum(vs))
                dec_proj[ev] += float(xp.sum(vp))
                dec_used[ev] += float(xp.sum(used))
            psi = apply_noise(psi, i, n, d, rng)
            ev += 1
        for o in obs_names:
            out[o].append(expect_terms(psi, terms[o], n))
        done += B
    res = {o: np.concatenate([xp.asnumpy(a) if GPU else a for a in out[o]]) for o in obs_names}
    if decompose:
        res["_dec"] = {"std": dec_std / shots, "proj": dec_proj / shots, "used": dec_used / shots}
    if frac_proj is not None:
        res["_frac_proj"] = frac_proj / shots
    return res


def exact_mean(ops, n, obs_name, local_bp=False):
    """Exact channel-averaged expectation via the dense backward pass (n <= 10; n <= 12 local)."""
    if local_bp:
        _, O0 = backprop_observables_local(ops, n, obs_matrix(obs_name, n))
    else:
        _, O0 = backprop_observables(ops, n, obs_matrix(obs_name, n))
    psi0 = xp.zeros(2 ** n, complex)
    psi0[0] = 1.0
    return float(xp.real(xp.vdot(psi0, O0 @ psi0)))
