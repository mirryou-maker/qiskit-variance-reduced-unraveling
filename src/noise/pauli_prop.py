"""Sparse truncated Pauli propagation (Heisenberg picture) for traj_kraus circuits (E9a).

An observable is a list of Pauli strings with real coefficients, O = sum_s c_s P_s. A string on n <= 32
qubits is one uint64 key: bit q is the X bit of qubit q, bit 32 + q its Z bit (I = 00, X = x, Y = x+z,
Z = z). Every gate U and channel E acts on O through its real Pauli transfer matrix
    T[p, q] = Tr(P_q  G(P_p)) / d,   G(O) = U^dag O U   or   G(O) = sum_k K^dag O K  (Heisenberg adjoint),
applied term by term on the touched qubits; duplicate strings are merged. After every op the identity and
the M largest-|c| non-identity terms are kept. The answer <0..0| O~_0 |0..0> is the sum of c over strings
without X bits.

Qubit numbering matches traj_ref: a ("g2", i, U) gate acts on (i, i+1) with the first tensor factor on i.
GPU (cupy) if available, numpy otherwise.
"""
from __future__ import annotations

import numpy as np

from . import traj_ref as T
from .traj_ref import GPU, I2, XM, YM, ZM, xp

PAULI = (I2, XM, YM, ZM)                       # local index 0 I, 1 X, 2 Y, 3 Z
XB = np.array([0, 1, 1, 0], np.uint64)          # X bit of local index
ZB = np.array([0, 0, 1, 1], np.uint64)          # Z bit of local index
LOW = np.uint64(0xFFFFFFFF)


def _paulis(k):
    """k-qubit Pauli matrices, index p = sum_j idx_j 4^j with j = 0 the first tensor factor."""
    out = []
    for p in range(4 ** k):
        m = np.array([[1.0]], complex)
        for j in range(k):
            m = np.kron(m, PAULI[(p >> (2 * j)) & 3])
        out.append(m)
    return out


def transfer(op):
    """Real PTM (4^k x 4^k) of a traj_kraus op in the Heisenberg picture."""
    kind, _, M = op
    if kind == "ch":
        Ks = [K for K in M.cands[0] if np.any(K)]
        k = 1
        G = lambda P: sum(K.conj().T @ P @ K for K in Ks)  # noqa: E731
    else:
        k = 1 if kind == "g1" else 2
        U = np.asarray(M)
        G = lambda P: U.conj().T @ P @ U  # noqa: E731
    Ps = _paulis(k)
    d = 2 ** k
    Tm = np.array([[np.trace(Q @ G(P)) / d for Q in Ps] for P in Ps])
    assert np.max(np.abs(Tm.imag)) < 1e-12, "PTM not real"
    return Tm.real


def _table(Tm, tol=1e-14):
    """Branch table: for every input p the nonzero (q, T[p, q]), padded to the max branching."""
    nz = [np.nonzero(np.abs(row) > tol)[0] for row in Tm]
    b = max(1, max(len(z) for z in nz))
    qs = np.zeros((len(Tm), b), np.int64)
    cs = np.zeros((len(Tm), b))
    for p, z in enumerate(nz):
        qs[p, :len(z)] = z
        cs[p, :len(z)] = Tm[p, z]
    outs = [int(z[0]) for z in nz if len(z)]
    inj = b == 1 and len(set(outs)) == len(outs)     # one-to-one: distinct strings stay distinct, no merge
    return xp.asarray(qs), xp.asarray(cs), b, inj


def _local_idx(keys, q):
    x = (keys >> np.uint64(q)) & np.uint64(1)
    z = (keys >> np.uint64(32 + q)) & np.uint64(1)
    return (x * (np.uint64(1) + z) + np.uint64(3) * z * (np.uint64(1) - x)).astype(np.int64)


def _set_local(keys, q, idx):
    xb, zb = xp.asarray(XB), xp.asarray(ZB)
    clear = ~((np.uint64(1) << np.uint64(q)) | (np.uint64(1) << np.uint64(32 + q)))
    return (keys & clear) | (xb[idx] << np.uint64(q)) | (zb[idx] << np.uint64(32 + q))


def _merge(keys, c):
    order = xp.argsort(keys)
    ks, cs = keys[order], c[order]
    if ks.size == 0:
        return ks, cs
    start = xp.concatenate([xp.ones(1, bool), ks[1:] != ks[:-1]])
    seg = xp.cumsum(start) - 1
    return ks[start], xp.bincount(seg, weights=cs)


def _truncate(keys, c, M):
    nz = c != 0
    keys, c = keys[nz], c[nz]
    if keys.size <= M + 1:
        return keys, c
    mag = xp.abs(c)
    mag = xp.where(keys == 0, xp.inf, mag)        # identity always kept, not counted in M
    idx = xp.argpartition(-mag, M)[:M + 1]
    return keys[idx], c[idx]


class PauliOp:
    def __init__(self, keys, coefs):
        self.keys, self.c = keys, coefs

    @classmethod
    def from_terms(cls, terms, n):
        assert n <= 32
        ks, cs = [], []
        for coef, term in terms:
            key = 0
            for q, m in term.items():
                j = next(j for j, P in enumerate(PAULI) if np.allclose(P, m))
                key |= int(XB[j]) << q | int(ZB[j]) << (32 + q)
            ks.append(key)
            cs.append(coef)
        k, c = _merge(xp.asarray(np.array(ks, np.uint64)), xp.asarray(np.array(cs, float)))
        return cls(k, c)

    def __len__(self):
        return int(self.keys.size)

    def apply(self, op, table, M):
        kind, i, _ = op
        qs, cs, b, inj = table
        p = _local_idx(self.keys, i)
        if kind == "g2":
            p = p + 4 * _local_idx(self.keys, i + 1)
        if b == 1:                                   # one output per input (permutation / scaling)
            q, w = qs[p, 0], cs[p, 0]
            keys, c = self._relabel(self.keys, kind, i, q), self.c * w
            if not inj:
                keys, c = _merge(keys, c)
        else:
            q, w = qs[p].reshape(-1), cs[p].reshape(-1)
            keys = xp.repeat(self.keys, b)
            c = xp.repeat(self.c, b) * w
            keep = w != 0
            keys, c, q = keys[keep], c[keep], q[keep]
            keys, c = _merge(self._relabel(keys, kind, i, q), c)
        self.keys, self.c = _truncate(keys, c, M)

    @staticmethod
    def _relabel(keys, kind, i, q):
        keys = _set_local(keys, i, q & 3)
        if kind == "g2":
            keys = _set_local(keys, i + 1, (q >> 2) & 3)
        return keys

    def zero_state_value(self):
        mask = (self.keys & LOW) == 0
        return float(xp.sum(self.c[mask]))


def propagate(ops, n, obs_name, M, tables=None):
    """Truncated PP of obs through ops (reverse order). Returns (value on |0..0>, max term count seen)."""
    tables = {} if tables is None else tables
    O = PauliOp.from_terms(T.obs_terms(obs_name, n), n)
    peak = len(O)
    for op in reversed(ops):
        key = id(op[2])
        if key not in tables:
            tables[key] = _table(transfer(op))
        O.apply(op, tables[key], M)
        peak = max(peak, len(O))
    return O.zero_state_value(), peak


def _host(a):
    return a.get() if GPU else np.asarray(a)


def _bucket(keys, nb):
    """Hash bucket of each key (Fibonacci hashing of the 64-bit key)."""
    return ((keys * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(nb)


def _apply_big(O, op, table, M, chunk, nb, rng, stats):
    """Same result as O.apply(op, table, M), with the branched term list never held whole on the device.

    Inputs are expanded in chunks of `chunk` terms on the device, merged within the chunk, split into `nb` hash
    buckets and moved to host RAM. Each bucket is then merged on the device (a key lives in exactly one bucket, so
    bucket merges are exact). The kept set is the M+1 largest |c| (identity always kept), selected exactly via a
    sampled magnitude threshold t: all terms with |c| >= t are gathered, and if fewer than M+1 survive, t is lowered.
    """
    kind, i, _ = op
    qs, cs, b, inj = table
    if b == 1 or len(O) * b <= chunk * 4:
        O.apply(op, table, M)
        return
    parts = [[] for _ in range(nb)]
    for s in range(0, len(O), chunk):
        kk, cc = O.keys[s:s + chunk], O.c[s:s + chunk]
        p = _local_idx(kk, i)
        if kind == "g2":
            p = p + 4 * _local_idx(kk, i + 1)
        q, w = qs[p].reshape(-1), cs[p].reshape(-1)
        keys = xp.repeat(kk, b)
        c = xp.repeat(cc, b) * w
        keep = w != 0
        keys, c, q = keys[keep], c[keep], q[keep]
        keys, c = _merge(O._relabel(keys, kind, i, q), c)
        bid = _bucket(keys, nb)
        order = xp.argsort(bid)
        bid, keys, c = bid[order], keys[order], c[order]
        bounds = _host(xp.searchsorted(bid, xp.arange(nb + 1, dtype=bid.dtype)))
        kh, ch = _host(keys), _host(c)
        for j in range(nb):
            if bounds[j + 1] > bounds[j]:
                parts[j].append((kh[bounds[j]:bounds[j + 1]], ch[bounds[j]:bounds[j + 1]]))
        del p, q, w, keys, c, keep, bid, order
    O.keys = O.c = None
    if GPU:
        xp.get_default_memory_pool().free_all_blocks()
    merged, total = [], 0
    for j in range(nb):
        if not parts[j]:
            continue
        k = np.concatenate([a for a, _ in parts[j]])
        c = np.concatenate([a for _, a in parts[j]])
        parts[j] = None
        kd, cd = _merge(xp.asarray(k), xp.asarray(c))
        nz = cd != 0
        merged.append((_host(kd[nz]), _host(cd[nz])))
        total += merged[-1][0].size
        del k, c, kd, cd, nz
    stats["peak_unique"] = max(stats.get("peak_unique", 0), total)
    if total <= M + 1:
        O.keys = xp.asarray(np.concatenate([a for a, _ in merged]))
        O.c = xp.asarray(np.concatenate([a for _, a in merged]))
        return
    ns = 4_000_000
    samp = np.concatenate([np.abs(cb[rng.integers(0, cb.size, max(1, int(round(ns * cb.size / total))))])
                           for _, cb in merged])
    frac = 1.0 - 1.3 * (M + 1) / total
    while True:
        t = np.quantile(samp, frac) if frac > 0 else 0.0
        sel = [(kb, cb) for kb, cb in ((kb[(np.abs(cb) >= t) | (kb == 0)], cb[(np.abs(cb) >= t) | (kb == 0)])
                                       for kb, cb in merged)]
        cnt = sum(kb.size for kb, _ in sel)
        if cnt >= M + 1 or frac <= 0:
            break
        stats["threshold_retries"] = stats.get("threshold_retries", 0) + 1
        frac = 1.0 - 2.0 * (1.0 - frac)
    keys = xp.asarray(np.concatenate([kb for kb, _ in sel]))
    c = xp.asarray(np.concatenate([cb for _, cb in sel]))
    del sel, merged
    O.keys, O.c = _truncate(keys, c, M)


def propagate_big(ops, n, obs_name, M, tables=None, chunk=5_000_000, nb=64, seed=0, stats=None):
    """propagate() for large M: identical truncation, branched lists staged in host RAM (see _apply_big)."""
    tables = {} if tables is None else tables
    stats = {} if stats is None else stats
    rng = np.random.default_rng(seed)
    O = PauliOp.from_terms(T.obs_terms(obs_name, n), n)
    peak = len(O)
    for op in reversed(ops):
        key = id(op[2])
        if key not in tables:
            tables[key] = _table(transfer(op))
        _apply_big(O, op, tables[key], M, chunk, nb, rng, stats)
        peak = max(peak, len(O))
    return O.zero_state_value(), peak


def sync():
    if GPU:
        xp.cuda.Stream.null.synchronize()
