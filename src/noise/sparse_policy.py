"""Greedy-unraveling policy statistics from a SPARSE back-propagated observable (E9b).

traj_kraus.run(greedy) needs, at every channel event on qubit i,
    N[b', b]        = <u_b'|u_b>
    R[a', b', a, b] = <a' u_b'| O~ |a u_b>
where u_b is psi with qubit i fixed to b. traj_kraus._NR builds R from a DENSE O~ (2^n x 2^n), impossible at
n = 20. Here O~ = sum_j c_j L_j (x) P_j is a truncated Pauli sum from pauli_prop (L_j the local Pauli on i,
P_j the rest), so
    R[a', b', a, b] = sum_j c_j L_j[a', a] G_j[b', b],    G_j[b', b] = <u_b'| P_j |u_b>,
and with P_j |x> = i^{nY_j} (-1)^{popcount(x & zm_j)} |x ^ xm_j>  (labelled Paulis, Y = i X Z per qubit)
    G_j[b', b] = i^{nY_j} sum_x conj(u_b'[x ^ xm_j]) (-1)^{popcount(x & zm_j)} u_b[x].
One CUDA block per (string, trajectory) accumulates the four sums in complex64. The policy only has to pick
a candidate; its numerical precision cannot bias the estimator (the choice depends only on the past), so
float32 is safe here and only affects which candidate is picked.

Flat index convention (traj_ref): qubit q is bit n - 1 - q of the state index.
"""
from __future__ import annotations

import numpy as np

from . import pauli_prop as P
from .traj_ref import GPU, xp

_SRC = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void qco_gsum(const complex<float>* psi, const unsigned int* xm, const unsigned int* zm,
                                   int D, int p, int nrest, complex<float>* out) {
    // grid (S, B); out[(t * S + j) * 4 + 2 * b' + b]
    int j = blockIdx.x, t = blockIdx.y, S = gridDim.x;
    const complex<float>* v = psi + (size_t)t * D;
    unsigned int X = xm[j], Z = zm[j], hb = 1u << p, lowm = hb - 1u;
    complex<float> a00(0, 0), a01(0, 0), a10(0, 0), a11(0, 0);
    for (int r = threadIdx.x; r < nrest; r += blockDim.x) {
        unsigned int x = ((r & ~lowm) << 1) | (r & lowm);
        float s = (__popc(x & Z) & 1) ? -1.f : 1.f;
        unsigned int y = x ^ X;
        complex<float> c0 = v[x] * s, c1 = v[x | hb] * s;
        complex<float> f0 = conj(v[y]), f1 = conj(v[y | hb]);
        a00 += f0 * c0; a01 += f0 * c1; a10 += f1 * c0; a11 += f1 * c1;
    }
    __shared__ complex<float> sh[4][256];
    sh[0][threadIdx.x] = a00; sh[1][threadIdx.x] = a01; sh[2][threadIdx.x] = a10; sh[3][threadIdx.x] = a11;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w)
            for (int k = 0; k < 4; ++k) sh[k][threadIdx.x] += sh[k][threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x < 4) out[((size_t)t * S + j) * 4 + threadIdx.x] = sh[threadIdx.x][0];
}
"""
# E9e ("v2"): tiled variant. One block per (tile of TILE rest indices, trajectory). The block loads its tile of
# psi (both values of qubit i) into shared memory ONCE; each warp then takes whole strings (j = warp, warp + 8,
# ...), its lanes sweep the tile and read the partner amplitudes psi[x ^ X_j] from global memory. Strings are
# sorted by X mask, so strings sharing the high bits of X (processed concurrently by neighbouring warps) read the
# same partner tile (L1/L2 hits): DRAM traffic per event ~ (1 + #distinct X tiles) state reads instead of ~2 S,
# and one warp reduction per (string, tile). Block partials go to part (B, nT, S, 4); the host sums over tiles
# (deterministic order).
_SRC2 = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void qco_gsum_t(const complex<float>* psi, const unsigned int* xm, const unsigned int* zm,
                                      const int* perm, const int* gstart, const int* glen, int ng,
                                      int S, int D, int p, int nrest, float* part) {
    __shared__ complex<float> s0[TILE], s1[TILE];
    const int tile = blockIdx.x, t = blockIdx.y, nT = gridDim.x;
    const complex<float>* v = psi + (size_t)t * D;
    const unsigned int hb = 1u << p, lowm = hb - 1u;
    const int base = tile * TILE;
    const int len = min(TILE, nrest - base);
    for (int e = threadIdx.x; e < len; e += blockDim.x) {
        const unsigned int r = base + e, x = ((r & ~lowm) << 1) | (r & lowm);
        s0[e] = v[x]; s1[e] = v[x | hb];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    // groups of <= GMAX strings with the SAME X: partner amplitudes and the four products are shared,
    // only the sign (-1)^{popcount(x & Z_k)} differs per string.
    for (int g = warp; g < ng; g += nw) {
        const int j0 = gstart[g], gl = glen[g];
        const unsigned int X = xm[j0];
        unsigned int Zk[GMAX];
        float acc[GMAX][8];
        #pragma unroll
        for (int k = 0; k < GMAX; ++k) {
            Zk[k] = (k < gl) ? zm[j0 + k] : 0u;
            #pragma unroll
            for (int c = 0; c < 8; ++c) acc[k][c] = 0.f;
        }
        for (int e = lane; e < len; e += 32) {
            const unsigned int r = base + e, x = ((r & ~lowm) << 1) | (r & lowm), y = x ^ X;
            const complex<float> c0 = s0[e], c1 = s1[e];
            const complex<float> f0 = conj(v[y]), f1 = conj(v[y | hb]);
            const complex<float> p00 = f0 * c0, p01 = f0 * c1, p10 = f1 * c0, p11 = f1 * c1;
            #pragma unroll
            for (int k = 0; k < GMAX; ++k) {
                const float s = (__popc(x & Zk[k]) & 1) ? -1.f : 1.f;
                acc[k][0] += s * p00.real(); acc[k][1] += s * p00.imag();
                acc[k][2] += s * p01.real(); acc[k][3] += s * p01.imag();
                acc[k][4] += s * p10.real(); acc[k][5] += s * p10.imag();
                acc[k][6] += s * p11.real(); acc[k][7] += s * p11.imag();
            }
        }
        #pragma unroll
        for (int k = 0; k < GMAX; ++k) {
            if (k >= gl) break;                                   // warp-uniform
            float q[8];
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                q[c] = acc[k][c];
                for (int off = 16; off > 0; off >>= 1) q[c] += __shfl_xor_sync(0xffffffffu, q[c], off);  // all lanes
            }
            if (lane < 8) {
                float qc = q[0];
                #pragma unroll
                for (int c = 1; c < 8; ++c) if (lane == c) qc = q[c];
                part[(((size_t)t * nT + tile) * S + perm[j0 + k]) * 8 + lane] = qc;
            }
        }
    }
}
"""
_KERN = {}
_LOC = np.array([np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
IMPL = "v1"          # "v1": E9b kernel (one block per string); "v2": tiled kernel (E9e)
TILE = 1024          # v2: rest indices per block
GMAX = 4             # v2: strings per same-X group processed together (register budget)


def _kernel():
    if "g" not in _KERN:
        _KERN["g"] = xp.RawKernel(_SRC, "qco_gsum")
    return _KERN["g"]


def _kernel2(tile, gmax):
    k = ("t", tile, gmax)
    if k not in _KERN:
        _KERN[k] = xp.RawKernel(_SRC2, "qco_gsum_t", options=(f"-DTILE={tile}", f"-DGMAX={gmax}"))
    return _KERN[k]


def _groups(ev, gmax):
    """Runs of equal X in the sorted string list, split into chunks of <= gmax (cached per event)."""
    key = ("groups", gmax)
    if key not in ev.cache:
        xs = xp.asnumpy(ev.xm_s) if GPU else ev.xm_s
        st, ln = [], []
        j = 0
        while j < len(xs):
            k = j
            while k < len(xs) and xs[k] == xs[j]:
                k += 1
            for a in range(j, k, gmax):
                st.append(a)
                ln.append(min(gmax, k - a))
            j = k
        ev.cache[key] = (xp.asarray(np.array(st, np.int32)), xp.asarray(np.array(ln, np.int32)), len(st))
    return ev.cache[key]


def G_tiled(ps, ev, n, tile=None, gmax=None):
    """G (B, S, 4) complex64 from the tiled kernel; ps is psi as complex64 (B, D)."""
    tile, gmax = tile or TILE, gmax or GMAX
    B, D = ps.shape
    nrest = D // 2
    nT = -(-nrest // tile)
    gs, gl, ng = _groups(ev, gmax)
    part = xp.empty((B, nT, ev.S, 4), xp.complex64)
    _kernel2(tile, gmax)((nT, B), (256,), (ps, ev.xm_s, ev.zm_s, ev.perm, gs, gl, np.int32(ng), np.int32(ev.S),
                                           np.int32(D), np.int32(n - 1 - ev.i), np.int32(nrest), part))
    return part.sum(axis=1)


class SparseEvent:
    """Precomputed data for one channel event: masks (flat index, qubit i removed) and W[j, a', a]."""

    def __init__(self, keys, c, i, n):
        keys = np.asarray(xp.asnumpy(keys) if GPU else keys, np.uint64)
        c = np.asarray(xp.asnumpy(c) if GPU else c, float)
        xb = np.array([(keys >> np.uint64(q)) & np.uint64(1) for q in range(n)], np.int64)        # (n, S)
        zb = np.array([(keys >> np.uint64(32 + q)) & np.uint64(1) for q in range(n)], np.int64)
        loc = (xb[i] * (1 + zb[i]) + 3 * zb[i] * (1 - xb[i]))                                      # 0 I 1 X 2 Y 3 Z
        rest = np.array([q != i for q in range(n)])
        nY = ((xb & zb)[rest]).sum(axis=0)
        sh = np.array([n - 1 - q for q in range(n)], np.int64)[:, None]
        xm = ((xb * rest[:, None]) << sh).sum(axis=0)
        zm = ((zb * rest[:, None]) << sh).sum(axis=0)
        W = (c * (1j ** nY))[:, None, None] * _LOC[loc]                                             # (S, a', a)
        self.i, self.S = i, len(c)
        self.xm = xp.asarray(xm.astype(np.uint32))
        self.zm = xp.asarray(zm.astype(np.uint32))
        order = np.lexsort((zm, xm))                      # v2: strings sharing X (and its high bits) adjacent
        self.xm_s = xp.asarray(xm[order].astype(np.uint32))
        self.zm_s = xp.asarray(zm[order].astype(np.uint32))
        self.perm = xp.asarray(order.astype(np.int32))    # sorted position -> original string index
        self.cache = {}
        self.W = xp.asarray(W.reshape(len(c), 4))                                                   # [j, 2a'+a]


def record_after(ops, n, obs_name, M):
    """Truncated PP of obs (pauli_prop, truncation after every op) recording O~ just AFTER each channel event
    in forward order, as SparseEvent objects. Returns (events, value on |0..0>, peak terms)."""
    tables = {}
    O = P.PauliOp.from_terms(P.T.obs_terms(obs_name, n), n)
    rec, peak = [], len(O)
    for op in reversed(ops):
        key = id(op[2])
        if key not in tables:
            tables[key] = P._table(P.transfer(op))
        if op[0] == "ch":
            rec.append(SparseEvent(O.keys, O.c, op[1], n))
        O.apply(op, tables[key], M)
        peak = max(peak, len(O))
    rec.reverse()
    return rec, O.zero_state_value(), peak


def NR_sparse(psi, ev, n, N, ps=None):
    """R (B, 2, 2, 2, 2) [a', b', a, b] from a SparseEvent; N (B, 2, 2) is passed through unchanged.
    ps: psi already cast to complex64 (v2 only; traj_kraus opt2 writes it in the N pass)."""
    B, D = psi.shape
    p = n - 1 - ev.i
    if GPU and IMPL == "v2":
        G = G_tiled(psi.astype(xp.complex64) if ps is None else ps, ev, n)
    elif GPU:
        ps = psi.astype(xp.complex64)
        G = xp.empty((B, ev.S, 4), xp.complex64)
        _kernel()((ev.S, B), (256,), (ps, ev.xm, ev.zm, np.int32(D), np.int32(p), np.int32(D // 2), G))
    else:
        G = _G_numpy(psi, ev, n)
    # R[t, a', b', a, b] = sum_j W[j, a', a] G[t, j, b', b]
    R = xp.einsum("jA,tjB->tBA", ev.W.astype(xp.complex64), G).astype(complex)   # (B, [b'b], [a'a])
    R = R.reshape(B, 2, 2, 2, 2).transpose(0, 3, 1, 4, 2)                          # [t, a', b', a, b]
    return N, R


def _G_numpy(psi, ev, n):
    """Reference (numpy, complex128) version of the kernel, for tests."""
    B, D = psi.shape
    p = n - 1 - ev.i
    hb = 1 << p
    xm, zm = xp.asnumpy(ev.xm) if GPU else ev.xm, xp.asnumpy(ev.zm) if GPU else ev.zm
    ps = xp.asnumpy(psi) if GPU else psi
    xs = np.array([x for x in range(D) if not (x & hb)], np.int64)
    G = np.zeros((B, ev.S, 4), complex)
    for j in range(ev.S):
        s = 1 - 2 * (np.array([bin(int(x) & int(zm[j])).count("1") & 1 for x in xs]))
        y = xs ^ int(xm[j])
        c0, c1 = ps[:, xs] * s, ps[:, xs | hb] * s
        f0, f1 = np.conj(ps[:, y]), np.conj(ps[:, y | hb])
        G[:, j] = np.stack([(f0 * c0).sum(1), (f0 * c1).sum(1), (f1 * c0).sum(1), (f1 * c1).sum(1)], 1)
    return xp.asarray(G)
