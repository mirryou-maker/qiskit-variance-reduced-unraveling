"""Native batched GPU trajectory sampler for layered dephasing circuits (Plan A, step 3).

Performance code for wall-clock comparisons; validated against src/noise/traj_ref.py.

Circuit model (matches traj_ref.ising_ops / brickwork_ops gate-for-gate)
-----------------------------------------------------------------------
stages: ("u1", [2x2]*n)              one single-qubit gate per qubit (traj_ref qubit order)
        ("zz", theta)                exp(-i theta/2 Z_i Z_{i+1}) on all bonds
        ("cz", [bond indices i])     CZ on (i, i+1)
        ("noise", p)                 dephasing on every qubit, events numbered in qubit order
traj_ref qubit q is stored at bit (n-1-q) of the amplitude index.

Unravelings per event: table[e] = 0 (std: Z jump with prob p) or 1 (projector: with prob 2p a
Z measurement, Born outcome). Event probabilities of *whether* something happens do not
depend on the state, only projector outcomes do.

Kernels (one read+write pass each over B x 2^n complex128 amplitudes)
  u1_group : a group of k<=10 bit positions (always containing bits 0..4 for coalescing) is
             loaded into shared memory and all its single-qubit gates are applied there.
  diag     : zz/cz phases x std jump signs x projector masks/renormalisation (all diagonal).
  hist     : per-trajectory joint Born distribution of its measured qubits (<= 8 per call,
             conditional chunks for more).
  obs_*    : per-trajectory expectation values.
Pending diagonal work is folded into one pass and flushed before the next u1 stage.
"""
from __future__ import annotations

import numpy as np

import cupy as cp

_SRC = r"""
typedef unsigned long long u64;

__device__ __forceinline__ u64 pdep_list(u64 x, const int* pos, int k) {
    u64 r = 0;
    for (int m = 0; m < k; ++m) r |= ((x >> m) & 1ULL) << pos[m];
    return r;
}

extern "C" __global__ void u1_group(double2* psi, const double2* g, const int* gpos,
                                    const int* opos, int k, int no, long long D,
                                    const int* active) {
    extern __shared__ double2 s[];
    const int half = 1 << (k - 1);
    const long long tile = blockIdx.x;
    const long long b = blockIdx.y;
    double2* p = psi + b * D;
    u64 base = pdep_list((u64)tile, opos, no);
    int tid = threadIdx.x;
    for (int l = tid; l < (1 << k); l += half) {
        u64 idx = base | pdep_list((u64)l, gpos, k);
        s[l] = p[idx];
    }
    __syncthreads();
    for (int m = 0; m < k; ++m) {
        if (active[m]) {
            int l0 = ((tid >> m) << (m + 1)) | (tid & ((1 << m) - 1));
            int l1 = l0 | (1 << m);
            double2 a = s[l0], c = s[l1];
            const double2* G = g + 4 * m;
            double2 r0, r1;
            r0.x = G[0].x * a.x - G[0].y * a.y + G[1].x * c.x - G[1].y * c.y;
            r0.y = G[0].x * a.y + G[0].y * a.x + G[1].x * c.y + G[1].y * c.x;
            r1.x = G[2].x * a.x - G[2].y * a.y + G[3].x * c.x - G[3].y * c.y;
            r1.y = G[2].x * a.y + G[2].y * a.x + G[3].x * c.y + G[3].y * c.x;
            s[l0] = r0; s[l1] = r1;
        }
        __syncthreads();
    }
    for (int l = tid; l < (1 << k); l += half) {
        u64 idx = base | pdep_list((u64)l, gpos, k);
        p[idx] = s[l];
    }
}

// kind: 0 none, 1 zz uniform theta over bondmask, 2 cz over bondmask
extern "C" __global__ void diag_pass(double2* psi, long long D, int kind, double theta,
                                     u64 bondmask, int nb, const u64* jmask, const u64* pmask,
                                     const u64* pval, const double* pscale) {
    const long long b = blockIdx.y;
    double2* p = psi + b * D;
    const u64 jm = jmask[b], pm = pmask[b], pv = pval[b];
    const double sc = pscale[b];
    for (long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x; j < D;
         j += (long long)gridDim.x * blockDim.x) {
        double2 a = p[j];
        double re = a.x, im = a.y;
        if (pm && (((u64)j & pm) != pv)) { re = 0.0; im = 0.0; }
        else {
            double f = sc;
            if (__popcll((u64)j & jm) & 1) f = -f;
            if (kind == 2 && (__popcll((u64)j & ((u64)j >> 1) & bondmask) & 1)) f = -f;
            re *= f; im *= f;
            if (kind == 1) {
                int anti = __popcll(((u64)j ^ ((u64)j >> 1)) & bondmask);
                double ph = -0.5 * theta * (double)(nb - 2 * anti);
                double c = cos(ph), sn = sin(ph);
                double r2 = re * c - im * sn, i2 = re * sn + im * c;
                re = r2; im = i2;
            }
        }
        a.x = re; a.y = im;
        p[j] = a;
    }
}

// rows: compacted list of trajectories; hist[r*256 + key]; condition (fm, fv) per row
extern "C" __global__ void hist_pass(const double2* psi, long long D, const int* rows,
                                     const u64* mmask, const u64* fmask, const u64* fval,
                                     double* hist) {
    __shared__ double h[256];
    const int r = blockIdx.y;
    const long long b = rows[r];
    const double2* p = psi + b * D;
    const u64 mm = mmask[r], fm = fmask[r], fv = fval[r];
    for (int t = threadIdx.x; t < 256; t += blockDim.x) h[t] = 0.0;
    __syncthreads();
    for (long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x; j < D;
         j += (long long)gridDim.x * blockDim.x) {
        if (((u64)j & fm) != fv) continue;
        double2 a = p[j];
        double w = a.x * a.x + a.y * a.y;
        int key = 0, c = 0;
        u64 m = mm;
        while (m) { u64 lsb = m & (~m + 1ULL); if ((u64)j & lsb) key |= (1 << c); ++c; m ^= lsb; }
        atomicAdd(&h[key], w);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 256; t += blockDim.x)
        if (h[t] != 0.0) atomicAdd(&hist[r * 256 + t], h[t]);
}

// Atomic-free Born masses: block (x, row, key t). Enumerates only amplitudes whose fixed bits
// (chunk | filter) equal (pdep(t, chunk) | fval): free index o -> insert zeros at fixed bits.
// Each amplitude is read once over all t. Block tree reduction, one global atomic per block.
extern "C" __global__ void hist_gather(const double2* psi, long long D, const int* rows,
                                       const u64* mmask, const u64* fmask, const u64* fval,
                                       double* hist) {
    __shared__ double red[256];
    const int r = blockIdx.y;
    const int t = blockIdx.z;
    const u64 mm = mmask[r], fm = fmask[r], fv = fval[r];
    const int nc = __popcll(mm);
    if (t >= (1 << nc)) return;
    const u64 fixed = mm | fm;
    const long long nfree = D >> __popcll(fixed);
    u64 v = fv & ~mm;
    { u64 m = mm; int c = 0; while (m) { u64 l = m & (~m + 1ULL); if ((t >> c) & 1) v |= l; ++c; m ^= l; } }
    const double2* p = psi + (long long)rows[r] * D;
    double acc = 0.0;
    for (long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x; o < nfree;
         o += (long long)gridDim.x * blockDim.x) {
        u64 j = (u64)o;
        u64 m = fixed;
        while (m) {
            int pos = __ffsll((long long)m) - 1;
            j = ((j >> pos) << (pos + 1)) | (j & ((1ULL << pos) - 1ULL));
            m &= m - 1;
        }
        double2 a = p[j | v];
        acc += a.x * a.x + a.y * a.y;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && red[0] != 0.0) atomicAdd(&hist[r * 256 + t], red[0]);
}

// diagonal observables: kind 0: z avg (fullmask, n), 1: zz avg (bondmask, nb), 2: single z (bit)
extern "C" __global__ void obs_diag(const double2* psi, long long D, int kind, u64 mask, int cnt,
                                    double* out) {
    __shared__ double red[256];
    const long long b = blockIdx.y;
    const double2* p = psi + b * D;
    double acc = 0.0;
    for (long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x; j < D;
         j += (long long)gridDim.x * blockDim.x) {
        double2 a = p[j];
        double w = a.x * a.x + a.y * a.y;
        double f;
        if (kind == 0) f = (double)(cnt - 2 * __popcll((u64)j & mask)) / cnt;
        else if (kind == 1) f = (double)(cnt - 2 * __popcll(((u64)j ^ ((u64)j >> 1)) & mask)) / cnt;
        else f = (((u64)j & mask) ? -1.0 : 1.0);
        acc += w * f;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&out[b], red[0]);
}

// X-type observables on a group tile: sum_m w[m] <X_{gpos[m]}>
extern "C" __global__ void obs_x_group(const double2* psi, const int* gpos, const int* opos,
                                       int k, int no, long long D, const double* w, double* out) {
    extern __shared__ double2 s[];
    __shared__ double red[512];
    const int half = 1 << (k - 1);
    const long long tile = blockIdx.x;
    const long long b = blockIdx.y;
    const double2* p = psi + b * D;
    u64 base = pdep_list((u64)tile, opos, no);
    int tid = threadIdx.x;
    for (int l = tid; l < (1 << k); l += half) s[l] = p[base | pdep_list((u64)l, gpos, k)];
    __syncthreads();
    double acc = 0.0;
    for (int m = 0; m < k; ++m) {
        if (w[m] != 0.0) {
            int l0 = ((tid >> m) << (m + 1)) | (tid & ((1 << m) - 1));
            int l1 = l0 | (1 << m);
            double2 a = s[l0], c = s[l1];
            acc += w[m] * 2.0 * (a.x * c.x + a.y * c.y);
        }
    }
    red[tid] = acc;
    __syncthreads();
    for (int st = half / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&out[b], red[0]);
}
"""

_mod = None


def _k(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=("-std=c++14",))
    return _mod.get_function(name)


I2 = np.eye(2, dtype=complex)
HM = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)


def rz(a):
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def rx(b):
    c, s = np.cos(b / 2), np.sin(b / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def dephasing_p(gt):
    return 0.5 * (1.0 - np.exp(-2.0 * gt))


def ising_circuit(n, L, gt, J=1.0, h=0.7, dt=0.25):
    p = dephasing_p(gt)
    st = [("u1", [HM] * n)]
    for _ in range(L):
        st += [("zz", 2 * J * dt), ("u1", [rx(2 * h * dt)] * n), ("noise", p)]
    return st


def brickwork_circuit(n, L, gt, seed=7, clifford=False):
    rng = np.random.default_rng(seed)
    p = dephasing_p(gt)

    def ang():
        a = rng.uniform(0, 2 * np.pi, size=(n, 2))
        if clifford:
            a = np.round(a / (np.pi / 2)) * (np.pi / 2)
        return a

    st = []
    for layer in range(L):
        a = ang()
        st.append(("u1", [rx(a[i, 1]) @ rz(a[i, 0]) for i in range(n)]))
        st.append(("cz", list(range(layer % 2, n - 1, 2))))
        st.append(("noise", p))
    a = ang()
    st.append(("u1", [rx(a[i, 1]) @ rz(a[i, 0]) for i in range(n)]))
    return st


def n_events(stages, n):
    return n * sum(1 for s in stages if s[0] == "noise")


class FastSim:
    """Batched trajectory sampler. ``table``: per-event 0/1 (std/projector), or None (all std)."""

    def __init__(self, n, kgroup=10, B=None):
        self.n = n
        self.D = 1 << n
        k = min(kgroup, n)
        self.groups = []
        if n <= k:
            self.groups.append(list(range(n)))
        else:
            self.groups.append(list(range(k)))
            low = list(range(5))
            rest = list(range(k, n))
            step = k - 5
            for s in range(0, len(rest), step):
                self.groups.append(low + rest[s:s + step])
        self._gdev = []
        for g in self.groups:
            o = [b for b in range(n) if b not in g]
            self._gdev.append((cp.asarray(np.array(g, np.int32)), cp.asarray(np.array(o, np.int32)),
                               len(g), len(o)))
        self.B = B or max(1, min(512, (1 << 31) // (16 * self.D)))

    # bit position of traj_ref qubit q
    def bit(self, q):
        return self.n - 1 - q

    def _u1(self, psi, mats, B):
        n = self.n
        for (gpos, opos, k, no), g in zip(self._gdev, self.groups):
            G = np.zeros((k, 4), complex)
            act = np.zeros(k, np.int32)
            for m, bpos in enumerate(g):
                q = n - 1 - bpos
                M = mats[q] if bpos >= 5 or g is self.groups[0] else None
                if M is None or np.allclose(M, I2):
                    G[m] = [1, 0, 0, 1]
                    continue
                G[m] = M.reshape(4)
                act[m] = 1
            if not act.any():
                continue
            _k("u1_group")((self.D >> k, B), (1 << (k - 1),),
                           (psi, cp.asarray(G), gpos, opos, np.int32(k), np.int32(no),
                            np.int64(self.D), cp.asarray(act)),
                           shared_mem=16 * (1 << k))

    def _diag(self, psi, B, kind, theta, bondmask, nb, jm, pm, pv, ps):
        threads = 256
        blocks = int(min(max(1, self.D // threads), 4096))
        _k("diag_pass")((blocks, B), (threads,),
                        (psi, np.int64(self.D), np.int32(kind), np.float64(theta),
                         np.uint64(bondmask), np.int32(nb), jm, pm, pv, ps))

    def _evolve(self, psi, stages, tab, rng, urow=None, noise=True):
        """Apply ``stages`` in place to the batch psi (B, D).

        tab: per-event 0/1 for the noise stages in ``stages`` (in order).
        urow: optional int array (B,) -> rows with the same id share their noise random
              numbers (used by the continuation estimator). noise=False skips noise stages.
        Returns the number of projector events applied.
        """
        n = self.n
        B = psi.shape[0]
        ev = 0
        st_ = {"pend": None, "dirty": False}
        jm = np.zeros(B, np.uint64)
        pm = np.zeros(B, np.uint64)
        pv = np.zeros(B, np.uint64)
        ps = np.ones(B)
        nproj = 0

        def flush():
            if st_["pend"] is None and not st_["dirty"]:
                return
            kind, th, bm, nb = st_["pend"] if st_["pend"] is not None else (0, 0.0, 0, 0)
            self._diag(psi, B, kind, th, bm, nb, cp.asarray(jm), cp.asarray(pm),
                       cp.asarray(pv), cp.asarray(ps))
            st_["pend"] = None
            jm[:] = 0
            pm[:] = 0
            pv[:] = 0
            ps[:] = 1.0
            st_["dirty"] = False

        bits = (1 << (n - 1 - np.arange(n))).astype(np.uint64)
        for st in stages:
            if st[0] == "u1":
                flush()
                self._u1(psi, st[1], B)
            elif st[0] in ("zz", "cz"):
                if st_["pend"] is not None:
                    flush()
                if st[0] == "zz":
                    st_["pend"] = (1, st[1], (1 << (n - 1)) - 1, n - 1)
                else:
                    bm = 0
                    for i in st[1]:
                        bm |= 1 << (n - 2 - i)  # bond (i, i+1) -> lower bit n-2-i
                    st_["pend"] = (2, 0.0, bm, 0)
            else:  # noise
                p = st[1]
                lam = tab[ev:ev + n]
                ev += n
                if not noise:
                    continue
                if urow is None:
                    u = rng.random((B, n))
                else:
                    u = rng.random((int(urow.max()) + 1, n))[urow]
                jump = (lam[None, :] == 0) & (u < p)
                meas = (lam[None, :] == 1) & (u < 2 * p)
                jm ^= (jump * bits[None, :]).sum(axis=1).astype(np.uint64)
                if meas.any():
                    mm = (meas * bits[None, :]).sum(axis=1).astype(np.uint64)
                    if (pm & mm).any():
                        # a bit re-measured before the pending projection was applied: the
                        # combined filter in hist_gather assumes disjoint masks -> apply first
                        flush()
                    rows = np.nonzero(mm)[0]
                    nproj += int(meas.sum())
                    # a projection still pending on a row is folded in (Born masses of the raw
                    # state under the pending filter; pending phases/signs do not change |a|^2)
                    val, prob = self._measure(psi, rows, mm[rows], rng, pm[rows], pv[rows])
                    pm[rows] |= mm[rows]
                    pv[rows] |= val
                    ps[rows] = 1.0 / np.sqrt(prob)
                st_["dirty"] = True
        flush()
        return nproj

    def run(self, stages, obs, shots, table=None, seed=0, return_stats=False):
        """Return per-trajectory estimates (np.ndarray) for observable obs in {x, zz, xc, zc, z}."""
        n, D = self.n, self.D
        ne = n_events(stages, n)
        tab = np.zeros(ne) if table is None else np.asarray(table, float)
        rng = np.random.default_rng(seed)
        out = []
        nproj_events = 0
        done = 0
        while done < shots:
            B = min(self.B, shots - done)
            psi = cp.zeros((B, D), complex)
            psi[:, 0] = 1.0
            nproj_events += self._evolve(psi, stages, tab, rng)
            out.append(self._obs(psi, B, obs))
            done += B
        x = np.concatenate(out)
        if return_stats:
            return x, {"proj_events_per_shot": nproj_events / shots}
        return x

    def continuation_table(self, stages, obs, reps=4, seed=0, tol=1e-3, abs_frac=1e-2,
                           max_rows=None):
        """Static std/projector table from the noiseless trajectory (matrix-free, scalable).

        At the noise stage of layer l, for every qubit q the noiseless state psi is split into
        P0 psi, P1 psi (Z_q eigen-components). The rows {psi, P0 psi, P1 psi (all q)} are
        continued through the REST of the circuit (later layers, std noise) sharing the same
        jump realisation within a repetition r, so with M_r(.) = <.|O|.> at the end
        m00 = E_r M_r(P0 psi), m11 = E_r M_r(P1 psi), Re m01 = (E_r M_r(psi) - m00 - m11) / 2
        are unbiased for the channel-averaged back-propagated observable (std jump
        probabilities do not depend on the state). Then A = 2 Re m01, C = m00 + m11,
        V_B = pi0 s0^2 + pi1 s1^2 - C^2 and std/proj increments as in traj_ref.
        Approximation: dephasing events of the same layer are not included in the
        continuation (they commute; only the decision quality is affected, never the bias).
        Negligible local gains keep std: projector only if q < (1 - tol) s AND
        s - q > abs_frac * max_e s_e (MC noise on near-zero increments must not flip events).
        Returns (table, std_increments, proj_increments).
        """
        n, D = self.n, self.D
        rng = np.random.default_rng(seed)
        ne = n_events(stages, n)
        zero_tab = np.zeros(ne)
        s_out = np.zeros(ne)
        q_out = np.zeros(ne)
        psi = cp.zeros((1, D), complex)
        psi[:, 0] = 1.0
        ev = 0
        max_rows = max_rows or max(1, (1 << 32) // (16 * D))
        for si, st in enumerate(stages):
            if st[0] != "noise":
                self._evolve(psi, [st], zero_tab, rng, noise=False)
                continue
            p = st[1]
            rest = stages[si + 1:]
            rest_ne = n_events(rest, n)
            # rows: for each rep r: psi, then (P0, P1) for every qubit -> 1 + 2n rows per rep
            per_rep = 1 + 2 * n
            mvals = np.zeros((reps, per_rep))
            pis = np.zeros(n)
            rep_chunk = max(1, max_rows // per_rep)
            qubit_chunk = n if per_rep <= max_rows else max(1, (max_rows - 1) // 2)
            for r0 in range(0, reps, rep_chunk):
                rr = min(rep_chunk, reps - r0)
                for q0 in range(0, n, qubit_chunk):
                    qs = list(range(q0, min(n, q0 + qubit_chunk)))
                    rows = []
                    urow = []
                    for r in range(rr):
                        rows.append(psi[0])
                        urow.append(r)
                        for q in qs:
                            b = self.bit(q)
                            idx = cp.arange(D)
                            m1 = ((idx >> b) & 1).astype(bool)
                            p0 = cp.where(m1, 0, psi[0])
                            rows += [p0, psi[0] - p0]
                            urow += [r, r]
                            if r0 == 0 and r == 0:
                                pis[q] = float(cp.sum(cp.abs(p0) ** 2))
                    batch = cp.stack(rows)
                    # same seed per (layer, rep) so that qubit chunks share jump realisations
                    sub = np.random.default_rng((seed, si, r0))
                    self._evolve(batch, rest, np.zeros(rest_ne), sub, urow=np.array(urow))
                    vals = self._obs(batch, batch.shape[0], obs).reshape(rr, 1 + 2 * len(qs))
                    mvals[r0:r0 + rr, 0] = vals[:, 0]
                    for k, q in enumerate(qs):
                        mvals[r0:r0 + rr, 1 + 2 * q] = vals[:, 1 + 2 * k]
                        mvals[r0:r0 + rr, 2 + 2 * q] = vals[:, 2 + 2 * k]
            M = mvals.mean(axis=0)
            for q in range(n):
                m00, m11 = M[1 + 2 * q], M[2 + 2 * q]
                A = M[0] - m00 - m11
                C = m00 + m11
                pi0, pi1 = pis[q], 1.0 - pis[q]
                s0 = m00 / pi0 if pi0 > 1e-14 else 0.0
                s1 = m11 / pi1 if pi1 > 1e-14 else 0.0
                VB = max(pi0 * s0 ** 2 + pi1 * s1 ** 2 - C ** 2, 0.0)
                s_out[ev + q] = 4 * p * (1 - p) * A ** 2
                q_out[ev + q] = 2 * p * (1 - 2 * p) * A ** 2 + 2 * p * VB
            ev += n
        thr = abs_frac * (s_out.max() if ne else 0.0)
        tab = ((q_out < (1 - tol) * s_out) & (s_out - q_out > thr)).astype(float)
        return tab, s_out, q_out

    def _measure(self, psi, rows, masks, rng, fm0=None, fv0=None):
        """Sample joint Born outcomes of the masked bits for the given trajectories.

        fm0/fv0: projection still pending on these rows (not yet applied to psi). Born masses are
        then taken over the filtered amplitudes, so the returned ``prob`` is the total mass of the
        combined (pending + new) projection of the raw state and 1/sqrt(prob) is the combined
        renormalisation. Vectorised over rows (no per-row Python loop).
        """
        R = len(rows)
        one = np.uint64(1)
        val = np.zeros(R, np.uint64)
        prob = np.ones(R)
        remaining = masks.astype(np.uint64).copy()
        fm = np.zeros(R, np.uint64) if fm0 is None else fm0.astype(np.uint64).copy()
        fv = np.zeros(R, np.uint64) if fv0 is None else fv0.astype(np.uint64).copy()
        threads = 256
        while remaining.any():
            # lowest <= 8 set bits of each row's remaining mask
            lsbs = np.zeros((8, R), np.uint64)
            rem = remaining.copy()
            for c in range(8):
                l = rem & (~rem + one)
                lsbs[c] = l
                rem ^= l
            chunk = np.bitwise_or.reduce(lsbs, axis=0)
            act = np.nonzero(chunk)[0]
            hist = cp.zeros((len(act), 256))
            nc_max = int(max(bin(int(c)).count("1") for c in chunk[act]))
            nfix_min = int(min(bin(int(c) | int(f)).count("1") for c, f in zip(chunk[act], fm[act])))
            nfree = self.D >> nfix_min
            blocks = int(min(max(1, nfree // (threads * 16)), 1024))
            _k("hist_gather")((blocks, len(act), 1 << nc_max), (threads,),
                              (psi, np.int64(self.D), cp.asarray(rows[act].astype(np.int32)),
                               cp.asarray(chunk[act]), cp.asarray(fm[act]), cp.asarray(fv[act]), hist))
            h = cp.asnumpy(hist)
            cum = np.cumsum(h, axis=1)
            tgt = rng.random(len(act)) * cum[:, -1]
            key = np.minimum((cum <= tgt[:, None]).sum(axis=1), 255)
            # never pick an empty bin (round-off at the top edge): step down to the last nonzero
            hz = h[np.arange(len(act)), key] <= 0
            if hz.any():
                for a in np.nonzero(hz)[0]:
                    nz = np.nonzero(h[a, :key[a] + 1] > 0)[0]
                    key[a] = nz[-1] if len(nz) else 0
            bitsval = np.zeros(len(act), np.uint64)
            la = lsbs[:, act]
            for c in range(8):
                bitsval |= np.where((key >> c) & 1, la[c], np.uint64(0))
            val[act] |= bitsval
            prob[act] = h[np.arange(len(act)), key]
            fm[act] |= chunk[act]
            fv[act] |= bitsval
            remaining[act] &= ~chunk[act]
        return val, prob

    def _obs(self, psi, B, obs):
        n, D = self.n, self.D
        out = cp.zeros(B)
        threads = 256
        blocks = int(min(max(1, D // threads), 4096))
        if obs in ("z", "zz", "zc"):
            if obs == "z":
                kind, mask, cnt = 0, (1 << n) - 1, n
            elif obs == "zz":
                kind, mask, cnt = 1, (1 << (n - 1)) - 1, n - 1
            else:
                kind, mask, cnt = 2, 1 << self.bit(n // 2), 1
            _k("obs_diag")((blocks, B), (threads,),
                           (psi, np.int64(D), np.int32(kind), np.uint64(mask), np.int32(cnt), out))
        else:
            w_all = np.zeros(n)
            if obs == "x":
                w_all[:] = 1.0 / n
            elif obs == "xc":
                w_all[self.bit(n // 2)] = 1.0
            else:
                raise ValueError(obs)
            covered = set()
            for (gpos, opos, k, no), g in zip(self._gdev, self.groups):
                w = np.array([w_all[b] if b not in covered else 0.0 for b in g])
                covered.update(g)
                if not w.any():
                    continue
                _k("obs_x_group")((D >> k, B), (1 << (k - 1),),
                                  (psi, gpos, opos, np.int32(k), np.int32(no), np.int64(D),
                                   cp.asarray(w), out), shared_mem=16 * (1 << k))
        return cp.asnumpy(out)
