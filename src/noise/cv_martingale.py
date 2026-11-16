"""Martingale control variate from the truncated back-propagated observable O~ (E13,
docs/plan_B_characterization_2026-11-06.md, "E13 pre-registration").

At channel event e on qubit i, with O~_e the M-term truncated observable just AFTER the event
(sparse_policy.record_after) and the scheme's candidate Kraus set {K_k} at that event, define for the state psi
before the event
    w_k   = <psi| K_k^dag K_k |psi>             (Born weights, exactly the ones used to sample the outcome)
    num_k = <psi| K_k^dag O~_e K_k |psi>        (from the same R tensor as the greedy rule, traj_kraus.increment_vars)
    m     = sum_{k: w_k > 0} num_k / sum_k w_k
and, after outcome k was sampled,
    D_e   = num_k / w_k - m                     (= <psi'|O~_e|psi'> - E[<psi'|O~_e|psi'> | psi], psi' the new state).
Given the past, E[D_e] = sum_k (w_k / sum w) num_k / w_k - m = 0 EXACTLY for any operator O~_e (truncated or not,
float32 or not): num_k is a fixed function of the pre-event state and the outcome law is the one used here.
So S = sum_e D_e has mean zero under every unraveling (fixed candidate or static table) and
    Y_beta = f - beta S,   f = <psi_T| O |psi_T>,
is unbiased for <O> for any beta that is independent of the trajectory it is applied to. With the exact O~ and
beta = 1, Y_1 = <0|O~_0|0> = <O> for every trajectory (telescoping; tested in validation/test_cv_martingale.py).

run_cv reproduces traj_kraus.run(..., fast=True, opt2=True) for ("fixed", c) and ("table", tab) with the SAME random
draws (one rng.random(B) per event, same Born weights), so the plain estimates f equal that run's estimates
trajectory by trajectory; it only adds the sparse R evaluation (sparse_policy.NR_sparse, the greedy rule's kernel) and
the per-event increment. traj_kraus.py and static_table.py are not modified.

Coefficient estimators (cv_estimates):
  beta1     : beta = 1 (no estimation, exactly unbiased)
  crossfit  : 2-fold cross-fitting by trajectory index parity; beta for fold j = Cov(f, S) / Var(S) on the OTHER fold
              (independent of the trajectories it is applied to -> exactly unbiased)
  insample  : beta = Cov(f, S) / Var(S) on all trajectories (O(1/N) bias; optimistic variance)
"""
from __future__ import annotations

import numpy as np

from . import sparse_policy as SP
from . import traj_kraus as K
from . import traj_ref as T
from .traj_ref import GPU, xp


def _kraus_step(psi, i, n, Ks, rng, N):
    """apply_kraus_fast (same code, same RNG draw) that also returns the sampled outcome index and the weights."""
    B, D = psi.shape
    w = xp.real(xp.einsum("xkaq,xkab,xqb->xk", xp.conj(Ks), Ks, N))
    w = xp.maximum(w, 0.0)
    cum = xp.cumsum(w, axis=1)
    uu = rng.random(B) * cum[:, -1]
    k = xp.minimum(xp.sum(cum < uu[:, None], axis=1), Ks.shape[1] - 1)
    idx = xp.arange(B)
    Kk = Ks[idx, k] / xp.sqrt(w[idx, k])[:, None, None]
    psi = K._mix2(psi.reshape(B, 2 ** i, 2, 2 ** (n - i - 1)), Kk).reshape(B, D)
    return psi, k, w


def increment(w, k, num):
    """D (B,) = num_k / w_k - sum_{w>0} num / sum w for the sampled k (see module docstring)."""
    B = w.shape[0]
    idx = xp.arange(B)
    pos = w > 0
    m = xp.where(pos, num, 0.0).sum(axis=1) / w.sum(axis=1)
    return num[idx, k] / w[idx, k] - m


def run_cv(ops, n, obs_names, shots, rec, scheme=("fixed", 0), seed=0, chunk=37, opt2=True):
    """Trajectories under ("fixed", c) or ("table", tab) with the martingale control variate of O~ = rec.

    rec: SparseEvent list (forward order) from sparse_policy.record_after(ops, n, target, M).
    Returns dict obs -> per-trajectory plain estimates f (as traj_kraus.run) and "_S" -> per-trajectory sum of
    increments (float64). Requires the opt2/v2 GPU path when GPU is available (as E10-E12); on CPU the numpy
    reference kernels are used (tests)."""
    opt2 = bool(opt2 and GPU)
    rng = xp.random.default_rng(seed)
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "ch"}
    chans = {id(op[2]): xp.asarray(op[2].cands) for op in ops if op[0] == "ch"}
    n_ev = sum(1 for op in ops if op[0] == "ch")
    assert len(rec) == n_ev
    if scheme[0] == "table":
        tab = [int(c) for c in np.asarray(scheme[1]).reshape(-1)]
        assert len(tab) == n_ev
    elif scheme[0] == "fixed":
        tab = [int(scheme[1])] * n_ev
    else:
        raise ValueError(scheme)
    terms = {o: T.obs_terms(o, n) for o in obs_names}
    out = {o: [] for o in obs_names}
    S_all = []
    done = 0
    while done < shots:
        B = min(chunk, shots - done)
        psi = xp.zeros((B, 2 ** n), complex)
        psi[:, 0] = 1.0
        ps64 = xp.empty(psi.shape, xp.complex64) if opt2 else None
        Ssum = xp.zeros(B)
        ev = 0
        for op in ops:
            if op[0] != "ch":
                psi = K.apply_gate_fast(psi, op, n, mats)
                continue
            i, cx = op[1], chans[id(op[2])]
            if opt2:
                Nev = K._N_fused(psi, i, n, ps=ps64)
                N, R = SP.NR_sparse(psi, rec[ev], n, Nev, ps=ps64 if SP.IMPL == "v2" else None)
            else:
                Nev = K._N_fast(psi, i, n)
                N, R = SP.NR_sparse(psi, rec[ev], n, Nev)
            cand = cx[tab[ev]]                                              # (K, 2, 2)
            num = xp.real(xp.einsum("kpq,kab,xpqab->xk", xp.conj(cand), cand, R))
            Ks = xp.broadcast_to(cand[None], (B,) + cand.shape)
            psi, k, w = _kraus_step(psi, i, n, Ks, rng, Nev)
            Ssum += increment(w, k, num)
            ev += 1
        for o in obs_names:
            e = K.expect_fast(psi, o, n)
            out[o].append(T.expect_terms(psi, terms[o], n) if e is None else e)
        S_all.append(Ssum)
        done += B
    res = {o: np.concatenate([xp.asnumpy(a) if GPU else np.asarray(a) for a in out[o]]) for o in obs_names}
    res["_S"] = np.concatenate([xp.asnumpy(a) if GPU else np.asarray(a) for a in S_all]).astype(np.float64)
    return res


def _beta(f, S):
    v = S.var(ddof=1)
    if not np.isfinite(v) or v <= 1e-300:
        return 0.0
    return float(np.cov(f, S, ddof=1)[0, 1] / v)


def cv_estimates(f, S):
    """Per-trajectory CV estimates Y for the three coefficient rules (module docstring) and the coefficients used."""
    f = np.asarray(f, np.float64)
    S = np.asarray(S, np.float64)
    ev = np.arange(len(f)) % 2 == 0
    b_even, b_odd = _beta(f[ev], S[ev]), _beta(f[~ev], S[~ev])
    bcf = np.where(ev, b_odd, b_even)                 # coefficient fitted on the OTHER fold
    bin_ = _beta(f, S)
    return dict(beta1=f - S, crossfit=f - bcf * S, insample=f - bin_ * S), \
        dict(beta_even=b_even, beta_odd=b_odd, beta_in=bin_)
