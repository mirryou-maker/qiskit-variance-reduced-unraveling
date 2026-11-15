"""Multi-target static per-location unraveling table (E11, docs/plan_B_characterization_2026-11-06.md,
"E11 pre-registration").

A single static table (one candidate per channel event, the same for every trajectory, fixed before production) chosen
for a SET of observables at once, so that one production run serves all of them (E10 tables are per target observable).

Construction. One pilot of `shots` trajectories under candidate 0 (std / jump), Born-sampled, own seed (the same pilot
loop and RNG draws as static_table.build_table(kind="pilot")). At every event e and for every observable o in the set,
with O~_e(o) the M-term truncated back-propagated observable (sparse_policy.record_after), the local increment variance
V_c(o, psi) (traj_kraus.increment_vars) is averaged over the pilot states: Vbar_c(o, e).
    V0hat(o)   = sum_e Vbar_0(o, e)       (pilot estimate of the candidate-0 estimator variance of o: the
                                           law-of-total-variance / martingale decomposition, see
                                           validation/test_static_table.py::test_pilot_variance_decomposition)
    S_c(e)     = sum_o Vbar_c(o, e) / V0hat(o)
    tab[e]     = argmin_c S_c(e), tie rule: candidate 0 unless S_best < S_0 - (1e-4 S_0 + 1e-15 sum_o 1 / V0hat(o))
Under the additive approximation Var_tab(o) ~ sum_e Vbar_{tab[e]}(o, e), sum_o S_{tab[e]}(e) summed over e is the sum
over o of the predicted variance ratio Var_tab(o) / Var_0(o); the objective is separable over events, so the per-event
argmin minimises it. Observables with V0hat(o) <= 1e-300 get weight 0 (deterministic; none in the E11 groups).
With one observable the table equals static_table.build_table(kind="pilot") for the same seed (the normalisation is a
positive constant and the tie tolerance is scaled with it).
Unbiased for every observable for the same reason as any static table: every candidate realises the same channel and
the choice does not depend on the trajectory.
"""
from __future__ import annotations

import numpy as np

from . import sparse_policy as SP
from . import traj_kraus as K
from .traj_ref import GPU, xp


def combine(Vm, V0hat=None):
    """Vm: dict obs -> (n_ev, C) mean increment variances. Returns (tab, S, V0hat) with the normalised criterion."""
    obs = list(Vm)
    if V0hat is None:
        V0hat = {o: float(np.asarray(Vm[o], float)[:, 0].sum()) for o in obs}
    w = {o: (1.0 / V0hat[o] if V0hat[o] > 1e-300 else 0.0) for o in obs}
    S = sum(np.asarray(Vm[o], float) * w[o] for o in obs)
    best = S.argmin(axis=1)
    sb = S[np.arange(len(S)), best]
    keep = sb >= S[:, 0] - (1e-4 * S[:, 0] + 1e-15 * sum(w.values()))
    return np.where(keep, 0, best).astype(int), S, V0hat


def pilot_vars(ops, n, recs, shots=74, seed=0, chunk=37, opt2=True):
    """recs: dict obs -> SparseEvent list (forward order). One candidate-0 pilot; returns dict obs -> Vm (n_ev, C)."""
    opt2 = bool(opt2 and GPU)
    rng = xp.random.default_rng(seed)
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "ch"}
    chans = {id(op[2]): xp.asarray(op[2].cands) for op in ops if op[0] == "ch"}
    n_ev = sum(1 for op in ops if op[0] == "ch")
    obs = list(recs)
    for o in obs:
        assert len(recs[o]) == n_ev
    C = max(op[2].cands.shape[0] for op in ops if op[0] == "ch")
    Vsum = {o: xp.zeros((n_ev, C)) for o in obs}
    done = 0
    while done < shots:
        B = min(chunk, shots - done)
        psi = xp.zeros((B, 2 ** n), complex)
        psi[:, 0] = 1.0
        ps64 = xp.empty(psi.shape, xp.complex64) if opt2 else None
        ev = 0
        for op in ops:
            if op[0] != "ch":
                psi = K.apply_gate_fast(psi, op, n, mats)
                continue
            i, cx = op[1], chans[id(op[2])]
            Nev = K._N_fused(psi, i, n, ps=ps64) if opt2 else K._N_fast(psi, i, n)
            for o in obs:
                if opt2:
                    N, R = SP.NR_sparse(psi, recs[o][ev], n, Nev, ps=ps64 if SP.IMPL == "v2" else None)
                else:
                    N, R = SP.NR_sparse(psi, recs[o][ev], n, Nev)
                V, _, _ = K.increment_vars(N, R, cx)
                Vsum[o][ev, :cx.shape[0]] += V.sum(axis=0)
            Ks = xp.broadcast_to(cx[0][None], (B,) + cx.shape[1:])
            psi = K.apply_kraus_fast(psi, i, n, Ks, rng, N=Nev)
            ev += 1
        done += B
    return {o: (xp.asnumpy(Vsum[o]) if GPU else Vsum[o]) / shots for o in obs}


def build_table_multi(ops, n, recs, shots=74, seed=0, chunk=37, opt2=True):
    """Multi-target static table. Returns (tab (n_ev,) int, Vm dict obs -> (n_ev, C), V0hat dict obs -> float)."""
    Vm = pilot_vars(ops, n, recs, shots=shots, seed=seed, chunk=chunk, opt2=opt2)
    tab, _, V0hat = combine(Vm)
    return tab, Vm, V0hat
