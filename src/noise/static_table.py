"""Static per-location unraveling tables (E10, docs/plan_B_characterization_2026-11-06.md).

A static table assigns ONE candidate unraveling to every channel event (location), the same for all trajectories, and
is fixed before the production run; traj_kraus.run(scheme=("table", tab)) then simulates with it at the cost of a fixed
scheme (no per-trajectory selection). Unbiased for every observable: every candidate realises the same channel and the
choice does not depend on the trajectory.

Construction (same local criterion as the greedy rule in traj_kraus.run): at event e on qubit i, with O~_e the M-term
truncated back-propagated observable (sparse_policy.record_after), the increment variance of candidate c on state psi is
V_c(psi) (traj_kraus.increment_vars). The table picks
    tab[e] = argmin_c  mean_{psi in pilot states at e}  V_c(psi),
with the greedy tie rule (candidate 0 unless strictly better by more than 1e-4 V_0 + 1e-15).
  kind "ideal": pilot states = the single noiseless trajectory (gates only, channels skipped), deterministic.
  kind "pilot": pilot states = `shots` trajectories under candidate 0 (std / jump), Born-sampled, own seed.
"""
from __future__ import annotations

import numpy as np

from . import sparse_policy as SP
from . import traj_kraus as K
from .traj_ref import GPU, xp


def tie_argmin(Vm):
    """Vm (n_ev, C) mean increment variances -> table (n_ev,) with the greedy tie rule (-> candidate 0)."""
    Vm = np.asarray(Vm, float)
    best = Vm.argmin(axis=1)
    vb = Vm[np.arange(len(Vm)), best]
    keep = vb >= Vm[:, 0] - (1e-4 * Vm[:, 0] + 1e-15)
    return np.where(keep, 0, best).astype(int)


def build_table(ops, n, rec, kind="pilot", shots=74, seed=0, chunk=37, opt2=True):
    """Static table from SparseEvent list ``rec`` (forward event order, from sparse_policy.record_after).

    Returns (tab (n_ev,) int, Vmean (n_ev, C) float). kind "ideal" ignores shots/seed."""
    if kind == "ideal":
        shots, chunk = 1, 1
    elif kind != "pilot":
        raise ValueError(kind)
    opt2 = bool(opt2 and GPU)
    rng = xp.random.default_rng(seed)
    mats = {id(op[2]): xp.asarray(op[2]) for op in ops if op[0] != "ch"}
    chans = {id(op[2]): xp.asarray(op[2].cands) for op in ops if op[0] == "ch"}
    n_ev = sum(1 for op in ops if op[0] == "ch")
    assert len(rec) == n_ev
    C = max(op[2].cands.shape[0] for op in ops if op[0] == "ch")
    Vsum = xp.zeros((n_ev, C))
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
            if opt2:
                Nev = K._N_fused(psi, i, n, ps=ps64)
                N, R = SP.NR_sparse(psi, rec[ev], n, Nev, ps=ps64 if SP.IMPL == "v2" else None)
            else:
                Nev = K._N_fast(psi, i, n)
                N, R = SP.NR_sparse(psi, rec[ev], n, Nev)
            V, _, _ = K.increment_vars(N, R, cx)
            Vsum[ev, :cx.shape[0]] += V.sum(axis=0)
            if kind == "pilot":
                Ks = xp.broadcast_to(cx[0][None], (B,) + cx.shape[1:])
                psi = K.apply_kraus_fast(psi, i, n, Ks, rng, N=Nev)
            ev += 1
        done += B
    Vm = xp.asnumpy(Vsum) / shots if GPU else Vsum / shots
    return tie_argmin(Vm), Vm
