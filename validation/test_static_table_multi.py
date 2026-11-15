"""Tests for the multi-target static table (E11): src/noise/static_table_multi.py."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import static_table as ST  # noqa: E402
from src.noise import static_table_multi as SM  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise.traj_ref import GPU  # noqa: E402

OBS = ["x", "zz", "xc", "zc"]
CASES = [("deph_sweep", "brickwork", "s7"), ("ad_phase", "brickwork", "s8"), ("ad_phase", "ising", "h1.4"),
         ("deph_sweep", "ising", "h0.7")]


def _nev(ops):
    return sum(1 for op in ops if op[0] == "ch")


@pytest.mark.parametrize("ch,fam,var", CASES)
@pytest.mark.parametrize("opt2", [False, True])
def test_single_observable_equals_single_target_table(ch, fam, var, opt2):
    """With one observable the multi-target table (same pilot seed) is the E10 single-target pilot table."""
    if opt2 and not GPU:
        pytest.skip("opt2 needs cupy")
    n = 7
    ops = K.build(fam, var, n, 4, 0.03, ch)
    for o in ("zz", "xc"):
        rec, _, _ = S.record_after(ops, n, o, 100)
        old = S.IMPL
        try:
            S.IMPL = "v2" if opt2 else "v1"
            t1, V1 = ST.build_table(ops, n, rec, "pilot", shots=74, seed=13, chunk=37, opt2=opt2)
            t2, V2, V0 = SM.build_table_multi(ops, n, {o: rec}, shots=74, seed=13, chunk=37, opt2=opt2)
        finally:
            S.IMPL = old
        assert np.array_equal(V1, V2[o]), (ch, fam, var, o)
        assert np.array_equal(t1, t2), (ch, fam, var, o, np.mean(t1 == t2))
        assert abs(V0[o] - V1[:, 0].sum()) <= 1e-12 * abs(V0[o])


def test_combine_tie_rule_and_scaling():
    Vm = np.array([[1.0, 1.0, 2.0], [1.0, 0.99999, 2.0], [1.0, 0.5, 0.4], [0.0, 0.0, 0.0]])
    for s in (1.0, 1e-6, 3e4):
        tab, _, _ = SM.combine({"a": Vm * s})
        assert list(tab) == list(ST.tie_argmin(Vm)) == [0, 0, 2, 0]
    # two observables: equal normalised weight; candidate 1 helps a, candidate 2 helps b more
    A = np.array([[1.0, 0.2, 1.0]])
    Bm = np.array([[1.0, 1.0, 0.1]]) * 1e-3
    tab, Sx, V0 = SM.combine({"a": A, "b": Bm})
    assert np.allclose(Sx, [[2.0, 1.2, 1.1]]) and list(tab) == [2]
    tab, _, _ = SM.combine({"a": A, "b": Bm}, V0hat={"a": 1.0, "b": 1.0})   # unnormalised: a dominates
    assert list(tab) == [1]


@pytest.mark.parametrize("ch,fam,var", CASES)
def test_multi_table_unbiased_all_observables(ch, fam, var):
    """Multi-target tables (M = 100 truncated O~, as E11) are unbiased for EVERY observable vs the exact mean, n = 5."""
    n, shots = 5, 20000
    ops = K.build(fam, var, n, 3, 0.1, ch)
    ex = {o: K.exact_mean(ops, n, o) for o in OBS}
    recs = {o: S.record_after(ops, n, o, 100)[0] for o in OBS}
    for j, sub in enumerate((OBS, ["zz", "xc"])):
        tab, _, _ = SM.build_table_multi(ops, n, {o: recs[o] for o in sub}, shots=200, seed=5 + j, chunk=100,
                                         opt2=False)
        assert len(tab) == _nev(ops)
        r = K.run(ops, n, OBS, shots, ("table", tab), seed=700 + j, chunk=5000, fast=True)
        for o in OBS:
            x = r[o]
            v = x.var(ddof=1)
            if v < 1e-14:
                assert abs(x.mean() - ex[o]) < 1e-8
                continue
            z = (x.mean() - ex[o]) / np.sqrt(v / shots)
            assert abs(z) < 4.5, (ch, fam, var, sub, o, z)


@pytest.mark.skipif(not GPU, reason="needs cupy")
def test_multi_opt2_matches_reference_path():
    n = 8
    ops = K.build("brickwork", "s7", n, 4, 0.03, "ad_phase")
    recs = {o: S.record_after(ops, n, o, 100)[0] for o in ("zz", "x", "zc")}
    try:
        S.IMPL = "v2"
        a_t, a_V, _ = SM.build_table_multi(ops, n, recs, shots=74, seed=9, chunk=37, opt2=True)
        S.IMPL = "v1"
        b_t, b_V, _ = SM.build_table_multi(ops, n, recs, shots=74, seed=9, chunk=37, opt2=False)
    finally:
        S.IMPL = "v1"
    for o in recs:
        assert np.allclose(a_V[o], b_V[o], rtol=1e-3, atol=1e-9)
    assert np.mean(a_t == b_t) >= 0.98
