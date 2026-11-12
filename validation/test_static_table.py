"""Tests for the static per-location table scheme (E10): traj_kraus.run(scheme=("table", tab)) and
src/noise/static_table.py."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import static_table as ST  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise.traj_ref import GPU  # noqa: E402

OBS = ["x", "zz", "xc", "zc"]
CASES = [("deph_sweep", "brickwork", "s7"), ("ad_phase", "brickwork", "s8"), ("ad_phase", "ising", "h1.4"),
         ("deph_sweep", "ising", "h0.7")]


def _nev(ops):
    return sum(1 for op in ops if op[0] == "ch")


def _ncand(ops):
    return next(op[2].cands.shape[0] for op in ops if op[0] == "ch")


@pytest.mark.parametrize("ch", ["deph_sweep", "ad_phase"])
@pytest.mark.parametrize("path", ["default", "fast", "opt2"])
def test_constant_table_reproduces_fixed_bit_for_bit(ch, path):
    if path == "opt2" and not GPU:
        pytest.skip("opt2 needs cupy")
    n = 6
    ops = K.build("brickwork", "s7", n, 3, 0.1, ch)
    kw = {"default": {}, "fast": {"fast": True}, "opt2": {"fast": True, "opt2": True}}[path]
    for c in range(_ncand(ops)):
        a = K.run(ops, n, OBS, 300, ("fixed", c), seed=11 + c, chunk=128, **kw)
        b = K.run(ops, n, OBS, 300, ("table", np.full(_nev(ops), c)), seed=11 + c, chunk=128, **kw)
        for o in OBS:
            assert np.array_equal(a[o], b[o]), (ch, path, c, o)


@pytest.mark.parametrize("ch,fam,var", CASES)
def test_static_tables_unbiased(ch, fam, var):
    """ideal / pilot tables (M = 100 truncated O~, as in E10) and a random table: unbiased for EVERY observable
    (exact back-propagation reference), and the exact reference agrees with the density matrix."""
    n, shots = 5, 20000
    ops = K.build(fam, var, n, 3, 0.1, ch)
    ex = {o: K.exact_mean(ops, n, o) for o in OBS}
    for o in ("x", "zc"):
        assert abs(ex[o] - K.dm_mean(ops, n, o)) < 1e-10
    rec, _, _ = S.record_after(ops, n, "xc", 100)
    t_id, _ = ST.build_table(ops, n, rec, "ideal")
    t_pi, _ = ST.build_table(ops, n, rec, "pilot", shots=200, seed=5, chunk=100)
    t_rn = np.random.default_rng(3).integers(0, _ncand(ops), _nev(ops))
    for j, tab in enumerate((t_id, t_pi, t_rn)):
        r = K.run(ops, n, OBS, shots, ("table", tab), seed=500 + j, chunk=5000, fast=True)
        for o in OBS:
            x = r[o]
            v = x.var(ddof=1)
            if v < 1e-14:
                assert abs(x.mean() - ex[o]) < 1e-8
                continue
            z = (x.mean() - ex[o]) / np.sqrt(v / shots)
            assert abs(z) < 4.5, (ch, fam, var, j, o, z)


@pytest.mark.parametrize("ch,fam,var", CASES[:3])
def test_pilot_variance_decomposition(ch, fam, var):
    """With the EXACT O~ (M = 4^n), sum_e E_std[V_0] over std pilot trajectories = Var of the std estimator
    (martingale decomposition): checks that the table criterion measures the variance it is meant to minimise."""
    n = 5
    ops = K.build(fam, var, n, 4, 0.1, ch)
    rec, _, _ = S.record_after(ops, n, "x", 4 ** n)
    _, Vm = ST.build_table(ops, n, rec, "pilot", shots=6000, seed=21, chunk=3000)
    r = K.run(ops, n, ["x"], 20000, ("fixed", 0), seed=22, chunk=5000, fast=True)
    v = r["x"].var(ddof=1)
    assert abs(Vm[:, 0].sum() / v - 1) < 0.1, (Vm[:, 0].sum(), v)


@pytest.mark.skipif(not GPU, reason="needs cupy")
def test_build_table_opt2_matches_reference_path():
    n = 8
    ops = K.build("brickwork", "s7", n, 4, 0.03, "ad_phase")
    rec, _, _ = S.record_after(ops, n, "zz", 100)
    try:
        S.IMPL = "v2"
        a_t, a_V = ST.build_table(ops, n, rec, "pilot", shots=74, seed=9, chunk=37, opt2=True)
        S.IMPL = "v1"
        b_t, b_V = ST.build_table(ops, n, rec, "pilot", shots=74, seed=9, chunk=37, opt2=False)
    finally:
        S.IMPL = "v1"
    assert np.allclose(a_V, b_V, rtol=1e-3, atol=1e-9)
    assert np.mean(a_t == b_t) >= 0.98


def test_tie_rule():
    Vm = np.array([[1.0, 1.0, 2.0], [1.0, 0.99999, 2.0], [1.0, 0.5, 0.4], [0.0, 0.0, 0.0]])
    assert list(ST.tie_argmin(Vm)) == [0, 0, 2, 0]
