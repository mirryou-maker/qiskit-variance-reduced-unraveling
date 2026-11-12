"""Tests for src/noise/pauli_prop.py (E9a sparse truncated Pauli propagation)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise import traj_ref as T  # noqa: E402

CIRCS = [("brickwork", "s7"), ("ising", "h0.7"), ("ising", "h1.4")]
OBS = ("x", "zz", "xc", "zc")


def test_ptm_properties():
    ops = K.build("brickwork", "s7", 4, 2, 0.1, "ad")
    for op in ops:
        Tm = P.transfer(op)
        if op[0] == "ch":
            assert np.allclose(Tm[0], np.eye(len(Tm))[0])           # unital in Heisenberg: E^dag(I) = I
        else:
            assert np.allclose(Tm @ Tm.T, np.eye(len(Tm)), atol=1e-12)  # unitary -> orthogonal PTM


def test_key_roundtrip():
    O = P.PauliOp.from_terms(T.obs_terms("zz", 5), 5)
    assert len(O) == 4
    O = P.PauliOp.from_terms([(1.0, {0: T.XM, 3: T.YM})], 5)
    k = int(O.keys[0])
    assert k == (1 << 0) | (1 << 3) | (1 << (32 + 3))


@pytest.mark.parametrize("ch", ["deph", "ad"])
@pytest.mark.parametrize("fam,var", CIRCS)
def test_untruncated_equals_exact(ch, fam, var):
    n, L = 6, 4
    for gt in (0.03, 0.3):
        ops = K.build(fam, var, n, L, gt, ch)
        for o in OBS:
            v, _ = P.propagate(ops, n, o, M=4 ** n)
            assert abs(v - K.exact_mean(ops, n, o)) < 1e-10, (ch, fam, var, gt, o)


@pytest.mark.parametrize("ch", ["deph", "ad", "deph_small"])
@pytest.mark.parametrize("fam,var", [("brickwork", "s7"), ("ising", "h1.4")])
def test_fast_kernels_match(ch, fam, var):
    n = 6
    ops = K.build(fam, var, n, 3, 0.1, ch)
    for scheme in (("fixed", 0), ("fixed", 1), ("greedy", None)):
        a = K.run(ops, n, list(OBS), 300, scheme, seed=5, chunk=128, obs_for_policy="xc")
        b = K.run(ops, n, list(OBS), 300, scheme, seed=5, chunk=128, obs_for_policy="xc", fast=True)
        for o in OBS:
            assert np.max(np.abs(a[o] - b[o])) < 1e-10, (ch, scheme, o)


@pytest.mark.parametrize("ch", ["deph", "ad"])
@pytest.mark.parametrize("fam,var", CIRCS)
def test_dm_fast_equals_exact(ch, fam, var):
    n = 6
    ops = K.build(fam, var, n, 4, 0.1, ch)
    got = K.dm_means_fast(ops, n, list(OBS))
    for o in OBS:
        assert abs(got[o] - K.exact_mean(ops, n, o)) < 1e-10, (ch, fam, o)


def test_truncation_bounds_and_converges():
    n = 8
    ops = K.build("brickwork", "s8", n, 6, 0.03, "deph")
    ex = K.exact_mean(ops, n, "x")
    errs = []
    for M in (10, 100, 1000, 4 ** n):
        v, peak = P.propagate(ops, n, "x", M=M)
        errs.append(abs(v - ex))
    assert errs[-1] < 1e-10
    # peak after truncation is bounded by (branching) x (M + 1); check the bound used in memory planning
    _, peak = P.propagate(ops, n, "x", M=100)
    assert peak <= 101
