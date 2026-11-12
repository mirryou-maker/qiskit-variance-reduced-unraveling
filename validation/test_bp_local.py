"""Local back-propagation (traj_ref.backprop_observables_local) must equal the dense routine.

Also: greedy trajectories with local_bp=True reproduce local_bp=False with the same seed
(decisions agree up to complex64 rounding, so we compare the estimator means/variances).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noise import traj_ref as T  # noqa: E402


def _np(a):
    return T.xp.asnumpy(a) if T.GPU else np.asarray(a)


CASES = [("brickwork", 6, 3, 0.05), ("brickwork", 8, 2, 0.1), ("ising", 6, 3, 0.1), ("ising", 7, 2, 0.03)]


def _ops(fam, n, L, gt):
    return T.brickwork_ops(n, L, gt) if fam == "brickwork" else T.ising_ops(n, L, gt)


@pytest.mark.parametrize("fam,n,L,gt", CASES)
@pytest.mark.parametrize("obs", ["x", "zz", "xc", "zc"])
def test_local_equals_dense(fam, n, L, gt, obs):
    ops = _ops(fam, n, L, gt)
    O = T.obs_matrix(obs, n)
    a_d, O0_d = T.backprop_observables(ops, n, O)
    a_l, O0_l = T.backprop_observables_local(ops, n, O)
    assert len(a_d) == len(a_l)
    assert np.max(np.abs(_np(O0_d) - _np(O0_l))) < 1e-10
    for x, y in zip(a_d, a_l):
        assert np.max(np.abs(_np(x) - np.asarray(_np(y)))) < 1e-5  # both stored complex64
    assert abs(T.exact_mean(ops, n, obs) - T.exact_mean(ops, n, obs, local_bp=True)) < 1e-10


@pytest.mark.parametrize("fam", ["brickwork", "ising"])
def test_greedy_local_same(fam):
    n, L, gt = 6, 3, 0.1
    ops = _ops(fam, n, L, gt)
    r1 = T.run(ops, n, ["xc"], 4000, ("greedy", None), seed=11, obs_for_policy="xc")
    r2 = T.run(ops, n, ["xc"], 4000, ("greedy", None), seed=11, obs_for_policy="xc", local_bp=True)
    assert np.max(np.abs(r1["xc"] - r2["xc"])) < 1e-9
    assert np.max(np.abs(r1["_frac_proj"] - r2["_frac_proj"])) < 1e-3
