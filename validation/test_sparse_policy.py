"""Tests for src/noise/sparse_policy.py (E9b): greedy statistics from a sparse Pauli-sum O~."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise.traj_ref import GPU, xp  # noqa: E402

CASES = [("deph_sweep", "brickwork", "s7"), ("ad_phase", "brickwork", "s8"), ("ad_phase", "ising", "h1.4")]


def _psi(B, n, seed):
    r = np.random.default_rng(seed)
    v = r.normal(size=(B, 2 ** n)) + 1j * r.normal(size=(B, 2 ** n))
    return xp.asarray(v / np.linalg.norm(v, axis=1, keepdims=True))


@pytest.mark.parametrize("ch,fam,var", CASES)
def test_R_matches_dense(ch, fam, var):
    n = 6
    ops = K.build(fam, var, n, 3, 0.1, ch)
    for o in ("x", "zz", "zc"):
        rec, _, _ = S.record_after(ops, n, o, M=4 ** n)               # untruncated
        dense, _ = K.backprop_local(ops, n, K.T.obs_matrix(o, n))
        idx = [k for k, op in enumerate(op for op in ops if op[0] == "ch")]
        chq = [op[1] for op in ops if op[0] == "ch"]
        psi = _psi(5, n, 1)
        for e in (0, len(idx) // 2, len(idx) - 1):
            N0, R0 = K._NR(psi, xp.asarray(dense[e]), chq[e], n)
            N1, R1 = S.NR_sparse(psi, rec[e], n, K._N_fast(psi, chq[e], n))
            scale = float(xp.max(xp.abs(R0)))
            assert float(xp.max(xp.abs(R0 - R1))) < 1e-5 * max(scale, 1.0), (ch, o, e)
            assert float(xp.max(xp.abs(N0 - N1))) < 1e-12


@pytest.mark.skipif(not GPU, reason="kernel test needs cupy")
def test_kernel_matches_numpy_reference():
    n = 7
    ops = K.build("brickwork", "s7", n, 3, 0.03, "ad_phase")
    rec, _, _ = S.record_after(ops, n, "x", M=50)
    psi = _psi(3, n, 2)
    for ev in rec[::7]:
        p = n - 1 - ev.i
        G = xp.empty((3, ev.S, 4), xp.complex64)
        S._kernel()((ev.S, 3), (256,), (psi.astype(xp.complex64), ev.xm, ev.zm, np.int32(2 ** n), np.int32(p),
                                        np.int32(2 ** (n - 1)), G))
        Gr = S._G_numpy(psi, ev, n)
        assert float(xp.max(xp.abs(G - Gr))) < 1e-5


@pytest.mark.skipif(not GPU, reason="kernel test needs cupy")
@pytest.mark.parametrize("n,tile", [(7, 32), (7, 1024), (10, 256), (12, 512), (12, 1024)])
def test_tiled_kernel_matches_numpy_reference(n, tile):
    """E9e v2 kernel: one short tile (n=7), several tile sizes, S up to ~100, with and without X grouping."""
    ops = K.build("brickwork", "s7", n, 4, 0.03, "ad_phase")
    f = S.G_tiled
    for o, M in (("x", 100), ("zz", 10)):
        rec, _, _ = S.record_after(ops, n, o, M=M)
        psi = _psi(3, n, 4)
        for ev in rec[::5]:
            for gmax in (1, 4):
                G = f(psi.astype(xp.complex64), ev, n, tile=tile, gmax=gmax)
                Gr = S._G_numpy(psi, ev, n)
                assert float(xp.max(xp.abs(G - Gr))) < 2e-5, (n, tile, o, gmax)


@pytest.mark.skipif(not GPU, reason="kernel test needs cupy")
@pytest.mark.parametrize("n", [5, 9, 12])
def test_N_fused_matches_N_fast(n):
    psi = _psi(4, n, 5)
    for i in range(n):
        assert float(xp.max(xp.abs(K._N_fused(psi, i, n) - K._N_fast(psi, i, n)))) < 1e-12


@pytest.mark.skipif(not GPU, reason="needs cupy")
@pytest.mark.parametrize("ch,fam,var", CASES)
def test_opt2_reproduces_e9b_path(ch, fam, var):
    """E9e path (v2 kernel + opt2) gives the same trajectories as the E9b path (same seed): same choices up to
    float32 near-ties, estimates equal to rounding."""
    n = 9
    ops = K.build(fam, var, n, 6, 0.03, ch)
    rec, _, _ = S.record_after(ops, n, "x", M=100)
    try:
        S.IMPL = "v1"
        a = K.run(ops, n, ["x", "zz"], 300, ("greedy", None), seed=7, chunk=100, after=rec, fast=True)
        f_a = K.run(ops, n, ["x"], 300, ("fixed", 1), seed=8, chunk=100, fast=True)
        S.IMPL = "v2"
        b = K.run(ops, n, ["x", "zz"], 300, ("greedy", None), seed=7, chunk=100, after=rec, fast=True, opt2=True)
        f_b = K.run(ops, n, ["x"], 300, ("fixed", 1), seed=8, chunk=100, fast=True, opt2=True)
    finally:
        S.IMPL = "v1"
    assert np.allclose(f_a["x"], f_b["x"], atol=1e-10)
    same = np.isclose(a["x"], b["x"], atol=1e-8)
    assert same.mean() >= 0.99, same.mean()
    assert np.allclose(a["_choice"], b["_choice"], atol=0.02)


@pytest.mark.parametrize("ch,fam,var", CASES)
def test_sparse_greedy_unbiased_and_close_to_dense(ch, fam, var):
    n = 6
    ops = K.build(fam, var, n, 3, 0.1, ch)
    rec, _, _ = S.record_after(ops, n, "x", M=4 ** n)
    ex = {o: K.exact_mean(ops, n, o) for o in ("x", "zz")}
    a = K.run(ops, n, ["x", "zz"], 4000, ("greedy", None), seed=3, chunk=1000, obs_for_policy="x", fast=True)
    b = K.run(ops, n, ["x", "zz"], 4000, ("greedy", None), seed=3, chunk=1000, after=rec, fast=True)
    for o in ("x", "zz"):
        z = (b[o].mean() - ex[o]) / np.sqrt(b[o].var() / len(b[o]) + 1e-300)
        assert abs(z) < 4.5, (ch, o, z)
    # same policy information (exact O~), float32 vs float32 dense: variances agree closely
    assert abs(b["x"].var() / a["x"].var() - 1) < 0.15
