"""Validation for src/noise/traj_kraus.py (E2 pre-registration checks 1-3)."""
import numpy as np
import pytest

from src.noise import traj_kraus as K
from src.noise import traj_ref as T
from src.noise.traj_ref import GPU, I2 as I2_, XM, xp


def _np(a):
    return a.get() if GPU else a


@pytest.mark.parametrize("ch", ["depol", "ad", "deph", "deph_sweep", "ad_phase"])
@pytest.mark.parametrize("gt", [0.01, 0.3])
def test_candidates_same_channel(ch, gt):
    chan = K.CHANNELS[ch](gt)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    outs = chan.superop_check(rho)
    for o in outs[1:]:
        assert np.allclose(o, outs[0], atol=1e-13)


CASES = [("brickwork", "s7", 2), ("brickwork", "s8", 3), ("ising", "h0.7", 2), ("ising", "h1.4", 3)]


@pytest.mark.parametrize("ch", ["depol", "ad"])
@pytest.mark.parametrize("fam,var,L", CASES)
def test_exact_mean_vs_density_matrix(ch, fam, var, L):
    n = 4
    ops = K.build(fam, var, n, L, 0.3, ch)
    for o in ("x", "zz", "xc", "zc"):
        assert abs(K.exact_mean(ops, n, o) - K.dm_mean(ops, n, o)) < 1e-10


@pytest.mark.parametrize("fam,var,L", CASES)
def test_dephasing_reproduces_traj_ref_increments(fam, var, L):
    n, gt = 5, 0.1
    ops_dp = T.brickwork_ops(n, L, gt, seed=int(var[1:])) if fam == "brickwork" else \
        T.ising_ops(n, L, gt, h=float(var[1:]))
    ops = K.with_channel(ops_dp, K.dephasing(gt))
    O = T.obs_matrix("xc", n)
    after_k, O0k = K.backprop_local(ops, n, O)
    after_r, O0r = T.backprop_observables_local(ops_dp, n, O)
    assert float(xp.max(xp.abs(O0k - O0r))) < 1e-10
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(64, 2 ** n)) + 1j * rng.normal(size=(64, 2 ** n))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    psi = xp.asarray(psi)
    cx = xp.asarray(K.dephasing(gt).cands)
    p = T.dephasing_p(gt)
    for ev in (0, len(after_k) // 2, len(after_k) - 1):
        i = [op[1] for op in ops if op[0] == "ch"][ev]
        Ot = xp.asarray(after_k[ev])
        N, R = K._NR(psi, Ot, i, n)
        V, m, _ = K.increment_vars(N, R, cx)
        A, C, VB = T.local_AV(psi, Ot, i, n)
        vs = 4 * p * (1 - p) * A ** 2
        vp = 2 * p * (1 - 2 * p) * A ** 2 + 2 * p * VB
        assert float(xp.max(xp.abs(V[:, 0] - vs))) < 1e-6
        assert float(xp.max(xp.abs(V[:, 1] - vp))) < 1e-6
        # common mean = <psi| E^dag(O~) |psi> = (1-p)<O~> + p<Z O~ Z> = C + (1-2p) A
        assert float(xp.max(xp.abs(m - (C + (1 - 2 * p) * A)))) < 1e-5


@pytest.mark.parametrize("ch", ["depol", "ad", "deph_sweep", "ad_phase"])
def test_unbiased(ch):
    n, shots = 4, 20000
    ops = K.build("brickwork", "s7", n, 3, 0.1, ch)
    obs = ["x", "zz", "xc", "zc"]
    ex = {o: K.exact_mean(ops, n, o) for o in obs}
    C = ops[[op[0] for op in ops].index("ch")][2].cands.shape[0]
    schemes = [("fixed", c) for c in range(C)] + [("greedy", None)]
    for s, sc in enumerate(schemes):
        r = K.run(ops, n, obs, shots, sc, seed=100 + s, obs_for_policy="xc")
        for o in obs:
            x = r[o]
            v = x.var(ddof=1)
            if v < 1e-14:
                assert abs(x.mean() - ex[o]) < 1e-8
                continue
            z = (x.mean() - ex[o]) / np.sqrt(v / shots)
            assert abs(z) < 4.5, (ch, sc, o, z)


@pytest.mark.parametrize("gt", [0.01, 0.1, 0.3])
def test_tjm_analog_equals_dephasing(gt):
    a = K.dephasing_tjm(gt)
    d = K.dephasing(gt)
    rng = np.random.default_rng(5)
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    ref = d.superop_check(rho)[0]
    for o in a.superop_check(rho):
        assert np.allclose(o, ref, atol=1e-12)


def test_restricted_greedy_matches_dephasing_greedy():
    n = 5
    ops_dp = T.brickwork_ops(n, 3, 0.1, seed=7)
    a = K.run(K.with_channel(ops_dp, K.dephasing_tjm(0.1)), n, ["xc"], 3000, ("greedy", None),
              seed=9, obs_for_policy="xc", greedy_cands=(0, 1))
    b = K.run(K.with_channel(ops_dp, K.dephasing(0.1)), n, ["xc"], 3000, ("greedy", None),
              seed=9, obs_for_policy="xc")
    assert np.allclose(a["xc"], b["xc"], atol=1e-10)


def test_pauli_round_trip_and_known_coefficients():
    n = 3
    rng = np.random.default_rng(11)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    H = xp.asarray(A + A.conj().T)
    assert float(xp.max(xp.abs(K.from_pauli(K.to_pauli(H, n), n) - H))) < 1e-12
    # traceless parts and hermiticity: every coefficient of a Hermitian operator is real
    assert float(xp.max(xp.abs(xp.imag(K.to_pauli(H, n))))) < 1e-12
    # X on qubit 0 has a single coefficient 1 in slot (q0 row=0, q0 col=1)
    Xq = xp.asarray(np.kron(np.kron(XM, I2_), I2_))
    c = K.to_pauli(Xq, n)
    assert abs(float(xp.sum(xp.abs(c))) - 1.0) < 1e-12


def test_truncation_large_m_is_exact_and_small_m_counts():
    n = 4
    ops = K.build("brickwork", "s7", n, 3, 0.1, "ad")
    O = T.obs_matrix("xc", n)
    a_ex, O0_ex = K.backprop_local(ops, n, O, host=False)
    a_tr, O0_tr = K.backprop_truncated(ops, n, O, 4 ** n, host=False)
    assert float(xp.max(xp.abs(O0_ex - O0_tr))) < 1e-10
    for x, y in zip(a_ex, a_tr):
        assert float(xp.max(xp.abs(x - y))) < 1e-5
    t = K.truncate_pauli(xp.asarray(T.obs_matrix("x", n), dtype=complex) + 0.3 * xp.eye(2 ** n), n, 2)
    c = K.to_pauli(t, n)
    assert int(xp.sum(xp.abs(c) > 1e-12)) == 3      # identity + 2


@pytest.mark.parametrize("M", [1, 10])
@pytest.mark.parametrize("ch", ["deph_tjm", "ad"])
def test_truncated_greedy_unbiased(ch, M):
    n, shots = 4, 20000
    ops = K.build("ising", "h0.7", n, 3, 0.1, ch)
    for o in ("x", "zz", "xc"):
        after, _ = K.backprop_truncated(ops, n, T.obs_matrix(o, n), M)
        r = K.run(ops, n, [o], shots, ("greedy", None), seed=400 + M, obs_for_policy=o, after=after)
        z = (r[o].mean() - K.exact_mean(ops, n, o)) / np.sqrt(r[o].var(ddof=1) / shots)
        assert abs(z) < 4.5, (ch, M, o, z)


def test_tjm_and_myopic_unbiased():
    n, shots = 4, 20000
    ops = K.build("ising", "h0.7", n, 3, 0.1, "deph_tjm")
    obs = ["x", "zz", "xc"]
    ex = {o: K.exact_mean(ops, n, o) for o in obs}
    runs = [("fixed", 2), ("fixed", 3)]
    for s, sc in enumerate(runs):
        r = K.run(ops, n, obs, shots, sc, seed=200 + s)
        for o in obs:
            z = (r[o].mean() - ex[o]) / np.sqrt(r[o].var(ddof=1) / shots)
            assert abs(z) < 4.5, (sc, o, z)
    for o in obs:
        r = K.run(ops, n, [o], shots, ("greedy", None), seed=300, obs_for_policy=o,
                  greedy_cands=(0, 1), myopic=True)
        z = (r[o].mean() - ex[o]) / np.sqrt(r[o].var(ddof=1) / shots)
        assert abs(z) < 4.5, ("myopic", o, z)
