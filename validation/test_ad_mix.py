"""Rev6 C: validation of the partially-mixed amplitude-damping unravelings (traj_kraus.ad_mix_set / amp_damp_mix).

1. Kraus completeness sum K^dag K = I for every candidate and theta (incl. a dense theta / phi grid).
2. Every candidate realises the AD channel (superoperator equal to photon counting on random rho).
3. Endpoints: theta = 0 is photon counting up to a sign; theta = pi/4 equals the ad_phase homodyne set at phi.
4. Indices 0-4 of ad_mix equal ad_phase exactly, and a fixed run of candidate c < 5 under ad_mix reproduces the
   ad_phase run trajectory by trajectory (same seed).
5. Unbiasedness: trajectory means of every new candidate (indices 5-10) agree with the exact density-matrix mean (n = 4, 5, 6;
   2 circuits; 4 observables) within |z| < 4.5.
"""
import numpy as np
import pytest

from src.noise import traj_kraus as K
from src.noise.traj_ref import I2


def _rand_rho(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    r = A @ A.conj().T
    return r / np.trace(r)


@pytest.mark.parametrize("gt", [0.01, 0.3, 2.0])
def test_completeness_and_channel(gt):
    for th in np.linspace(0, np.pi / 2, 17):
        for phi in np.linspace(0, 2 * np.pi, 9):
            S = K.ad_mix_set(gt, th, phi)
            assert np.allclose(sum(k.conj().T @ k for k in S), I2, atol=1e-13)
            ref = K.ad_mix_set(gt, 0.0, 0.0)
            for s in range(3):
                rho = _rand_rho(s)
                a = sum(k @ rho @ k.conj().T for k in S)
                b = sum(k @ rho @ k.conj().T for k in ref)
                assert np.allclose(a, b, atol=1e-13)
    ch = K.amp_damp_mix(gt)                      # Channel.__post_init__ asserts completeness for all 11 candidates
    assert ch.cands.shape[0] == 11
    outs = ch.superop_check(_rand_rho(7))
    for o in outs[1:]:
        assert np.allclose(o, outs[0], atol=1e-13)


@pytest.mark.parametrize("gt", [0.01, 0.3])
def test_endpoints_and_prefix(gt):
    ph = K.amp_damp_phase(gt)
    mx = K.amp_damp_mix(gt)
    assert np.array_equal(mx.cands[:5], ph.cands)
    assert mx.labels[:5] == ph.labels
    S0 = K.ad_mix_set(gt, 0.0, 0.3)
    # theta = 0: {K0, -e^{i phi} K1} = photon counting up to a global phase of the jump operator
    assert np.allclose(S0[0], ph.cands[0][0]) and np.allclose(S0[1], -np.exp(0.3j) * ph.cands[0][1])
    for j, phi in enumerate((0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
        S = K.ad_mix_set(gt, np.pi / 4, phi)
        assert np.allclose(S[0], ph.cands[1 + j][0]) and np.allclose(S[1], ph.cands[1 + j][1])
    # the new candidates are genuinely different from both endpoints
    for c in (5, 6, 7):
        assert not np.allclose(np.abs(mx.cands[c]), np.abs(mx.cands[0]))
        assert not np.allclose(mx.cands[c], mx.cands[2])


def test_prefix_runs_identical():
    n = 5
    a = K.build("brickwork", "s7", n, 3, 0.1, "ad_phase")
    b = K.build("brickwork", "s7", n, 3, 0.1, "ad_mix")
    obs = ["x", "zz", "xc", "zc"]
    for c in range(5):
        ra = K.run(a, n, obs, 64, ("fixed", c), seed=11 + c, chunk=32)
        rb = K.run(b, n, obs, 64, ("fixed", c), seed=11 + c, chunk=32)
        for o in obs:
            assert np.allclose(ra[o], rb[o], atol=1e-12)


CASES = [(4, "s7", 3), (5, "s8", 3), (6, "s7", 2)]


@pytest.mark.parametrize("n,var,L", CASES)
def test_unbiased_vs_density_matrix(n, var, L):
    gt, shots = 0.2, 20000
    ops = K.build("brickwork", var, n, L, gt, "ad_mix")
    obs = ["x", "zz", "xc", "zc"]
    ex = {o: K.dm_mean(ops, n, o) for o in obs}
    zs = []
    for c in range(5, 11):
        r = K.run(ops, n, obs, shots, ("fixed", c), seed=500 + 10 * n + c, chunk=4096)
        for o in obs:
            x = np.asarray(r[o], float)
            se = x.std(ddof=1) / np.sqrt(len(x))
            if se < 1e-12:
                assert abs(x.mean() - ex[o]) < 1e-9
                continue
            z = (x.mean() - ex[o]) / se
            zs.append(z)
            assert abs(z) < 4.5, (n, var, c, o, x.mean(), ex[o], z)
    assert len(zs) >= 12


@pytest.mark.parametrize("gt", [0.01, 0.3])
def test_angles(gt):
    mx = K.amp_damp_mix(gt)
    g = 1.0 - np.exp(-gt)
    th = list(K.AD_MIX_THETAS) + [np.arctan(x * np.sqrt(g)) for x in K.AD_MIX_X2]
    for j, t in enumerate(th):
        S = K.ad_mix_set(gt, t, K.AD_MIX_PHI)
        assert np.allclose(mx.cands[5 + j][0], S[0]) and np.allclose(mx.cands[5 + j][1], S[1])
