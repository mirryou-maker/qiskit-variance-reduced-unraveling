"""Tests for the martingale control variate (E13, src/noise/cv_martingale.py).

1. run_cv reproduces traj_kraus.run trajectory by trajectory (same seed -> same draws) for fixed candidates and
   static tables, on the fast path (CPU/GPU) and the opt2 path (GPU).
2. With the EXACT O~ (M >= 4^n, untruncated) and beta = 1, f - S equals the exact mean for every trajectory.
3. With the M = 100 truncated O~ (as in E13): S has mean zero and every CV estimate (beta1, crossfit, insample) is
   unbiased vs the exact mean, for all 4 observables' targets, fixed candidates and pilot tables, both channels.
4. Table + CV consistency: a constant table under run_cv equals the fixed scheme under run_cv bit for bit.
5. Cross-fitting uses the other fold's coefficient (unit check of cv_estimates).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noise import cv_martingale as CV  # noqa: E402
from src.noise import sparse_policy as S  # noqa: E402
from src.noise import static_table as ST  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise.traj_ref import GPU  # noqa: E402

OBS = ["x", "zz", "xc", "zc"]
CASES = [("deph_sweep", "brickwork", "s7"), ("ad_phase", "brickwork", "s8"), ("ad_phase", "ising", "h1.4"),
         ("deph_sweep", "ising", "h0.7"), ("ad_phase", "brickwork", "s11")]


def _nev(ops):
    return sum(1 for op in ops if op[0] == "ch")


def _ncand(ops):
    return next(op[2].cands.shape[0] for op in ops if op[0] == "ch")


@pytest.fixture(autouse=True)
def _impl():
    old = S.IMPL
    S.IMPL = "v2" if GPU else "v1"
    yield
    S.IMPL = old


@pytest.mark.parametrize("ch", ["deph_sweep", "ad_phase"])
@pytest.mark.parametrize("path", ["fast", "opt2"])
def test_plain_estimates_reproduce_traj_kraus(ch, path):
    if path == "opt2" and not GPU:
        pytest.skip("opt2 needs cupy")
    n = 6
    ops = K.build("brickwork", "s7", n, 3, 0.1, ch)
    rec, _, _ = S.record_after(ops, n, "zz", 100)
    opt2 = path == "opt2"
    tabs = [("fixed", c) for c in range(_ncand(ops))]
    tabs.append(("table", np.random.default_rng(1).integers(0, _ncand(ops), _nev(ops))))
    for j, sc in enumerate(tabs):
        a = K.run(ops, n, OBS, 200, sc, seed=31 + j, chunk=64, fast=True, opt2=opt2)
        b = CV.run_cv(ops, n, OBS, 200, rec, sc, seed=31 + j, chunk=64, opt2=opt2)
        for o in OBS:
            assert np.allclose(a[o], b[o], rtol=0, atol=1e-12), (ch, path, j, o, np.abs(a[o] - b[o]).max())


@pytest.mark.parametrize("ch,fam,var", CASES)
def test_exact_otilde_gives_zero_variance(ch, fam, var):
    n = 5
    ops = K.build(fam, var, n, 3, 0.1, ch)
    nrand = 0
    for o in ("x", "zc"):
        ex = K.exact_mean(ops, n, o)
        rec, v0, _ = S.record_after(ops, n, o, 4 ** n)
        assert abs(v0 - ex) < 1e-10
        for sc in [("fixed", 0), ("fixed", _ncand(ops) - 1),
                   ("table", np.random.default_rng(2).integers(0, _ncand(ops), _nev(ops)))]:
            r = CV.run_cv(ops, n, [o], 300, rec, sc, seed=7, chunk=150)
            y = r[o] - r["_S"]
            # NR_sparse forms R in complex64 (GPU and CPU) -> ~1e-6 per event
            tol = 2e-4
            assert np.abs(y - ex).max() < tol, (ch, fam, var, o, sc[0], np.abs(y - ex).max())
            nrand += r[o].var() > 1e-6        # count cases where the plain estimator is genuinely random
    assert nrand >= 3, nrand


@pytest.mark.parametrize("ch,fam,var", CASES)
def test_cv_unbiased_truncated(ch, fam, var):
    n, shots = 5, 20000
    ops = K.build(fam, var, n, 3, 0.1, ch)
    ex = {o: K.exact_mean(ops, n, o) for o in OBS}
    for j, o in enumerate(OBS):
        rec, _, _ = S.record_after(ops, n, o, 100)          # truncated (4^5 = 1024 > 100)
        tab, _ = ST.build_table(ops, n, rec, "pilot", shots=74, seed=40 + j, chunk=37)
        for sc in [("fixed", 0), ("fixed", 1), ("table", tab)]:
            r = CV.run_cv(ops, n, OBS, shots, rec, sc, seed=900 + j, chunk=5000)
            Sx = r["_S"]
            assert abs(Sx.mean()) < 4.5 * Sx.std(ddof=1) / np.sqrt(shots) + 1e-12, (o, sc[0])
            Y, _ = CV.cv_estimates(r[o], Sx)
            for lab, y in Y.items():
                v = y.var(ddof=1)
                if v < 1e-14:
                    assert abs(y.mean() - ex[o]) < 1e-6
                    continue
                z = (y.mean() - ex[o]) / np.sqrt(v / shots)
                assert abs(z) < 4.5, (ch, fam, var, o, sc[0], lab, z)
            for q in OBS:                                     # plain estimates of every observable still unbiased
                v = r[q].var(ddof=1)
                if v > 1e-14:
                    assert abs((r[q].mean() - ex[q]) / np.sqrt(v / shots)) < 4.5


@pytest.mark.parametrize("ch", ["deph_sweep", "ad_phase"])
def test_constant_table_equals_fixed_under_cv(ch):
    n = 6
    ops = K.build("brickwork", "s8", n, 3, 0.03, ch)
    rec, _, _ = S.record_after(ops, n, "x", 100)
    for c in range(_ncand(ops)):
        a = CV.run_cv(ops, n, OBS, 150, rec, ("fixed", c), seed=5 + c, chunk=50)
        b = CV.run_cv(ops, n, OBS, 150, rec, ("table", np.full(_nev(ops), c)), seed=5 + c, chunk=50)
        for o in OBS + ["_S"]:
            assert np.array_equal(a[o], b[o]), (ch, c, o)


def test_crossfit_uses_other_fold():
    rng = np.random.default_rng(0)
    S_ = rng.normal(size=1000)
    f = 2.0 * S_ + rng.normal(size=1000) * 0.1 + 3.0
    Y, b = CV.cv_estimates(f, S_)
    ev = np.arange(1000) % 2 == 0
    assert np.allclose(Y["crossfit"][ev], f[ev] - b["beta_odd"] * S_[ev])
    assert np.allclose(Y["crossfit"][~ev], f[~ev] - b["beta_even"] * S_[~ev])
    assert np.allclose(Y["beta1"], f - S_)
    assert abs(b["beta_in"] - 2.0) < 0.05 and Y["crossfit"].var() < 0.02


def test_cv_reduces_variance_with_good_otilde():
    """Sanity (not a claim): with the exact O~ the CV variance is ~0 while the plain variance is not; with M = 100
    at n = 5 (about 10 % of the 1024 strings) the crossfit CV must not be worse than the plain estimator by > 10 %."""
    n = 5
    ops = K.build("brickwork", "s7", n, 3, 0.1, "ad_phase")
    rec, _, _ = S.record_after(ops, n, "zz", 100)
    r = CV.run_cv(ops, n, ["zz"], 4000, rec, ("fixed", 0), seed=3, chunk=2000)
    Y, _ = CV.cv_estimates(r["zz"], r["_S"])
    assert Y["crossfit"].var() <= 1.1 * r["zz"].var()
