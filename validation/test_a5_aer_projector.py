"""Correctness of the public-API projector unraveling in Qiskit-Aer (CPU, small n).

* product state: per-shot mean and variance match the closed forms for both methods;
* entangling circuits (brickwork, Trotter TFIM): both methods are unbiased against
  the density_matrix reference (|mean - exact| < 4 SE).

Run:  pytest -q validation/test_a5_aer_projector.py
"""
import os
import sys

import numpy as np
import pytest
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(__file__))
import aer_unravel_circuits as auc  # noqa: E402

SHOTS = 40000


def _run(qc, seed=11):
    r = AerSimulator(method="statevector", device="CPU").run(qc, shots=SHOTS, seed_simulator=seed).result()
    assert r.success, r.status
    return np.asarray(r.data(0)["ev"], float)


@pytest.mark.parametrize("gt", [0.1, 0.5, 1.5])
@pytest.mark.parametrize("method", auc.METHODS)
def test_product_closed_form(method, gt):
    n = 4
    v = _run(auc.product_circuit(n, gt, method))
    ex = auc.product_exact(n, gt)
    var_ex = ex["var_" + method]
    assert abs(v.mean() - ex["mean"]) < 4.5 * np.sqrt(var_ex / SHOTS)
    # relative SE of a sample variance ~ sqrt(2/N) for near-Gaussian, larger for Bernoulli;
    # 10 % is > 5 sigma at N = 40000 for these parameters
    assert abs(v.var(ddof=1) / var_ex - 1) < 0.10


@pytest.mark.parametrize("family", ["brickwork", "ising"])
@pytest.mark.parametrize("obs", ["x", "zz"])
@pytest.mark.parametrize("method", auc.METHODS)
def test_entangled_unbiased(family, obs, method):
    n, layers, gt = 5, 4, 0.2
    qc, _ = auc.FAMILIES[family](n, layers, gt, method, seed=7, obs=obs)
    v = _run(qc)
    exact = auc.family_exact(family, n, layers, gt, seed=7, obs=obs)
    se = v.std(ddof=1) / np.sqrt(SHOTS)
    assert abs(v.mean() - exact) < 4 * se + 1e-12


def test_projector_kraus_is_canonicalised():
    """A projector Kraus mixture appended as a QuantumError gives the *standard* variance."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import IGate
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import QuantumError

    n, gt = 4, 1.5
    p = auc.dephasing_p(gt)
    P0 = np.diag([1.0, 0.0]).astype(complex)
    P1 = np.diag([0.0, 1.0]).astype(complex)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n):
        qc.append(QuantumError([(IGate(), 1 - 2 * p), (Kraus([P0, P1]), 2 * p)]), [i])
    qc.save_expectation_value(auc.mean_x_obs(n), list(range(n)), pershot=True, label="ev")
    v = _run(qc)
    ex = auc.product_exact(n, gt)
    assert abs(v.var(ddof=1) / ex["var_std"] - 1) < 0.10
