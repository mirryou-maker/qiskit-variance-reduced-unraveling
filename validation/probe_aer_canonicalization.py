"""Probe: does Aer preserve a custom Kraus unraveling, or canonicalize the channel?"""
import numpy as np
from qiskit_aer.noise import kraus_error, amplitude_damping_error

p = 0.4751
I2 = np.eye(2, dtype=complex)
Z2 = np.diag([1, -1]).astype(complex)
PIp = np.diag([1, 0]).astype(complex)
PIm = np.diag([0, 1]).astype(complex)
prj = [np.sqrt(1 - 2 * p) * I2, np.sqrt(2 * p) * PIp, np.sqrt(2 * p) * PIm]

e = kraus_error(prj)
d = e.to_dict()
print("=== projector Kraus error, as Aer stores it ===")
for i, c in enumerate(d["instructions"]):
    print("  component", i, ":", [op["name"] for op in c])
print("  n_components:", len(d["instructions"]),
      " probs:", [round(x, 4) for x in d["probabilities"]])

ad = amplitude_damping_error(0.6)
da = ad.to_dict()
print("\n=== amplitude damping (non-unital) for contrast ===")
for i, c in enumerate(da["instructions"]):
    print("  component", i, ":", [op["name"] for op in c])
print("  probs:", [round(x, 4) for x in da["probabilities"]])
