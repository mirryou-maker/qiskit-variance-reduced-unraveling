import numpy as np
from qiskit_aer.noise import kraus_error

# analog unraveling: two-point unitary rotation e^{+-i theta Z}, each with weight 1/2
theta = 0.3
Z = np.diag([1, -1]).astype(complex)
Up = np.cos(theta)*np.eye(2) + 1j*np.sin(theta)*Z
Um = np.cos(theta)*np.eye(2) - 1j*np.sin(theta)*Z
w = 0.5
kraus = [np.sqrt(w)*Up, np.sqrt(w)*Um]
e = kraus_error(kraus)
d = e.to_dict()
print("=== analog (unitary two-point) Kraus error, as Aer stores it ===")
for i, c in enumerate(d["instructions"]):
    print("  component", i, ":", [op["name"] for op in c])
print("  n_components:", len(d["instructions"]), " probs:", [round(x,4) for x in d["probabilities"]])
