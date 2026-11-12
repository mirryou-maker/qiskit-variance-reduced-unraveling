import sys; sys.path.insert(0,'/mnt/d/Claude-Code-R/Qiskit-CO/src'); sys.path.insert(0,'/mnt/d/Claude-Code-R/Qiskit-CO/validation')
import numpy as np, cupy as cp
import test_a1_tracedistance as t
from qiskit.quantum_info import Statevector
ops=t.build_ops(6); s=t.GPUTrajectorySim(6,"cupy")
for name,qs,ps in ops:
    if name=="cx": s.apply_cx(*qs)
    elif name=="h": s.apply_1q(s.g["H"],qs[0])
    else: s.apply_1q(t.ry(s.xp,ps[0]),qs[0])
f=abs(np.vdot(np.asarray(Statevector(t.qiskit_circuit(6,ops))),cp.asnumpy(s.statevector())))**2
print("1-F =",1-f)
