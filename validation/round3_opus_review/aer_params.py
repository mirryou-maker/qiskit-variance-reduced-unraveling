import numpy as np
from qiskit_aer.noise import kraus_error
from qiskit_aer.noise.errors import QuantumError
P=(1-np.exp(-3))/2
I2=np.eye(2,dtype=complex); Pp=np.diag([1,0]).astype(complex); Pm=np.diag([0,1]).astype(complex)
e=kraus_error([np.sqrt(1-2*P)*I2,np.sqrt(2*P)*Pp,np.sqrt(2*P)*Pm])
d=e.to_dict()
for op in d["instructions"][0]:
    print(op["name"]); 
    for m in op.get("params",[]): print(np.round(np.array(m),4))
import inspect
print(inspect.signature(kraus_error))
src=inspect.getsource(QuantumError.__init__)
print(src[:3000])
