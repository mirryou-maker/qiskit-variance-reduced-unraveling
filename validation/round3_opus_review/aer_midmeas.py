# Projector unraveling via Aer PUBLIC API: ancilla coin (prob 2p) + conditional mid-circuit Z-measurement
import numpy as np, time
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
GT=1.5; P=(1-np.exp(-2*GT))/2; n=4
q=QuantumRegister(n,"q"); a=QuantumRegister(1,"a")
coin=ClassicalRegister(n,"coin"); junk=ClassicalRegister(n,"junk")
qc=QuantumCircuit(q,a,coin,junk)
for i in range(n): qc.h(q[i])
ang=2*np.arcsin(np.sqrt(2*P))
for i in range(n):
    qc.reset(a[0]); qc.ry(ang,a[0]); qc.measure(a[0],coin[i])
    with qc.if_test((coin[i],1)):
        qc.measure(q[i],junk[i])          # Z-basis collapse with Born rule
obs=SparsePauliOp.from_sparse_list([("X",[i],1.0/n) for i in range(n)],num_qubits=n+1)
qc.save_expectation_value(obs, list(range(n+1)), pershot=True, label="ev")
for dev in ["CPU","GPU"]:
    sim=AerSimulator(method="statevector",device=dev)
    r=sim.run(qc,shots=40000,seed_simulator=1).result()
    v=np.array(r.data(0)["ev"],float)
    print(dev,"mean",v.mean().round(4),"exact",np.exp(-2*GT).round(4),"var",v.var(ddof=1).round(5),
          "exact_proj",(np.exp(-3)*(1-np.exp(-3))/n).round(5),"exact_std",((1-np.exp(-6))/n).round(5))
