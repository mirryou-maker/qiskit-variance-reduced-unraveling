import numpy as np, time, sys
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, kraus_error, pauli_error
GT=1.5; P=(1-np.exp(-2*GT))/2; n=int(sys.argv[1]) if len(sys.argv)>1 else 10
dev=sys.argv[2] if len(sys.argv)>2 else "CPU"
I2=np.eye(2,dtype=complex); Z2=np.diag([1,-1]).astype(complex)
Pp=np.diag([1,0]).astype(complex); Pm=np.diag([0,1]).astype(complex)
kr={"pauli":None,"std_kraus":[np.sqrt(1-P)*I2,np.sqrt(P)*Z2],
    "proj_kraus":[np.sqrt(1-2*P)*I2,np.sqrt(2*P)*Pp,np.sqrt(2*P)*Pm]}
obs=SparsePauliOp.from_sparse_list([("X",[q],1.0/n) for q in range(n)],num_qubits=n)
for name,k in kr.items():
  for extra in [{}, {"fusion_enable":False}]:
    qc=QuantumCircuit(n)
    for q in range(n): qc.h(q)
    for q in range(n): qc.id(q)
    qc.save_expectation_value(obs, range(n), pershot=True, label="ev")
    nm=NoiseModel()
    err=pauli_error([("Z",P),("I",1-P)]) if k is None else kraus_error(k)
    nm.add_all_qubit_quantum_error(err,["id"])
    kw=dict(method="statevector",device=dev,noise_model=nm,**extra)
    if dev=="GPU": kw["batched_shots_gpu"]=True
    sim=AerSimulator(**kw)
    shots=20000
    t0=time.time(); r=sim.run(qc,shots=shots,seed_simulator=20260719).result(); dt=time.time()-t0
    v=np.array(r.data(0)["ev"],dtype=float)
    print(f"{name:10s} {str(extra):24s} len={len(v)} mean={v.mean():.4f} var={v.var(ddof=1):.5f} N_to_0.01={v.var(ddof=1)/1e-4:.0f} uniq={len(np.unique(np.round(v,6)))} t={dt:.2f}s")
