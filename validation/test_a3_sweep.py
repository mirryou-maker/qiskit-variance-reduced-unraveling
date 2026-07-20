"""A3 characterization sweep: variance-reduced unraveling across noise strength,
circuit class, and qubit count, on the GPU dense-statevector engine.

Model: per-qubit Z-dephasing (gamma=1), observable O = mean_i <X_i> ({X,Z}=0).
Unravelings: standard (Pauli flip), projector (collapse), analog (kicks).
Exact event sampling per idle window (no time discretization).

Circuit classes (probe how coherent dynamics erodes the projector absorbing window):
  C1 idle    : prep |+>^n, single dephasing window            (absorbing window intact)
  C2 ghz     : prep GHZ, single dephasing window              (entangled)
  C3 rotated : |+>^n, RX rotations interleaved between 3 windows (coherent mixing)

Outputs:
  A3a regime map  : variance vs gamma*t for the 3 unravelings (C1, n=12)
  A3b circuit map : N-to-target-SE per unraveling for {C1,C2,C3} (n=12, strong noise)
  A3c n-scaling   : N-to-target + wall-clock + memory vs n (C1, strong noise)
  A3d correctness : ensemble rho vs Aer density_matrix (n=8) -> unbiasedness of all unravelings

Run:  python /mnt/d/Claude-Code-R/Qiskit-CO/validation/test_a3_sweep.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trajectory.gpu_trajectory import GPUTrajectorySim  # noqa: E402

SEED = 20260719
THETA0 = 0.30
TARGET_SE = 0.01


def apply_noise_window(sim, q, gamma, tau, unravel, rng):
    if unravel == "standard":
        if rng.poisson(gamma * tau) % 2 == 1:
            sim.apply_1q(sim.g["Z"], q)
    elif unravel == "projector":
        if rng.random() < (1.0 - np.exp(-2.0 * gamma * tau)):
            plus = rng.random() < sim.prob_plus("Z", q)
            sim.apply_projector("Z", q, plus)
    elif unravel == "analog":
        lam = gamma / np.sin(THETA0) ** 2
        for _ in range(rng.poisson(lam * tau)):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            sim.apply_analog_kick("Z", q, sign * THETA0)


def build_stages(kind, n, T):
    if kind == "idle":
        return [("gates", [("h", q, 0.0) for q in range(n)]), ("noise", T)]
    if kind == "ghz":
        g = [("h", 0, 0.0)] + [("cx", i, i + 1) for i in range(n - 1)]
        return [("gates", g), ("noise", T)]
    if kind == "rotated":
        st = [("gates", [("h", q, 0.0) for q in range(n)]), ("noise", T / 3)]
        for _ in range(2):
            st += [("gates", [("rx", q, 0.6) for q in range(n)]), ("noise", T / 3)]
        return st
    raise ValueError(kind)


def run_traj(sim, stages, gamma, unravel, rng):
    n = sim.n
    sim.reset()
    for typ, payload in stages:
        if typ == "gates":
            for op in payload:
                if op[0] == "h":
                    sim.apply_1q(sim.g["H"], op[1])
                elif op[0] == "rx":
                    from trajectory.gpu_trajectory import rx
                    sim.apply_1q(rx(sim.xp, op[2]), op[1])
                elif op[0] == "cx":
                    sim.apply_cx(op[1], op[2])
        else:  # noise window of duration payload
            for q in range(n):
                apply_noise_window(sim, q, gamma, payload, unravel, rng)
    return sum(sim.expval_1q(sim.g["X"], q) for q in range(n)) / n


def collect(sim, stages, gamma, unravel, N, rng):
    vals = np.empty(N)
    for i in range(N):
        vals[i] = run_traj(sim, stages, gamma, unravel, rng)
    return vals


def n_to_target(vals):
    return (vals.std(ddof=1) / TARGET_SE) ** 2


def ratio(a, b):
    return a / b if b > 1e-9 else float("inf")


def a3a_regime_map():
    print("=== A3a regime map: variance vs gamma*t (C1 idle, n=12, N=2000) ===")
    n, N = 12, 2000
    sim = GPUTrajectorySim(n, backend="cupy")
    rng = np.random.default_rng(SEED)
    grid = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
    print(f"{'gt':>5} {'var_std':>9} {'var_proj':>9} {'var_analog':>11} "
          f"{'proj/std':>9} {'anlg/std':>9}")
    rows = []
    for gt in grid:
        v = {u: collect(sim, build_stages("idle", n, gt), 1.0, u, N, rng).var()
             for u in ("standard", "projector", "analog")}
        print(f"{gt:>5} {v['standard']:>9.4f} {v['projector']:>9.4f} "
              f"{v['analog']:>11.4f} {v['projector']/v['standard']:>9.3f} "
              f"{v['analog']/v['standard']:>9.3f}")
        rows.append((gt, v))
    return rows


def a3b_circuit_map():
    print("\n=== A3b circuit-class map: N-to-target-SE (n=12, gt=1.5, N=2000) ===")
    n, N, gt = 12, 2000, 1.5
    sim = GPUTrajectorySim(n, backend="cupy")
    rng = np.random.default_rng(SEED + 2)
    print(f"{'circuit':>9} {'N_std':>8} {'N_proj':>8} {'N_analog':>9} "
          f"{'speedup_proj':>13} {'speedup_analog':>15}")
    for kind in ("idle", "ghz", "rotated"):
        st = build_stages(kind, n, gt)
        Nn = {u: n_to_target(collect(sim, st, 1.0, u, N, rng))
              for u in ("standard", "projector", "analog")}
        print(f"{kind:>9} {Nn['standard']:>8.0f} {Nn['projector']:>8.0f} "
              f"{Nn['analog']:>9.0f} {ratio(Nn['standard'], Nn['projector']):>13.1f} "
              f"{ratio(Nn['standard'], Nn['analog']):>15.1f}")


def a3c_n_scaling():
    print("\n=== A3c n-scaling: N-to-target + wall-clock + memory (C1, gt=1.5) ===")
    import cupy as cp
    gt, N = 1.5, 600
    rng = np.random.default_rng(SEED + 3)
    print(f"{'n':>3} {'dim':>10} {'N_std':>8} {'N_proj':>8} {'speedup':>8} "
          f"{'t_proj_s':>9} {'gpu_mem_MB':>11}")
    for n in (8, 12, 16, 20):
        sim = GPUTrajectorySim(n, backend="cupy")
        st = build_stages("idle", n, gt)
        Ns = n_to_target(collect(sim, st, 1.0, "standard", N, rng))
        t0 = time.time()
        Np = n_to_target(collect(sim, st, 1.0, "projector", N, rng))
        tp = time.time() - t0
        mem = (cp.get_default_memory_pool().used_bytes()) / 1e6
        print(f"{n:>3} {2**n:>10} {Ns:>8.0f} {Np:>8.0f} {ratio(Ns, Np):>8.1f} "
              f"{tp:>9.1f} {mem:>11.1f}")
        del sim
        cp.get_default_memory_pool().free_all_blocks()


def a3d_correctness():
    print("\n=== A3d correctness vs Aer density_matrix (n=8, GHZ + dephasing) ===")
    import cupy as cp
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, pauli_error
    n, gt, N = 8, 1.0, 15000
    p = (1.0 - np.exp(-2.0 * gt)) / 2.0  # dephasing time gt -> Z-error prob
    # Aer reference: GHZ then Z-error prob p on each qubit (via identity gate with error)
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for q in range(n):
        qc.id(q)
    qc.save_density_matrix()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(pauli_error([("Z", p), ("I", 1 - p)]), ["id"])
    rho_ref = cp.asarray(np.asarray(DensityMatrix(
        AerSimulator(method="density_matrix", noise_model=nm)
        .run(qc, shots=1, seed_simulator=SEED).result().data(0)["density_matrix"]).data))
    st = build_stages("ghz", n, gt)
    for unravel in ("standard", "projector", "analog"):
        sim = GPUTrajectorySim(n, backend="cupy")
        rng = np.random.default_rng(SEED + 7)
        rho = cp.zeros((2 ** n, 2 ** n), dtype=cp.complex128)
        for _ in range(N):
            sim.reset()
            # run stages, accumulate final statevector
            run_traj(sim, st, 1.0, unravel, rng)
            psi = sim.statevector()
            rho += cp.outer(psi, cp.conj(psi))
        rho /= N
        d = rho - rho_ref
        d = 0.5 * (d + d.conj().T)
        td = float(0.5 * cp.sum(cp.abs(cp.linalg.eigvalsh(d))))
        print(f"  {unravel:>10}: trace distance to Aer rho = {td:.4f} "
              f"{'PASS' if td < 0.02 else 'CHECK'}")


def main():
    a3a_regime_map()
    a3b_circuit_map()
    a3c_n_scaling()
    a3d_correctness()


if __name__ == "__main__":
    main()
