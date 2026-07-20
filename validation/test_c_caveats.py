"""C: close the two A3 caveats.

C1 (GHZ): use the GHZ stabilizer observable X^{otimes n} (= the |0..0>-|1..1> coherence, decays as
   e^{-2 n gamma t}, anticommutes with each Z_i) instead of <X_i> (trivially 0). Projector should
   again reduce variance (Bernoulli estimator: mean m, Var_proj = m(1-m) vs Var_std ~ 1-m^2).

C2 (erosion): interleave RY(theta) rotations between noise windows. RY does NOT commute with the X
   observable (X -> X cos - Z sin), so after a projector collapse to a Z-eigenstate (<X>=0), the next
   RY rotates <X> back to non-zero, eroding the absorbing window. Sweep theta: projector speedup
   should decline from the idle (theta=0) value toward ~1 as coherent mixing grows.

Run:  python -u /mnt/d/Claude-Code-R/Qiskit-CO/validation/test_c_caveats.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trajectory.gpu_trajectory import GPUTrajectorySim, rx, ry  # noqa: E402
import cupy as cp  # noqa: E402

SEED = 20260719
THETA0 = 0.30
TARGET_SE = 0.01


def noise_window(sim, q, gamma, tau, unravel, rng):
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


def expval_global_X(sim):
    """<psi| X^{otimes n} |psi> = sum_x conj(psi_x) psi_{~x} (bit-flip = index complement)."""
    psi = sim.statevector()
    return float(cp.real(cp.sum(cp.conj(psi) * psi[::-1])))


def n_to_target(vals):
    return (vals.std(ddof=1) / TARGET_SE) ** 2


def c1_ghz_stabilizer():
    print("=== C1: GHZ with stabilizer observable X^{otimes n} (n=10) ===")
    n, N = 10, 2000
    sim = GPUTrajectorySim(n, backend="cupy")
    print(f"{'gt':>5} {'mean':>8} {'exact':>8} {'var_std':>9} {'var_proj':>9} "
          f"{'var_analog':>11} {'spdup_proj':>11} {'spdup_analog':>12}")
    for gt in (0.05, 0.10, 0.15):
        exact = np.exp(-2 * n * gt)
        stats = {}
        for u in ("standard", "projector", "analog"):
            rng = np.random.default_rng(SEED + 1)
            vals = np.empty(N)
            for i in range(N):
                sim.reset()
                sim.apply_1q(sim.g["H"], 0)
                for q in range(n - 1):
                    sim.apply_cx(q, q + 1)
                for q in range(n):
                    noise_window(sim, q, 1.0, gt, u, rng)
                vals[i] = expval_global_X(sim)
            stats[u] = (vals.mean(), vals.var(), n_to_target(vals))
        Ns, Np, Na = stats["standard"][2], stats["projector"][2], stats["analog"][2]
        print(f"{gt:>5} {stats['projector'][0]:>8.4f} {exact:>8.4f} "
              f"{stats['standard'][1]:>9.4f} {stats['projector'][1]:>9.4f} "
              f"{stats['analog'][1]:>11.4f} {Ns/Np:>11.1f} {Ns/Na:>12.1f}")


def c2_erosion():
    print("\n=== C2: window erosion by interleaved RY(theta) (n=8, gt=1.5, obs mean<X_i>) ===")
    n, N, gt = 8, 2000, 1.5
    sim = GPUTrajectorySim(n, backend="cupy")
    tau = gt / 3.0
    print(f"{'theta':>7} {'N_std':>8} {'N_proj':>8} {'speedup_proj':>13} "
          f"{'mean':>8}   (theta=0 => idle, absorbing intact)")
    for theta in (0.0, 0.3, 0.6, 1.0, np.pi / 2):
        res = {}
        for u in ("standard", "projector"):
            rng = np.random.default_rng(SEED + 2)
            vals = np.empty(N)
            for i in range(N):
                sim.reset()
                for q in range(n):
                    sim.apply_1q(sim.g["H"], q)
                for w in range(3):
                    for q in range(n):
                        noise_window(sim, q, 1.0, tau, u, rng)
                    if w < 2 and theta != 0.0:
                        for q in range(n):
                            sim.apply_1q(ry(sim.xp, theta), q)
                vals[i] = sum(sim.expval_1q(sim.g["X"], q) for q in range(n)) / n
            res[u] = (n_to_target(vals), vals.mean())
        print(f"{theta:>7.3f} {res['standard'][0]:>8.0f} {res['projector'][0]:>8.0f} "
              f"{res['standard'][0]/res['projector'][0]:>13.1f} {res['projector'][1]:>8.4f}")


def main():
    c1_ghz_stabilizer()
    c2_erosion()


if __name__ == "__main__":
    main()
