"""A2 validation: variance-reduced unravelings reproduce TJM Theorem 1.

Scenario (single Pauli-Lindblad dephasing channel, gamma=1):
  master eq  rho' = gamma (Z rho Z - rho),  initial |+>,  observable O = X, {X,Z}=0.
Closed forms:
  mean          <X>(t) = e^{-2 gamma t}
  standard var  Var = 1 - e^{-4 gamma t}                 (telegraph +-1)
  projector var Var = e^{-2 gamma t}(1 - e^{-2 gamma t}) (Bernoulli, Theorem 1)
  analog: same mean, lowest variance in the weak-noise regime.

Exact event sampling (no time discretization): number/first-time of jumps sampled
analytically, then the engine applies the corresponding operators to the statevector.

Part 1 (numpy, single qubit): reproduce the closed forms, high N.
Part 2 (cupy, n qubits on GPU): trajectories-to-target-SE, standard vs projector.

Run:  python /mnt/d/Claude-Code-R/Qiskit-CO/validation/test_a2_variance.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trajectory.gpu_trajectory import GPUTrajectorySim  # noqa: E402

SEED = 20260719
THETA0 = 0.30  # analog kick angle


def traj_expX(sim, q, mode, gT, rng):
    """One trajectory: prepare |+> on qubit q, apply channel events, return <X_q>."""
    sim.apply_1q(sim.g["H"], q)  # |0> -> |+>
    if mode == "standard":
        k = rng.poisson(gT)               # gamma=1 -> Poisson mean gT
        if k % 2 == 1:                    # Z^k = Z if odd (parity), I if even
            sim.apply_1q(sim.g["Z"], q)
    elif mode == "projector":
        if rng.random() < (1.0 - np.exp(-2.0 * gT)):   # a jump occurred in [0,T]
            plus = rng.random() < sim.prob_plus("Z", q)
            sim.apply_projector("Z", q, plus)          # collapse -> absorbing
    elif mode == "analog":
        lam = 1.0 / np.sin(THETA0) ** 2                # generator match: lam*sin^2 = gamma
        k = rng.poisson(lam * gT)
        for _ in range(k):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            sim.apply_analog_kick("Z", q, sign * THETA0)
    return sim.expval_1q(sim.g["X"], q)


def part1_closed_form():
    print("=== Part 1: single-qubit closed-form reproduction (numpy) ===")
    N = 20000
    grid = [0.1, 0.25, 0.5, 1.0, 1.5]
    sim = GPUTrajectorySim(1, backend="numpy")
    rng = np.random.default_rng(SEED)
    hdr = (f"{'gt':>5} {'mode':>10} {'mean_emp':>9} {'mean_cf':>8} "
           f"{'var_emp':>9} {'var_cf':>8}")
    print(hdr)
    results = []
    for gt in grid:
        mean_cf = np.exp(-2 * gt)
        var_cf = {
            "standard": 1 - np.exp(-4 * gt),
            "projector": np.exp(-2 * gt) * (1 - np.exp(-2 * gt)),
            "analog": None,
        }
        for mode in ("standard", "projector", "analog"):
            xs = np.empty(N)
            for i in range(N):
                sim.reset()
                xs[i] = traj_expX(sim, 0, mode, gt, rng)
            me, ve = xs.mean(), xs.var()
            vcf = var_cf[mode]
            vcf_s = f"{vcf:8.4f}" if vcf is not None else "     n/a"
            print(f"{gt:>5} {mode:>10} {me:>9.4f} {mean_cf:>8.4f} {ve:>9.4f} {vcf_s}")
            results.append((gt, mode, me, mean_cf, ve, vcf))
        print()
    # assertions
    ok = True
    for gt, mode, me, mcf, ve, vcf in results:
        if abs(me - mcf) > 0.02:
            ok = False; print(f"  MEAN MISMATCH {mode} gt={gt}: {me:.4f} vs {mcf:.4f}")
        if vcf is not None and abs(ve - vcf) > 0.02:
            ok = False; print(f"  VAR MISMATCH {mode} gt={gt}: {ve:.4f} vs {vcf:.4f}")
    # projector < standard in strong noise
    strong = [r for r in results if r[0] == 1.5]
    v_std = [r[4] for r in strong if r[1] == "standard"][0]
    v_prj = [r[4] for r in strong if r[1] == "projector"][0]
    print(f"[strong noise gt=1.5] Var standard={v_std:.4f}  projector={v_prj:.4f}  "
          f"reduction x{v_std/v_prj:.1f}")
    print(f"[Part 1] {'PASS' if ok else 'FAIL'} (empirical matches closed form)\n")
    return results


def part2_gpu_savings():
    print("=== Part 2: GPU multi-qubit trajectories-to-target-SE (cupy) ===")
    import cupy as cp
    n = 10
    gt = 1.5                       # strong-noise regime (projector favored)
    target_se = 0.01
    checkpoints = [500, 2000, 8000, 32000]
    sim = GPUTrajectorySim(n, backend="cupy")
    rng = np.random.default_rng(SEED + 1)
    mean_cf = np.exp(-2 * gt)
    print(f"n={n} qubits (2^{n}={2**n}-dim SV on GPU), gt={gt}, "
          f"observable=mean_i<X_i>, closed-form mean={mean_cf:.4f}")
    for mode in ("standard", "projector"):
        t0 = time.time()
        vals = np.empty(max(checkpoints))
        for t in range(max(checkpoints)):
            sim.reset()
            # independent dephasing channel + |+> on each qubit
            xacc = 0.0
            for q in range(n):
                xacc += traj_expX(sim, q, mode, gt, rng)
            vals[t] = xacc / n
        dt = time.time() - t0
        print(f"  [{mode}] {max(checkpoints)} traj in {dt:.1f}s")
        print(f"    {'N':>7} {'mean':>8} {'SE':>9} {'N_to_SE<=0.01(est)':>20}")
        for N in checkpoints:
            v = vals[:N]
            se = v.std(ddof=1) / np.sqrt(N)
            n_needed = (v.std(ddof=1) / target_se) ** 2
            print(f"    {N:>7} {v.mean():>8.4f} {se:>9.5f} {n_needed:>20.0f}")
        print()


def main():
    part1_closed_form()
    part2_gpu_savings()


if __name__ == "__main__":
    main()
