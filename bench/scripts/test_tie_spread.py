"""E9d acceptance diagnostic: is the reference propagate() ill-conditioned (near-ties at the truncation boundary)?

(1) min relative gap between the (M+1)-th and (M+2)-th largest |c| over all truncations of the reference run;
(2) spread of the reference when every coefficient gets a relative perturbation of 1e-15 before each op (the size of
    summation-order rounding differences). A spread comparable to |big - ref| shows the difference is rounding-level
    ambiguity of the truncated answer, not an error of propagate_big.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402
from src.noise import traj_ref as T  # noqa: E402
from src.noise.traj_ref import xp  # noqa: E402

CASES = [("ad", "ising", "h0.7", 12, 16, 0.01, "zz", 10_000), ("ad", "ising", "h1.4", 20, 16, 0.01, "zc", 100_000),
         ("ad", "ising", "h0.7", 12, 16, 0.01, "zz", 100_000), ("deph", "brickwork", "s7", 20, 16, 0.01, "zz", 100_000)]


def run(ops, n, o, M, seed=None, gaps=None):
    rs = xp.random.RandomState(seed) if seed is not None else None
    O = P.PauliOp.from_terms(T.obs_terms(o, n), n)
    tables = {}
    orig = P._truncate

    def trunc(keys, c, MM):
        if gaps is not None:
            nz = c != 0
            mag = xp.abs(c[nz])
            mag = mag[keys[nz] != 0]
            if mag.size > MM + 1:
                s = xp.sort(mag)[::-1]
                gaps.append(float((s[MM - 1] - s[MM]) / s[MM - 1]))     # kept M non-identity vs first dropped
        return orig(keys, c, MM)

    P._truncate = trunc
    try:
        for op in reversed(ops):
            k = id(op[2])
            if k not in tables:
                tables[k] = P._table(P.transfer(op))
            if rs is not None:
                O.c = O.c * (1 + 1e-15 * rs.standard_normal(len(O)))
            O.apply(op, tables[k], M)
    finally:
        P._truncate = orig
    return O.zero_state_value()


def main():
    for ch, fam, var, n, L, gt, o, M in CASES:
        ops = K.build(fam, var, n, L, gt, ch)
        gaps = []
        ref = run(ops, n, o, M, gaps=gaps)
        big, _ = P.propagate_big(ops, n, o, M, chunk=max(200, M // 7), nb=8)
        vals = np.array([run(ops, n, o, M, seed=s) for s in range(6)])
        g = np.array(gaps)
        print(f"{ch} {fam} {var} n{n} L{L} gt{gt} {o} M={M}: |big-ref| {abs(big - ref):.1e}; perturbed-ref spread "
              f"{vals.max() - vals.min():.1e} (max |pert-ref| {np.abs(vals - ref).max():.1e}); boundary rel. gap min "
              f"{g.min() if g.size else float('nan'):.1e}, #gaps<1e-12: {int((g < 1e-12).sum())}/{g.size}", flush=True)


if __name__ == "__main__":
    main()
