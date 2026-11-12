"""E9d acceptance test (plan doc, "E9d pre-registration"): propagate_big == propagate on E9a TN groups.

Tiny chunks and few buckets force the staged path at small M. Pass: |diff| <= 1e-10 for every case.
Run: python bench/scripts/test_propagate_big.py
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import pauli_prop as P  # noqa: E402
from src.noise import traj_kraus as K  # noqa: E402

CASES = [("deph", "brickwork", "s7", 20, 16, 0.01, "zz"), ("ad", "brickwork", "s8", 20, 8, 0.03, "x"),
         ("ad", "ising", "h1.4", 20, 16, 0.01, "zc"), ("deph", "ising", "h1.4", 20, 8, 0.01, "xc"),
         ("ad", "brickwork", "s7", 24, 16, 0.01, "zz"), ("deph", "brickwork", "s8", 24, 32, 0.03, "x"),
         ("ad", "ising", "h0.7", 12, 16, 0.01, "zz")]


def main():
    worst, ok = 0.0, True
    for ch, fam, var, n, L, gt, o in CASES:
        ops = K.build(fam, var, n, L, gt, ch)
        for M in (1_000, 10_000, 100_000):
            t0 = time.perf_counter()
            v0, p0 = P.propagate(ops, n, o, M)
            t1 = time.perf_counter()
            st = {}
            v1, p1 = P.propagate_big(ops, n, o, M, chunk=max(200, M // 7), nb=8, stats=st)
            t2 = time.perf_counter()
            d = abs(v0 - v1)
            worst = max(worst, d)
            ok &= d <= 1e-10
            print(f"{ch:4s} {fam:9s} {var:4s} n{n} L{L} gt{gt} {o:2s} M={M:>6d}: ref {v0:+.12f} big {v1:+.12f} "
                  f"|d| {d:.1e} peak {p0}/{p1} uniq {st.get('peak_unique')} retries {st.get('threshold_retries', 0)} "
                  f"({t1 - t0:.1f}s / {t2 - t1:.1f}s)", flush=True)
    print(f"\nworst |diff| {worst:.2e} -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
