"""Check continuation_events (matrix-free) against ideal_events (dense) at small n."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.noise import traj_ref as T  # noqa: E402

for fam, n, L, gt, h in (("ising", 6, 4, 0.1, 0.7), ("ising", 6, 4, 0.03, 1.4), ("brick", 6, 4, 0.1, 0),
                         ("brick", 8, 4, 0.03, 0)):
    ops = T.ising_ops(n, L, gt, h=h) if fam == "ising" else T.brickwork_ops(n, L, gt)
    for o in ("x", "zz", "xc", "zc"):
        s, q = T.ideal_events(ops, n, o)
        s2, q2 = T.continuation_events(ops, n, o, reps=256, seed=1)
        t1, t2 = q < (1 - 1e-3) * s, q2 < (1 - 1e-3) * s2
        sig = (s + q) > 1e-6 * (s + q).max()
        print(f"{fam} n={n} L={L} gt={gt} {o:3} relerr std={np.abs(s2-s).sum()/max(s.sum(),1e-30):.3f} "
              f"proj={np.abs(q2-q).sum()/max(q.sum(),1e-30):.3f} table agree(sig events)="
              f"{np.mean(t1[sig]==t2[sig]) if sig.any() else float('nan'):.3f} n_sig={sig.sum()}/{len(s)}")
