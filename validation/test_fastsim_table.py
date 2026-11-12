"""Compare the scalable continuation table (fastsim) with the dense noiseless-trajectory table
(traj_ref.ideal_events) at n=8: event agreement and the variance each table achieves."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.noise import fastsim as F  # noqa: E402
from src.noise import traj_ref as T  # noqa: E402

n = 8
sim = F.FastSim(n)
for fam in ("ising", "brickwork"):
    for L, gt in ((4, 0.03), (4, 0.1), (8, 0.03)):
        ops = T.ising_ops(n, L, gt) if fam == "ising" else T.brickwork_ops(n, L, gt)
        st = F.ising_circuit(n, L, gt) if fam == "ising" else F.brickwork_circuit(n, L, gt)
        for o in ("x", "zz", "xc", "zc"):
            s, q = T.ideal_events(ops, n, o)
            td = (q < (1 - 1e-3) * s).astype(float)
            tf, _, _ = sim.continuation_table(st, o, reps=8, seed=3)
            sig = (s + q) > 1e-6 * max((s + q).max(), 1e-300)
            agree = np.mean(td[sig] == tf[sig]) if sig.any() else float("nan")
            xs = sim.run(st, o, 20000, seed=1)
            xd = sim.run(st, o, 20000, table=td, seed=2)
            xf = sim.run(st, o, 20000, table=tf, seed=3)
            vs, vd, vf = xs.var(ddof=1), xd.var(ddof=1), xf.var(ddof=1)
            r = (lambda a, b: a / b if b > 1e-14 else float("nan"))
            print(f"{fam:9} L={L} gt={gt:<5} {o:3} agree={agree:.3f} frac_proj dense={td.mean():.2f} "
                  f"fast={tf.mean():.2f}  Var_std/Var: dense-table={r(vs, vd):.3f} fast-table={r(vs, vf):.3f}",
                  flush=True)
