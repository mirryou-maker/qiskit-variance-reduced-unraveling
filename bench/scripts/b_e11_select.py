"""E11 group selection (docs/plan_B_characterization_2026-11-06.md, "E11 pre-registration"): the E9a run on FRESH
circuits, with the E9a code unchanged (bench/scripts/b_e9a.py is imported and only its VARIANTS list is replaced).

Fresh variants (not used at n = 20 in E9a–E10): brickwork s9, s10; Ising h1.0, h1.2. n = 20, L 4/8/16/32,
gamma*t 0.01/0.03/0.1, channels deph / ad, observables x, zz, xc, zc. PP at M 1e2..1e6, std reference N_ref = 1000.
Classification: analyze_e11.py select (E9a rule, verified to reproduce b_e9a_summary.csv on the E9a data).
Usage (ubuntu94, env qsim13): python bench/scripts/b_e11_select.py --out bench/results/b_e11_select.csv
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location("b_e9a", os.path.join(os.path.dirname(__file__), "b_e9a.py"))
E9A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E9A)

FRESH = [("brickwork", "s9"), ("brickwork", "s10"), ("ising", "h1.0"), ("ising", "h1.2")]

if __name__ == "__main__":
    E9A.VARIANTS = FRESH
    if "--n" not in sys.argv:
        sys.argv += ["--n", "20"]
    E9A.main()
