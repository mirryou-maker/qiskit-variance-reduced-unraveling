"""E10-V100 vs E10 (ubuntu94, RTX 2060): pre-registered agreement judgement (docs/plan_B_characterization_2026-11-06.md,
"E10-V100 pre-registration").
Inputs: bench/results/b_e10_summary.csv (+ b_e10_medians.csv, b_e10_raw/) from ubuntu94 and b_e10_summary_v100.csv
(+ b_e10_medians_v100.csv, b_e10v_raw/) from analyze_e10.py on the V100 run.
Prints per channel x table: median W_bf / R_bf / t_static/t_f on both hosts, the per-group ratio W_V100 / W_u94 (median,
98.75% group bootstrap), the table identity across hosts (same seeds), and the verdict for the replication rule.
Usage: python bench/scripts/compare_e10_hosts.py > bench/results/b_e10_hosts.txt
"""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "bench/results")
_spec = importlib.util.spec_from_file_location("analyze_e10", os.path.join(os.path.dirname(__file__), "analyze_e10.py"))
A10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A10)
KEYS = A10.KEYS


def main():
    rng = np.random.default_rng(2027)
    u = pd.read_csv(os.path.join(R, "b_e10_summary.csv"))
    v = pd.read_csv(os.path.join(R, "b_e10_summary_v100.csv"))
    mu = pd.read_csv(os.path.join(R, "b_e10_medians.csv"))
    mv = pd.read_csv(os.path.join(R, "b_e10_medians_v100.csv"))
    m = u.merge(v, on=KEYS, suffixes=("_u", "_v"))
    print(f"groups: ubuntu94 {len(u)}, V100 {len(v)}, common {len(m)}")
    print(f"t_f(best) V100 / u94: median {np.median(m.t_f_v / m.t_f_u):.3f}")
    print("\nPer channel x table, all groups: median W_bf [98.75% group] (u94 | V100); R_bf; t_s/t_f; "
          "per-group W_V100/W_u94 [98.75%]; same best fixed; table identical across hosts")
    for t in A10.TABLES:
        for ch, C in m.groupby("channel"):
            a = mu[(mu.subset == "all98") & (mu.table == t) & (mu.channel == ch)].iloc[0]
            b = mv[(mv.subset == "all98") & (mv.table == t) & (mv.channel == ch)].iloc[0]
            rr, rlo, rhi = A10.med_ci(C[f"{t}_W_bf_v"] / C[f"{t}_W_bf_u"], rng)
            same_tab = []
            for _, r in C.iterrows():
                ks = A10.key_str(*(r[c] for c in A10.CFG))
                tu = np.load(os.path.join(R, "b_e10_raw", f"{ks}_{t}_{r['obs']}.npz"))["tab"]
                tv = np.load(os.path.join(R, "b_e10v_raw", f"{ks}_{t}_{r['obs']}.npz"))["tab"]
                same_tab.append(float(np.mean(tu == tv)))
            same_tab = np.array(same_tab)
            print(f"  {t:5s} {ch:4s} W_bf {a.W_bf:.3f} [{a.W_bf_lo:.3f}, {a.W_bf_hi:.3f}] | {b.W_bf:.3f} "
                  f"[{b.W_bf_lo:.3f}, {b.W_bf_hi:.3f}] {{{b.W_bf_clo:.3f}, {b.W_bf_chi:.3f}}}; R_bf {a.R_bf:.3f} | "
                  f"{b.R_bf:.3f}; t_s/t_f {C[f'{t}_tratio_u'].median():.3f} | {C[f'{t}_tratio_v'].median():.3f}; "
                  f"W_V/W_u {rr:.3f} [{rlo:.3f}, {rhi:.3f}]; same best {np.mean(C.best_u == C.best_v):.3f}; "
                  f"table identical {int((same_tab == 1).sum())}/{len(same_tab)} (mean entry agreement {same_tab.mean():.4f})")
    print("\nREPLICATION RULE (pre-registered): E10's positive cell (ad, pilot) is replicated on V100 iff its V100 median "
          "W_bf > 1 with 98.75% group-bootstrap lower bound > 1 (decision-line draw, rng 7, as analyze_e10).")
    for t in A10.TABLES:
        for ch, C in v.groupby("channel"):
            mm, lo, _ = A10.med_ci(C[f"{t}_W_bf"], np.random.default_rng(7))
            Cu = u[u.channel == ch]
            mu_, lou, _ = A10.med_ci(Cu[f"{t}_W_bf"], np.random.default_rng(7))
            pu = "PASS" if (mu_ > 1 and lou > 1) else "fail"
            pv = "PASS" if (mm > 1 and lo > 1) else "fail"
            print(f"  {ch:4s} {t:5s} u94 median {mu_:.3f} lower {lou:.3f} -> {pu}; V100 median {mm:.3f} lower {lo:.3f} -> {pv}"
                  f"{'  (agrees)' if pu == pv else '  (DISAGREES)'}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
