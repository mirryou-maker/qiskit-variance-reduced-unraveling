# Adaptive trajectory unravelings for noisy-circuit simulation: characterization code and data

Code, measurement scripts and result tables for a study of Monte-Carlo trajectory unravelings of
noisy quantum circuits on a GPU dense-statevector engine. A manuscript is in preparation; it is not
part of this repository.

Author: Chun-Yeol You, Department of Physics and Chemistry, DGIST, Daegu 42988, Republic of Korea
(ORCID [0000-0001-9549-8611](https://orcid.org/0000-0001-9549-8611), cyyou@dgist.ac.kr).

## Summary

Trajectory (quantum-jump) simulation of noisy circuits is limited by estimator variance. The unraveling
(the choice of jump operators per noise location) changes that variance without changing the expectation
value. This repository implements an exact variance criterion for the choice of unraveling, based on a
back-propagated observable (the increments of the Doob martingale of the trajectory estimator), and
measures what it buys in wall-clock time against the best fixed unraveling.

Current findings (n = 20 qubits, see the result files named below for the numbers):

- Per-trajectory greedy adaptive selection lowers variance, but gave no significant wall-clock gain over
  the best fixed unraveling once its cost is included (E9b-E9e: `bench/results/b_e9e_analysis.txt`,
  `b_e9e_v100_analysis.txt`).
- Static per-location tables built once from a short pilot gave a wall-clock gain for amplitude damping in
  pre-specified tests (E10: `b_e10_analysis.txt`; E11: `b_e11_analysis.txt`), concentrated in brickwork
  circuits (post hoc split: `pra_family_split.txt`), and no gain for dephasing. Sensitivity to the pilot
  size and truncation: `b_e12_analysis.txt`. Timing replication on a second GPU model:
  `b_e10_v100_analysis.txt`.
- For most test conditions, Pauli propagation with a larger cap is the cheaper way to reach the same
  accuracy (post hoc, `pra_rev3b_misc.txt`, part C). The trajectory methods here are therefore not a
  general speed-up claim.
- Earlier statements that this code reaches a target accuracy with an order of magnitude fewer
  trajectories than Qiskit-Aer, and that unravelings cannot be supplied through Qiskit-Aer's public API,
  are withdrawn; they were too strong and are not supported by the later experiments. Results under
  `bench/results/a*` and the `validation/probe_*`/`test_a*` scripts belong to that earlier stage and are
  kept for provenance only.
- E13 (pre-specified, 94 fresh groups where Pauli propagation does not converge at a 10^7-term cap; P100):
  the static table again gave a wall-clock gain for amplitude damping, median 1.45 (98.75% CI 1.30-1.58),
  but not for dephasing (0.89, secondary). The Doob-martingale control variate lowered the variance within a
  run (2.1x ad, 1.5x deph) but cost 3.5-3.8x per trajectory, so it lost in wall-clock terms in every
  pre-specified test (`bench/results/logs_iremb_e13/e13_analyze_main.txt`, `b_e13_summary.csv`).

## Repository layout

```
src/noise/                 trajectory engine components: traj_kraus.py (Kraus trajectories, schemes),
                           static_table.py / static_table_multi.py (static per-location tables),
                           sparse_policy.py, pauli_prop.py (Pauli propagation), fastsim.py, traj_ref.py
src/trajectory/            gpu_trajectory.py: earlier dense-statevector trajectory engine (cupy)
validation/                pytest tests of the engine and Aer cross-check scripts
bench/scripts/             experiment runners (b_e*.py, a_*.py, PBS scripts) and analysis scripts
bench/results/             tracked result tables (CSV) and analysis printouts (TXT), with host/idle records
```

## Setup and tests

```bash
python3 -m venv ~/qsim-venv && source ~/qsim-venv/bin/activate
pip install numpy scipy pandas matplotlib pytest cupy-cuda12x          # GPU runs need cupy
pip install "qiskit>=1.1,<1.3" "qiskit-aer-gpu==0.15.1"                # only for the Aer cross-checks
pytest validation/test_traj_kraus.py validation/test_static_table.py \
       validation/test_static_table_multi.py validation/test_sparse_policy.py \
       validation/test_pauli_prop.py validation/test_fastsim.py
```

Aer cross-check scripts (`validation/test_a*.py`, `validation/probe_*.py`) need Qiskit-Aer and, for the
GPU path, a CUDA GPU. Timing results were measured on RTX 2060, RTX 3070 Ti and Tesla V100 hosts; each
result file records the host and an idle check taken before the run.

## Reproducing the analyses

Each experiment has a runner (`bench/scripts/b_e*.py`) and an analysis script whose docstring states its
inputs, outputs and usage:

| Experiment | Runner | Analysis | Printout / tables in `bench/results/` |
|---|---|---|---|
| E9b greedy vs fixed | `b_e9b.py` | `analyze_e9b.py` | `b_e9b_analysis.txt` |
| E9e optimized greedy | `b_e9e.py`, `b_e9e_fallback.py` | `analyze_e9e.py` | `b_e9e_analysis.txt` |
| E10 static tables | `b_e10.py` | `analyze_e10.py` | `b_e10_analysis.txt`, `b_e10_v100_analysis.txt` |
| E11 confirmatory | `b_e11_select.py`, `b_e11.py` | `analyze_e11.py` | `b_e11_analysis.txt` |
| E12 sensitivity | `b_e12.py` | `analyze_e12.py` | `b_e12_analysis.txt` |
| Post hoc | | `pra_family_split.py`, `pra_rev3b_misc.py`, `pra_numbers.py`, `pra_baseline_cost.py`, ... | `pra_*.txt` / `pra_*.csv` |

The scripts `pra_family_split.py` and `pra_rev3b_misc.py` read only the tracked summary CSVs and can be
re-run directly from a clone. The `analyze_e*.py` scripts and some `pra_*.py` scripts also read the
per-trajectory arrays (see below); the tracked `*_analysis.txt` and `*_summary.csv` files are their outputs.

## Data availability

The tracked CSV/TXT files contain every summary statistic reported. The per-trajectory arrays
(`bench/results/*_raw/`, about 1.5 GB of `.npz`, excluded by `.gitignore`) and the dated planning document
that fixes the pre-specified decision rules for E10-E12 are available from the author on request.
The manuscript, its figures and internal notes are not distributed through this repository.

## AI-assisted development

The engine, benchmark harness and analysis scripts were developed with an AI coding assistant (Claude
Code, Anthropic) under the author's direction and review; reported numbers come from executing the
scripts in this repository.

## License

Code is released for review and reproduction purposes. Please contact the author regarding reuse.
