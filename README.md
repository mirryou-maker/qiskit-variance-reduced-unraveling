# Variance-Reduced Trajectory Unravelings for GPU Noisy Quantum-Circuit Simulation

Reproducibility artifact for the manuscript

> **Variance-Reduced Trajectory Unravelings for GPU Noisy Quantum-Circuit Simulation:
> Characterization and a Qiskit-Aer Integration Gap**
> Chun-Yeol You, Department of Physics and Chemistry, DGIST, Daegu 42988, Republic of Korea
> ORCID [0000-0001-9549-8611](https://orcid.org/0000-0001-9549-8611) · cyyou@dgist.ac.kr

## What this is

Monte-Carlo trajectory (quantum-jump) simulation of noisy quantum circuits is limited by estimator
variance, not memory. This repository implements **projector** and **analog** variance-reduced
unravelings on a **GPU dense-statevector** trajectory engine and characterizes them against
Qiskit-Aer's exact `density_matrix` reference and its production `batched_shots_gpu` path.

Headline results (single consumer GPU, RTX 5060 Ti 8 GB):

- Projector unraveling reaches a target standard error with **20.8× fewer trajectories** than
  Qiskit-Aer at *n* = 10, holding at **19–26×** across *n* = 8–20.
- A **regime map** places analog sampling optimal at weak noise and projector at strong noise,
  crossing near γ*t* ≈ 0.35.
- All unravelings are **unbiased**: the reconstructed density matrix converges to Aer's exact result
  at the 1/√N Monte-Carlo rate.
- **Systems finding:** Qiskit-Aer applies noise at the *channel* level and reconstructs a canonical
  Kraus decomposition at apply time, discarding any user-supplied unraveling — so variance-reduced
  unravelings cannot be delivered through its public API. A minimal change is specified.

## Layout

```
src/trajectory/gpu_trajectory.py   GPU dense-statevector trajectory engine (cupy)
validation/                        correctness, variance, sweep, head-to-head, Aer probes
bench/results/*.csv                raw measured data (seeds + environment in headers)
paper/                             manuscript (IEEE TQE + arXiv versions), figures, supplementary
```

See `paper/supplementary.md` for the full environment manifest, code map, run order, raw-data
dictionary, and derivations.

## Quick start

```bash
python3 -m venv ~/qsim-venv && source ~/qsim-venv/bin/activate
pip install "qiskit>=1.1,<1.3" "qiskit-aer-gpu==0.15.1" cupy-cuda12x numpy matplotlib
python -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"
# expected: ('CPU', 'GPU')

python -u validation/test_a1_tracedistance.py   # correctness (1/sqrt(N) convergence)
python -u validation/test_a4_vs_aer.py          # head-to-head vs Qiskit-Aer
python -u paper/figures/make_figures.py         # regenerate all figures
```

## Building the paper

`paper/main.tex` targets IEEE TQE and requires the IEEE template files
(`ieeeaccess.cls`, plus the branding images), which are **not redistributed here** for copyright
reasons — download them from the
[IEEE Author Center](https://journals.ieeeauthorcenter.ieee.org/).

`paper/arxiv.tex` is a self-contained preprint version using the standard `article` class and needs
no IEEE files:

```bash
cd paper
python make_arxiv.py                       # regenerate from main.tex
pdflatex arxiv && bibtex arxiv && pdflatex arxiv && pdflatex arxiv
```

## Note on AI-assisted development

The engine, benchmark harness, and figure scripts were developed with an AI coding assistant
(Claude Code, Anthropic) under the author's direction and review; all reported numbers were produced
by executing the scripts in this repository and verified by the author. See the Acknowledgment and
the methodology section of the manuscript.

## License

Code is released for review and reproduction purposes. Please contact the author regarding reuse.
