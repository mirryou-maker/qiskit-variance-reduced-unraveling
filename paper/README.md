# Paper — IEEE TQE draft

**Target venue:** IEEE Transactions on Quantum Engineering (TQE). Format: IEEEtran, `journal` mode.

## Files
- `main.tex` — manuscript draft (T0 scope: current measured results on a consumer GPU).
- `references.bib` — bibliography (TJM, Atlas, quEStab, Qiskit-Aer, MPDO/LPDO, etc.).
- `figures/*.png` — figures, generated from measured data by `figures/make_figures.py`.
- `explainer_ko.md` — non-specialist Korean explainer (companion, not part of submission).

## Regenerate figures (from measured data)
```bash
# in WSL venv (qiskit 1.2.4 / qiskit-aer-gpu 0.15.1 / cupy / matplotlib)
python paper/figures/make_figures.py
```
Data sources: `bench/results/{a1_convergence,a3_sweep,a4_vs_aer,c_caveats}.csv`.

## Build the PDF
No LaTeX is installed locally (texlive needs root apt, unavailable here). Options:
- **Overleaf**: upload `main.tex`, `references.bib`, and `figures/` — compiles as-is.
- **Local (once texlive is installed):**
  ```bash
  cd paper
  pdflatex main && bibtex main && pdflatex main && pdflatex main
  ```

## Status / TODO before submission
- [ ] Author names, affiliations, funding.
- [ ] Fill exact citation details for arXiv entries (TJM 2607.01323, LPDO, MPDO) at submission.
- [ ] (T1) Add A100/H100 results to n~32 + absolute wall-clock (upgrades competitiveness).
- [ ] (T2, optional) Aer C++ PR result if pursued.
- [ ] Artifact-evaluation appendix (env manifest, seeds, scripts) per TQE policy.
- [ ] Re-check the venue's AI-tooling disclosure policy at submission time (CLAUDE.md §2).
