# arXiv Submission — Final Check

Build verified: `arxiv.pdf`, **20 pages, 0 LaTeX errors, 0 undefined citations/references,
50 bibliography entries**. Source is self-contained (`article` class; no publisher class or branding
files required).

**Venue-neutral by design.** The preprint contains no mention of IEEE or of any specific journal, the
bibliography uses the generic `unsrt` style, and the author-biography section is omitted, so the
same source can accompany a submission to any venue. Float placement is constrained with `placeins` so
figures appear next to their discussion rather than drifting to the end.

## 1. What to upload where

Both submission systems are satisfied by a **self-contained `.tex` with the bibliography inlined**
(`make_selfcontained.py`), so neither a `.bib` nor a BibTeX run is required. Build the archives with
`python make_packages.py`.

| Target | Archive | Contents |
|:--|:--|:--|
| **arXiv** | `submission_arxiv.tar.gz` | `arxiv_selfcontained.tex` + `figures/` |
| **Journal (IEEE)** | `submission_ieee.zip` | `main_ieee.tex`, `ieeeaccess.cls`, branding PNGs, `figures/` |

**Never upload the compiled PDF to arXiv** — it is auto-rejected ("appears to have been produced by
TeX"); arXiv needs the source to build HTML/MathML/EPUB and accessible formats
(<https://info.arxiv.org/help/faq/whytex.html>).

**Do not mix up the two versions.** `arxiv_selfcontained.tex` uses the `article` class and `unsrt`
references; `main_ieee.tex` uses `ieeeaccess` and IEEE-formatted references. Submitting the arXiv
file to the journal produces the wrong layout.

### Failures seen and how they were fixed

| Symptom | Cause | Fix |
|:--|:--|:--|
| arXiv: "appears to have been produced by TeX" | the PDF was uploaded | upload the source archive |
| arXiv: latex→dvi→ps attempted, PNG figures fail | no `\pdfoutput` | `\pdfoutput=1` placed before `\documentclass` |
| Journal: "I couldn't open database file references.bib", 50 empty citations | the publisher runs BibTeX but `references.bib` was not uploaded | bibliography inlined into the `.tex`; no BibTeX needed |
| Journal: `unsrt.bst` in the log | the arXiv preprint was uploaded instead of the journal source | upload `main_ieee.tex` |
| Missing `Logo.png` on a case-sensitive server | the class requests `Logo.png`/`notaglineLogo.png`, the official template ships `logo.png`/`notaglinelogo.png` | the zip contains **both letter cases** |

### Verification performed

Each archive was extracted into a clean directory and compiled with **pdflatex three times, no
BibTeX, no `.bib`** — exactly what the submission systems do:

- `main_ieee.tex` -> 10 pages, 0 errors, 0 undefined citations/references, 0 missing files
- `arxiv_selfcontained.tex` -> 20 pages, 0 errors, 0 undefined citations/references

### Rebuilding

```bash
cd paper
python make_arxiv.py                                                  # arxiv.tex from main.tex
pdflatex main -jobname=main_tqe4 && bibtex main_tqe4                  # journal .bbl
pdflatex arxiv -jobname=arxiv3   && bibtex arxiv3                     # preprint .bbl
python make_selfcontained.py                                          # inline both bibliographies
python make_packages.py                                               # build both archives
```

## 3. arXiv metadata

**Primary category:** `quant-ph`
**Cross-lists:** `cs.DC` (distributed/parallel computing) and/or `physics.comp-ph`

**Comments field (suggested):**

```
20 pages, 6 figures, 4 tables. Submitted for publication.
Code and raw data: https://github.com/mirryou-maker/qiskit-variance-reduced-unraveling
```

**Abstract for the metadata box** (plain text; arXiv accepts inline `$...$` math but not
`\emph`/`\texttt`):

```
Monte-Carlo trajectory (quantum-jump) methods are the practical route to simulating noisy quantum
circuits once the exact density-matrix method is precluded by its $4^n$ memory cost. Their
bottleneck is estimator variance: resolving one expectation value can demand thousands of
trajectories. Recent tensor-network work shows that variance-reduced unravelings -- projector and
analog sampling -- sharply cut this variance, but only on CPU matrix-product-state backends, with no
path into production tooling. We implement both unravelings on a GPU dense-statevector trajectory
engine and validate them against the exact density matrix (ideal-circuit fidelity $1-2.2 \times
10^{-16}$; $1/\sqrt{N}$ convergence; all unravelings unbiased to trace distance $<0.01$). On a
single consumer GPU, projector unraveling reaches a target standard error with $20.8\times$ fewer
trajectories than Qiskit-Aer's batched_shots_gpu at $n=10$, a factor that holds at $19$-$26\times$
across $n=8$-$20$. A regime map places analog sampling optimal at weak noise and projector at strong
noise, crossing near $\gamma t \approx 0.35$. We further report a systems finding: Qiskit-Aer
applies noise at the channel level and reconstructs a canonical Kraus decomposition at apply time,
discarding any user-supplied unraveling, so variance-reduced unravelings cannot be delivered through
its public API. Because Aer's Born-rule collapse machinery already exists, we specify a minimal
change that would unlock the technique in production.
```

**License:** arXiv's default (non-exclusive license to distribute) is compatible with later IEEE
publication. Do **not** choose CC BY if you intend to transfer copyright to IEEE.

**ORCID:** 0000-0001-9549-8611 — link it in the arXiv account so the paper is claimed automatically.

## 4. IEEE preprint policy (compatible)

IEEE permits posting a submitted version to arXiv. After acceptance, update the arXiv entry with the
IEEE copyright/citation line, e.g.:

> © 20XX IEEE. Personal use of this material is permitted. … Accepted for publication in *IEEE
> Transactions on Quantum Engineering*, DOI: 10.1109/TQE.XXXX.XXXXXXX.

Posting the preprint **before or at the same time as** the TQE submission is standard and does not
count as prior publication.

## 5. Content review — passed

- No venue-specific or submission-internal wording leaked into the preprint (checked).
- No "Impact Statement"; no author biography; no venue-specific wording.
- Body is continuous prose; no bulleted lists.
- All 6 figures have detailed captions; figure numbering matches narrative order.
- AI-usage disclosure present (Acknowledgment + methodology section) — good practice for a preprint.
- Reference [50] is marked "under review"; update if it is accepted before posting.
- Reference [19] cites *Phys. Rev. A* **110**, 032604 (2024) — please confirm this against the
  original record, since an arXiv identifier of 2508.07610 (2025) was previously associated with it.

## 6. Rebuild command

```bash
cd paper
python make_arxiv.py                 # regenerate arxiv.tex from main.tex
pdflatex arxiv && bibtex arxiv && pdflatex arxiv && pdflatex arxiv
```
