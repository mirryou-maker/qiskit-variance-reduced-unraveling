r"""Produce self-contained manuscript sources with the bibliography inlined.

Publisher submission systems (IEEE ScholarOne, Editorial Manager, ...) run BibTeX themselves and
will fail if `references.bib` is not uploaded. arXiv, conversely, does NOT run BibTeX and needs a
`.bbl`. Inlining the resolved `thebibliography` block into the `.tex` satisfies both: no `.bib`, no
`.bbl`, no BibTeX pass required.

Usage (after a normal pdflatex+bibtex build has produced the .bbl files):

    python make_selfcontained.py

Outputs:
    main_ieee.tex           <- journal version  (ieeeaccess class, IEEEtran-formatted references)
    arxiv_selfcontained.tex <- preprint version (article class, unsrt-formatted references)
"""
import io
import os
import re
import sys


def inline(tex_path, bbl_path, out_path):
    if not os.path.exists(bbl_path):
        sys.exit(f"missing {bbl_path} - run pdflatex+bibtex first")
    tex = io.open(tex_path, encoding="utf-8").read()
    bbl = io.open(bbl_path, encoding="utf-8").read()

    # keep only the thebibliography environment from the .bbl
    start = bbl.index(r"\begin{thebibliography}")
    end = bbl.index(r"\end{thebibliography}") + len(r"\end{thebibliography}")
    bib_block = bbl[start:end]

    # replace \bibliographystyle{...} ... \bibliography{...} with the literal block
    pattern = re.compile(
        r"\\bibliographystyle\{[^}]*\}\s*\n\s*\\bibliography\{[^}]*\}")
    if not pattern.search(tex):
        sys.exit(f"could not find the \\bibliographystyle/\\bibliography pair in {tex_path}")
    header = ("%% Bibliography inlined by make_selfcontained.py: no .bib and no BibTeX run needed.\n"
              "%% Regenerate with: pdflatex <job> && bibtex <job> && python make_selfcontained.py\n")
    tex = pattern.sub(lambda _: header + bib_block, tex, count=1)

    io.open(out_path, "w", encoding="utf-8").write(tex)
    n = bib_block.count(r"\bibitem")
    print(f"wrote {out_path}  ({n} references inlined)")


if __name__ == "__main__":
    # journal version: .bbl produced with IEEEtran.bst
    inline("main.tex", "main.bbl", "main_ieee.tex")
    # preprint version: .bbl produced with unsrt.bst
    inline("arxiv.tex", "arxiv.bbl", "arxiv_selfcontained.tex")
