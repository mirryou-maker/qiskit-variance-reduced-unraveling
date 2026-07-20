r"""Generate an arXiv-style preprint (arxiv.tex) from the IEEE TQE source (main.tex).

The body text is reused verbatim; only the IEEE-specific preamble and front/back matter are
replaced by a plain single-column `article` layout. Re-run whenever main.tex changes:

    python make_arxiv.py
    pdflatex arxiv ; bibtex arxiv ; pdflatex arxiv ; pdflatex arxiv
"""
import io
import re

SRC, DST = "main.tex", "arxiv.tex"
s = io.open(SRC, encoding="utf-8").read()

# ---- extract the pieces we keep -----------------------------------------
abstract = s[s.index(r"\begin{abstract}") + len(r"\begin{abstract}"):
             s.index(r"\end{abstract}")].strip()

kw = s[s.index(r"\begin{keywords}") + len(r"\begin{keywords}"):
       s.index(r"\end{keywords}")].strip()

body_start = s.index(r"\section{Introduction}")
body_end = s.index(r"\bibliographystyle{IEEEtran}")
body = s[body_start:body_end]

# (the author biography is intentionally omitted from the preprint)

# ---- body fixes for a one-column article --------------------------------
body = body.replace(r"\PARstart{C}{lassical}", "Classical")
body = body.replace(r"\begin{figure*}", r"\begin{figure}")
body = body.replace(r"\end{figure*}", r"\end{figure}")
body = body.replace(r"\columnwidth", r"\linewidth")
# the wide concept figure was sized for a two-column spread
body = body.replace(r"\includegraphics[width=0.92\textwidth]{fig1_concept.png}",
                    r"\includegraphics[width=\linewidth]{fig1_concept.png}")
# tables were tightened for the narrow IEEE column; relax them here
body = body.replace(r"\footnotesize" + "\n" + r"\setlength{\tabcolsep}{3.5pt}", "")
body = body.replace(r"\footnotesize" + "\n" + r"\setlength{\tabcolsep}{4pt}", "")
# allowbreak hints are unnecessary at this measure
body = body.replace(r"\texttt{batched\_\allowbreak shots\_\allowbreak gpu}",
                    r"\texttt{batched\_shots\_gpu}")
body = body.replace(r"\texttt{qiskit.\allowbreak quantum\_info.\allowbreak Statevector}",
                    r"\texttt{qiskit.quantum\_info.Statevector}")
abstract = abstract.replace(r"\texttt{batched\_\allowbreak shots\_\allowbreak gpu}",
                            r"\texttt{batched\_shots\_gpu}")
# IEEE cross-reference style -> plain
body = re.sub(r"Sec\.~\\ref", r"Section~\\ref", body)
# let floats settle near their callouts instead of drifting to the end
body = body.replace(r"\begin{figure}[t]", r"\begin{figure}[!htbp]")
body = body.replace(r"\begin{table}[t]", r"\begin{table}[!htbp]")
# keep the preprint venue-neutral (it may end up at a non-IEEE journal)
body = body.replace("In accordance with IEEE policy, the author discloses",
                    "The author discloses")

# ---- assemble -----------------------------------------------------------
out = r"""%% arXiv preprint version, generated from main.tex by make_arxiv.py -- do not edit directly.
\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage{placeins}

% keep floats close to where they are discussed
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\let\oldsection\section
\renewcommand{\section}{\FloatBarrier\oldsection}

\newcommand{\ket}[1]{\lvert #1 \rangle}
\newcommand{\bra}[1]{\langle #1 \rvert}
\newcommand{\expv}[1]{\langle #1 \rangle}

\title{\bfseries Variance-Reduced Trajectory Unravelings for GPU Noisy Quantum-Circuit
Simulation: Characterization and a Qiskit-Aer Integration Gap}

\author{Chun-Yeol You\thanks{Department of Physics and Chemistry, DGIST, Daegu 42988,
Republic of Korea. ORCID: 0000-0001-9549-8611. E-mail: \texttt{cyyou@dgist.ac.kr}}}

\date{\today}

\begin{document}
\maketitle

\begin{abstract}
%(ABSTRACT)
\end{abstract}

\noindent\textbf{Keywords:} %(KEYWORDS)

\vspace{1em}

%(BODY)

\bibliographystyle{unsrt}
\bibliography{references}

\end{document}
"""
out = out.replace("%(ABSTRACT)", abstract)
out = out.replace("%(KEYWORDS)", kw)
out = out.replace("%(BODY)", body)

io.open(DST, "w", encoding="utf-8").write(out)
print("wrote", DST, "-", len(out), "chars")
