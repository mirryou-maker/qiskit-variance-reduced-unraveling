"""Generate paper figures from the measured Track-A data (English labels).

Data values are the measured results recorded in bench/results/*.csv:
  a3_sweep.csv (regime map, n-scaling), a4_vs_aer.csv, a1_convergence.csv, c_caveats.csv.
Outputs PNGs into this directory. Colorblind-safe palette; large fonts.

Run:  python -u /mnt/d/Claude-Code-R/Qiskit-CO/paper/figures/make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(__file__)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
    "grid.alpha": 0.3, "figure.autolayout": True,
})
# Okabe-Ito colorblind-safe
C = {"std": "#000000", "proj": "#0072B2", "analog": "#D55E00",
     "aer": "#009E73", "ref": "#999999"}


def fig1_concept():
    """Conceptual diagram: the Qiskit(-Aer) stack and where this work intervenes."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 6.0); ax.axis("off"); ax.grid(False)

    GREY, EDGE = "#EDEDED", "#666666"
    HL = "#D6E9F8"          # highlight fill for our contributions
    HLE = C["proj"]          # highlight edge

    def box(x, y, w, h, text, fill=GREY, edge=EDGE, lw=1.4, fs=10.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=fill, edgecolor=edge, linewidth=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, linespacing=1.35)

    def arrow(x, y0, y1):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
                                     color=EDGE, linewidth=1.3))

    X, W = 0.35, 4.0
    # --- stack ---
    box(X, 5.15, W, 0.62, "Quantum circuit / algorithm\n(VQE, QAOA, error mitigation)")
    arrow(X + W / 2, 5.13, 4.87)
    box(X, 4.20, W, 0.62, "NoiseModel — device physics\n($T_1$, $T_2$, Pauli-Lindblad, crosstalk)")
    arrow(X + W / 2, 4.18, 3.95)

    # Aer backend container
    ax.add_patch(FancyBboxPatch((X - 0.12, 0.95), W + 0.24, 2.95,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#FAFAFA", edgecolor=EDGE, linewidth=1.6, linestyle="--"))
    ax.text(X + W / 2, 3.72, "Qiskit-Aer simulation backend", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#333333")

    box(X + 0.15, 2.90, W - 0.30, 0.60, "Noise channel representation\n(Kraus / Lindblad)",
        fill=HL, edge=HLE, lw=2.0)
    arrow(X + W / 2, 2.88, 2.68)
    box(X + 0.15, 2.02, W - 0.30, 0.62, "Unraveling & trajectory sampling",
        fill=HL, edge=HLE, lw=2.0, weight="bold")
    arrow(X + W / 2, 2.00, 1.80)
    box(X + 0.15, 1.15, W - 0.30, 0.62, "GPU dense-statevector kernel")

    arrow(X + W / 2, 0.93, 0.75)
    box(X, 0.10, W, 0.62, "Estimator: expectation values / counts")

    # --- annotations ---
    def note(x, y, txt, color, edge_color):
        ax.text(x, y, txt, ha="left", va="center", fontsize=10.2, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor=edge_color,
                          linewidth=1.6))

    def link(x0, y0, x1, y1, color):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                     color=color, linewidth=1.8,
                                     connectionstyle="arc3,rad=-0.12"))

    link(X + W + 0.10, 3.20, 5.20, 3.55, HLE)
    note(5.35, 3.62,
         "② THIS WORK — diagnosis\nAer canonicalizes the channel here,\n"
         "discarding any user-supplied unraveling\n→ blocker identified + minimal fix",
         HL, HLE)

    link(X + W + 0.10, 2.33, 5.20, 2.05, HLE)
    note(5.35, 2.05,
         "① THIS WORK — method\nvariance-reduced unraveling\n(projector / analog)\n"
         "→ $\\sim$21$\\times$ fewer trajectories,\n    same verified accuracy",
         HL, HLE)

    link(X + W + 0.10, 1.46, 5.20, 0.72, "#999999")
    note(5.35, 0.72,
         "prior GPU work optimizes here\n(cache blocking, multi-shot batching,\ncuStateVec, ROCm)",
         "#F2F2F2", "#999999")

    fig.savefig(os.path.join(OUT, "fig1_concept.png"))
    plt.close(fig)


def fig1_regime_map():
    gt = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00, 1.50]
    proj = [0.439, 0.385, 0.325, 0.242, 0.198, 0.127, 0.045]
    analog = [0.206, 0.313, 0.371, 0.417, 0.492, 0.534, 0.477]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gt, proj, "o-", color=C["proj"], lw=2.5, ms=8, label="projector")
    ax.plot(gt, analog, "s-", color=C["analog"], lw=2.5, ms=8, label="analog")
    ax.axhline(1.0, color=C["std"], ls="--", lw=1.5, label="standard (baseline = 1)")
    ax.axvline(0.35, color=C["ref"], ls=":", lw=1.5)
    ax.text(0.36, 0.62, r"crossover $\approx$ 0.35", color="#555", fontsize=11)
    ax.set_xlabel(r"noise strength  $\gamma t$  (larger = stronger noise)")
    ax.set_ylabel("variance ratio vs. standard  (lower is better)")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper center", ncol=3, fontsize=10, framealpha=0.9,
              columnspacing=1.0, handletextpad=0.4)
    fig.savefig(os.path.join(OUT, "fig3_regime_map.png"))
    plt.close(fig)


def fig2_vs_aer():
    labels = ["Qiskit-Aer\n(standard)", "ours\n(standard)", "ours\n(projector)"]
    N = [998, 997, 48]
    # lighter fills with darker edges keep the annotation legible over the bars
    fills = ["#8FD3C2", "#BFBFBF", C["proj"]]
    edges = [C["aer"], "#4D4D4D", C["proj"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, N, color=fills, edgecolor=edges, linewidth=1.8, width=0.6)
    for b, v in zip(bars, N):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    # annotation placed in the empty upper-right area, with a boxed white background
    ax.annotate(r"$\mathbf{\sim}$21$\times$ fewer trajectories" + "\nfor the same accuracy",
                xy=(2, 90), xytext=(1.62, 780),
                ha="center", va="center", fontsize=12.5, fontweight="bold", color="#B00020",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#B00020", linewidth=1.8),
                arrowprops=dict(arrowstyle="-|>", color="#B00020", lw=2.2,
                                shrinkA=6, shrinkB=4,
                                connectionstyle="arc3,rad=0.18"))
    ax.set_ylabel("trajectories to reach target standard error\n(lower is better)")
    ax.set_ylim(0, 1200)
    fig.savefig(os.path.join(OUT, "fig6_vs_aer.png"))
    plt.close(fig)


def fig3_scaling():
    n = [8, 12, 16, 20]
    speedup = [19.1, 17.7, 25.7, 25.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n, speedup, "D-", color=C["proj"], lw=2.5, ms=10)
    ax.axhline(1.0, color=C["std"], ls="--", lw=1.5, label="no gain (1x)")
    for xi, yi in zip(n, speedup):
        ax.text(xi, yi + 1.2, f"{yi:.0f}x", ha="center", fontsize=11)
    ax.set_xlabel(r"qubit count  $n$  (state dimension $2^n$)")
    ax.set_ylabel("projector trajectory-reduction factor")
    ax.set_xticks(n)
    ax.set_ylim(0, 30)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(OUT, "fig4_scaling.png"))
    plt.close(fig)


def fig4_convergence():
    N = np.array([200, 1000, 5000, 20000, 60000])
    td = np.array([0.10072, 0.02555, 0.01676, 0.00618, 0.00417])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N, td, "o-", color=C["proj"], lw=2.5, ms=8, label="measured error")
    ref = td[0] * np.sqrt(N[0] / N)
    ax.loglog(N, ref, "--", color=C["ref"], lw=2, label=r"theory ($1/\sqrt{N}$)")
    ax.set_xlabel(r"number of trajectories  $N$  (log scale)")
    ax.set_ylabel("trace distance to exact $\\rho$  (log, lower is better)")
    ax.legend()
    fig.savefig(os.path.join(OUT, "fig2_convergence.png"))
    plt.close(fig)


def fig5_erosion():
    theta = [0.0, 0.3, 0.6, 1.0, 1.571]
    speedup = [19.7, 9.2, 2.5, 1.8, 3.5]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(theta, speedup, "o-", color=C["proj"], lw=2.5, ms=9)
    ax.axhline(1.0, color=C["std"], ls="--", lw=1.5, label="no gain (1x)")
    ax.set_xlabel(r"interleaved coherent rotation  $\theta$  (more 'compute')")
    ax.set_ylabel("projector speedup factor")
    ax.annotate("idle / decoherence\ndominated (max gain)", xy=(0.0, 19.7),
                xytext=(0.25, 16), fontsize=11, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("compute\ndominated (gain shrinks)", xy=(1.0, 1.8),
                xytext=(0.7, 6), fontsize=11, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_ylim(0, 22)
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUT, "fig5_erosion.png"))
    plt.close(fig)


def main():
    fig1_concept()
    fig1_regime_map()
    fig2_vs_aer()
    fig3_scaling()
    fig4_convergence()
    fig5_erosion()
    print("figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print("  ", f)


if __name__ == "__main__":
    main()
