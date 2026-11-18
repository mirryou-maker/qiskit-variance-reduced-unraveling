"""Rev7 (review 7, M3): numerical check of the closed-form amplitude-damping increment variance against the code.

Closed form (derived for the paper, Sec. II C / Sec. S2 of the Supplemental Material). Amplitude damping with
g = 1 - exp(-gt), K0 = diag(1, sqrt(1-g)), K1 = sqrt(g) s-, s- = |0><1| on the damped qubit; unitary mixing
K'0 = cos(th) K0 + e^{i ph} sin(th) K1, K'1 = sin(th) K0 - e^{i ph} cos(th) K1 (th = 0 counting, th = pi/4 homodyne).
With m = <E^dag(Ot)>, p1 = <s+ s->, C = <s+ Ot s-> - m p1, B = <K0 Ot s-> - m <s->, and w'0 = <K'0^dag K'0>,
    V(th, ph) = g [ sin(2 th) Re(e^{i ph} B) - sqrt(g) cos(2 th) C ]^2 / ( w'0 (1 - w'0) )        (exact).
Special cases: V_count = g C^2 / (p1 (1 - g p1)),  V_hom(ph) = 4 g [Re(e^{i ph} B)]^2 / (1 - 4 g [Re(e^{i ph}<s->)]^2).
Leading order in g: homodyne at phase ph beats counting iff 4 p1 [Re(e^{i ph} B)]^2 < C^2.

The code's increment variances (traj_kraus._NR + increment_vars, the functions used by greedy and the static tables)
are compared with the closed form for random states and random Hermitian Ot (n = 3..6), and for one realistic case
(brickwork circuit, back-propagated observable from backprop_local, trajectory states from a counting run).
CPU only (numpy); run on ubuntu94 in env `sci` (no cupy -> numpy backend).
Usage: python pra_rev7_ad_criterion_check.py <repo root> <output .txt>
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from src.noise import traj_kraus as K  # noqa: E402

assert K.xp is np, "run without cupy (numpy backend)"
out = open(sys.argv[2], "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


def closed_form(psi, Ot, i, n, gt, th, ph):
    """V(th, ph) from the closed form, for one state psi (D,) and dense Hermitian Ot (D, D), damped qubit i."""
    g = 1.0 - np.exp(-gt)
    D = 2 ** n
    lo, hi = 2 ** i, 2 ** (n - i - 1)

    def onq(M):  # embed a 2x2 operator on qubit i (same index convention as traj_kraus._NR)
        return np.kron(np.kron(np.eye(lo), M), np.eye(hi))

    sm = onq(np.array([[0, 1], [0, 0]], complex))
    K0 = onq(np.diag([1.0, np.sqrt(1 - g)]).astype(complex))
    K1 = np.sqrt(g) * sm
    ev = lambda A: np.vdot(psi, A @ psi)
    m = np.real(ev(K0.conj().T @ Ot @ K0 + K1.conj().T @ Ot @ K1))
    p1 = np.real(ev(sm.conj().T @ sm))
    C = np.real(ev(sm.conj().T @ Ot @ sm)) - m * p1
    B = ev(K0 @ Ot @ sm) - m * ev(sm)
    c, s, e = np.cos(th), np.sin(th), np.exp(1j * ph)
    Kp0 = c * K0 + e * s * K1
    w0 = np.real(ev(Kp0.conj().T @ Kp0))
    num = g * (np.sin(2 * th) * np.real(e * B) - np.sqrt(g) * np.cos(2 * th) * C) ** 2
    return num / (w0 * (1 - w0)), dict(m=m, p1=p1, C=C, B=B, g=g, sm=ev(sm))


def code_V(psi, Ot, i, n, cands):
    N, R = K._NR(psi[None, :], Ot, i, n)
    V, m, _ = K.increment_vars(N, R, np.asarray(cands))
    return V[0], m[0]


rng = np.random.default_rng(20261118)
log("== Random states and random Hermitian Ot ==")
worst = 0.0
lo_agree = []
for n in (3, 4, 5, 6):
    for trial in range(40):
        D = 2 ** n
        psi = rng.normal(size=D) + 1j * rng.normal(size=D)
        psi /= np.linalg.norm(psi)
        A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
        Ot = (A + A.conj().T) / 2
        i = int(rng.integers(n))
        for gt in (0.001, 0.01, 0.1, 1.0):
            ch = K.amp_damp_mix(gt)          # 0 counting, 1-4 homodyne phi in {0, pi/4, pi/2, 3pi/4}, 5-10 mixed at pi/4
            g = 1.0 - np.exp(-gt)
            ths = [0.0] + [np.pi / 4] * 4 + list(K.AD_MIX_THETAS) + [float(np.arctan(x * np.sqrt(g))) for x in K.AD_MIX_X2]
            phs = [0.0, 0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4] + [K.AD_MIX_PHI] * 6
            Vc, mc = code_V(psi, Ot, i, n, ch.cands)
            for c in range(len(ths)):
                Vf, info = closed_form(psi, Ot, i, n, gt, ths[c], phs[c])
                rel = abs(Vf - Vc[c]) / max(abs(Vc[c]), 1e-300)
                worst = max(worst, rel)
            assert abs(info["m"] - mc) < 1e-10
            # leading-order criterion: homodyne phase ph beats counting iff 4 p1 Re(e^{i ph} B)^2 < C^2
            if gt == 0.01:
                for c in range(1, 5):
                    exact = Vc[c] < Vc[0]
                    lead = 4 * info["p1"] * np.real(np.exp(1j * phs[c]) * info["B"]) ** 2 < info["C"] ** 2
                    lo_agree.append(exact == lead)
log(f"cases: n = 3..6, 40 random (psi, Ot) each, gt in (0.001, 0.01, 0.1, 1.0), 11 candidates")
log(f"max relative |V_code - V_closed| over all candidates: {worst:.2e}")
log(f"leading-order criterion vs exact comparison (gt = 0.01, 4 phases): agree in {sum(lo_agree)} of {len(lo_agree)}")

log("== Realistic case: brickwork s11, n = 8, L = 16, gt = 0.01, exact back-propagated Ot, counting trajectories ==")
n, L, gt = 8, 16, 0.01
ops = K.build("brickwork", "s11", n, L, gt, "ad_mix")
worst2, nev, nchk = 0.0, 0, 0
best_counts = np.zeros(11, int)
for obs in ("x", "zz", "xc", "zc"):
    O = K.T.obs_matrix(obs, n)
    after, _ = K.backprop_local(ops, n, O, host=False)
    mats = {id(op[2]): np.asarray(op[2]) for op in ops if op[0] != "ch"}
    # forward counting trajectories; at each channel event compare code and closed form on the pre-event state
    for traj in range(4):
        r = np.random.default_rng(1000 + traj)
        psi = np.zeros(2 ** n, complex)
        psi[0] = 1.0
        k = 0
        for op in ops:
            kind, q, M = op
            if kind == "ch":
                Ot = np.asarray(after[k], complex)
                k += 1
                g = 1.0 - np.exp(-gt)
                ths = [0.0] + [np.pi / 4] * 4 + list(K.AD_MIX_THETAS) + [float(np.arctan(x * np.sqrt(g))) for x in K.AD_MIX_X2]
                phs = [0.0, 0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4] + [K.AD_MIX_PHI] * 6
                Vc, _ = code_V(psi, Ot, q, n, M.cands)
                for c in range(11):
                    Vf, _ = closed_form(psi, Ot, q, n, gt, ths[c], phs[c])
                    if abs(Vc[c]) > 1e-14:
                        worst2 = max(worst2, abs(Vf - Vc[c]) / abs(Vc[c]))
                        nchk += 1
                best_counts[int(np.argmin(Vc))] += 1
                nev += 1
                psi = K.apply_kraus(psi[None, :], q, n, np.asarray(M.cands[0])[None], r)[0]
            else:
                psi = K.T.apply_gate(psi[None, :], op, n, mats)[0]
log(f"events checked: {nev}; candidate values compared (V > 1e-14): {nchk}; max relative difference {worst2:.2e}")
log("per-event argmin over the 11 candidates (counts; 0 counting, 1-4 homodyne, 5-10 mixed): " + " ".join(map(str, best_counts)))
out.close()
