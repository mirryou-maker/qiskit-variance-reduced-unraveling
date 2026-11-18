"""Rev7 (review 7, M3): numerical check of the closed-form amplitude-damping increment variance against the code.

Closed form (derived for the paper, Sec. II C / Sec. S2 of the Supplemental Material). Amplitude damping with
g = 1 - exp(-gt), K0 = diag(1, sqrt(1-g)), K1 = sqrt(g) s-, s- = |0><1| on the damped qubit; unitary mixing
K'0 = cos(th) K0 + e^{i ph} sin(th) K1, K'1 = sin(th) K0 - e^{i ph} cos(th) K1 (th = 0 counting, th = pi/4 homodyne).
With m = <E^dag(Ot)>, p1 = <s+ s->, C = <s+ Ot s-> - m p1, B = <K0 Ot s-> - m <s->, and w'0 = <K'0^dag K'0>,
    V(th, ph) = g [ sin(2 th) Re(e^{i ph} B) - sqrt(g) cos(2 th) C ]^2 / ( w'0 (1 - w'0) )        (exact).
Special cases: V_count = g C^2 / (p1 (1 - g p1)),  V_hom(ph) = 4 g [Re(e^{i ph} B)]^2 / (1 - 4 g [Re(e^{i ph}<s->)]^2).
Leading order in g: homodyne at phase ph beats counting iff 4 p1 [Re(e^{i ph} B)]^2 < C^2.
Exact (multiply through by the positive denominators; p1 > 0, 0 < g < 1, |<s->| <= 1/2):
    V_hom(ph) < V_count  iff  4 p1 (1 - g p1) [Re(e^{i ph} B)]^2 < C^2 {1 - 4 g [Re(e^{i ph}<s->)]^2}.
Zero-variance angle: the bracket of V(th, ph) is -sqrt(g) C at th = 0 and +sqrt(g) C at th = pi/2, so it vanishes at
th* in [0, pi/2) with 2 th* = atan2(sqrt(g) C, Re(e^{i ph} B)) mod pi (tan 2 th* = sqrt(g) C / Re(e^{i ph} B));
th* < pi/4 iff C Re(e^{i ph} B) > 0.  (th, ph + pi) is a relabeling of (pi/2 - th, ph).

The code's increment variances (traj_kraus._NR + increment_vars, the functions used by greedy and the static tables)
are compared with the closed form for random states and random Hermitian Ot (n = 3..6), and for one realistic case
(brickwork circuit, back-propagated observable from backprop_local, trajectory states from a counting run).
Added after the independent Sec. II D check (2026-11-18; the first seven output lines are unchanged): the exact
criterion on the same 640 cases; exact vs leading-order criteria against the computed comparison (code
increment_vars, cross-checked by a direct dense evaluation of Eq. (2)) on one random ensemble at several gt; the
zero-variance angle th* (random cases and the realistic events); the relabeling symmetry; and the p1 = 0 edge case.
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


def theta_star(C, R, g):
    """Zero-variance angle in [0, pi/2): sin(2 th) R - sqrt(g) cos(2 th) C = 0."""
    return (np.arctan2(np.sqrt(g) * C, R) % np.pi) / 2


def brute_V(psi, Ot, i, n, Ks):
    """Eq. (2) directly from dense Kraus operators (independent of traj_kraus); terms with w = 0 contribute 0."""
    lo, hi = 2 ** i, 2 ** (n - i - 1)
    full = [np.kron(np.kron(np.eye(lo), Kq), np.eye(hi)) for Kq in Ks]
    w = [np.real(np.vdot(Kf @ psi, Kf @ psi)) for Kf in full]
    nu = [np.real(np.vdot(Kf @ psi, Ot @ (Kf @ psi))) for Kf in full]
    m = sum(nu)
    return sum((nu[j] - w[j] * m) ** 2 / w[j] for j in range(len(Ks)) if w[j] > 1e-14)


rng = np.random.default_rng(20261118)
log("== Random states and random Hermitian Ot ==")
worst = 0.0
lo_agree = []
ex_agree = []
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
                    Rs = np.real(np.exp(1j * phs[c]) * info["sm"])
                    exf = (4 * info["p1"] * (1 - g * info["p1"]) * np.real(np.exp(1j * phs[c]) * info["B"]) ** 2
                           < info["C"] ** 2 * (1 - 4 * g * Rs ** 2))
                    ex_agree.append(exact == exf)
log(f"cases: n = 3..6, 40 random (psi, Ot) each, gt in (0.001, 0.01, 0.1, 1.0), 11 candidates")
log(f"max relative |V_code - V_closed| over all candidates: {worst:.2e}")
log(f"leading-order criterion vs exact comparison (gt = 0.01, 4 phases): agree in {sum(lo_agree)} of {len(lo_agree)}")

log("== Realistic case: brickwork s11, n = 8, L = 16, gt = 0.01, exact back-propagated Ot, counting trajectories ==")
n, L, gt = 8, 16, 0.01
ops = K.build("brickwork", "s11", n, L, gt, "ad_mix")
worst2, nev, nchk = 0.0, 0, 0
best_counts = np.zeros(11, int)
real_side, real_zero_worst, real_zero_n = 0, 0.0, 0
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
                    Vf, info = closed_form(psi, Ot, q, n, gt, ths[c], phs[c])
                    if abs(Vc[c]) > 1e-14:
                        worst2 = max(worst2, abs(Vf - Vc[c]) / abs(Vc[c]))
                        nchk += 1
                best_counts[int(np.argmin(Vc))] += 1
                # zero-variance angle at the E13-mix phase pi/4 (info from the last closed_form call, ph = pi/4)
                Rb = np.real(np.exp(1j * K.AD_MIX_PHI) * info["B"])
                real_side += int(info["C"] * Rb > 0)
                ths_ = theta_star(info["C"], Rb, g)
                Vz, _ = code_V(psi, Ot, q, n, np.asarray([K.ad_mix_set(gt, ths_, K.AD_MIX_PHI)]))
                if Vc[0] > 1e-14:
                    real_zero_worst = max(real_zero_worst, Vz[0] / Vc[0])
                    real_zero_n += 1
                nev += 1
                psi = K.apply_kraus(psi[None, :], q, n, np.asarray(M.cands[0])[None], r)[0]
            else:
                psi = K.T.apply_gate(psi[None, :], op, n, mats)[0]
log(f"events checked: {nev}; candidate values compared (V > 1e-14): {nchk}; max relative difference {worst2:.2e}")
log("per-event argmin over the 11 candidates (counts; 0 counting, 1-4 homodyne, 5-10 mixed): " + " ".join(map(str, best_counts)))
log(f"zero-variance angle th* at ph = pi/4 on these events: C Re(e^(i pi/4) B) > 0 (th* < pi/4) in {real_side} of {nev}; "
    f"max V(th*)/V_count over {real_zero_n} events with V_count > 1e-14: {real_zero_worst:.2e}")

log("== Exact criterion (added after the Sec. II D check) ==")
log(f"exact criterion vs exact comparison on the 640 cases above (gt = 0.01, 4 phases): agree in {sum(ex_agree)} of {len(ex_agree)}")

rng2 = np.random.default_rng(20261119)
NS, n2 = 2000, 3
ens = []
for _ in range(NS):
    D = 2 ** n2
    psi = rng2.normal(size=D) + 1j * rng2.normal(size=D)
    psi /= np.linalg.norm(psi)
    A = rng2.normal(size=(D, D)) + 1j * rng2.normal(size=(D, D))
    ens.append((psi, (A + A.conj().T) / 2, int(rng2.integers(n2))))
PHS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
log(f"ensemble: {NS} random (psi, Ot, noisy qubit) at n = {n2} (default_rng(20261119)), same ensemble at every gt; "
    f"4 homodyne phases -> {4 * NS} comparisons per gt; computed comparison = code increment_vars (candidates 0-4)")
for gt in (0.001, 0.01, 0.1, 0.5, 1.0):
    g = 1.0 - np.exp(-gt)
    ch = K.amp_damp_phase(gt)
    n_ex = n_lo = n_bf = tot = 0
    for psi, Ot, i in ens:
        Vc, _ = code_V(psi, Ot, i, n2, ch.cands)
        _, info = closed_form(psi, Ot, i, n2, gt, 0.0, 0.0)
        Vb0 = brute_V(psi, Ot, i, n2, [np.asarray(ch.cands[0][j]) for j in range(2)])
        for c, ph in enumerate(PHS, start=1):
            comp = Vc[c] < Vc[0]
            Vbc = brute_V(psi, Ot, i, n2, [np.asarray(ch.cands[c][j]) for j in range(2)])
            R = np.real(np.exp(1j * ph) * info["B"])
            Rs = np.real(np.exp(1j * ph) * info["sm"])
            exf = 4 * info["p1"] * (1 - g * info["p1"]) * R ** 2 < info["C"] ** 2 * (1 - 4 * g * Rs ** 2)
            lead = 4 * info["p1"] * R ** 2 < info["C"] ** 2
            n_ex += int(exf == comp)
            n_lo += int(lead != comp)
            n_bf += int((Vbc < Vb0) == comp)
            tot += 1
    log(f"gt = {gt}: exact criterion agrees in {n_ex} of {tot}; leading-order criterion disagrees in {n_lo} of {tot}; "
        f"direct dense Eq. (2) comparison agrees with code comparison in {n_bf} of {tot}")

log("== Zero-variance angle th* (tan 2 th* = sqrt(g) C / Re(e^{i ph} B), th* in [0, pi/2)) ==")
rng3 = np.random.default_rng(20261120)
worst_code = worst_brute = sym_worst = 0.0
side_ok = ntot = n_low = 0
for trial in range(400):
    n3 = int(rng3.integers(2, 6))
    D = 2 ** n3
    psi = rng3.normal(size=D) + 1j * rng3.normal(size=D)
    psi /= np.linalg.norm(psi)
    A = rng3.normal(size=(D, D)) + 1j * rng3.normal(size=(D, D))
    Ot = (A + A.conj().T) / 2
    i = int(rng3.integers(n3))
    gt = float(rng3.choice([0.001, 0.01, 0.1, 0.5, 1.0]))
    g = 1.0 - np.exp(-gt)
    ph = float(rng3.uniform(0, np.pi))
    _, info = closed_form(psi, Ot, i, n3, gt, 0.0, ph)
    R = np.real(np.exp(1j * ph) * info["B"])
    ths_ = theta_star(info["C"], R, g)
    Vcnt = brute_V(psi, Ot, i, n3, K.ad_mix_set(gt, 0.0, ph))
    Vz_b = brute_V(psi, Ot, i, n3, K.ad_mix_set(gt, ths_, ph))
    Vz_c, _ = code_V(psi, Ot, i, n3, np.asarray([K.ad_mix_set(gt, ths_, ph)]))
    worst_brute = max(worst_brute, Vz_b / Vcnt)
    worst_code = max(worst_code, Vz_c[0] / Vcnt)
    side_ok += int((ths_ < np.pi / 4) == (info["C"] * R > 0))
    n_low += int(ths_ < np.pi / 4)
    th_r = float(rng3.uniform(0, np.pi / 2))
    Va = brute_V(psi, Ot, i, n3, K.ad_mix_set(gt, np.pi / 2 - th_r, ph))
    Vb = brute_V(psi, Ot, i, n3, K.ad_mix_set(gt, th_r, ph + np.pi))
    sym_worst = max(sym_worst, abs(Va - Vb) / max(abs(Va), 1e-300))
    ntot += 1
log(f"{ntot} random cases (n = 2..5, gt in (0.001, 0.01, 0.1, 0.5, 1.0), ph uniform in [0, pi)): "
    f"max V(th*)/V_count {worst_brute:.2e} (direct dense Eq. (2)), {worst_code:.2e} (code increment_vars)")
log(f"th* < pi/4 iff C Re(e^(i ph) B) > 0: holds in {side_ok} of {ntot} (th* < pi/4 in {n_low})")
log(f"relabeling V(pi/2 - th, ph) = V(th, ph + pi), random th: max relative difference {sym_worst:.2e}")

log("== Edge case p1 = 0 (noisy qubit in |0>) ==")
psi = np.kron(np.array([1.0, 0.0]), np.array([0.6, 0.8j])).astype(complex)
A = rng3.normal(size=(4, 4)) + 1j * rng3.normal(size=(4, 4))
Ot = (A + A.conj().T) / 2
Vc, _ = code_V(psi, Ot, 0, 2, K.amp_damp_phase(0.01).cands)
log("V (counting, 4 homodyne phases) at gt = 0.01: " + " ".join(f"{v:.1e}" for v in Vc))
out.close()
