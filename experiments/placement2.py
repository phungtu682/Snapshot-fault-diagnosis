"""DMP-2: sensor placement by minimising a Bhattacharyya bound on the fault-type error.

With noise independent between PMUs, squared noise-normalised distances add over sensors. For an
event e (fault bus, R_f), D_e^2(x) = sum_b x_b m_eb^2, where m_eb is the smallest distance between two
fault types at bus b; the sum of per-bus minima lower-bounds the true minimum over pairs, so the bound
P_err(e) <= 0.5 exp(-D_e^2 / 8) stays valid under the Gaussian equal-covariance approximation.
sum_e w_e (1 - exp(-D_e^2/8)) is monotone submodular in the sensor set (concave of a modular function),
so greedy selection has the (1 - 1/e) guarantee; a MILP with tangent cuts refines it where tractable.
Uses only the simulation design grid: no labelled data, no classifier, no threshold."""
from itertools import combinations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, vstack

A = np.exp(2j * np.pi / 3)


def signature(Vp, Vf):
    """Per-bus FPS-consistent signature (..., 3) phasors -> (..., 8)."""
    with np.errstate(all="ignore"):
        ratio = np.nan_to_num(np.abs(Vf) / np.abs(Vp), nan=1.0)
        shift = np.nan_to_num(np.angle(Vf / Vp), nan=0.0)
        a, b, c = Vf[..., 0], Vf[..., 1], Vf[..., 2]
        V0, V1, V2 = (a + b + c) / 3, (a + A * b + A ** 2 * c) / 3, (a + A ** 2 * b + A * c) / 3
        z0 = np.nan_to_num(np.abs(V0) / np.abs(V1), nan=0.0)
        z2 = np.nan_to_num(np.abs(V2) / np.abs(V1), nan=0.0)
    return np.concatenate([ratio, shift, z0[..., None], z2[..., None]], -1)


def margins2(cols, Vp, Vf, meta, mag_sigma=0.0032951, ang_sigma_deg=0.18879, draws=40, seed=0):
    """Returns events, squared margins (E, B) and prior weights (E,): R_f weight is the log-width of its bin."""
    rng = np.random.RandomState(seed)
    P_, B = len(meta), len(cols)
    Mp, Mf = np.full((P_, B, 3), np.nan + 0j), np.full((P_, B, 3), np.nan + 0j)
    v = cols >= 0
    Mp[:, v], Mf[:, v] = Vp[:, cols[v]], Vf[:, cols[v]]
    mu = signature(Mp, Mf)
    acc, acc2 = np.zeros_like(mu), np.zeros_like(mu)
    for _ in range(draws):
        n1 = (1 + rng.normal(0, mag_sigma, Mp.shape)) * np.exp(1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mp.shape)))
        n2 = (1 + rng.normal(0, mag_sigma, Mf.shape)) * np.exp(1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mf.shape)))
        s = signature(Mp * n1, Mf * n2)
        acc += s
        acc2 += s ** 2
    sd = np.sqrt(np.maximum(acc2 / draws - (acc / draws) ** 2, 0)) + 1e-4
    groups = {}
    for i, (fb, c, rf) in enumerate(meta):
        groups.setdefault((fb, rf), []).append((c, i))
    events = sorted(e for e, g in groups.items() if len({c for c, _ in g}) > 1)
    m2 = np.full((len(events), B), np.inf)
    for k, e in enumerate(events):
        for (c1, i), (c2, j) in combinations(groups[e], 2):
            if c1 != c2:
                pooled = (sd[i] ** 2 + sd[j] ** 2) / 2
                m2[k] = np.minimum(m2[k], (((mu[i] - mu[j]) ** 2) / pooled).sum(1))
    rfs = np.array(sorted({e[1] for e in events}))
    edges = np.sqrt(rfs[:-1] * rfs[1:])
    width = np.diff(np.log(np.concatenate([[rfs[0]], edges, [rfs[-1]]])))
    wmap = dict(zip(rfs, width))
    w = np.array([wmap[e[1]] for e in events])
    return events, m2, w / w.sum()


def bound(m2, w, obs):
    """Weighted Bhattacharyya error bound of a sensor set."""
    return float((w * 0.5 * np.exp(-m2[:, list(obs)].sum(1) / 8)).sum())


def bound_n1(m2, w, obs):
    """Mean bound after losing one sensor (uniform over the set)."""
    return float(np.mean([bound(m2, w, [o for o in obs if o != x]) for x in obs]))


def _greedy_swap(score, n, k):
    chosen = []
    for _ in range(k):
        chosen.append(min((b for b in range(n) if b not in chosen), key=lambda b: score(chosen + [b])))
    improved = True
    while improved:
        improved = False
        for i in range(k):
            for b in range(n):
                if b not in chosen:
                    c = chosen[:i] + [b] + chosen[i + 1:]
                    if score(c) < score(chosen) - 1e-12:
                        chosen, improved = c, True
    return sorted(chosen)


def dmp2(m2, w, k, n1=False, time_limit=60):
    """Exact-as-possible minimiser. N-1 uses greedy+swap (the worst-sensor term is not linear)."""
    n = m2.shape[1]
    if n1:
        return _greedy_swap(lambda o: bound_n1(m2, w, o), n, k), None
    start = _greedy_swap(lambda o: bound(m2, w, o), n, k)
    # tangent cuts of 0.5 exp(-D/8): t_e >= f(D0) + f'(D0) (D - D0)
    E = len(w)
    rows, lo = [], []
    A_ = csr_matrix(m2)
    for D0 in (0, 4, 8, 16, 32, 64):
        f, g = 0.5 * np.exp(-D0 / 8), -0.5 / 8 * np.exp(-D0 / 8)
        rows.append(hstack([csr_matrix(-g * A_), identity(E, format="csr")]))
        lo.append(np.full(E, f - g * D0))
    Ac = vstack(rows + [hstack([csr_matrix(np.ones((1, n))), csr_matrix((1, E))])]).tocsr()
    lb = np.concatenate(lo + [[k]])
    ub = np.concatenate([np.full(len(lb) - 1, np.inf), [k]])
    c = np.concatenate([np.zeros(n), w])
    res = milp(c, constraints=LinearConstraint(Ac, lb, ub), integrality=np.concatenate([np.ones(n), np.zeros(E)]),
               bounds=Bounds(0, np.concatenate([np.ones(n), np.full(E, 1.0)])),
               options={"time_limit": time_limit})
    gap = getattr(res, "mip_gap", None)
    if res.x is not None:
        cand = sorted(np.where(res.x[:n] > 0.5)[0].tolist())
        if len(cand) == k and bound(m2, w, cand) < bound(m2, w, start):
            start = cand
    return start, gap
