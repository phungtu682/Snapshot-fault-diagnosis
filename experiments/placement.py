"""Sensor-placement methods for snapshot-based fault-type classification.

Proposed: Discriminability-Margin Placement (DMP). From a simulation-only design grid
(every fault bus x fault type x faulted phases x R_f level), the noise-normalised margin
between the closest signatures of two *different* fault types is computed at each
candidate bus. A bus "covers" a (location, R_f) event when that margin exceeds tau.
k sensors are placed by an exact max-coverage ILP; the N-1 variant requires every
covered event to be seen by at least two sensors. No classifier is trained.

Baselines: random, topology (farthest-point), observability ILP (PMU placement),
monitor-reach-area ILP (sag detection, Olguin & Bollen style), information greedy
(leave-one-out 1-NN error), accuracy greedy (wrapper), exhaustive optimum."""
from itertools import combinations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import simlib

RF_GRID = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
CLASSES = ["SLG", "LL", "3PH"]
A = np.exp(2j * np.pi / 3)


# ------------------------------------------------------------------ design grid
def design_grid(n_bus, rf_grid=RF_GRID):
    if hasattr(simlib, "precomputed_grid"):          # offline mode (simlib_offline.py, no OpenDSS)
        return simlib.precomputed_grid(n_bus, rf_grid)
    fd = simlib.Feeder(n_bus)
    Vp, Vf, meta = [], [], []
    for b in fd.buses:
        ph = fd.bus_phases[b]
        for c in CLASSES:
            need = {"SLG": 1, "LL": 2, "3PH": 3}[c]
            if len(ph) < need:
                continue
            for phs in combinations(ph, need):
                for rf in rf_grid:
                    p, f, ok = fd.fault(b, c, list(phs), rf)
                    if ok:
                        Vp.append(p)
                        Vf.append(f)
                        meta.append((b, c, rf))
    cols = np.array([[fd.node_index.get(f"{b}.{p}", -1) for p in (1, 2, 3)] for b in fd.buses])
    return fd.buses, cols, np.array(Vp), np.array(Vf), meta


def bus_signature(Vp, Vf):
    """Per-bus phase-agnostic signature (..., 3) phasors -> (..., 5): sorted dips, V0/V1, V2/V1."""
    with np.errstate(all="ignore"):
        dip = np.nan_to_num(1 - np.abs(Vf) / np.abs(Vp), nan=0.0)
        a, b, c = Vf[..., 0], Vf[..., 1], Vf[..., 2]
        V0, V1, V2 = (a + b + c) / 3, (a + A * b + A ** 2 * c) / 3, (a + A ** 2 * b + A * c) / 3
        z0 = np.nan_to_num(np.abs(V0) / np.abs(V1), nan=0.0)
        z2 = np.nan_to_num(np.abs(V2) / np.abs(V1), nan=0.0)
    return np.concatenate([-np.sort(-dip, -1), z0[..., None], z2[..., None]], -1)


def margins(cols, Vp, Vf, meta, mag_sigma=0.0032951, ang_sigma_deg=0.18879, draws=40, seed=0):
    """Noise-normalised class margin per (event, bus) and noise-free sag depth per (event, bus).
    Event = (fault bus, R_f). Returns event list, margin matrix (E, B) and sag matrix (E, B)."""
    rng = np.random.RandomState(seed)
    P, B = len(meta), len(cols)
    def take(V):
        M = np.full((P, B, 3), np.nan + 0j)
        valid = cols >= 0
        M[:, valid] = V[:, cols[valid]]
        return M
    Mp, Mf = take(Vp), take(Vf)
    mu = bus_signature(Mp, Mf)
    sd = np.zeros_like(mu)
    acc, acc2 = np.zeros_like(mu), np.zeros_like(mu)
    for _ in range(draws):
        n1 = (1 + rng.normal(0, mag_sigma, Mp.shape)) * np.exp(1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mp.shape)))
        n2 = (1 + rng.normal(0, mag_sigma, Mf.shape)) * np.exp(1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mf.shape)))
        s = bus_signature(Mp * n1, Mf * n2)
        acc += s
        acc2 += s ** 2
    sd = np.sqrt(np.maximum(acc2 / draws - (acc / draws) ** 2, 0)) + 1e-4
    sag = np.nan_to_num(np.nanmax(1 - np.abs(Mf) / np.abs(Mp), axis=2), nan=0.0)

    events = sorted({(m[0], m[2]) for m in meta})
    ev_idx = {e: i for i, e in enumerate(events)}
    marg = np.full((len(events), B), np.inf)
    sag_e = np.zeros((len(events), B))
    by_event = {}
    for i, (fb, c, rf) in enumerate(meta):
        by_event.setdefault((fb, rf), []).append((c, i))
        sag_e[ev_idx[(fb, rf)]] = np.maximum(sag_e[ev_idx[(fb, rf)]], sag[i])
    for e, items in by_event.items():
        for (c1, i), (c2, j) in combinations(items, 2):
            if c1 == c2:
                continue
            pooled = np.sqrt((sd[i] ** 2 + sd[j] ** 2) / 2)
            d = np.linalg.norm((mu[i] - mu[j]) / pooled, axis=1)
            marg[ev_idx[e]] = np.minimum(marg[ev_idx[e]], d)
    if not np.isfinite(marg).all():
        marg[~np.isfinite(marg)] = 0.0  # events with a single feasible fault type carry no class information
    return events, marg, sag_e


# ------------------------------------------------------------------ ILP max coverage
def max_coverage(cover, k, redundancy=1, tie=None):
    """Choose k columns of the boolean (E, B) cover matrix maximising the number of rows
    covered at least `redundancy` times (exact, scipy MILP / HiGHS). `tie` (B,) breaks ties."""
    E, B = cover.shape
    c = np.concatenate([np.zeros(B), -np.ones(E)])
    if tie is not None:
        c[:B] = -1e-3 * tie / (np.abs(tie).max() + 1e-12)
    rows = [np.concatenate([np.ones(B), np.zeros(E)])]
    lb, ub = [k], [k]
    for e in range(E):
        r = np.zeros(B + E)
        r[:B] = cover[e]
        r[B + e] = -redundancy
        rows.append(r)
        lb.append(0)
        ub.append(np.inf)
    res = milp(c, constraints=LinearConstraint(np.array(rows), lb, ub),
               integrality=np.ones(B + E), bounds=Bounds(0, 1), options={"time_limit": 120})
    return sorted(np.where(res.x[:B] > 0.5)[0].tolist())


def dmp(marg, k, tau=3.0, n1=False):
    cover = marg >= tau
    return max_coverage(cover.astype(float), k, redundancy=2 if n1 else 1,
                        tie=np.clip(marg, 0, 3 * tau).sum(0))


def mra(sag, k, threshold=0.1):
    return max_coverage((sag >= threshold).astype(float), k, tie=sag.sum(0))


def observability(n_buses, edges, k):
    cover = np.eye(n_buses)
    for a, b in edges:
        cover[a, b] = cover[b, a] = 1
    return max_coverage(cover, k, tie=cover.sum(0))


def topology(n_buses, edges, sub, k):
    from e2_scale import hops_from
    dist = np.array([hops_from(i, n_buses, edges) for i in range(n_buses)])
    chosen = [sub]
    while len(chosen) < k:
        chosen.append(int(np.argmax(dist[:, chosen].min(1))))
    return chosen


def greedy(score_fn, n_buses, k):
    chosen, rest = [], list(range(n_buses))
    for _ in range(k):
        best = max(rest, key=lambda b: score_fn(chosen + [b]))
        chosen.append(best)
        rest.remove(best)
    return chosen
