"""E01 placement, reimplemented on our voltage-only snapshot (TASK_04 / N1).

Source: Kaliuzhnyi et al., "..." Electric Power Systems Research 265 (2027) 114472,
PDF at TASK_01_output/pdf/E01_kaliuzhnyi2027.pdf. Every equation used is cited inline by
its number and page, as TASK_04 section 5 requires.

What is reproduced
------------------
E01 solve: maximise, over the PMU set P, the MINIMUM pairwise Mahalanobis distance
between fault scenarios under a Gaussian measurement-error model  (Eq. 25, p.6):

    max_P  min_{theta_i != theta_j}  D_M(theta_i, theta_j; P)

where D_M^2 = (mu_i - mu_j)^T R^{-1} (mu_i - mu_j)  (Eq. 5, p.4), mu being the
noise-free scenario signature at the sensors in P and R the measurement covariance.

Their link to the Bayes error is exact for equal-covariance Gaussians with equal priors
(Eq. 19, p.5):  P_e = Q(D_M / 2).  It becomes the STOPPING RULE for adding PMUs
(Eq. 27, p.7):  Q(D_M^min / 2) <= eps,  with eps = 1e-3 adopted for classification (p.12).

They solve it GREEDILY, one PMU at a time, because they state that an exhaustive search
over configurations "is therefore computationally intractable" (p.6). We keep their greedy
rule because that IS their method; where we differ is only the input representation
(voltage-only snapshot at k buses, vs their V+I sequence phasors at PMUs).

Deviations from E01, and why (all documented, none silent)
----------------------------------------------------------
1. **Representation.** E01 measures V and I in three sequence components (R^12 per PMU,
   p.9, p.12). We apply their criterion to OUR feature space — the per-bus voltage
   signature used by DMP (`placement.bus_signature`, 5 features: sorted dips, V0/V1,
   V2/V1) — because the point of N1 is "what does their criterion achieve on our
   problem", not "reproduce their paper".
2. **Covariance.** E01 needs the full matrix R. Our per-bus signature has 5 dimensions
   estimated from `draws` Monte-Carlo noise replicates, so R is estimated as a per-bus
   diagonal covariance from those draws. With 5 dims and a diagonal R, R^{-1} is the
   reciprocal of the per-feature variance, which is exactly the whitening E01 performs
   explicitly ("Euclidean distance between conditional means in the whitened space
   y = R^{-1/2} z", p.4).
3. **Greedy move.** E01 picks the next PMU by "the spatial concentration of statistically
   indistinguishable fault scenarios ... the feeder substation downstream of which the
   identified concentration region is located" (p.7) — a coarse, region-level rule that
   needs a substation partition we do not have for an arbitrary distribution feeder.
   Used here instead: the classic max-min greedy, i.e. add the bus that maximises
   D_M^min of the enlarged set. This is the same objective as Eq. 25, optimised greedily,
   which is strictly *stronger* than their region rule and therefore a **conservative**
   (favourable to E01) baseline. Noted so nobody reads N1 as "E01 underperforms".
4. **Stopping rule** is implemented exactly (Eq. 27) and reported as the k at which it
   fires, which is what TASK_04 asks for.

A finding that must be reported, not hidden
-------------------------------------------
Applied verbatim to our problem, E01's max-min objective **collapses to D_M^min ~ 0 for
every sensor set**, so the stopping rule never fires and the objective cannot distinguish
placements at all. This is not an implementation bug. On IEEE-13, 41 % of the design-grid
fault events produce a maximal voltage dip below 0.02 pu at every bus, and 32 % below
0.01 pu — i.e. below the 1 % TVE noise floor. The worst offender is a fault at the
substation bus itself (dip 3e-6 pu). Because Eq. 25 takes a MINIMUM over all scenario
pairs, one physically invisible pair pins the whole objective at zero, and no sensor set
can lift it. Measured directly: D_M^min is 0.0025 with all 14 buses observed.

This is exactly the pathology the manuscript attributes to max-min objectives, and it is
what a coverage formulation removes: DMP counts how many events are adequately separated
and simply does not credit an event that no sensor can see. N1 therefore reports both the
raw E01 criterion (`strict=True`, the paper's literal formulation) and a variant restricted
to events that are visible at all (`strict=False`), so the comparison is diagnostic rather
than merely a score line. The restriction is documented in the notebook and in DONE.md;
it is NOT presented as "E01's method".
"""
import numpy as np
from scipy.stats import norm

import k4lib  # noqa: F401  (imported first: it puts the experiments dir on sys.path)
import placement as P


def _whitened_signatures(cols, Vp, Vf, mag_sigma, ang_sigma_deg, draws=40, seed=0):
    """Noise-free signature mu and per-feature noise std, as E01's whitening (p.4).

    Returns mu (n_sim, B, F) and sd (n_sim, B, F), F = 5 features of bus_signature.
    """
    rng = np.random.RandomState(seed)
    Pn, B = len(Vp), len(cols)
    Mp = np.full((Pn, B, 3), np.nan + 0j)
    Mf = np.full((Pn, B, 3), np.nan + 0j)
    v = cols >= 0
    Mp[:, v], Mf[:, v] = Vp[:, cols[v]], Vf[:, cols[v]]
    mu = P.bus_signature(Mp, Mf)

    acc, acc2 = np.zeros_like(mu), np.zeros_like(mu)
    for _ in range(draws):
        n1 = (1 + rng.normal(0, mag_sigma, Mp.shape)) * np.exp(
            1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mp.shape)))
        n2 = (1 + rng.normal(0, mag_sigma, Mf.shape)) * np.exp(
            1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mf.shape)))
        s = P.bus_signature(Mp * n1, Mf * n2)
        acc += s
        acc2 += s ** 2
    sd = np.sqrt(np.maximum(acc2 / draws - (acc / draws) ** 2, 0.0)) + 1e-6
    return mu, sd


def scenario_pairs(meta):
    """Group simulations into events and list the (i, j) pairs of DIFFERENT fault classes.

    E01's min is over scenario pairs theta_i != theta_j. Two simulations at the same
    (bus, R_f) but different class are a genuinely separable pair, so those form the
    pair set; a pair identical in class carries no class information.
    """
    by_event = {}
    for i, (fb, c, rf) in enumerate(meta):
        by_event.setdefault((fb, rf), []).append((c, i))
    pairs = []
    for e, items in by_event.items():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if items[a][0] != items[b][0]:
                    pairs.append((items[a][1], items[b][1]))
    return pairs


def dm_matrix(mu, sd, pairs, obs):
    """Mahalanobis distance of every class-different pair for the sensor set `obs`.

    mu : (n_sim, B, F)   sd : (n_sim, B, F)   obs : list of bus indices
    Returns (n_pairs,) distances, whitened per E01 Eq. (5)/(p.4).
    """
    obs = list(obs)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    # sum over selected buses of squared whitened differences; independent noise across
    # buses makes the squared Mahalanobis distances additive (same argument as DMP-2).
    d2 = np.zeros(len(pairs))
    for b in obs:
        diff = mu[ii, b, :] - mu[jj, b, :]
        pooled = 0.5 * (sd[ii, b, :] ** 2 + sd[jj, b, :] ** 2) + 1e-12
        d2 += (diff ** 2 / pooled).sum(1)
    return np.sqrt(np.maximum(d2, 0.0))


def dm_min(mu, sd, pairs, obs):
    """D_M^min of a sensor set: the worst-pair Mahalanobis distance (Eq. 25, p.6)."""
    return float(dm_matrix(mu, sd, pairs, obs).min())


def bayes_from_dm(dm):
    """P_e = Q(D_M / 2), Eq. (19) p.5, and the Q^-1 form of the stopping rule."""
    return float(norm.sf(np.asarray(dm) / 2.0))


def dm_for_pe(eps):
    """D_M required for P_e <= eps, inverting P_e = Q(D_M/2)  (Eq. 27, p.7)."""
    return float(2.0 * norm.isf(eps))


def e01_place(mu, sd, pairs, candidates, k_max, eps=1e-3, first=None, pair_mask=None):
    """Sequential greedy max-min Mahalanobis placement with E01's Bayes stopping rule.

    Paper: Eq. (25) objective, Eq. (27) stopping rule, greedy sequential addition (p.6-7).

    candidates : bus indices allowed (all measurable buses)
    first      : bus forced first (E01 fixes the first PMU at the feeder substation, p.7);
                 None disables that constraint so the greedy starts from scratch.
    pair_mask  : optional boolean mask over `pairs` selecting which pairs enter the
                 minimum. Used for the diagnostic 'visible-only' variant; leave None for
                 the literal Eq. (25) formulation.

    Returns dict with the selected buses in order, D_M^min after each addition, the
    corresponding P_e, and the k at which the stopping rule fired (None if never).
    """
    chosen = [] if first is None else [int(first)]
    path = []

    def dmin(obs):
        d = dm_matrix(mu, sd, pairs, obs)
        if pair_mask is not None:
            d = d[pair_mask]
        return float(d.min()) if len(d) else float("inf")

    if chosen:
        d = dmin(chosen)
        path.append({"k": 1, "bus": int(chosen[0]), "dm": d, "pe": bayes_from_dm(d)})
    while len(chosen) < k_max:
        best, best_d = None, -np.inf
        for b in candidates:
            if b in chosen:
                continue
            d = dmin(chosen + [b])
            if d > best_d:
                best, best_d = int(b), d
        if best is None:
            break
        chosen.append(best)
        path.append({"k": len(chosen), "bus": best, "dm": best_d,
                     "pe": bayes_from_dm(best_d)})
    stop_k = None
    for rec in path:
        if rec["pe"] <= eps:
            stop_k = rec["k"]
            break
    return {"buses": chosen, "path": path, "eps": eps, "stop_k": stop_k,
            "dm_star": dm_for_pe(eps)}


def visible_pair_mask(cols, Vp, Vf, meta, pairs, noise_floor_pu=0.01):
    """Pairs in which at least one simulation is visible above the noise floor somewhere.

    A pair where BOTH simulations have a maximal per-bus dip below `noise_floor_pu` cannot
    be separated by any sensor set under that noise level (the signal is inside the noise),
    so including it in a MIN objective pins the objective at ~0. See the module docstring.
    """
    B = len(cols)
    Mp = np.full((len(Vp), B, 3), np.nan + 0j)
    Mf = np.full((len(Vp), B, 3), np.nan + 0j)
    v = cols >= 0
    Mp[:, v], Mf[:, v] = Vp[:, cols[v]], Vf[:, cols[v]]
    with np.errstate(all="ignore"):
        dip = np.nan_to_num(1 - np.abs(Mf) / np.abs(Mp), nan=0.0)
    worst = dip.max(axis=(1, 2))
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    return (worst[ii] >= noise_floor_pu) | (worst[jj] >= noise_floor_pu)


# =============================================================================
# Fair-comparison extensions (Claude, 2026-11-10)
# -----------------------------------------------------------------------------
# The 'strict' run above applies Eq. (25) to the whole design grid, R_f up to 100 ohm,
# including faults that no voltage sensor can see. E01 itself is evaluated with
# R_f <= 10 ohm. Reporting only 'strict' would make E01 fail for a reason that is not
# part of E01, so N1 reports three variants side by side:
#   literal   : Eq. (25) on the full grid, DMP's per-bus signature (old 'strict')
#   e01_range : same, but pairs restricted to E01's fault-resistance range (R_f <= 10)
#   e01_phasor: E01's own measurement representation, i.e. the whitened complex phasor
#               (real and imaginary part of V_fault / V_pre per phase, p.4, p.9), pairs
#               over ALL scenarios of different class (Eq. 25 min over theta_i != theta_j),
#               R_f <= 10
# Pairs in 'same_event' mode are limited to the same (bus, R_f), which is narrower than
# E01 and is labelled as such.
# =============================================================================

def phasor_signatures(cols, Vp, Vf, mag_sigma, ang_sigma_deg, draws=40, seed=0):
    """Noise-free mean and per-feature std of the complex ratio V_f / V_pre per phase.

    Features per bus: Re and Im of rho_phi for phi = a, b, c (6 values; missing phase -> 0
    with a large std so it carries no distance). This is E01's whitened-phasor distance
    (Eq. 5, p.4) applied to our voltage-only snapshot.
    """
    rng = np.random.RandomState(seed)
    Pn, B = len(Vp), len(cols)
    Mp = np.full((Pn, B, 3), np.nan + 0j)
    Mf = np.full((Pn, B, 3), np.nan + 0j)
    v = cols >= 0
    Mp[:, v], Mf[:, v] = Vp[:, cols[v]], Vf[:, cols[v]]

    def feat(A, Bm):
        with np.errstate(all="ignore"):
            r = Bm / A
        return np.concatenate([r.real, r.imag], axis=2)        # (Pn, B, 6)

    mu = feat(Mp, Mf)
    acc, acc2 = np.zeros_like(mu), np.zeros_like(mu)
    for _ in range(draws):
        n1 = (1 + rng.normal(0, mag_sigma, Mp.shape)) * np.exp(
            1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mp.shape)))
        n2 = (1 + rng.normal(0, mag_sigma, Mf.shape)) * np.exp(
            1j * np.deg2rad(rng.normal(0, ang_sigma_deg, Mf.shape)))
        s = feat(Mp * n1, Mf * n2)
        acc += np.nan_to_num(s)
        acc2 += np.nan_to_num(s) ** 2
    sd = np.sqrt(np.maximum(acc2 / draws - (acc / draws) ** 2, 0.0)) + 1e-9
    missing = ~np.isfinite(mu)
    mu = np.nan_to_num(mu)
    sd[missing] = 1e6
    return mu, sd


def scenario_pairs_v2(meta, mode="same_event", rf_max=None, max_pairs=400000, seed=0):
    """Class-different scenario pairs.

    mode='same_event': same (bus, R_f) only (as the original scenario_pairs).
    mode='all'       : every pair of simulations with different fault class (E01 Eq. 25);
                       randomly subsampled to `max_pairs` (seeded) when larger, which is
                       reported in the result.
    rf_max           : keep only simulations with R_f <= rf_max (E01 range: 10 ohm).
    Returns (pairs, info).
    """
    idx = [i for i, m in enumerate(meta) if rf_max is None or m[2] <= rf_max + 1e-12]
    info = {"mode": mode, "rf_max": rf_max, "n_sims": len(idx)}
    if mode == "same_event":
        sub = [meta[i] for i in idx]
        p = scenario_pairs(sub)
        pairs = [(idx[a], idx[b]) for a, b in p]
        info["n_pairs"] = len(pairs)
        info["subsampled"] = False
        return pairs, info
    cls = np.array([meta[i][1] for i in idx])
    ia, ib = np.triu_indices(len(idx), 1)
    keep = cls[ia] != cls[ib]
    ia, ib = ia[keep], ib[keep]
    info["n_pairs_total"] = int(len(ia))
    if len(ia) > max_pairs:
        sel = np.random.RandomState(seed).choice(len(ia), max_pairs, replace=False)
        ia, ib = ia[sel], ib[sel]
        info["subsampled"] = True
    else:
        info["subsampled"] = False
    idx = np.asarray(idx)
    pairs = list(zip(idx[ia].tolist(), idx[ib].tolist()))
    info["n_pairs"] = len(pairs)
    return pairs, info


def noise_floor_pu(mag_sigma, n_sigma=3.0):
    """Visibility floor tied to the noise model instead of a hard-coded 0.01 pu."""
    return n_sigma * mag_sigma


E01_VARIANTS = {
    # name        representation  pair mode     R_f max
    "literal":    ("signature",   "same_event", None),
    "e01_range":  ("signature",   "same_event", 10.0),
    "e01_phasor": ("phasor",      "all",        10.0),
}
