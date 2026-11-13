"""N2 helpers: the detectability law R_f*(sigma, Z) and its calibration.

Manuscript basis (main_Ver_10_clean.tex, sec. 3.0b):
  * eq. (eq_dip):   d_b(R_f) ~ kappa_bq / R_f          (per unit, |V_q| ~ 1)
  * separation condition:  |d_b^c - d_b^c'| >= tau * sqrt(2) * sigma_m
  * eq. (eq_rfstar): R_f* ~ kappa_bq / (tau * sqrt(2) * sigma_m)

The relation is a PROPORTIONALITY, not an equality: it fixes the exponent on sigma and the
linear dependence on the transfer impedance, but not the constant. So the honest thing to
do is (a) test the scaling prediction against simulation and (b) calibrate the constant
once, on one feeder, then predict the other two without refitting. That is exactly what N2
does, and it is what makes the test falsifiable.

Definition of kappa used here
-----------------------------
d_b ~ kappa_bq / R_f, so kappa_bq is the per-unit dip that a fault of 1 ohm at bus q
produces at the sensor bus b. Measured directly from the design grid: for every
(fault bus q, sensor bus b) pair, kappa_bq = R_f * d_b, using the smallest R_f in the grid
(where the 1/R_f approximation is best).

Detectability threshold
-----------------------
The dip contrast between two fault classes at the same (q, R_f) must exceed the noise:
  Delta_d(R_f) = kappa^sep_bq / R_f  >=  tau * sqrt(2) * sigma_m
where kappa^sep is the *separation* transfer impedance: the largest |kappa^c - kappa^c'|
over class pairs at that (q, b). Then
  R_f* = kappa^sep_bq / (tau * sqrt(2) * sigma_m).
"""
import numpy as np

A = np.exp(2j * np.pi / 3)


def _dips(cols, Vp, Vf):
    """Per-bus maximal per-phase dip, (n_sim, n_bus)."""
    B = len(cols)
    Mp = np.full((len(Vp), B, 3), np.nan + 0j)
    Mf = np.full((len(Vp), B, 3), np.nan + 0j)
    v = cols >= 0
    Mp[:, v], Mf[:, v] = Vp[:, cols[v]], Vf[:, cols[v]]
    with np.errstate(all="ignore"):
        dip = np.nan_to_num(1 - np.abs(Mf) / np.abs(Mp), nan=0.0)
    return dip.max(axis=2)          # (n_sim, B): max dip across phases at each bus


def kappa_tables(cols, Vp, Vf, meta, rf_min_fit=1.0):
    """Transfer impedance kappa[bus, fault_bus, class] from the design grid.

    The law d_b ~ kappa/R_f is asymptotic in R_f: for R_f >> |Z_qq| (manuscript sec. 3.0b)
    the dip decays as 1/R_f, but at small R_f the dip SATURATES because the faulted bus
    voltage collapses and d_b cannot exceed ~1 pu. Estimating kappa from the smallest R_f
    would therefore be wrong by a factor of ~30 (verified: at Rf=0.01 the measured dip is
    0.55 while kappa/Rf = 0.31 for a kappa that is correct at Rf>=1).

    So kappa is estimated by a robust fit over the region where the asymptotic law holds:
    pool d_b * R_f over all R_f >= `rf_min_fit` and take the median, which is insensitive to
    the few large-R_f samples that fall below the numerical floor of the simulation.

    Returns (kappa, fault_buses, classes) with kappa shape (B, n_fault_bus, n_class).
    """
    dip = _dips(cols, Vp, Vf)
    fault_buses = sorted({m[0] for m in meta})
    classes = sorted({m[1] for m in meta})
    fb_i = {b: i for i, b in enumerate(fault_buses)}
    cl_i = {c: i for i, c in enumerate(classes)}
    B = dip.shape[1]
    samples = [[[] for _ in classes] for _ in range(len(fault_buses))]
    for i, (fb, c, rf) in enumerate(meta):
        if rf >= rf_min_fit:
            samples[fb_i[fb]][cl_i[c]].append(dip[i] * rf)
    kappa = np.zeros((B, len(fault_buses), len(classes)))
    for f in range(len(fault_buses)):
        for c in range(len(classes)):
            rows = samples[f][c]
            if rows:
                kappa[:, f, c] = np.median(np.array(rows), axis=0)
    return kappa, fault_buses, classes


def fit_quality(cols, Vp, Vf, meta, rf_min_fit=1.0, n_report=5):
    """How well does d_b ~ kappa/R_f hold? Returns per-R_f measured vs predicted means."""
    dip = _dips(cols, Vp, Vf)
    kappa, fbs, cls = kappa_tables(cols, Vp, Vf, meta, rf_min_fit=rf_min_fit)
    pos = {b: i for i, b in enumerate(fbs)}
    kmean = kappa.mean(axis=(1, 2))
    out = []
    for rf in sorted({m[2] for m in meta}):
        idx = [i for i, m in enumerate(meta) if abs(m[2] - rf) < 1e-12]
        if not idx:
            continue
        meas = dip[idx].mean()
        pred = float(np.mean([kmean[pos[meta[i][0]]] / rf for i in idx]))
        out.append({"rf": rf, "measured": float(meas), "predicted": pred,
                    "ratio": float(meas / pred) if pred else None})
    return out


def separation_kappa(kappa):
    """Largest class-to-class difference of kappa at each (sensor bus, fault bus).

    kappa : (B, n_fault_bus, n_class) -> (B, n_fault_bus)
    """
    B, F, C = kappa.shape
    out = np.zeros((B, F))
    for c1 in range(C):
        for c2 in range(c1 + 1, C):
            out = np.maximum(out, np.abs(kappa[:, :, c1] - kappa[:, :, c2]))
    return out


def rf_star(kappa_sep, obs, tau=3.0, mag_sigma=0.01):
    """R_f* for a sensor set, per fault bus: max over the chosen sensors (best coverage).

    R_f* = kappa_sep / (tau * sqrt(2) * sigma_m)     (manuscript eq_rfstar)
    """
    k = kappa_sep[list(obs), :].max(axis=0)
    return k / (tau * np.sqrt(2.0) * mag_sigma)


def predicted_accuracy_curve(rf_bins, rf_star_per_bus, fault_bus_of_sample, rf_of_sample):
    """Predicted 'detectable' mask: a sample is predicted correct iff R_f <= R_f*(its bus).

    This is the sharp-threshold version of the law. It is deliberately crude: the point of
    N2 is to test whether the 1/sigma scaling and the rank order of R_f* across buses are
    right, not to build a classifier.
    """
    rs = np.asarray(rf_star_per_bus)[np.asarray(fault_bus_of_sample)]
    return np.asarray(rf_of_sample) <= rs
