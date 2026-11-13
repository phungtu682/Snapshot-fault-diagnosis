"""Measurement model and snapshot feature representations (AGG-FPS, currents, graph input)."""
import numpy as np

A = np.exp(2j * np.pi / 3)
CLASSES3 = ["SLG", "LL", "3PH"]
CLASSES4 = ["SLG", "LL", "LLG", "3PH"]


class Dataset:
    def __init__(self, path, task="3class"):
        d = np.load(path, allow_pickle=False)
        keep = np.isin(d["ftype"], CLASSES3 if task == "3class" else CLASSES4)
        self.classes = CLASSES3 if task == "3class" else CLASSES4
        self.Vpre, self.Vf = d["Vpre"][keep], d["Vf"][keep]
        self.Ipre, self.If = d["Ipre"][keep], d["If"][keep]
        self.ftype, self.fbus, self.rf = d["ftype"][keep], d["bus"][keep], d["rf"][keep]
        self.y = np.array([self.classes.index(t) for t in self.ftype])
        self.buses = [str(b) for b in d["buses"]]
        self.lines = [str(l) for l in d["lines"]]
        self.line_buses = [tuple(map(str, lb)) for lb in d["line_buses"]]
        self.substation = str(d["substation"])
        node_names = [str(n) for n in d["node_names"]]
        idx = {n: i for i, n in enumerate(node_names)}
        # node table: (n_bus, 3) column indices into V arrays, -1 where the phase does not exist
        self.node_cols = np.array([[idx.get(f"{b}.{p}", -1) for p in (1, 2, 3)] for b in self.buses])
        self.n = len(self.y)

    def bus_ids(self, names):
        return [self.buses.index(b) for b in names]


def measure(ds, obs, rng, mag_sigma=0.0032951, ang_sigma_deg=0.18879):
    """Noisy PMU phasors at the observed buses -> (Vp, Vf) arrays of shape (n, n_obs, 3), NaN = no phase.
    Default noise (1 % magnitude, 0.5 deg) corresponds to the 1 % TVE class of IEEE C37.118.1."""
    cols = ds.node_cols[obs]
    out = []
    for V in (ds.Vpre, ds.Vf):
        M = np.full((ds.n,) + cols.shape, np.nan + 0j, dtype=np.complex128)
        valid = cols >= 0
        M[:, valid] = V[:, cols[valid]]
        if mag_sigma > 0 or ang_sigma_deg > 0:
            noise = (1 + rng.normal(0, mag_sigma, M.shape)) * np.exp(1j * np.deg2rad(rng.normal(0, ang_sigma_deg, M.shape)))
            M = M * noise
        out.append(M)
    return out


def _seq(M):
    """Symmetrical components of (n, n_obs, 3) phasors; NaN where a bus is not three-phase."""
    a, b, c = M[..., 0], M[..., 1], M[..., 2]
    return (a + b + c) / 3, (a + A * b + A ** 2 * c) / 3, (a + A ** 2 * b + A * c) / 3


def agg(Vp, Vf):
    """Paper baseline: four aggregate statistics over all observed node magnitudes."""
    mf, d = np.abs(Vf), np.abs(Vp) - np.abs(Vf)
    return np.column_stack([np.nanmean(mf, (1, 2)), np.nanstd(mf, (1, 2)),
                            np.nanmax(np.abs(d), (1, 2)), np.nansum(d ** 2, (1, 2))])


def ppa(Vp, Vf):
    """AGG statistics computed separately for phases a, b, c (12 features)."""
    feats = []
    for p in range(3):
        mf, d = np.abs(Vf[:, :, p:p + 1]), np.abs(Vp[:, :, p:p + 1]) - np.abs(Vf[:, :, p:p + 1])
        with np.errstate(all="ignore"):
            f = np.column_stack([np.nanmean(mf, (1, 2)), np.nanstd(mf, (1, 2)),
                                 np.nanmax(np.abs(d), (1, 2)), np.nansum(d ** 2, (1, 2))])
        feats.append(np.nan_to_num(f, nan=0.0))
    return np.hstack(feats)


def seq(Vp, Vf):
    """Phase-agnostic protection-style features: sorted per-phase dips at the most
    disturbed observed bus and symmetrical-component ratios (V0/V1, V2/V1)."""
    with np.errstate(all="ignore"):
        ratio = np.abs(Vf) / np.abs(Vp)
        dip = np.nan_to_num(1 - ratio, nan=0.0)
        worst = np.argmax(dip.sum(2), axis=1)
        dips_sorted = -np.sort(-dip[np.arange(len(dip)), worst], axis=1)
        V0, V1, V2 = _seq(Vf)
        z0, z2 = np.abs(V0) / np.abs(V1), np.abs(V2) / np.abs(V1)
        V0p, V1p, _ = _seq(Vp)
        v1drop = 1 - np.abs(V1) / np.abs(V1p)
        f = np.column_stack([dips_sorted,
                             np.nanmax(z0, 1), np.nanmean(z0, 1), np.nanmax(z2, 1), np.nanmean(z2, 1),
                             np.nanmax(v1drop, 1), np.nanmean(v1drop, 1)])
    return np.hstack([np.nan_to_num(f, nan=0.0), agg(Vp, Vf)])


def fps(Vp, Vf):
    """Full snapshot: per-node magnitude ratio and angle shift, plus per-bus V0/V1, V2/V1."""
    with np.errstate(all="ignore"):
        ratio = np.nan_to_num(np.abs(Vf) / np.abs(Vp), nan=1.0)
        shift = np.nan_to_num(np.angle(Vf / Vp), nan=0.0)
        V0, V1, V2 = _seq(Vf)
        z = np.nan_to_num(np.concatenate([np.abs(V0) / np.abs(V1), np.abs(V2) / np.abs(V1)], 1), nan=0.0)
    n = len(Vp)
    return np.hstack([ratio.reshape(n, -1), shift.reshape(n, -1), z])


REPS = {"AGG": agg, "PPA": ppa, "SEQ": seq, "FPS": fps}


def head_line(ds):
    """Index of the line leaving the substation bus (feeder-head current measurement)."""
    for k, (a, b) in enumerate(ds.line_buses):
        if ds.substation in (a.lower(), b.lower()):
            return k
    return 0


def currents(ds, rng, line_ids=None, mag_sigma=0.0032951, ang_sigma_deg=0.18879):
    """Feeder-head (and optional extra line) current features: sorted per-phase |If|/|Ipre|,
    I0/I1 and I2/I1 of the fault current."""
    line_ids = line_ids or [head_line(ds)]
    feats = []
    for k in line_ids:
        Ip, If = ds.Ipre[:, k, :].astype(complex), ds.If[:, k, :].astype(complex)
        noise = lambda s: (1 + rng.normal(0, mag_sigma, s)) * np.exp(1j * np.deg2rad(rng.normal(0, ang_sigma_deg, s)))
        Ip, If = Ip * noise(Ip.shape), If * noise(If.shape)
        with np.errstate(all="ignore"):
            r = -np.sort(-np.nan_to_num(np.abs(If) / np.abs(Ip), nan=0.0, posinf=0.0), axis=1)
            I0 = np.abs(If.sum(1)) / 3
            I1 = np.abs(If[:, 0] + A * If[:, 1] + A ** 2 * If[:, 2]) / 3
            I2 = np.abs(If[:, 0] + A ** 2 * If[:, 1] + A * If[:, 2]) / 3
            f = np.column_stack([r, I0 / I1, I2 / I1])
        feats.append(np.nan_to_num(f, nan=0.0, posinf=0.0))
    return np.hstack(feats)


def graph_inputs(Vp, Vf):
    """Per-bus node features for the GNN: ratio(3), angle shift(3), V0/V1, V2/V1, phase mask(3)."""
    with np.errstate(all="ignore"):
        ratio = np.nan_to_num(np.abs(Vf) / np.abs(Vp), nan=1.0)
        shift = np.nan_to_num(np.angle(Vf / Vp), nan=0.0)
        V0, V1, V2 = _seq(Vf)
        z0 = np.nan_to_num(np.abs(V0) / np.abs(V1), nan=0.0)[..., None]
        z2 = np.nan_to_num(np.abs(V2) / np.abs(V1), nan=0.0)[..., None]
    mask = (~np.isnan(Vp.real)).astype(float)
    return np.concatenate([ratio, shift, z0, z2, mask], axis=2)


def bayes_lower_bound(X, y, n_classes):
    """Cover-Hart lower bound on the Bayes error from the leave-one-out 1-NN error."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    knn = KNeighborsClassifier(2).fit(Xs, y)
    _, nb = knn.kneighbors(Xs)
    e1 = np.mean(y[nb[:, 1]] != y)
    M = n_classes
    lb = (M - 1) / M * (1 - np.sqrt(max(0.0, 1 - M / (M - 1) * e1)))
    return e1, lb
