"""k4lib - shared helpers for the six TASK_04 Kaggle notebooks.

Design goals
------------
1. **Self-contained on Kaggle.** The notebooks import this module and `features.py`/
   `models.py`/`placement.py` from the attached dataset. Nothing here needs OpenDSS:
   the design grids and feeder adjacencies are precomputed (`extra/grid_*.npz`,
   `extra/edges_*.json`).
2. **Resumable.** Every experiment writes one JSON per finished unit
   (feeder x k x seed x config) into the Kaggle working directory and skips units that
   already exist. A session that dies after 12 h continues where it stopped.
3. **Comparable.** Same noise levels, same fold seeds, same classifier settings as the
   manuscript's e13/e14/e16, so numbers are directly comparable.

Conventions used throughout
---------------------------
* `obs` is always a **list of bus indices** into `ds.buses` (not names).
* A "unit" is identified by a dict `key`; its cache path is derived from it, so the same
  unit name always maps to the same file.
* Every result dict carries `"meta"` with the noise level, seeds and versions, so a stale
  cache from a different protocol can be detected.
"""
import json
import os
import platform
import re
import sys
import time

import numpy as np

# --------------------------------------------------------------------- paths
# On Kaggle the dataset is mounted read-only at /kaggle/input/<slug>/; locally we read
# from the repository. Both are handled by resolving once at import time.
CANDIDATES = [
    os.environ.get("K4_DATA", ""),
    "/kaggle/input/snapshot-fd-code-data/experiments",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"),
    os.path.dirname(os.path.abspath(__file__)),
]


def _resolve_data_dir():
    for c in CANDIDATES:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "features.py")):
            return os.path.abspath(c)
    raise RuntimeError(
        "could not find the experiments directory; set K4_DATA to the extracted "
        "snapshot-fd-code-data/experiments folder. Tried: " + repr(CANDIDATES)
    )


DATA = _resolve_data_dir()
if DATA not in sys.path:
    sys.path.insert(0, DATA)

SIM = os.path.join(DATA, "data", "sim")
DERFAULT = os.path.join(DATA, "data", "derfault", "derfault.npz")


def _resolve_extra():
    """Directory holding the precomputed grids/adjacencies.

    Search order: explicit override, the dataset layout used on Kaggle, the repo, then
    the local staging folder this task builds (`kaggle_dataset/extra`).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    tried = [os.environ.get("K4_EXTRA", ""),
             os.path.join(DATA, "extra"),
             "/kaggle/input/snapshot-fd-code-data/experiments/extra",
             os.path.join(here, "_precomputed"),
             os.path.join(here, "kaggle_dataset", "extra")]
    for c in tried:
        if c and os.path.isfile(os.path.join(c, "grid_ieee13.npz")):
            return os.path.abspath(c)
    raise RuntimeError(
        "could not find the precomputed grids (grid_ieee13.npz). Build them with "
        "_precompute.py and ship them inside the Kaggle dataset as experiments/extra/. "
        "Set K4_EXTRA to override. Tried: " + repr(tried))


EXTRA = _resolve_extra()

# --------------------------------------------------------------------- results
# Results go to a writable directory: /kaggle/working/results on Kaggle, or
# TASK_04_output/results locally. Set K4_RESULTS to override.
RES = os.environ.get("K4_RESULTS") or (
    "/kaggle/working/results" if os.path.isdir("/kaggle/working") else
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
)
os.makedirs(RES, exist_ok=True)

# --------------------------------------------------------------------- constants
# --------------------------------------------------------------------- noise framework
# Phasor measurement error model (fix by Claude, 2026-11-10).
#
# A phasor X is measured as X * (1 + e_m) * exp(j e_a), e_m ~ N(0, s_m^2), e_a ~ N(0, s_a^2).
# For small errors TVE = |X_meas - X| / |X| ~= sqrt(e_m^2 + e_a^2).
#
# IEEE C37.118.1 states 1 % TVE as a LIMIT, not as a standard deviation. The manuscript's
# old model (s_m = 1 %, s_a = 0.5 deg) exceeds 1 % TVE in 56 % of draws (99th percentile
# 2.9 %), i.e. it simulates a device that fails the standard. It is kept below as "legacy"
# only to show how much this changes the conclusions.
#
# Class definition used here: a device "meets x % TVE" when the 99th percentile of its TVE
# is x %. With the error split equally between magnitude and angle (s_m = s_a in radians),
# TVE / s is Rayleigh distributed, so P99 = s * sqrt(-2 ln 0.01) = 3.035 s, hence
# s = x / 3.035.  1 % TVE -> s_m = 0.330 %, s_a = 0.189 deg.
#
# A time-synchronisation error dt (std, seconds) adds an angle error 2*pi*f*dt, combined in
# quadrature with s_a (60 Hz: 1 us -> 0.0216 deg).
import math as _math


def tve_class(tve_pct, p=0.99, f_hz=60.0, sync_us=0.0):
    """(s_m, s_a_deg) for a device whose TVE p-quantile equals `tve_pct` percent.

    Magnitude and angle share the budget equally. If `sync_us` > 0, part of the angle
    budget is taken by the timing error, so the remaining angle noise is reduced.
    """
    s = (tve_pct / 100.0) / _math.sqrt(-2.0 * _math.log(1.0 - p))
    s_sync = 2 * _math.pi * f_hz * sync_us * 1e-6
    s_ang = _math.sqrt(max(s * s - s_sync * s_sync, 0.0)) if sync_us else s
    s_ang_total = _math.sqrt(s_ang ** 2 + s_sync ** 2)
    return (s, _math.degrees(s_ang_total))


def tve_quantiles(s_m, s_a_deg, n=200000, seed=0):
    """Monte-Carlo TVE statistics (percent) of a (s_m, s_a) model: median, P95, P99, P(>1 %)."""
    r = np.random.RandomState(seed)
    m = r.normal(0, s_m, n)
    a = np.deg2rad(r.normal(0, s_a_deg, n))
    t = np.abs((1 + m) * np.exp(1j * a) - 1) * 100
    return dict(median=float(np.median(t)), p95=float(np.quantile(t, .95)),
                p99=float(np.quantile(t, .99)), frac_above_1pct=float((t > 1).mean()))


# Reference levels of the framework. Keys "1% TVE" and "uPMU" are kept so every notebook
# picks up the corrected definition without edits.
#   uPMU   : manuscript value (0.02 %, 0.002 deg). NOTE: results/e10_lbnl_noise.json measures
#            about 0.0025 % and 0.0006 deg at half-cycle lag, i.e. lower; the manuscript value
#            is kept as a conservative upper estimate and must be justified in the paper.
#   1% TVE : C37.118.1 steady-state limit, P99 definition above.
#   3% TVE : fault-on regime level used by Sun et al. 2026 (IEEE TIM), same P99 definition.
#   legacy : the old manuscript model, for the "does the definition matter" comparison.
NOISE = {
    "uPMU": (0.0002, 0.002),
    "1% TVE": tve_class(1.0),
    "3% TVE": tve_class(3.0),
    "legacy 1%/0.5deg": (0.01, 0.5),
}
NOISE_MAIN = ["1% TVE", "uPMU"]          # levels every experiment must report
# Noise used to DESIGN placements (DMP margins). The manuscript designs at 1 % TVE; with
# the corrected class definition this is NOISE["1% TVE"], not the legacy (0.01, 0.5).
DESIGN_NOISE = NOISE["1% TVE"]

# Classifier used for the main sweeps. Deliberately the same as e14/e16 so placement
# comparisons are fold-for-fold comparable.
FAST = dict(n_estimators=150, learning_rate=0.1, num_leaves=31,
            n_jobs=-1, verbose=-1, random_state=0)

FEEDERS = {13: "ieee13", 34: "ieee34", 123: "ieee123"}
# k ranges: from the manuscript and e16 (DMP-2 sweep)
KS = {13: [2, 3, 4, 5], 34: [2, 3, 4, 6], 123: [3, 4, 6, 8, 9]}

# Optional work-slicing, so a 12 h session can be split without editing notebook bodies.
# A notebook's SETUP cell sets ONLY_FEEDERS / ONLY_KS; these two helpers apply them.
ONLY_FEEDERS = []
ONLY_KS = []


def feeder_list(smoke):
    """Feeders this run should cover: the smoke default (13), else all, honouring slicing."""
    fs = [13] if smoke else [13, 34, 123]
    if ONLY_FEEDERS:
        fs = [f for f in fs if int(f) in {int(x) for x in ONLY_FEEDERS}]
        if not fs:
            raise ValueError(f"ONLY_FEEDERS={ONLY_FEEDERS} excludes every feeder in {fs}")
    return fs


def k_list(feeder, smoke):
    """k values this run should cover for `feeder`, honouring ONLY_KS."""
    ks = [3] if smoke else list(KS[feeder])
    if ONLY_KS:
        ks = [k for k in ks if int(k) in {int(x) for x in ONLY_KS}]
        if not ks:
            raise ValueError(f"ONLY_KS={ONLY_KS} excludes every k for feeder {feeder} "
                             f"(available {KS[feeder]})")
    return ks


def ks_max(feeder, smoke):
    """Largest k the run needs for `feeder` (used to size E01 greedy placement)."""
    return max(k_list(feeder, smoke))

CKPT_DIRS = ("results", "kaggle")


# --------------------------------------------------------------------- caching
def _slug(key):
    """Filesystem-safe deterministic name for a unit key."""
    if isinstance(key, str):
        return re.sub(r"[^A-Za-z0-9_.=-]+", "_", key)
    parts = []
    for k in sorted(key):
        parts.append(f"{k}={key[k]}")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", "__".join(parts))[:180]


def unit_path(exp, key, root=None):
    d = os.path.join(root or RES, exp)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, _slug(key) + ".json")


def done(exp, key, root=None):
    return os.path.isfile(unit_path(exp, key, root))


def load_unit(exp, key, root=None):
    p = unit_path(exp, key, root)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def save_unit(exp, key, obj, root=None):
    p = unit_path(exp, key, root)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(tmp, p)          # atomic: a killed session never leaves a half-written unit
    return p


def run_units(exp, keys, fn, root=None, verbose=True):
    """Run fn(key) for every key not already cached. Returns {key_string: result}.

    This is the only resumption mechanism: a unit is either complete on disk or it re-runs.
    """
    out, t0, n_new = {}, time.time(), 0
    keys = list(keys)
    for i, key in enumerate(keys):
        cached = load_unit(exp, key, root)
        if cached is not None:
            out[_slug(key)] = cached
            if verbose:
                print(f"[{i+1}/{len(keys)}] cached  {_slug(key)}", flush=True)
            continue
        t = time.time()
        res = fn(key)
        n_new += 1
        save_unit(exp, key, res, root)
        out[_slug(key)] = res
        if verbose:
            print(f"[{i+1}/{len(keys)}] NEW     {_slug(key)}  {time.time()-t:.1f}s", flush=True)
    if verbose:
        print(f"-- {exp}: {n_new} new units, {len(keys)-n_new} cached, "
              f"{time.time()-t0:.1f}s --", flush=True)
    return out


# --------------------------------------------------------------------- data
def dataset(feeder):
    """Load one of our simulated feeders (3-class task)."""
    import features as F
    return F.Dataset(os.path.join(SIM, f"{FEEDERS.get(feeder, feeder)}.npz"))


def derfault():
    """Load DER-Fault (3-class task: contamination is filtered by features.Dataset)."""
    import features as F
    return F.Dataset(DERFAULT)


def _feeder_tag(feeder):
    """Normalise a feeder id ('ieee13', 13, '13') to the file tag ('ieee13')."""
    s = str(feeder)
    return s if s.startswith("ieee") else f"ieee{s}"


def feeder_edges(feeder):
    """Adjacency as index pairs, matching ds.buses order."""
    p = os.path.join(EXTRA, f"edges_{_feeder_tag(feeder)}.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"missing precomputed adjacency {p}; run _precompute.py")
    d = json.load(open(p))
    return [tuple(e) for e in d["edges"]], d["bus_order"]


def design_grid(feeder):
    """Precomputed design grid -> (buses, cols, Vp, Vf, meta) with meta = (bus, class, rf)."""
    name = _feeder_tag(feeder)
    p = os.path.join(EXTRA, f"grid_{name}.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"missing precomputed grid {p}; run _precompute.py")
    z = np.load(p, allow_pickle=False)
    meta = list(zip([str(x) for x in z["meta_bus"]],
                    [str(x) for x in z["meta_class"]],
                    [float(x) for x in z["meta_rf"]]))
    return [str(b) for b in z["buses"]], z["cols"], z["Vp"], z["Vf"], meta


def rows_for_buses(ds, buses):
    """Map grid bus names to indices in `ds` (both come from the same OpenDSS model)."""
    pos = {b: i for i, b in enumerate(ds.buses)}
    missing = [b for b in buses if b not in pos]
    if missing:
        raise KeyError(f"{len(missing)} grid buses absent from dataset, e.g. {missing[:5]}")
    return [pos[b] for b in buses]


def baselines(feeder, k):
    """Placement baselines recorded by e14/e16 for this feeder and k.

    Returns {method_name: {buses, acc, acc_sd, hif, n1, p_acc, p_hif, ...}}. These were
    produced by `e14_placement_baselines.py` with the same folds and the same FAST model, so
    re-running them inside TASK_04 would only burn session time. N1 reports E01 next to them
    instead of recomputing them, and says so in the notebook.
    """
    tag = _feeder_tag(feeder)
    out = {}
    p14 = os.path.join(DATA, "results", "e14_placement_baselines.json")
    if os.path.isfile(p14):
        d = json.load(open(p14))
        for key in (str(k), k):
            if key in d.get(tag, {}):
                out.update(d[tag][key])
                break
    p16 = os.path.join(DATA, "results", "e16_dmp2.json")
    if os.path.isfile(p16):
        d = json.load(open(p16))
        for key in (str(k), k):
            if key in d.get(tag, {}):
                for name, v in d[tag][key].items():
                    if isinstance(v, dict) and "acc" in v:
                        out.setdefault(name, v)
                break
    return out


# --------------------------------------------------------------------- measurement
def measure(ds, obs, seed, noise):
    """Noisy pre/fault phasors at the observed buses. `noise` is a key of NOISE or a tuple."""
    import features as F
    ms, asg = NOISE[noise] if isinstance(noise, str) else noise
    return F.measure(ds, list(obs), np.random.RandomState(seed),
                     mag_sigma=ms, ang_sigma_deg=asg)


def features_of(ds, obs, seed, noise, rep="FPS"):
    """Feature matrix for one sensor set / noise level / representation."""
    import features as F
    Vp, Vf = measure(ds, obs, seed, noise)
    return F.REPS[rep](Vp, Vf)


# --------------------------------------------------------------------- evaluation
def folds_acc(X, y, rf=None, n_splits=5, seeds=(0, 1), model=None, return_oof=False):
    """Stratified k-fold accuracy repeated over seeds.

    Returns (accs, hif, oof) where `hif` is the mean accuracy restricted to
    fault resistances above 10 ohm (as in e14/e16) and `oof` the out-of-fold predictions
    of the first repetition (needed for paired tests and for N6's confidence scores).
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    if model is None:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**FAST)

    accs, hifs, oof, proba = [], [], None, None
    for s in seeds:
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
        pred = np.empty_like(y)
        pr = np.zeros((len(y), len(np.unique(y))))
        for tr, te in skf.split(X, y):
            m = clone(model)
            m.fit(X[tr], y[tr])
            p = np.asarray(m.predict(X[te])).ravel()
            pred[te] = p
            if hasattr(m, "predict_proba"):
                cl = list(getattr(m, "classes_", np.unique(y)))
                pr[te] = m.predict_proba(X[te])[:, [list(cl).index(c) for c in np.unique(y)]]
            else:
                pr[te, p] = 1.0
            accs.append(accuracy_score(y[te], p))
            if rf is not None and np.any(rf[te] > 10):
                hifs.append(accuracy_score(y[te][rf[te] > 10], p[rf[te] > 10]))
        if oof is None:
            oof, proba = pred, pr
    out = {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
           "hif": float(np.mean(hifs)) if hifs else float("nan")}
    if return_oof:
        out["oof"], out["proba"] = oof, proba
    return out


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank p-value, robust to identical vectors."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 6 or np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


# --------------------------------------------------------------------- environment
def print_env():
    """Print versions at notebook start; a mismatch changes the numbers."""
    import scipy
    import sklearn
    print(f"python      {platform.python_version()}")
    print(f"numpy       {np.__version__}")
    print(f"scipy       {scipy.__version__}")
    print(f"scikit-learn{sklearn.__version__}")
    for mod in ("lightgbm", "xgboost", "pandas", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"{mod:12s}{getattr(m, '__version__', '?')}")
        except Exception:
            print(f"{mod:12s}not installed")
    try:
        import lightgbm
        print(f"lightgbm    {lightgbm.__version__}")
    except Exception:
        pass
    print(f"data dir    {DATA}")
    print(f"results dir {RES}")
    print(f"cpus        {os.cpu_count()}")
    try:
        print(f"mem         {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.1f} GB")
    except Exception:
        pass


META = {
    "noise": NOISE,
    "fast": FAST,
    "ks": {str(k): v for k, v in KS.items()},
    "levels_note": "noise framework v2 (2026-11-10): x% TVE = 99th percentile of TVE, equal mag/angle split; legacy = old manuscript model (0.01, 0.5 deg)",
}


def meta():
    """Provenance stamped into every result file."""
    import scipy
    import sklearn
    return {
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "python": platform.python_version(), "config": META,
    }
