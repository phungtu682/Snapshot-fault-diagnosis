"""Offline replacement of simlib (no OpenDSS): serves the precomputed feeder graphs and
noise-free design grids from the snapshot dataset (experiments/extra)."""
import glob, json, os
import numpy as np
_EX = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra", "grid_ieee13.npz")]
assert _EX, "experiments/extra is missing"
EXTRA = os.path.dirname(_EX[0])
_TAG = {13: "ieee13", 34: "ieee34", 123: "ieee123", 240: "derfault"}
HERE = os.path.dirname(os.path.abspath(__file__))
FEEDERS = {}


def _grid(n_bus):
    return np.load(os.path.join(EXTRA, f"grid_{_TAG[int(n_bus)]}.npz"), allow_pickle=False)


def feeder_graph(n_bus):
    d = json.load(open(os.path.join(EXTRA, f"edges_{_TAG[int(n_bus)]}.json")))
    o = d["bus_order"]
    return [(o[a], o[b]) for a, b in d["edges"]]


class Feeder:
    def __init__(self, n_bus):
        z = _grid(n_bus)
        self.buses = [str(b) for b in z["buses"]]
        cols = z["cols"]
        self.bus_phases = {b: [p + 1 for p in range(3) if cols[i, p] >= 0] for i, b in enumerate(self.buses)}


def precomputed_grid(n_bus, rf_grid):
    z = _grid(n_bus)
    have = sorted({round(float(r), 6) for r in z["meta_rf"]})
    want = sorted({round(float(r), 6) for r in rf_grid})
    assert have == want, (n_bus, have, want)
    meta = list(zip([str(x) for x in z["meta_bus"]], [str(x) for x in z["meta_class"]],
                    [float(x) for x in z["meta_rf"]]))
    return [str(b) for b in z["buses"]], z["cols"], z["Vp"], z["Vf"], meta
