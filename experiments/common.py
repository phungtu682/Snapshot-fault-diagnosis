"""Shared paths, sensor subsets and result I/O for the experiment scripts."""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "data", "sim")
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# Three-bus subset used in the manuscript (substation, feeder end, lateral hub)
PAPER_3BUS = {13: ["650", "675", "684"]}


def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def graph_edges(ds, n_bus):
    import simlib
    pos = {b: i for i, b in enumerate(ds.buses)}
    return [(pos[a], pos[b]) for a, b in simlib.feeder_graph(n_bus) if a in pos and b in pos]


def masked_graph_inputs(ds, Vp, Vf, obs):
    """Graph features for all buses with only the observed ones populated (+ observed flag)."""
    import features as F
    G = F.graph_inputs(Vp, Vf)
    full = np.zeros((ds.n, len(ds.buses), G.shape[2] + 1))
    full[:, obs, :-1] = G
    full[:, obs, -1] = 1
    return full
