"""E8 - External validation on the public PowerGraph cascading-failure benchmark.

Same questions as for the distribution feeders, on transmission grids with physics-based
(AC Cascades) labels: is a single pre-outage snapshot enough to flag a cascading failure
(binary, DNS > 0 or tripping), how much is lost by aggregate statistics versus the full
snapshot, and how much by observing only a fraction of buses. Balanced accuracy is used
because unstable states are rare. Published GNN results (Varbella et al.) are the reference."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import models as M
from common import HERE, save

GRIDS = ["ieee24", "ieee39", "uk", "ieee118"]
# Best published GNN balanced accuracy, binary task (PowerGraph, NeurIPS D&B 2024, Table 5)
PUBLISHED_GNN = {"ieee24": 0.9828, "ieee39": 0.9880, "uk": 0.9951, "ieee118": 0.9982}


def aggregate(node, edge):
    """AGG-style summary of the snapshot: statistics of V, net injections and branch loading."""
    V, P, S = node[..., 2], node[..., 0], node[..., 1]
    flow = np.abs(edge[..., 0] + 1j * edge[..., 1])
    rating = np.where(np.abs(edge[..., 3]) > 0, np.abs(edge[..., 3]), np.nan)
    with np.errstate(all="ignore"):
        load = np.nan_to_num(flow / rating, nan=0.0)
    out_lines = (np.abs(edge).sum(2) == 0).sum(1)
    return np.column_stack([V.mean(1), V.std(1), V.min(1), np.abs(P).sum(1), np.abs(S).sum(1),
                            load.mean(1), load.max(1), np.sort(load, 1)[:, -3:].mean(1), out_lines])


def evaluate(X, y, models=("LR", "RF", "LGBM")):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for name in models:
        scores = []
        for tr, te in cv.split(X, y):
            m = M.zoo()[name]
            if name == "LGBM":
                m.set_params(class_weight="balanced")
            elif name == "RF":
                m.set_params(class_weight="balanced_subsample", n_estimators=200)
            else:
                m.set_params(logisticregression__class_weight="balanced")
            m.fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], m.predict(X[te])))
        res[name] = scores
    return res


def main():
    out = {"published_gnn_binary": PUBLISHED_GNN}
    for g in GRIDS:
        d = np.load(f"{HERE}/data/powergraph/{g}.npz")
        node, edge, y = d["node"], d["edge"], d["y_bi"]
        nbus = node.shape[1]
        reps = {
            "aggregate": aggregate(node, edge),
            "full_node": node.reshape(len(y), -1),
            "full_node_edge": np.hstack([node.reshape(len(y), -1), edge.reshape(len(y), -1)]),
        }
        rng = np.random.RandomState(0)
        for frac in (0.1, 0.25, 0.5):
            k = max(1, int(round(frac * nbus)))
            obs = rng.choice(nbus, k, replace=False)
            reps[f"node_{int(frac * 100)}pct"] = node[:, obs, :].reshape(len(y), -1)
        r = {"n": len(y), "unstable_share": float(y.mean()), "n_bus": nbus}
        for name, X in reps.items():
            r[name] = evaluate(X, y)
            print(g, name, {k: round(float(np.mean(v)), 4) for k, v in r[name].items()}, flush=True)
        out[g] = r
    save("e8_powergraph", out)


if __name__ == "__main__":
    main()
