"""E1 - Representation ladder on the IEEE 13-node feeder (claim C1).

For all-bus and three-bus sensing, every representation AGG-FPS is evaluated with seven
learned classifiers, the non-learned relay rule (SEQ) and a GCN (graph input), using one
fixed dataset, 5-fold CV x 2 repetitions and PMU-class noise. A noise-free FPS run and the
Cover-Hart Bayes-error lower bound separate representation loss from measurement loss."""
import numpy as np
from scipy.stats import wilcoxon

import features as F
import models as M
from common import SIM, PAPER_3BUS, save, graph_edges, masked_graph_inputs


def main():
    ds = F.Dataset(f"{SIM}/ieee13.npz")
    edges = graph_edges(ds, 13)
    out = {"n": ds.n, "class_counts": np.bincount(ds.y)}
    for sensing, obs in [("all", list(range(len(ds.buses)))), ("3bus", ds.bus_ids(PAPER_3BUS[13]))]:
        rng = np.random.RandomState(0)
        Vp, Vf = F.measure(ds, obs, rng)
        res = {}
        for rep, fn in F.REPS.items():
            X = fn(Vp, Vf)
            e1, lb = F.bayes_lower_bound(X, ds.y, 3)
            res[rep] = {"dim": X.shape[1], "nn1_err": e1, "bayes_lb": lb, "models": {}}
            for name, mdl in M.zoo().items():
                acc, f1, _ = M.cross_validate(mdl, X, ds.y)
                res[rep]["models"][name] = {"acc": acc, "f1": f1}
                print(sensing, rep, name, f"{acc.mean() * 100:.1f}", flush=True)
            if rep == "SEQ":
                acc, f1, _ = M.cross_validate(M.RelayRule(), X, ds.y)
                res[rep]["models"]["Relay rule"] = {"acc": acc, "f1": f1}
                print(sensing, rep, "Relay", f"{acc.mean() * 100:.1f}", flush=True)
        G = masked_graph_inputs(ds, Vp, Vf, obs)
        acc, f1, _ = M.cross_validate(M.GCNClassifier(edges=edges, n_bus=len(ds.buses), epochs=60), G, ds.y)
        res["GNN"] = {"acc": acc, "f1": f1}
        print(sensing, "GCN", f"{acc.mean() * 100:.1f}", flush=True)

        # measurement-noise-free reference for the richest representation
        Vp0, Vf0 = F.measure(ds, obs, rng, mag_sigma=0, ang_sigma_deg=0)
        clean = {}
        for rep in ("AGG", "FPS"):
            X0 = F.REPS[rep](Vp0, Vf0)
            acc, f1, _ = M.cross_validate(M.zoo()["LGBM"], X0, ds.y)
            clean[rep] = {"acc": acc, "bayes_lb": F.bayes_lower_bound(X0, ds.y, 3)[1]}
            print(sensing, "noise-free", rep, f"{acc.mean() * 100:.1f}", flush=True)
        res["noise_free"] = clean

        # paired test: best learned model on AGG vs FPS (same folds)
        a1, a4 = res["AGG"]["models"]["LGBM"]["acc"], res["FPS"]["models"]["LGBM"]["acc"]
        res["wilcoxon_AGG_vs_FPS_LGBM_p"] = wilcoxon(a1, a4).pvalue
        out[sensing] = res
    save("e1_ladder", out)


if __name__ == "__main__":
    main()
