"""E4b - Proposed Discriminability-Margin Placement (DMP) vs placement baselines.

Selection uses only (i) the simulation design grid (DMP, MRA), (ii) topology (observability,
farthest-point) or (iii) an independent selection dataset (information / accuracy greedy).
All placements are then scored on the same evaluation dataset with LightGBM 5-fold CV:
overall accuracy, accuracy on high-impedance faults (R_f > 10 ohm), and N-1 accuracy
(one sensor lost, its fault-time value frozen at pre-fault), averaged over the lost sensor."""
import time
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import features as F
import placement as P
from common import SIM, PAPER_3BUS, save, graph_edges

FAST = dict(n_estimators=150, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1, random_state=0)
K_RANGE = {13: [2, 3, 4, 5], 34: [2, 3, 4, 6], 123: [3, 4, 6, 8]}


def evaluate(ds, obs, rep="FPS", seeds=(0,)):
    Vp, Vf = F.measure(ds, obs, np.random.RandomState(0))
    X = F.REPS[rep](Vp, Vf)
    acc, hif, n1 = [], [], []
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, ds.y):
            m = LGBMClassifier(**FAST).fit(X[tr], ds.y[tr])
            ok = m.predict(X[te]) == ds.y[te]
            acc.append(ok.mean())
            hif.append(ok[ds.rf[te] > 10].mean())
            lost = []
            for j in range(len(obs)):
                Vf_l = Vf[te].copy()
                Vf_l[:, j] = Vp[te][:, j]
                lost.append((m.predict(F.REPS[rep](Vp[te], Vf_l)) == ds.y[te]).mean())
            n1.append(np.mean(lost))
    return {"acc": float(np.mean(acc)), "acc_sd": float(np.std(acc)), "hif": float(np.mean(hif)),
            "n1": float(np.mean(n1))}


def info_score(ds, obs):
    Vp, Vf = F.measure(ds, obs, np.random.RandomState(1))
    Xs = StandardScaler().fit_transform(F.fps(Vp, Vf))
    _, nb = KNeighborsClassifier(2).fit(Xs, ds.y).kneighbors(Xs)
    return -np.mean(ds.y[nb[:, 1]] != ds.y)


def wrapper_score(ds, obs):
    Vp, Vf = F.measure(ds, obs, np.random.RandomState(1))
    m = LGBMClassifier(**{**FAST, "n_estimators": 80})
    return cross_val_score(m, F.fps(Vp, Vf), ds.y, cv=StratifiedKFold(3, shuffle=True, random_state=0)).mean()


def main():
    out = {}
    for nb in (13, 34, 123):
        t0 = time.time()
        ds = F.Dataset(f"{SIM}/ieee{nb}.npz")
        sel = F.Dataset(f"{SIM}/ieee{nb}_sel.npz")
        assert sel.buses == ds.buses
        edges = graph_edges(ds, nb)
        nbus = len(ds.buses)
        sub = ds.buses.index(ds.substation)
        buses, cols, Vp, Vf, meta = P.design_grid(nb)
        assert list(buses) == ds.buses
        events, marg, sag = P.margins(cols, Vp, Vf, meta)
        res = {"n_design_sims": len(meta), "n_events": len(events), "k": {}}
        print(nb, "design sims", len(meta), "events", len(events), f"{time.time() - t0:.0f}s", flush=True)
        for k in K_RANGE[nb]:
            placements = {
                "DMP (proposed)": P.dmp(marg, k),
                "DMP N-1 (proposed)": P.dmp(marg, k, n1=True),
                "MRA ILP": P.mra(sag, k),
                "Observability ILP": P.observability(nbus, edges, k),
                "Topology": P.topology(nbus, edges, sub, k),
                "Information greedy": P.greedy(lambda o: info_score(sel, o), nbus, k),
                "Accuracy greedy": P.greedy(lambda o: wrapper_score(sel, o), nbus, k),
            }
            if nb == 13 and k == 3:
                placements["Manuscript subset"] = ds.bus_ids(PAPER_3BUS[13])
            rk = {}
            for name, obs in placements.items():
                rk[name] = {"buses": [ds.buses[i] for i in obs], **evaluate(ds, obs)}
                print(nb, k, f"{name:20s}", rk[name]["buses"],
                      f"acc {rk[name]['acc'] * 100:.1f} HIF {rk[name]['hif'] * 100:.1f} N-1 {rk[name]['n1'] * 100:.1f}",
                      flush=True)
            rng = np.random.RandomState(11)
            rand = [evaluate(ds, list(rng.choice(nbus, k, replace=False))) for _ in range(10)]
            rk["Random (10 draws)"] = {key: float(np.mean([r[key] for r in rand])) for key in ("acc", "hif", "n1")}
            rk["Random (10 draws)"]["acc_range"] = [float(min(r["acc"] for r in rand)), float(max(r["acc"] for r in rand))]
            print(nb, k, "Random", rk["Random (10 draws)"], flush=True)
            res["k"][k] = rk
        # sensitivity of DMP to the margin threshold tau
        res["tau_sensitivity"] = {tau: evaluate(ds, P.dmp(marg, K_RANGE[nb][1], tau=tau)) for tau in (2.0, 3.0, 5.0)}
        out[f"ieee{nb}"] = res
        save("e4b_dmp", out)


if __name__ == "__main__":
    main()
