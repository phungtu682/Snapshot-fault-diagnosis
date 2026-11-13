"""E9 - Independent validation on DER-Fault (Iowa 240-bus, real-utility-based feeder, PV/EV/V2G).

(a) Representation ladder AGG-FPS x classifiers with all 93 PMUs, PMU noise and noise-free.
(b) Accuracy by fault resistance and by DER scenario (S1 no DER ... S6 high EV/V2G).
(c) Cross-scenario generalisation: train without DER (S1), test on DER scenarios S2-S6.
(d) Sensor placement among the 93 PMU-capable buses: proposed DMP (design grid simulated
    on the public Iowa 240 OpenDSS model) vs the dataset authors' NSGA-II observability
    placements (20 / 36 PMUs), MRA, observability ILP, random and greedy baselines.
    Selection-based methods use a 30 % selection split; scores use the remaining 70 %."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

import features as F
import models as M
import placement as P
from common import HERE, save

NPZ = f"{HERE}/data/derfault/derfault.npz"
FAST = dict(n_estimators=150, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1, random_state=0)


class Sub:
    """Row subset view of a features.Dataset."""
    def __init__(self, ds, idx):
        self.__dict__.update(ds.__dict__)
        for k in ("Vpre", "Vf", "Ipre", "If", "ftype", "fbus", "rf", "y"):
            setattr(self, k, getattr(ds, k)[idx])
        self.n = len(idx)


def cv_acc(X, y, rf=None, model=None):
    ok = np.zeros(len(y), bool)
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = (model or LGBMClassifier(**FAST)).fit(X[tr], y[tr])
        o = m.predict(X[te]) == y[te]
        ok[te] = o
        accs.append(o.mean())
    return np.array(accs), ok


def main():
    ds = F.Dataset(NPZ)
    scen = np.load(NPZ)["scenario"][np.isin(np.load(NPZ)["ftype"], F.CLASSES3)]
    allb = list(range(len(ds.buses)))
    out = {"n": ds.n, "n_pmu_buses": len(ds.buses), "classes": ds.classes}

    # (a) ladder
    lad = {}
    for noise in (0.01, 0.0):
        Vp, Vf = F.measure(ds, allb, np.random.RandomState(0), mag_sigma=noise / 3.0349, ang_sigma_deg=np.degrees(noise / 3.0349))
        for rep in ("AGG", "PPA", "SEQ", "FPS"):
            X = F.REPS[rep](Vp, Vf)
            names = ("RF", "SVM", "LR", "kNN", "MLP", "XGB", "LGBM") if noise else ("LGBM",)
            for name in names:
                acc, _, _ = M.cross_validate(M.zoo()[name], X, ds.y, seeds=(0,))
                lad[f"{noise}_{rep}_{name}"] = acc
                print("ladder", noise, rep, name, f"{acc.mean() * 100:.1f}", flush=True)
            if rep == "SEQ" and noise:
                acc, _, _ = M.cross_validate(M.RelayRule(), X, ds.y, seeds=(0,))
                lad[f"{noise}_SEQ_Relay rule"] = acc
                print("ladder relay", f"{acc.mean() * 100:.1f}", flush=True)
    out["ladder"] = lad

    # (b) by R_f and by DER scenario, AGG vs FPS, all PMUs, PMU noise
    Vp, Vf = F.measure(ds, allb, np.random.RandomState(0))
    edges = [0, 0.04, 1, 5, 10, 25.1]
    rfb = np.digitize(ds.rf, edges[1:-1])
    br = {}
    for rep in ("AGG", "SEQ", "FPS"):
        _, ok = cv_acc(F.REPS[rep](Vp, Vf), ds.y)
        br[rep] = {"by_rf": [float(ok[rfb == b].mean()) for b in range(5)],
                   "by_scenario": {s: float(ok[scen == s].mean()) for s in sorted(set(scen))}}
        print("breakdown", rep, br[rep], flush=True)
    out["breakdown"] = {"rf_edges": edges, **br}

    # (c) train on S1 (no DER), test on each DER scenario
    gen = {}
    for rep in ("AGG", "SEQ", "FPS"):
        X = F.REPS[rep](Vp, Vf)
        m = LGBMClassifier(**FAST).fit(X[scen == "S1"], ds.y[scen == "S1"])
        gen[rep] = {s: float((m.predict(X[scen == s]) == ds.y[scen == s]).mean()) for s in sorted(set(scen)) if s != "S1"}
        print("S1->", rep, gen[rep], flush=True)
    out["generalisation_from_S1"] = gen

    # (d) placement
    idx = np.arange(ds.n)
    sel_i, ev_i = train_test_split(idx, test_size=0.7, stratify=ds.y, random_state=0)
    sel, ev = Sub(ds, sel_i), Sub(ds, ev_i)
    gbuses, cols, dVp, dVf, meta = P.design_grid(240, rf_grid=[0.05, 0.5, 2, 5, 10, 15, 25])
    events, marg, sag = P.margins(cols, dVp, dVf, meta)
    cand = [gbuses.index(b) for b in ds.buses]  # PMU-capable buses only
    marg_c, sag_c = marg[:, cand], sag[:, cand]
    cfg = __import__("pandas").read_csv(f"{HERE}/data/derfault/pmu_placement_configurations.csv")

    def score(d, obs, rep="FPS"):
        Vp_, Vf_ = F.measure(d, obs, np.random.RandomState(0))
        acc, ok = cv_acc(F.REPS[rep](Vp_, Vf_), d.y)
        return {"acc": float(acc.mean()), "hif": float(ok[d.rf > 5].mean())}

    def wrap(o):
        Vp_, Vf_ = F.measure(sel, o, np.random.RandomState(1))
        X = F.fps(Vp_, Vf_)
        tr, te = train_test_split(np.arange(sel.n), test_size=0.3, stratify=sel.y, random_state=0)
        return (LGBMClassifier(**{**FAST, "n_estimators": 60}).fit(X[tr], sel.y[tr]).predict(X[te]) == sel.y[te]).mean()

    def info(o):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        Vp_, Vf_ = F.measure(sel, o, np.random.RandomState(1))
        Xs = StandardScaler().fit_transform(F.fps(Vp_, Vf_))
        _, nb = KNeighborsClassifier(2).fit(Xs, sel.y).kneighbors(Xs)
        return -np.mean(sel.y[nb[:, 1]] != sel.y)

    pl = {"n_design_sims": len(meta)}
    for k in (3, 5, 10, 20, 36):
        cands = {
            "DMP (proposed)": P.dmp(marg_c, k),
            "DMP N-1 (proposed)": P.dmp(marg_c, k, n1=True),
            "MRA ILP": P.mra(sag_c, k),
            "Information greedy": P.greedy(info, len(ds.buses), k),
        }
        if k <= 10:
            cands["Accuracy greedy"] = P.greedy(wrap, len(ds.buses), k)
        if f"PMU_{k}" in cfg:
            cands[f"NSGA-II (dataset authors, {k} PMU)"] = ds.bus_ids([b for b in cfg[f"PMU_{k}"].dropna() if b in ds.buses])
        rk = {}
        for name, obs in cands.items():
            rk[name] = {"buses": [ds.buses[i] for i in obs], **score(ev, obs)}
            print("placement", k, name, f"acc {rk[name]['acc'] * 100:.1f} HIF {rk[name]['hif'] * 100:.1f}", flush=True)
        rng = np.random.RandomState(5)
        rr = [score(ev, list(rng.choice(len(ds.buses), k, replace=False))) for _ in range(5)]
        rk["Random (5 draws)"] = {"acc": float(np.mean([r["acc"] for r in rr])), "hif": float(np.mean([r["hif"] for r in rr]))}
        print("placement", k, "Random", rk["Random (5 draws)"], flush=True)
        pl[k] = rk
        out["placement"] = pl
        save("e9_derfault", out)
    out["all_93"] = score(ev, allb)
    save("e9_derfault", out)


if __name__ == "__main__":
    main()
