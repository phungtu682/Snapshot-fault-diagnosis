"""E16 - DMP-2 (error-bound placement) against all placement baselines, chosen without seeing evaluation data.

Paired Wilcoxon tests use identical folds and noise. Baseline bus sets come from E4b and E14 (IEEE feeders)
and E9 (DER-Fault). Also checks how well the bound predicts the measured error on random sensor sets."""
import json
import time
import numpy as np
from scipy.stats import spearmanr, wilcoxon

import features as F
import placement as P
import placement2 as Q
from common import HERE, RES, SIM, save
from e14_placement_baselines import folds

KS = {13: [2, 3, 4, 5], 34: [2, 3, 4, 6], 123: [3, 4, 6, 8, 9]}
load = lambda n: json.load(open(f"{RES}/{n}.json"))


def compare(ds, sets, ref="DMP-2 (proposed)"):
    fr = {n: folds(ds, ds.bus_ids(b)) for n, b in sets.items()}
    out = {}
    for n, (a, h, n1) in fr.items():
        out[n] = {"buses": sets[n], "acc": float(a.mean()), "hif": float(h.mean()), "n1": float(n1.mean())}
        if n != ref:
            for lab, x, y in (("acc", fr[ref][0], a), ("hif", fr[ref][1], h), ("n1", fr[ref][2], n1)):
                out[n][f"p_{lab}"] = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
                out[n][f"diff_{lab}"] = float((x - y).mean())
    return out


def main():
    e4b, e14 = load("e4b_dmp"), load("e14_placement_baselines")
    out = {}
    for nb, ks in KS.items():
        ds = F.Dataset(f"{SIM}/ieee{nb}.npz")
        buses, cols, Vp, Vf, meta = P.design_grid(nb)
        assert list(buses) == ds.buses
        _, m2, w = Q.margins2(cols, Vp, Vf, meta)
        res = {}
        for k in ks:
            t = time.time()
            ob, gap = Q.dmp2(m2, w, k)
            sec = time.time() - t
            ob1, _ = Q.dmp2(m2, w, k, n1=True)
            sets = {"DMP-2 (proposed)": [ds.buses[i] for i in ob], "DMP-2 N-1 (proposed)": [ds.buses[i] for i in ob1]}
            for src in (e4b[f"ieee{nb}"]["k"].get(str(k), {}), e14[f"ieee{nb}"].get(str(k), {})):
                for n, v in src.items():
                    if "buses" in v and n not in sets:
                        sets[n] = v["buses"]
            r = compare(ds, sets)
            r["DMP-2 (proposed)"].update(select_s=sec, mip_gap=gap, bound=Q.bound(m2, w, ob))
            res[k] = r
            for n, v in r.items():
                print(nb, k, f"{n:22s} acc {v['acc'] * 100:.2f} HIF {v['hif'] * 100:.2f} N-1 {v['n1'] * 100:.2f} "
                      f"p_acc {v.get('p_acc', 1):.3f} p_hif {v.get('p_hif', 1):.3f}", flush=True)
        # does the bound predict measured error?
        rng = np.random.RandomState(3)
        pts = []
        for _ in range(40):
            k = int(rng.choice(ks))
            o = sorted(rng.choice(len(ds.buses), k, replace=False).tolist())
            Vp_, Vf_ = F.measure(ds, o, np.random.RandomState(0))
            from sklearn.model_selection import cross_val_score
            from lightgbm import LGBMClassifier
            acc = cross_val_score(LGBMClassifier(n_estimators=100, n_jobs=-1, verbose=-1), F.fps(Vp_, Vf_), ds.y, cv=3).mean()
            pts.append((Q.bound(m2, w, o), 1 - acc, k))
        pts = np.array(pts)
        rho = spearmanr(pts[:, 0], pts[:, 1])
        res["bound_vs_error"] = {"points": pts.tolist(), "spearman": float(rho.statistic), "p": float(rho.pvalue)}
        print(nb, "bound vs error Spearman", rho, flush=True)
        out[f"ieee{nb}"] = res
        save("e16_dmp2", out)

    # DER-Fault: design grid on the public Iowa 240 model, candidates limited to PMU-capable buses
    from e9_derfault import NPZ
    ds = F.Dataset(NPZ)
    gb, cols, Vp, Vf, meta = P.design_grid(240, rf_grid=[0.05, 0.5, 2, 5, 10, 15, 25])
    _, m2, w = Q.margins2(cols, Vp, Vf, meta)
    m2 = m2[:, [gb.index(b) for b in ds.buses]]
    e9 = load("e9_derfault")["placement"]
    res = {}
    for k in (3, 5, 10, 20, 36):
        ob, gap = Q.dmp2(m2, w, k, time_limit=120)
        sets = {"DMP-2 (proposed)": [ds.buses[i] for i in ob]}
        for n, v in e9[str(k)].items():
            if "buses" in v:
                sets[n] = v["buses"]
        r = compare(ds, sets)
        res[k] = r
        for n, v in r.items():
            print("der", k, f"{n:36s} acc {v['acc'] * 100:.2f} HIF {v['hif'] * 100:.2f} p_acc {v.get('p_acc', 1):.3f}", flush=True)
        out["derfault"] = res
        save("e16_dmp2", out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
