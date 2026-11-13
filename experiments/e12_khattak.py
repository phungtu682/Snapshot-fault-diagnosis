"""E12 - Head-to-head with Khattak & Choudhry (arXiv 2509.02192), the closest prior work:
greedy SVM-wrapper PMU selection (FSNR-SVM) on IEEE 34 / 123 with 5 / 9 PMUs.

Their published bus sets are scored against DMP and random sets of the same size under
(i) their conditions (R_f 0.01-10 ohm, no measurement noise), (ii) the measured uPMU noise
floor over the full R_f range and (iii) the 1 % TVE envelope over the full R_f range.
Their 34-bus set contains a DER bus (busPV1) absent from the standard feeder; it is dropped
and both sets use k = 4 there."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

import features as F
import placement as P
from common import SIM, save
from e9_derfault import Sub

FAST = dict(n_estimators=150, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1, random_state=0)
KHATTAK = {34: ["800", "848", "850", "854"],
           123: ["149", "83", "151", "300", "95", "250", "66", "56", "450"]}
CONDITIONS = {"Khattak (Rf 0.01-10, no noise)": ((0.01, 10.0), 0.0, 0.0),
              "uPMU measured (full Rf)": ((0, 1e9), 0.0002, 0.002),
              "1% TVE (full Rf)": ((0, 1e9), 0.0032951, 0.18879)}


def score(ds, obs, rep, sm, sa):
    Vp, Vf = F.measure(ds, obs, np.random.RandomState(0), mag_sigma=sm, ang_sigma_deg=sa)
    X = F.REPS[rep](Vp, Vf)
    ok = np.zeros(ds.n, bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, ds.y):
        ok[te] = LGBMClassifier(**FAST).fit(X[tr], ds.y[tr]).predict(X[te]) == ds.y[te]
    return {"acc": float(ok.mean()), "hif": float(ok[ds.rf > 10].mean()) if (ds.rf > 10).any() else None}


def main():
    out = {}
    for nb, theirs in KHATTAK.items():
        full = F.Dataset(f"{SIM}/ieee{nb}.npz")
        theirs = [b for b in theirs if b in full.buses]
        k = len(theirs)
        _, cols, Vp, Vf, meta = P.design_grid(nb)
        _, marg, _ = P.margins(cols, Vp, Vf, meta)
        rng = np.random.RandomState(7)
        sets = {"Khattak FSNR-SVM": full.bus_ids(theirs), "DMP (proposed)": P.dmp(marg, k),
                "DMP N-1 (proposed)": P.dmp(marg, k, n1=True)}
        sets.update({f"Random {i}": list(rng.choice(len(full.buses), k, replace=False)) for i in range(5)})
        res = {"k": k, "buses": {n: [full.buses[i] for i in o] for n, o in sets.items()}}
        print(nb, res["buses"], flush=True)
        for cond, ((lo, hi), sm, sa) in CONDITIONS.items():
            ds = Sub(full, np.where((full.rf >= lo) & (full.rf <= hi))[0])
            for rep in ("SEQ", "FPS"):
                r = {n: score(ds, o, rep, sm, sa) for n, o in sets.items()}
                rand = [r.pop(f"Random {i}") for i in range(5)]
                r["Random (5 draws)"] = {"acc": float(np.mean([x["acc"] for x in rand])),
                                         "hif": None if rand[0]["hif"] is None else float(np.mean([x["hif"] for x in rand]))}
                res[f"{cond}|{rep}"] = r
                for n, v in r.items():
                    hif = "" if v["hif"] is None else f" HIF {v['hif'] * 100:.1f}"
                    print(f"{nb} {cond:32s} {rep} {n:20s} acc {v['acc'] * 100:.1f}{hif}", flush=True)
        out[f"ieee{nb}"] = res
        save("e12_khattak", out)


if __name__ == "__main__":
    main()
