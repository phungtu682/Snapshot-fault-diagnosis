"""E17 - Robustness checks of E15 repeated over five random splits and noise draws, so that
every entry of the robustness table has a standard deviation (three DMP PMUs, FPS, LightGBM)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

import features as F
from common import SIM, save
from e15_tasks_robustness import DMP3, FAST, LEVELS

REPEATS = range(5)


def fit_score(Xtr, ytr, Xte, yte, seed):
    return float((LGBMClassifier(**{**FAST, "random_state": seed}).fit(Xtr, ytr).predict(Xte) == yte).mean())


def main():
    out = {}
    for nb in (13, 34, 123):
        ds = F.Dataset(f"{SIM}/ieee{nb}.npz")
        tri = ds.bus_ids(DMP3[nb])
        raw = np.load(f"{SIM}/ieee{nb}.npz")
        scale = raw["scale"][np.isin(raw["ftype"], F.CLASSES3)]
        low, high = np.where(scale < 1.0)[0], np.where(scale >= 1.0)[0]
        r = {k: [] for k in ("reference", "load_shift", "upmu_to_tve", "n1", "n2_mean", "n2_worst", "shuffled")}
        for s in REPEATS:
            tve = F.measure(ds, tri, np.random.RandomState(100 + s), *LEVELS["1% TVE"])
            upmu = F.measure(ds, tri, np.random.RandomState(200 + s), *LEVELS["uPMU"])
            X, Xu = F.fps(*tve), F.fps(*upmu)
            tr, te = train_test_split(np.arange(ds.n), test_size=0.3, stratify=ds.y, random_state=s)
            m = LGBMClassifier(**{**FAST, "random_state": s}).fit(X[tr], ds.y[tr])
            r["reference"].append(float((m.predict(X[te]) == ds.y[te]).mean()))
            r["upmu_to_tve"].append(fit_score(Xu[tr], ds.y[tr], X[te], ds.y[te], s))
            r["load_shift"].append(fit_score(X[low], ds.y[low], X[high], ds.y[high], s))
            r["shuffled"].append(fit_score(X[tr], np.random.RandomState(s).permutation(ds.y)[tr], X[te], ds.y[te], s))
            Vp, Vf = tve
            lost1, lost2 = [], []
            for i in range(3):
                Vl = Vf[te].copy()
                Vl[:, i] = Vp[te][:, i]
                lost1.append(float((m.predict(F.fps(Vp[te], Vl)) == ds.y[te]).mean()))
                for j in range(i + 1, 3):
                    Vl = Vf[te].copy()
                    Vl[:, [i, j]] = Vp[te][:, [i, j]]
                    lost2.append(float((m.predict(F.fps(Vp[te], Vl)) == ds.y[te]).mean()))
            r["n1"].append(float(np.mean(lost1)))
            r["n2_mean"].append(float(np.mean(lost2)))
            r["n2_worst"].append(float(np.min(lost2)))
        out[f"ieee{nb}"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "runs": v} for k, v in r.items()}
        for k, v in out[f"ieee{nb}"].items():
            print(nb, f"{k:12s} {100 * v['mean']:.1f} +- {100 * v['sd']:.1f}", flush=True)
        save("e17_robust_sd", out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
