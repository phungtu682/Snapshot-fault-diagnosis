"""E3 - Fault impedance x measurement noise (claim C3).

Accuracy per fault-resistance bin for PMU noise levels from ideal to 2 %, for the paper
representation (AGG) and the full snapshot (FPS), under all-bus and three-bus sensing.
This locates the detectability boundary in the (R_f, noise) plane."""
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import features as F
import models as M
from common import SIM, PAPER_3BUS, save

NOISE = [0.0, 0.001, 0.003, 0.01, 0.02]          # magnitude sigma; angle sigma = 50x in degrees
RF_EDGES = np.logspace(-3, 2, 11)


def main():
    ds = F.Dataset(f"{SIM}/ieee13.npz")
    bins = np.digitize(ds.rf, RF_EDGES[1:-1])
    out = {"rf_edges": RF_EDGES, "noise": NOISE, "bin_counts": np.bincount(bins)}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for sensing, obs in [("all", list(range(len(ds.buses)))), ("3bus", ds.bus_ids(PAPER_3BUS[13]))]:
        out[sensing] = {}
        for rep in ("AGG", "SEQ", "FPS"):
            grid = []
            for s in NOISE:
                Vp, Vf = F.measure(ds, obs, np.random.RandomState(1), mag_sigma=s / 3.0349, ang_sigma_deg=np.degrees(s / 3.0349))
                X = F.REPS[rep](Vp, Vf)
                ok = cross_val_predict(M.zoo()["LGBM"], X, ds.y, cv=cv) == ds.y
                grid.append([float(ok[bins == b].mean()) for b in range(len(RF_EDGES) - 1)] + [float(ok.mean())])
                print(sensing, rep, s, f"{ok.mean() * 100:.1f}", flush=True)
            out[sensing][rep] = grid
    save("e3_rf_noise", out)


if __name__ == "__main__":
    main()
