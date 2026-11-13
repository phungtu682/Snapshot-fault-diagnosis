"""E11 - Re-evaluation under the empirically measured micro-PMU noise floor (E10).

E10 shows that the pre/post snapshot ratio on a real uPMU varies by ~0.01-0.02 % in
magnitude and ~0.002 deg in angle over 1-15 cycles, far below the 1 % TVE envelope.
Because snapshot features are ratios at the same device, fixed calibration errors of the
PT/CT chain cancel. Here every feeder (and DER-Fault) is re-scored at three noise levels:
measured uPMU floor, a 5x conservative uPMU level, and the 1 % TVE envelope."""
import numpy as np

import features as F
import models as M
from common import SIM, HERE, PAPER_3BUS, save, graph_edges
from e2_scale import topo_3bus

LEVELS = {"uPMU measured": (0.0002, 0.002), "uPMU x5": (0.001, 0.01), "1% TVE": (0.0032951, 0.18879)}
RF_EDGES = [0, 1, 10, 50, 100.1]


def main():
    out = {"levels": LEVELS, "rf_edges": RF_EDGES}
    sets = [(f"ieee{nb}", f"{SIM}/ieee{nb}.npz", nb) for nb in (13, 34, 123)]
    sets.append(("derfault (Iowa 240)", f"{HERE}/data/derfault/derfault.npz", None))
    for name, path, nb in sets:
        ds = F.Dataset(path)
        if nb == 13:
            tri = ds.bus_ids(PAPER_3BUS[13])
        elif nb:
            tri = topo_3bus(ds, graph_edges(ds, nb))
        else:
            dmp = __import__("common").load("e9_derfault")["placement"]["3"]["DMP (proposed)"]["buses"]
            tri = ds.bus_ids(dmp)
        bins = np.digitize(ds.rf, RF_EDGES[1:-1])
        res = {}
        for sensing, obs in [("all", list(range(len(ds.buses)))), ("3bus", tri)]:
            for lvl, (sm, sa) in LEVELS.items():
                Vp, Vf = F.measure(ds, obs, np.random.RandomState(0), mag_sigma=sm, ang_sigma_deg=sa)
                for rep in ("AGG", "SEQ", "FPS"):
                    acc, _, oof = M.cross_validate(M.zoo()["LGBM"], F.REPS[rep](Vp, Vf), ds.y, seeds=(0,))
                    ok = oof == ds.y
                    res[f"{sensing}|{lvl}|{rep}"] = {"acc": float(acc.mean()), "acc_sd": float(acc.std(ddof=1)), "folds": acc.tolist(),
                                                    "by_rf": [float(ok[bins == b].mean()) if (bins == b).any() else None
                                                              for b in range(4)]}
                    print(name, sensing, lvl, rep, f"{acc.mean() * 100:.1f}", res[f"{sensing}|{lvl}|{rep}"]["by_rf"], flush=True)
        out[name] = res
        save("e11_upmu_noise", out)


if __name__ == "__main__":
    main()
