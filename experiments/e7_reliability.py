"""E7 - Reliability-oriented evaluation.

(a) Selective prediction: split-conformal prediction sets (LAC score) and confidence
    thresholding -> accuracy vs coverage, i.e. how many faults can be classified
    automatically at a target accuracy and how many must be escalated.
(b) Measurement-system robustness on the three-bus set: loss of one PMU (its
    measurements frozen at pre-fault values) and PMU time misalignment (phase offset).
(c) Inverter-based DER penetration 0 / 30 / 60 % (current-limited Generator model 7)."""
import numpy as np
from sklearn.model_selection import train_test_split

import features as F
import models as M
from common import SIM, PAPER_3BUS, save

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


def selective(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    Xtr, Xcal, ytr, ycal = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=0)
    m = M.zoo()["LGBM"].fit(Xtr, ytr)
    pcal, pte = m.predict_proba(Xcal), m.predict_proba(Xte)
    scores = 1 - pcal[np.arange(len(ycal)), ycal]
    conf = []
    for a in ALPHAS:
        q = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - a)) / len(scores), method="higher")
        sets = pte >= 1 - q
        single = sets.sum(1) == 1
        conf.append({"alpha": a, "set_coverage": float(sets[np.arange(len(yte)), yte].mean()),
                     "singleton_share": float(single.mean()),
                     "singleton_acc": float((pte[single].argmax(1) == yte[single]).mean())})
    pmax, ok = pte.max(1), pte.argmax(1) == yte
    order = np.argsort(-pmax)
    cov = np.linspace(0.1, 1, 19)
    curve = [float(ok[order[:max(1, int(c * len(ok)))]].mean()) for c in cov]
    return {"conformal": conf, "coverage": cov, "acc_at_coverage": curve}


def robustness(ds):
    obs = ds.bus_ids(PAPER_3BUS[13])
    rng = np.random.RandomState(0)
    Vp, Vf = F.measure(ds, obs, rng)
    idx = np.arange(ds.n)
    tr, te = train_test_split(idx, test_size=0.3, stratify=ds.y, random_state=0)
    out = {}
    for rep in ("SEQ", "FPS"):
        m = M.zoo()["LGBM"].fit(F.REPS[rep](Vp[tr], Vf[tr]), ds.y[tr])
        base = float((m.predict(F.REPS[rep](Vp[te], Vf[te])) == ds.y[te]).mean())
        r = {"nominal": base, "lost_pmu": {}, "time_skew": {}}
        for j, b in enumerate(PAPER_3BUS[13]):
            Vf_l = Vf[te].copy()
            Vf_l[:, j] = Vp[te][:, j]  # lost PMU -> stale pre-fault value
            r["lost_pmu"][b] = float((m.predict(F.REPS[rep](Vp[te], Vf_l)) == ds.y[te]).mean())
        for deg in (1, 2, 5, 10):
            Vf_s = Vf[te].copy()
            Vf_s[:, 1] = Vf_s[:, 1] * np.exp(1j * np.deg2rad(deg))  # one PMU's clock offset
            r["time_skew"][deg] = float((m.predict(F.REPS[rep](Vp[te], Vf_s)) == ds.y[te]).mean())
        out[rep] = r
        print("robust", rep, r, flush=True)
    return out


def der():
    out = {}
    edges = np.logspace(-3, 2, 6)
    for pen in (0, 30, 60):
        ds = F.Dataset(f"{SIM}/ieee13_der{pen}.npz")
        bins = np.digitize(ds.rf, edges[1:-1])
        r = {}
        for sensing, obs in [("all", list(range(len(ds.buses)))), ("3bus", ds.bus_ids(PAPER_3BUS[13]))]:
            Vp, Vf = F.measure(ds, obs, np.random.RandomState(0))
            for rep in ("AGG", "SEQ", "FPS"):
                acc, _, oof = M.cross_validate(M.zoo()["LGBM"], F.REPS[rep](Vp, Vf), ds.y)
                ok = oof == ds.y
                r[f"{sensing}_{rep}"] = {"acc": acc, "by_rf": [float(ok[bins == b].mean()) for b in range(5)]}
                print("DER", pen, sensing, rep, f"{acc.mean() * 100:.1f}", flush=True)
        out[pen] = r
    out["rf_edges"] = edges
    return out


def main():
    ds = F.Dataset(f"{SIM}/ieee13.npz")
    out = {}
    for sensing, obs in [("3bus", ds.bus_ids(PAPER_3BUS[13])), ("all", list(range(len(ds.buses))))]:
        Vp, Vf = F.measure(ds, obs, np.random.RandomState(0))
        out[f"selective_{sensing}"] = {rep: selective(F.REPS[rep](Vp, Vf), ds.y) for rep in ("AGG", "SEQ", "FPS")}
        for rep, s in out[f"selective_{sensing}"].items():
            print("selective", sensing, rep, [(c["alpha"], round(c["singleton_share"], 3), round(c["singleton_acc"], 3))
                                              for c in s["conformal"]], flush=True)
    out["robustness"] = robustness(ds)
    out["der"] = der()
    save("e7_reliability", out)


if __name__ == "__main__":
    main()
