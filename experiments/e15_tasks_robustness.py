"""E15 - Harder tasks and distribution shift (LGBM; three DMP buses and all buses).

(a) label-shuffling sanity check; (b) 4-class (with LLG) and 10-class phase-resolved fault type;
(c) fault-bus location; (d) per-class recall, confusion matrix and expected calibration error;
(e) learning curve; (f) noise mismatch between training and test; (g) load shift (train at low
load, test at high load); (h) loss of two PMUs; (i) transfer between feeders with SEQ."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import features as F
import placement as P
from common import SIM, save

FAST = dict(n_estimators=150, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1, random_state=0)
LEVELS = {"1% TVE": (0.0032951, 0.18879), "uPMU": (0.0002, 0.002)}
DMP3 = {13: ["634", "680", "692"], 34: ["830", "888", "890"], 123: ["250", "93", "300_open"]}


def cv(X, y, splits=5):
    ok, proba = np.zeros(len(y), bool), np.zeros((len(y), len(np.unique(y))))
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(splits, shuffle=True, random_state=0).split(X, y):
        m = LGBMClassifier(**FAST).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])
        pred[te] = proba[te].argmax(1)
        ok[te] = pred[te] == y[te]
    return ok, proba, pred


def ece(proba, ok, bins=15):
    conf = proba.max(1)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(ok[s].mean() - conf[s].mean())
    return float(e)


def phase_label(ftype, phases):
    return np.array([f"{t}-{p}" if t != "3PH" else "3PH" for t, p in zip(ftype, phases)])


def main():
    out = {}
    for nb in (13, 34, 123):
        res = {}
        ds = F.Dataset(f"{SIM}/ieee{nb}.npz")
        tri = ds.bus_ids(DMP3[nb])
        allb = list(range(len(ds.buses)))
        Vp, Vf = F.measure(ds, tri, np.random.RandomState(0))
        X4 = F.fps(Vp, Vf)

        # (a) sanity check
        rng = np.random.RandomState(0)
        res["shuffled_labels"] = float(cv(X4, rng.permutation(ds.y))[0].mean())
        print(nb, "shuffled", res["shuffled_labels"], flush=True)

        # (b) harder label sets, (c) location
        raw = np.load(f"{SIM}/ieee{nb}.npz")
        d4 = F.Dataset(f"{SIM}/ieee{nb}.npz", task="4class")
        keep4 = np.isin(raw["ftype"], F.CLASSES4)
        ph = phase_label(raw["ftype"][keep4], raw["phases"][keep4])
        labs, cnt = np.unique(ph, return_counts=True)
        rare = labs[cnt < 10]
        y_ph = np.searchsorted(labs, ph)
        mask = ~np.isin(ph, rare)
        y_ph = np.unique(y_ph[mask], return_inverse=True)[1]
        fb, fcnt = np.unique(ds.fbus, return_counts=True)
        m_loc = np.isin(ds.fbus, fb[fcnt >= 10])
        y_loc = np.unique(ds.fbus[m_loc], return_inverse=True)[1]
        for lvl, (sm, sa) in LEVELS.items():
            r = {}
            for sens, obs3, obs4 in (("3bus", tri, d4.bus_ids(DMP3[nb])), ("all", allb, list(range(len(d4.buses))))):
                a, b = F.measure(d4, obs4, np.random.RandomState(0), mag_sigma=sm, ang_sigma_deg=sa)
                c, d = F.measure(ds, obs3, np.random.RandomState(0), mag_sigma=sm, ang_sigma_deg=sa)
                for rep in ("AGG", "SEQ", "FPS"):
                    r[f"{sens}|{rep}|4class"] = float(cv(F.REPS[rep](a, b), d4.y)[0].mean())
                    r[f"{sens}|{rep}|phase"] = float(cv(F.REPS[rep](a, b)[mask], y_ph)[0].mean())
                    # SEQ is phase-agnostic by design, so it cannot identify phases: reported to show this.
                    print(nb, lvl, sens, rep, "4c", f"{r[f'{sens}|{rep}|4class'] * 100:.1f}",
                          "phase", f"{r[f'{sens}|{rep}|phase'] * 100:.1f}", flush=True)
                r[f"{sens}|FPS|location"] = float(cv(F.fps(c, d)[m_loc], y_loc, splits=3)[0].mean())
                print(nb, lvl, sens, "location", f"{r[f'{sens}|FPS|location'] * 100:.1f}", flush=True)
            res[lvl] = r
        res["n_phase_classes"] = int(y_ph.max() + 1)
        res["n_location_classes"] = int(y_loc.max() + 1)

        # (d) per-class behaviour and calibration
        for rep in ("AGG", "FPS"):
            ok, proba, pred = cv(F.REPS[rep](Vp, Vf), ds.y)
            res[f"class_{rep}"] = {"recall": recall_score(ds.y, pred, average=None).tolist(), "ece": ece(proba, ok),
                                   "confusion": confusion_matrix(ds.y, pred).tolist(), "classes": ds.classes}
            print(nb, rep, "recall", res[f"class_{rep}"]["recall"], "ECE", res[f"class_{rep}"]["ece"], flush=True)

        # (e) learning curve
        tr, te = train_test_split(np.arange(ds.n), test_size=0.3, stratify=ds.y, random_state=0)
        curve = {}
        for n in (100, 250, 500, 1000, 2000, 4000, len(tr)):
            n = min(n, len(tr))
            sub = np.random.RandomState(0).choice(tr, n, replace=False)
            curve[n] = {rep: float((LGBMClassifier(**FAST).fit(F.REPS[rep](Vp, Vf)[sub], ds.y[sub])
                                    .predict(F.REPS[rep](Vp, Vf)[te]) == ds.y[te]).mean()) for rep in ("AGG", "SEQ", "FPS")}
        res["learning_curve"] = curve
        print(nb, "learning", curve, flush=True)

        # (f) noise mismatch
        mm = {}
        data = {l: F.measure(ds, tri, np.random.RandomState(i), mag_sigma=s[0], ang_sigma_deg=s[1])
                for i, (l, s) in enumerate(LEVELS.items())}
        for a in LEVELS:
            for b in LEVELS:
                m = LGBMClassifier(**FAST).fit(F.fps(*data[a])[tr], ds.y[tr])
                mm[f"{a}->{b}"] = float((m.predict(F.fps(*data[b])[te]) == ds.y[te]).mean())
        res["noise_mismatch"] = mm
        print(nb, "mismatch", mm, flush=True)

        # (g) load shift
        scale = raw["scale"][np.isin(raw["ftype"], F.CLASSES3)]
        low, high = np.where(scale < 1.0)[0], np.where(scale >= 1.0)[0]
        ls = {}
        for rep in ("AGG", "SEQ", "FPS"):
            X = F.REPS[rep](Vp, Vf)
            ls[rep] = float((LGBMClassifier(**FAST).fit(X[low], ds.y[low]).predict(X[high]) == ds.y[high]).mean())
        res["load_shift_low_to_high"] = ls
        print(nb, "load shift", ls, flush=True)

        # (h) two PMUs lost
        m = LGBMClassifier(**FAST).fit(X4[tr], ds.y[tr])
        n2 = []
        for i in range(3):
            for j in range(i + 1, 3):
                Vl = Vf[te].copy()
                Vl[:, [i, j]] = Vp[te][:, [i, j]]
                n2.append(float((m.predict(F.fps(Vp[te], Vl)) == ds.y[te]).mean()))
        res["n2_mean"], res["n2_min"] = float(np.mean(n2)), float(np.min(n2))
        print(nb, "N-2", res["n2_mean"], res["n2_min"], flush=True)
        out[f"ieee{nb}"] = res
        save("e15_tasks_robustness", out)

    # (i) transfer between feeders: SEQ has the same dimension on every feeder
    tf = {}
    sets = {nb: F.Dataset(f"{SIM}/ieee{nb}.npz") for nb in DMP3}
    feats = {nb: F.seq(*F.measure(d, d.bus_ids(DMP3[nb]), np.random.RandomState(0))) for nb, d in sets.items()}
    for a in DMP3:
        m = LGBMClassifier(**FAST).fit(feats[a], sets[a].y)
        for b in DMP3:
            if a != b:
                tf[f"{a}->{b}"] = float((m.predict(feats[b]) == sets[b].y).mean())
    out["transfer_SEQ"] = tf
    print("transfer", tf, flush=True)
    save("e15_tasks_robustness", out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
