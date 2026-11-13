"""E13 - Extended classifier baselines on every feeder (three buses and all buses, 1 % TVE and uPMU noise).

Adds to the E1 zoo: LDA, QDA, Gaussian naive Bayes, ExtraTrees, a stacking ensemble, a tabular
1D-CNN and a model-based template matcher that needs no labelled data (nearest noise-free design-grid
signature). Reports accuracy and macro-F1 per fold, mean rank (Friedman test) and McNemar tests against LGBM."""
import numpy as np
from scipy.stats import chi2, friedmanchisquare
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import models as M
import placement as P
from common import SIM, PAPER_3BUS, save, graph_edges
from e2_scale import topo_3bus

NOISE = {"1% TVE": (0.0032951, 0.18879), "uPMU": (0.0002, 0.002)}


class CNN1D(BaseEstimator, ClassifierMixin):
    """Small 1D-CNN over the standardised feature vector (representative deep baseline)."""
    def __init__(self, epochs=40, lr=2e-3, seed=0):
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        self.sc_ = StandardScaler().fit(X)
        Xt = torch.tensor(self.sc_.transform(X), dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y, dtype=torch.long)
        self.classes_ = np.unique(y)
        self.net_ = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.GELU(),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
                                  nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Dropout(0.2),
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, len(self.classes_)))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-5)
        for _ in range(self.epochs):
            self.net_.train()
            for b in torch.randperm(len(yt)).split(256):
                opt.zero_grad()
                nn.functional.cross_entropy(self.net_(Xt[b]), yt[b]).backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        self.net_.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.sc_.transform(X), dtype=torch.float32).unsqueeze(1)
            return self.net_(Xt).argmax(1).numpy()


def extra_models():
    base = [("rf", RandomForestClassifier(100, n_jobs=-1, random_state=0)),
            ("mlp", M.zoo()["MLP"]), ("lgbm", M.zoo()["LGBM"].set_params(n_estimators=150))]
    return {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05)),
        "NB": make_pipeline(StandardScaler(), GaussianNB()),
        "ExtraTrees": ExtraTreesClassifier(300, n_jobs=-1, random_state=0),
        "Stacking": StackingClassifier(base, LogisticRegression(max_iter=1000), cv=3),
        "CNN1D": CNN1D(),
    }


def template_predict(nb, ds, obs, Vp, Vf):
    """Model-based baseline: nearest noise-free design-grid signature, no labelled measurements."""
    _, cols, dVp, dVf, meta = GRID[nb]
    sel = cols[[GRID_BUSES[nb].index(ds.buses[i]) for i in obs]]
    Mp, Mf = np.full((len(meta),) + sel.shape, np.nan + 0j), np.full((len(meta),) + sel.shape, np.nan + 0j)
    v = sel >= 0
    Mp[:, v], Mf[:, v] = dVp[:, sel[v]], dVf[:, sel[v]]
    T = P.bus_signature(Mp, Mf).reshape(len(meta), -1)
    S = P.bus_signature(Vp, Vf).reshape(len(Vp), -1)
    lab = np.array([ds.classes.index(m[1]) for m in meta])
    out = np.zeros(len(S), int)
    for i in range(0, len(S), 500):
        out[i:i + 500] = lab[((S[i:i + 500, None] - T[None]) ** 2).sum(2).argmin(1)]
    return out


GRID, GRID_BUSES = {}, {}


def mcnemar(a_ok, b_ok):
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return float(chi2.sf((abs(b - c) - 1) ** 2 / max(b + c, 1), 1))


def main():
    out = {}
    fold_table = []
    for nb in (13, 34, 123):
        ds = F.Dataset(f"{SIM}/ieee{nb}.npz")
        GRID[nb] = P.design_grid(nb)
        GRID_BUSES[nb] = list(GRID[nb][0])
        tri = ds.bus_ids(PAPER_3BUS[13]) if nb == 13 else topo_3bus(ds, graph_edges(ds, nb))
        res = {}
        for lvl, (sm, sa) in NOISE.items():
            Vp, Vf = F.measure(ds, tri, np.random.RandomState(0), mag_sigma=sm, ang_sigma_deg=sa)
            for rep in ("AGG", "SEQ", "FPS"):
                X = F.REPS[rep](Vp, Vf)
                models = {**{k: M.zoo()[k] for k in ("RF", "SVM", "LR", "kNN", "MLP", "XGB", "LGBM")}, **extra_models()}
                preds, r = {}, {}
                for name, mdl in models.items():
                    accs, f1s, oof = [], [], np.zeros(ds.n, int)
                    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, ds.y):
                        p = np.asarray(clone(mdl).fit(X[tr], ds.y[tr]).predict(X[te])).ravel()
                        oof[te] = p
                        accs.append(accuracy_score(ds.y[te], p))
                        f1s.append(f1_score(ds.y[te], p, average="macro"))
                    preds[name] = oof == ds.y
                    r[name] = {"acc": accs, "f1": f1s, "hif": float(preds[name][ds.rf > 10].mean())}
                    print(nb, lvl, rep, name, f"{np.mean(accs) * 100:.1f} F1 {np.mean(f1s):.3f}", flush=True)
                for name in preds:
                    if name != "LGBM":
                        r[name]["mcnemar_vs_LGBM"] = mcnemar(preds["LGBM"], preds[name])
                res[f"{lvl}|{rep}"] = r
                fold_table.append((nb, lvl, rep, {k: np.mean(v["acc"]) for k, v in r.items()}))
            t = template_predict(nb, ds, tri, Vp, Vf) == ds.y
            res[f"{lvl}|Template"] = {"acc": float(t.mean()), "hif": float(t[ds.rf > 10].mean())}
            print(nb, lvl, "Template (no labels)", f"{t.mean() * 100:.1f}", flush=True)
        out[f"ieee{nb}"] = res
        names = list(fold_table[0][3])
        mat = np.array([[row[3][n] for n in names] for row in fold_table])
        out["mean_rank"] = dict(zip(names, (-mat).argsort(1).argsort(1).mean(0) + 1))
        out["friedman_p"] = float(friedmanchisquare(*mat.T).pvalue)
        save("e13_classifiers", out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
