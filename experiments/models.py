"""Classifier zoo, physics rule baseline, GNN, and the shared cross-validation protocol."""
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def zoo(seed=0):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    sc = StandardScaler
    return {
        "RF": RandomForestClassifier(300, n_jobs=-1, random_state=seed),
        "SVM": make_pipeline(sc(), SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)),
        "LR": make_pipeline(sc(), LogisticRegression(max_iter=2000)),
        "kNN": make_pipeline(sc(), KNeighborsClassifier(5, weights="distance")),
        "MLP": make_pipeline(sc(), MLPClassifier((64, 32), max_iter=1000, early_stopping=True, random_state=seed)),
        "XGB": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, n_jobs=-1, random_state=seed, verbosity=0),
        "LGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=seed,
                               n_jobs=-1, verbose=-1),
    }


class RelayRule(BaseEstimator, ClassifierMixin):
    """Non-learned protection logic on SEQ columns 0-4 (sorted dips d1>=d2>=d3, max V0/V1):
    ground involvement (V0/V1 > tau) -> SLG; otherwise all three phases depressed alike
    (d3 > theta * d1) -> 3PH, else LL. The two thresholds are tuned on the training fold."""

    def __init__(self, taus=np.linspace(0.01, 0.6, 60), thetas=np.linspace(0.1, 0.95, 35)):
        self.taus, self.thetas = taus, thetas

    @staticmethod
    def _rule(X, tau, theta):
        d1, d3, z0 = X[:, 0], X[:, 2], X[:, 3]
        return np.where(z0 > tau, 0, np.where(d3 > theta * d1, 2, 1))

    def fit(self, X, y):
        best = (-1, None)
        for t in self.taus:
            for th in self.thetas:
                acc = np.mean(self._rule(X, t, th) == y)
                if acc > best[0]:
                    best = (acc, (t, th))
        self.tau_, self.theta_ = best[1]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._rule(X, self.tau_, self.theta_)


def cross_validate(model, X, y, n_splits=5, seeds=(0, 1), groups_for_report=None):
    """Stratified k-fold repeated over seeds. Returns per-fold accuracy, macro-F1 and, if
    groups_for_report is given, the out-of-fold predictions of the first repetition."""
    accs, f1s, oof = [], [], None
    for s in seeds:
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
        pred = np.empty_like(y)
        for tr, te in skf.split(X, y):
            m = clone(model)
            m.fit(X[tr], y[tr])
            p = np.asarray(m.predict(X[te])).ravel()
            pred[te] = p
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro"))
        if oof is None:
            oof = pred
    return np.array(accs), np.array(f1s), oof


def ci95(a):
    a = np.asarray(a)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def time_inference(model, X, n_single=300):
    """Per-sample latency (single-row calls, single thread where configurable) and batch throughput."""
    m = model
    if hasattr(m, "set_params"):
        for k in ("n_jobs", "randomforestclassifier__n_jobs"):
            try:
                m.set_params(**{k: 1})
            except ValueError:
                pass
    t = time.perf_counter()
    for i in range(n_single):
        m.predict(X[i % len(X)][None, :])
    single = (time.perf_counter() - t) / n_single * 1e6
    t = time.perf_counter()
    m.predict(X)
    batch = (time.perf_counter() - t) / len(X) * 1e6
    return single, batch


# ---------------------------------------------------------------- GNN (dense GCN, PyTorch)
class GCNClassifier(BaseEstimator, ClassifierMixin):
    """Two-layer GCN over the feeder graph with masked mean+max readout.
    X is (n, n_bus, F); unobserved buses carry zero features and an 'observed' flag."""

    def __init__(self, edges=None, n_bus=None, hidden=64, epochs=150, lr=3e-3, seed=0):
        self.edges, self.n_bus, self.hidden, self.epochs, self.lr, self.seed = edges, n_bus, hidden, epochs, lr, seed

    def _adj(self):
        import torch
        A = np.eye(self.n_bus)
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        d = 1 / np.sqrt(A.sum(1))
        return torch.tensor(A * d[:, None] * d[None, :], dtype=torch.float32)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.classes_ = np.unique(y)
        F = X.shape[2]
        self.mu_ = X.reshape(-1, F).mean(0)
        self.sd_ = X.reshape(-1, F).std(0) + 1e-6
        Ahat = self._adj()

        class Net(nn.Module):
            def __init__(s, F, H, C):
                super().__init__()
                s.l1, s.l2 = nn.Linear(F, H), nn.Linear(H, H)
                s.out = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, C))

            def forward(s, x):
                h = torch.relu(Ahat @ s.l1(x))
                h = torch.relu(Ahat @ s.l2(h))
                return s.out(torch.cat([h.mean(1), h.max(1).values], 1))

        self.net_ = Net(F, self.hidden, len(self.classes_))
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32)
        yt = torch.tensor(np.searchsorted(self.classes_, y))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 256):
                idx = perm[b:b + 256]
                opt.zero_grad()
                loss_fn(self.net_(Xt[idx]), yt[idx]).backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32)
            return self.classes_[self.net_(Xt).argmax(1).numpy()]
