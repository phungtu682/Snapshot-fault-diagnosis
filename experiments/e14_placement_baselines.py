"""E14 - More placement baselines, with fold-level Wilcoxon tests against DMP.

Added: QR-pivot sparse sensor placement (Manohar et al. style, on the design-grid signatures),
D-optimal greedy (log-det of the signature covariance), mutual-information greedy (mRMR style),
SVM forward selection with neighbour refinement (FSNR-SVM, Khattak & Choudhry), a genetic
algorithm with an LDA wrapper fitness, and simulated annealing on the DMP coverage objective.
Selection time is recorded for every method."""
import time
import numpy as np
from scipy.linalg import qr
from scipy.stats import wilcoxon
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import features as F
import placement as P
from common import SIM, save, graph_edges

FAST = dict(n_estimators=150, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1, random_state=0)
K = {13: [3, 5], 34: [3], 123: [3, 6, 9]}


def folds(ds, obs):
    Vp, Vf = F.measure(ds, obs, np.random.RandomState(0))
    X = F.fps(Vp, Vf)
    acc, hif, n1 = [], [], []
    for s in (0, 1):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, ds.y):
            m = LGBMClassifier(**FAST).fit(X[tr], ds.y[tr])
            ok = m.predict(X[te]) == ds.y[te]
            acc.append(ok.mean())
            hif.append(ok[ds.rf[te] > 10].mean())
            lost = []
            for j in range(len(obs)):
                Vl = Vf[te].copy()
                Vl[:, j] = Vp[te][:, j]
                lost.append((m.predict(F.fps(Vp[te], Vl)) == ds.y[te]).mean())
            n1.append(np.mean(lost))
    return np.array(acc), np.array(hif), np.array(n1)


def signatures(cols, Vp, Vf):
    """(simulations, buses, 5) noise-free signature tensor of the design grid."""
    Mp, Mf = np.full((len(Vp),) + cols.shape, np.nan + 0j), np.full((len(Vp),) + cols.shape, np.nan + 0j)
    v = cols >= 0
    Mp[:, v], Mf[:, v] = Vp[:, cols[v]], Vf[:, cols[v]]
    return P.bus_signature(Mp, Mf)


def qr_pivot(S, k):
    A = S.reshape(len(S), S.shape[1], -1).mean(2)
    A = A - A.mean(0)
    U = np.linalg.svd(A, full_matrices=False)[0][:, :k]
    _, _, piv = qr(U.T @ A, pivoting=True, mode="economic")
    return sorted(piv[:k].tolist())


def d_optimal(S, k):
    A = S.reshape(len(S), S.shape[1], -1)
    A = A - A.mean(0)
    chosen = []
    for _ in range(k):
        best, bv = None, -np.inf
        for b in range(S.shape[1]):
            if b in chosen:
                continue
            Z = A[:, chosen + [b]].reshape(len(A), -1)
            v = np.linalg.slogdet(np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1]))[1]
            if v > bv:
                best, bv = b, v
        chosen.append(best)
    return chosen


def mi_greedy(sel, k):
    Vp, Vf = F.measure(sel, list(range(len(sel.buses))), np.random.RandomState(1))
    per_bus = [F.fps(Vp[:, [b]], Vf[:, [b]]) for b in range(len(sel.buses))]
    idx = np.random.RandomState(0).choice(sel.n, min(1500, sel.n), replace=False)
    rel = np.array([mutual_info_classif(x[idx], sel.y[idx], random_state=0).sum() for x in per_bus])
    z = [StandardScaler().fit_transform(x).mean(1) for x in per_bus]
    C = np.abs(np.corrcoef(np.array(z)))
    chosen = [int(rel.argmax())]
    while len(chosen) < k:
        score = rel / (rel.max() + 1e-9) - C[:, chosen].mean(1)
        score[chosen] = -np.inf
        chosen.append(int(score.argmax()))
    return chosen


def fsnr_svm(sel, k, edges):
    """Forward selection with an RBF-SVM (3-fold on a 1200-sample subset) and neighbour swap."""
    idx = np.random.RandomState(0).choice(sel.n, min(1200, sel.n), replace=False)
    Vp, Vf = F.measure(sel, list(range(len(sel.buses))), np.random.RandomState(1))
    nbr = {b: set() for b in range(len(sel.buses))}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)

    def fit(o):
        m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
        return cross_val_score(m, F.fps(Vp[idx][:, o], Vf[idx][:, o]), sel.y[idx], cv=3).mean()
    chosen, cur = [], 0
    while len(chosen) < k:
        best = max((b for b in range(len(sel.buses)) if b not in chosen), key=lambda b: fit(chosen + [b]))
        chosen.append(best)
        cur = fit(chosen)
        if len(chosen) > 3:
            for n in nbr[best] - set(chosen):
                v = fit(chosen[:-1] + [n])
                if v > cur:
                    chosen[-1], cur = n, v
    return chosen


def ga_lda(sel, k, pop=24, gens=15):
    rng = np.random.RandomState(0)
    idx = rng.choice(sel.n, min(2000, sel.n), replace=False)
    Vp, Vf = F.measure(sel, list(range(len(sel.buses))), np.random.RandomState(1))
    nb, cache = len(sel.buses), {}

    def fit(g):
        key = tuple(sorted(g))
        if key not in cache:
            m = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            cache[key] = cross_val_score(m, F.fps(Vp[idx][:, list(key)], Vf[idx][:, list(key)]), sel.y[idx], cv=3).mean()
        return cache[key]
    P_ = [list(rng.choice(nb, k, replace=False)) for _ in range(pop)]
    for _ in range(gens):
        P_ = sorted(P_, key=fit, reverse=True)
        kids = P_[:pop // 4]
        while len(kids) < pop:
            a, b = rng.choice(pop // 2, 2, replace=False)
            g = list(rng.choice(list(set(P_[a]) | set(P_[b])), k, replace=False))
            if rng.rand() < 0.3:
                g[rng.randint(k)] = int(rng.choice([x for x in range(nb) if x not in g]))
            kids.append(g)
        P_ = kids
    return [int(x) for x in max(P_, key=fit)]


def sa_cover(marg, k, iters=4000, tau=3.0):
    rng = np.random.RandomState(0)
    a = marg >= tau
    cur = list(rng.choice(a.shape[1], k, replace=False))
    val = a[:, cur].any(1).sum()
    best, bv = list(cur), val
    for i in range(iters):
        c = list(cur)
        c[rng.randint(k)] = int(rng.choice([x for x in range(a.shape[1]) if x not in cur]))
        v = a[:, c].any(1).sum()
        if v >= val or rng.rand() < np.exp((v - val) / (5 * (1 - i / iters) + 1e-3)):
            cur, val = c, v
            if v > bv:
                best, bv = list(c), v
    return best


def main():
    out = {}
    for nb, ks in K.items():
        ds = F.Dataset(f"{SIM}/ieee{nb}.npz")
        sel = F.Dataset(f"{SIM}/ieee{nb}_sel.npz")
        edges = graph_edges(ds, nb)
        _, cols, Vp, Vf, meta = P.design_grid(nb)
        _, marg, sag = P.margins(cols, Vp, Vf, meta)
        S = signatures(cols, Vp, Vf)
        res = {}
        for k in ks:
            methods = {
                "DMP (proposed)": lambda: P.dmp(marg, k),
                "DMP N-1 (proposed)": lambda: P.dmp(marg, k, n1=True),
                "MRA ILP": lambda: P.mra(sag, k),
                "QR pivot": lambda: qr_pivot(S, k),
                "D-optimal": lambda: d_optimal(S, k),
                "MI greedy": lambda: mi_greedy(sel, k),
                "FSNR-SVM": lambda: fsnr_svm(sel, k, edges),
                "GA-LDA": lambda: ga_lda(sel, k),
                "SA coverage": lambda: sa_cover(marg, k),
            }
            base = None
            rk = {}
            for name, fn in methods.items():
                t = time.time()
                obs = [int(i) for i in fn()]
                sec = time.time() - t
                a, h, n1 = folds(ds, obs)
                if base is None:
                    base = (a, h, n1)
                rk[name] = {"buses": [ds.buses[i] for i in obs], "select_s": sec, "acc": float(a.mean()), "acc_sd": float(a.std()),
                            "hif": float(h.mean()), "n1": float(n1.mean())}
                if name != "DMP (proposed)":
                    for lab, x, y in (("acc", base[0], a), ("hif", base[1], h), ("n1", base[2], n1)):
                        rk[name][f"p_{lab}"] = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
                print(nb, k, f"{name:20s}", rk[name]["buses"], f"{sec:.1f}s acc {a.mean() * 100:.1f} HIF {h.mean() * 100:.1f} "
                      f"N-1 {n1.mean() * 100:.1f}", flush=True)
            res[k] = rk
            out[f"ieee{nb}"] = res
            save("e14_placement_baselines", out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
