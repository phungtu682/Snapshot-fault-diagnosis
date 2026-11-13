"""E5 - Feature importance and value of current measurements (claim C5).

(a) Paired comparison (same folds, Wilcoxon) of voltage-only vs voltage + feeder-head
current features for AGG, SEQ, FPS under three-bus and all-bus sensing.
(b) SHAP (TreeExplainer, XGBoost) and permutation importance of the SEQ + current set,
replacing the earlier regenerate-and-drop ablation, which confounded sampling noise."""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import features as F
import models as M
from common import SIM, PAPER_3BUS, save

SEQ_NAMES = ["dip1", "dip2", "dip3", "V0/V1 max", "V0/V1 mean", "V2/V1 max", "V2/V1 mean",
            "V1 drop max", "V1 drop mean", "mean|V|", "std|V|", "max|dV|", "sum dV^2"]
I_NAMES = ["I ratio 1", "I ratio 2", "I ratio 3", "I0/I1", "I2/I1"]


def main():
    ds = F.Dataset(f"{SIM}/ieee13.npz")
    out = {}
    for sensing, obs in [("3bus", ds.bus_ids(PAPER_3BUS[13])), ("all", list(range(len(ds.buses))))]:
        rng = np.random.RandomState(0)
        Vp, Vf = F.measure(ds, obs, rng)
        Ix = F.currents(ds, rng)
        out[sensing] = {}
        for rep in ("AGG", "SEQ", "FPS"):
            X = F.REPS[rep](Vp, Vf)
            a_v, _, _ = M.cross_validate(M.zoo()["LGBM"], X, ds.y, seeds=(0, 1, 2))
            a_vi, _, _ = M.cross_validate(M.zoo()["LGBM"], np.hstack([X, Ix]), ds.y, seeds=(0, 1, 2))
            out[sensing][rep] = {"V": a_v, "V+I": a_vi, "p": wilcoxon(a_v, a_vi).pvalue}
            print(sensing, rep, f"V {a_v.mean() * 100:.1f}  V+I {a_vi.mean() * 100:.1f}",
                  f"p={out[sensing][rep]['p']:.3g}", flush=True)
        a_i, _, _ = M.cross_validate(M.zoo()["LGBM"], Ix, ds.y, seeds=(0, 1, 2))
        out[sensing]["I_only"] = a_i
        print(sensing, "I only", f"{a_i.mean() * 100:.1f}", flush=True)

    # importance on the three-bus SEQ + current representation
    import shap
    rng = np.random.RandomState(0)
    obs = ds.bus_ids(PAPER_3BUS[13])
    Vp, Vf = F.measure(ds, obs, rng)
    X = np.hstack([F.seq(Vp, Vf), F.currents(ds, rng)])
    names = SEQ_NAMES + I_NAMES
    Xtr, Xte, ytr, yte = train_test_split(X, ds.y, test_size=0.25, stratify=ds.y, random_state=0)
    xgb = M.zoo()["XGB"].fit(Xtr, ytr)
    sv = shap.TreeExplainer(xgb).shap_values(Xte)
    sv = np.abs(np.asarray(sv))
    class_axis = [i for i, s in enumerate(sv.shape) if s == 3 and i != 0]
    mean_abs = sv.mean(axis=(0, class_axis[-1])) if class_axis else sv.mean(0)
    pi = permutation_importance(xgb, Xte, yte, n_repeats=20, random_state=0, n_jobs=1)
    out["importance"] = {"names": names, "shap": mean_abs, "perm_mean": pi.importances_mean,
                         "perm_std": pi.importances_std}
    for k in np.argsort(-mean_abs)[:8]:
        print(f"  {names[k]:14s} SHAP {mean_abs[k]:.3f}  perm {pi.importances_mean[k]:.3f}", flush=True)
    save("e5_importance", out)


if __name__ == "__main__":
    main()
