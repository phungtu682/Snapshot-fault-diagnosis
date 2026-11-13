"""E4 - Sensor placement (claim C4).

(a) Exhaustive evaluation of all C(14,3) = 364 three-bus subsets of the IEEE 13-node
feeder for AGG and SEQ, giving the full distribution (best / median / worst) and the rank
of the manuscript's subset. (b) Greedy forward selection vs random placement as the
number of sensors k grows, for AGG, SEQ and FPS. (c) Greedy curve on the 34-node feeder."""
from itertools import combinations
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import features as F
from common import SIM, PAPER_3BUS, save

FAST = dict(n_estimators=150, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1, random_state=0)
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def score(ds, obs, rep, rng_seed=0):
    Vp, Vf = F.measure(ds, list(obs), np.random.RandomState(rng_seed))
    return cross_val_score(LGBMClassifier(**FAST), F.REPS[rep](Vp, Vf), ds.y, cv=CV).mean()


def greedy(ds, rep, kmax):
    chosen, curve = [], []
    rest = list(range(len(ds.buses)))
    for _ in range(kmax):
        best = max(rest, key=lambda b: score(ds, chosen + [b], rep))
        chosen.append(best)
        rest.remove(best)
        curve.append(score(ds, chosen, rep))
        print(rep, "greedy", [ds.buses[i] for i in chosen], f"{curve[-1] * 100:.1f}", flush=True)
    return [ds.buses[i] for i in chosen], curve


def random_curve(ds, rep, kmax, n_draws=10):
    rng = np.random.RandomState(7)
    return [[score(ds, rng.choice(len(ds.buses), k, replace=False), rep) for _ in range(n_draws)]
            for k in range(1, kmax + 1)]


def main():
    ds = F.Dataset(f"{SIM}/ieee13.npz")
    out = {"buses": ds.buses, "paper_subset": PAPER_3BUS[13]}
    paper = tuple(sorted(ds.bus_ids(PAPER_3BUS[13])))
    for rep in ("AGG", "SEQ"):
        res = {}
        for c in combinations(range(len(ds.buses)), 3):
            res[",".join(ds.buses[i] for i in c)] = score(ds, c, rep)
        vals = np.array(list(res.values()))
        key = ",".join(ds.buses[i] for i in paper)
        out[f"exhaustive_{rep}"] = {"scores": res, "paper": res[key],
                                    "paper_percentile": float((vals < res[key]).mean() * 100)}
        print(rep, "exhaustive best", max(res, key=res.get), f"{vals.max() * 100:.1f}",
              "median", f"{np.median(vals) * 100:.1f}", "worst", f"{vals.min() * 100:.1f}",
              "paper", f"{res[key] * 100:.1f}", flush=True)
    for rep in ("AGG", "SEQ", "FPS"):
        out[f"greedy_{rep}"] = greedy(ds, rep, 8)
        out[f"random_{rep}"] = random_curve(ds, rep, 8)
        out[f"all_{rep}"] = score(ds, range(len(ds.buses)), rep)
    ds34 = F.Dataset(f"{SIM}/ieee34.npz")
    out["ieee34_greedy_SEQ"] = greedy(ds34, "SEQ", 6)
    out["ieee34_random_SEQ"] = random_curve(ds34, "SEQ", 6, n_draws=5)
    out["ieee34_all_SEQ"] = score(ds34, range(len(ds34.buses)), "SEQ")
    save("e4_placement", out)


if __name__ == "__main__":
    main()
