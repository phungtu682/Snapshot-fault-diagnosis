"""E2 - Network scale (claim C2): identical protocol on IEEE 13-, 34- and 123-node feeders.

Sensing: all buses, and three buses chosen by a topology rule (substation + two mutually
distant feeder ends). Accuracy is also resolved by graph distance (hops) between the
faulted bus and the nearest sensor, to show where scale-related degradation comes from."""
from collections import deque
import numpy as np

import features as F
import models as M
from common import SIM, save, graph_edges


def hops_from(src, n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    d = np.full(n, 10 ** 6)
    d[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] > d[u] + 1:
                d[v] = d[u] + 1
                q.append(v)
    return d


def topo_3bus(ds, edges):
    n = len(ds.buses)
    s = ds.buses.index(ds.substation)
    ds_ = hops_from(s, n, edges)
    first = int(np.argmax(ds_))
    d1 = hops_from(first, n, edges)
    second = int(np.argmax(np.minimum(ds_, d1)))
    return [s, first, second]


def main():
    out = {}
    for nb in (13, 34, 123):
        ds = F.Dataset(f"{SIM}/ieee{nb}.npz")
        edges = graph_edges(ds, nb)
        n = len(ds.buses)
        dist = np.array([hops_from(i, n, edges) for i in range(n)])
        fbus = np.array(ds.bus_ids(list(ds.fbus)))
        tri = topo_3bus(ds, edges)
        res = {"n_buses": n, "n_samples": ds.n, "three_bus": [ds.buses[i] for i in tri]}
        for sensing, obs in [("all", list(range(n))), ("3bus", tri)]:
            Vp, Vf = F.measure(ds, obs, np.random.RandomState(0))
            near = dist[fbus][:, obs].min(1)
            res[sensing] = {}
            for rep in ("AGG", "SEQ", "FPS"):
                X = F.REPS[rep](Vp, Vf)
                r = {}
                for name in ("RF", "LGBM"):
                    acc, f1, oof = M.cross_validate(M.zoo()[name], X, ds.y)
                    r[name] = {"acc": acc, "f1": f1}
                    if name == "LGBM":
                        ok = oof == ds.y
                        r["by_hops"] = {int(h): [float(ok[near == h].mean()), int((near == h).sum())]
                                        for h in np.unique(near)}
                    print(nb, sensing, rep, name, f"{acc.mean() * 100:.1f}", flush=True)
                res[sensing][rep] = r
        out[f"ieee{nb}"] = res
    save("e2_scale", out)


if __name__ == "__main__":
    main()
