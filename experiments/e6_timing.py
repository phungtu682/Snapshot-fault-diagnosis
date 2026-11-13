"""E6 - Inference latency (claim C6): single-sample (one thread) and batch, three-bus SEQ."""
import os
import platform
import numpy as np
from sklearn.model_selection import train_test_split

import features as F
import models as M
from common import SIM, PAPER_3BUS, save, graph_edges, masked_graph_inputs


def main():
    ds = F.Dataset(f"{SIM}/ieee13.npz")
    obs = ds.bus_ids(PAPER_3BUS[13])
    Vp, Vf = F.measure(ds, obs, np.random.RandomState(0))
    X = F.seq(Vp, Vf)
    Xtr, Xte, ytr, yte = train_test_split(X, ds.y, test_size=0.25, stratify=ds.y, random_state=0)
    out = {"cpu": platform.processor(), "cores": os.cpu_count()}
    mdls = dict(M.zoo())
    mdls["Relay rule"] = M.RelayRule()
    for name, m in mdls.items():
        m.fit(Xtr, ytr)
        single, batch = M.time_inference(m, Xte)
        out[name] = {"single_us": single, "batch_us": batch, "acc": float((m.predict(Xte) == yte).mean())}
        print(f"{name:10s} single {single:8.1f} us  batch {batch:7.2f} us", flush=True)
    G = masked_graph_inputs(ds, Vp, Vf, obs)
    Gtr, Gte = train_test_split(G, test_size=0.25, stratify=ds.y, random_state=0)
    g = M.GCNClassifier(edges=graph_edges(ds, 13), n_bus=len(ds.buses), epochs=60).fit(Gtr, ytr)
    import torch
    torch.set_num_threads(1)
    single, batch = M.time_inference(g, Gte)
    out["GCN"] = {"single_us": single, "batch_us": batch, "acc": float((g.predict(Gte) == yte).mean())}
    print(f"GCN        single {single:8.1f} us  batch {batch:7.2f} us", flush=True)
    save("e6_timing", out)


if __name__ == "__main__":
    main()
