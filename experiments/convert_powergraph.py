"""Converts the PowerGraph cascading-failure .mat (v7.3 cell arrays) files to compact .npz.

Per instance: node features (Pnet, Snet, V) per bus, edge features (Pij, Qij, X, rating)
per branch after the initiating outage, binary/multiclass labels and demand not served."""
import os
import sys
import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "powergraph", "dataset_cascades")


def cells(path, key):
    with h5py.File(path, "r") as h:
        d = h[key]
        refs = d[0, :] if d.shape[0] == 1 else d[:, 0]
        return [np.array(h[r]) for r in refs]


def convert(grid):
    raw = os.path.join(ROOT, grid, grid, "raw")
    Bf = np.stack([c.T for c in cells(os.path.join(raw, "Bf.mat"), "B_f_tot")]).astype(np.float32)  # (N, nbus, 3)
    Ef = np.stack([c.T for c in cells(os.path.join(raw, "Ef.mat"), "E_f_post")]).astype(np.float32)  # (N, nbr, 4)
    ybi = np.array([c.ravel()[0] for c in cells(os.path.join(raw, "of_bi.mat"), "output_features")]).astype(int)
    ymc = np.array([np.argmax(c.ravel()) for c in cells(os.path.join(raw, "of_mc.mat"), "category")])
    with h5py.File(os.path.join(raw, "of_reg.mat"), "r") as h:
        dns = np.array(h["dns_MW"]).ravel()
    with h5py.File(os.path.join(raw, "blist.mat"), "r") as h:
        blist = np.array(h["bList"]).T.astype(int)  # (nbr, 2), 1-based bus numbers
    out = os.path.join(os.path.dirname(ROOT), f"{grid}.npz")
    np.savez_compressed(out, node=Bf, edge=Ef, y_bi=ybi, y_mc=ymc, dns=dns, blist=blist)
    print(grid, Bf.shape, Ef.shape, "unstable share", ybi.mean().round(4), "classes", np.bincount(ymc))


if __name__ == "__main__":
    for g in sys.argv[1:] or ["ieee24", "ieee39", "uk", "ieee118"]:
        convert(g)
