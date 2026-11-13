"""Converts DER-Fault (Razick & Musilek, Zenodo 22904220; Iowa 240-bus) to the npz layout
written by simlib.generate, so that features.Dataset and all experiment code apply unchanged.

Only the 93 PMU buses are present. Fault types are mapped to SLG / LL / LLG / 3PH
(phase-specific labels are merged); normal cases are kept with ftype 'NORMAL'."""
import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "derfault")
MAP = {"LGA": "SLG", "LGB": "SLG", "LGC": "SLG", "LLAB": "LL", "LLBC": "LL", "LLAC": "LL",
       "LLABG": "LLG", "LLBCG": "LLG", "LLACG": "LLG", "LLLG": "3PH", "Normal": "NORMAL"}


def main():
    meta = pd.read_csv(os.path.join(D, "labels_metadata_DERfaults.csv"))
    cols = pd.read_csv(os.path.join(D, "DERfaults_measurements.csv"), nrows=0).columns
    vcols = [c for c in cols if c.startswith("bus")]
    head = [c for c in cols if c.startswith("Line.cb_101_T1_")]
    data = pd.read_csv(os.path.join(D, "DERfaults_measurements.csv"), usecols=["SampleID"] + vcols + head,
                       dtype=np.float32)
    data = data.set_index("SampleID").loc[meta.SampleID]

    nodes = sorted({c.split("_")[0] for c in vcols}, key=lambda n: (n.split(".")[0], int(n.split(".")[1])))
    buses = sorted({n.split(".")[0] for n in nodes})
    def phasor(prefix, node, state):
        m, a = data[f"{node}_{prefix}mag_{state}"].values, data[f"{node}_{prefix}ang_{state}"].values
        return (m * np.exp(1j * np.deg2rad(a))).astype(np.complex64)
    Vpre = np.stack([phasor("V", n, "pre") for n in nodes], 1)
    Vf = np.stack([phasor("V", n, "post") for n in nodes], 1)
    # per-unit on each node's median pre-fault magnitude (nominal phase voltage proxy)
    base = np.median(np.abs(Vpre), axis=0)
    Vpre, Vf = Vpre / base, Vf / base
    Ipre = np.zeros((len(meta), 1, 3), np.complex64)
    If = np.zeros_like(Ipre)
    for p in (1, 2, 3):
        Ipre[:, 0, p - 1] = phasor("I", f"Line.cb_101_T1_Ph{p}", "pre") if f"Line.cb_101_T1_Ph{p}_Imag_pre" in data else 0
        If[:, 0, p - 1] = phasor("I", f"Line.cb_101_T1_Ph{p}", "post") if f"Line.cb_101_T1_Ph{p}_Imag_post" in data else 0

    np.savez_compressed(
        os.path.join(D, "derfault.npz"), Vpre=Vpre, Vf=Vf, Ipre=Ipre, If=If,
        ftype=meta.Fault_Type.map(MAP).to_numpy(dtype=str), bus=meta.Fault_Bus.to_numpy(dtype=str),
        phases=meta.Fault_Type.to_numpy(dtype=str), rf=meta.Resistance.values.astype(float),
        scale=meta.Representative_Load_Index.values.astype(float), scenario=meta.Scenario_ID.to_numpy(dtype=str),
        node_names=np.array(nodes), buses=np.array(buses), lines=np.array(["cb_101"]),
        line_buses=np.array([["bus1", "bus1"]]), substation="bus1", der_buses=np.array([], dtype=str),
        der_penetration=0.0, seed=0)
    print("derfault", Vpre.shape, pd.Series(meta.Fault_Type.map(MAP)).value_counts().to_dict())


if __name__ == "__main__":
    main()
