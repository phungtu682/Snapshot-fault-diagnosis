"""Recompute the DMP design grid and noise-normalised margins for the IEEE 34-node feeder.

Saved to data/dmp34.npz:
  buses (B,), events (E,2) object -> (fault bus, R_f), marg (E,B), sag (E,B), meta list,
  plus the raw phasors of the design grid for the signature-cloud panels.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
import placement as P  # noqa: E402

t0 = time.time()
buses, cols, Vp, Vf, meta = P.design_grid(34)
print(f"design grid: {len(meta)} simulations in {time.time() - t0:.0f}s", flush=True)

events, marg, sag = P.margins(cols, Vp, Vf, meta)
print(f"margins: {len(events)} events x {len(buses)} buses in {time.time() - t0:.0f}s", flush=True)

sel = P.dmp(marg, 3)
print("DMP k=3 ->", [buses[i] for i in sel], flush=True)
np.savez_compressed(
    os.path.join(HERE, "data", "dmp34.npz"),
    buses=np.array(buses), marg=marg, sag=sag,
    events=np.array([[e[0], e[1]] for e in events], dtype=object),
    meta=np.array([[m[0], m[1], m[2]] for m in meta], dtype=object),
    cols=cols, Vp=Vp.astype(np.complex64), Vf=Vf.astype(np.complex64), sel=np.array(sel),
)
print("saved", round(time.time() - t0), "s", flush=True)
