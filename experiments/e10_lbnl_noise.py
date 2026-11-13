"""E10 - Empirical pre/post-snapshot noise from real micro-PMU data (LBNL Open uPMU, bus a6).

A snapshot classifier compares phasors at two instants (pre-fault and fault). Without a
fault, the same comparison measures the 'noise floor': PMU error plus natural load and
voltage fluctuation over the gap. For gaps of 1-30 cycles we report the spread of the
magnitude ratio and of the phase-angle change (common-mode frequency drift removed by
subtracting the three-phase mean), to compare with the 1 % / 0.5 deg model used in E1-E9."""
import gzip
import zlib
import numpy as np

from common import HERE, save

D = f"{HERE}/data/lbnl"
LAGS = [1, 2, 6, 12, 30]  # samples at 120 Hz -> 0.5 to 15 cycles of 60 Hz


def read(name):
    """Read a truncated .gz slice; the last partial line is dropped."""
    raw = open(f"{D}/a6_{name}.part.gz", "rb").read()
    txt = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw).decode(errors="ignore")
    rows = [l.split(",") for l in txt.splitlines()[:-1] if "," in l]
    t = np.array([int(r[0]) for r in rows], dtype=np.int64)
    v = np.array([float(r[1]) for r in rows])
    return t, v


def main():
    mags, angs, t0 = [], [], None
    for p in (1, 2, 3):
        tm, m = read(f"L{p}MAG")
        ta, a = read(f"L{p}ANG")
        mags.append((tm, m))
        angs.append((ta, a))
    n = min(min(len(x[1]) for x in mags), min(len(x[1]) for x in angs))
    t = mags[0][0][:n]
    M = np.stack([m[1][:n] for m in mags], 1)
    Ang = np.unwrap(np.deg2rad(np.stack([a[1][:n] for a in angs], 1)), axis=0)
    hours = (t[-1] - t[0]) / 1e9 / 3600
    out = {"hours": hours, "samples": int(n), "nominal_V": float(np.median(M)), "lags": {}}
    for lag in LAGS:
        r = np.log(M[lag:] / M[:-lag])
        dA = Ang[lag:] - Ang[:-lag]
        dA = dA - dA.mean(1, keepdims=True)  # remove common-mode frequency drift
        q = lambda x: [float(np.percentile(np.abs(x), k)) for k in (50, 95, 99, 99.9)]
        out["lags"][lag] = {"cycles": lag / 2, "mag_std": float(r.std()), "mag_abs_pct": q(r),
                            "ang_std_deg": float(np.rad2deg(dA.std())), "ang_abs_pct_deg": q(np.rad2deg(dA))}
        print(f"lag {lag:3d} ({lag / 2:4.1f} cyc): |V| ratio std {r.std() * 100:.3f}%  "
              f"p99 {np.percentile(np.abs(r), 99) * 100:.3f}%   angle std {np.rad2deg(dA.std()):.3f} deg", flush=True)
    print(f"{hours:.2f} h of data, nominal {out['nominal_V']:.0f} V", flush=True)
    save("e10_lbnl_noise", out)


if __name__ == "__main__":
    main()
