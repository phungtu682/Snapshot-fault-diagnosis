# Grid-code inverter fault model for the OpenDSS snapshot generator (part H, comment C8).
#
# Each inverter is a three-phase voltage-dependent current source built from three single-phase
# OpenDSS Isource elements. Its phase currents are recomputed from the terminal voltage and the
# network is re-solved until the terminal voltage changes by less than TOL pu, following the
# iterative current-source approach of Seuss et al. (2015, OpenDSS) and the voltage-controlled
# current source used in phasor short-circuit studies (Furlaneto et al. 2021; Haddadi et al. 2024).
#
# Strategies during the fault (positive-sequence voltage |V1| below VRT_ON):
#   "limit"     constant active power, positive sequence only, current limited to IMAX
#   "kfactor"   reactive-current priority: dIq1 = K * (VRT_ON - |V1|) * Ir, rest active (k = 3, Haddadi 2024)
#   "kfactor_neg" as "kfactor" plus negative-sequence current I2 = K * |V2| * Ir leading V2 by 90 deg
#               (IEEE 2800 range 90-100 deg, Javid et al. 2025)
#   "mc"        momentary cessation: no current when |V1| < MC_V (0.5 pu, Omran et al. 2025), else "limit"
# Current limit: IMAX = 1.2 Ir by default (1.1-1.3 pu typical, PSRC 2020); reactive and negative-sequence
# currents have priority and are scaled down together when they alone exceed IMAX.
#
import numpy as np
import opendssdirect as dss
import simlib

A = np.exp(2j * np.pi / 3)
K = 3.0
VRT_ON = 0.9
MC_V = 0.5
TOL = 1e-3
MAX_IT = 40
DAMP = 0.5


def seq(v):
    a, b, c = v
    return (a + b + c) / 3, (a + A * b + A * A * c) / 3, (a + A * A * b + A * c) / 3


def phases(i1, i2):
    return np.array([i1 + i2, A * A * i1 + A * i2, A * i1 + A * A * i2])


class InvFeeder(simlib.Feeder):
    def __init__(self, n_bus, der_penetration, strategy="kfactor", imax=1.2, seed=0):
        super().__init__(n_bus, der_penetration=0.0, seed=seed)
        self.strategy, self.imax = strategy, imax
        # same inverter sites as simlib (three-phase buses, not the substation, seeded)
        rng = np.random.RandomState(seed)
        three_ph = [b for b in self.buses if len(self.bus_phases[b]) == 3 and b != simlib.SUBSTATION[n_bus]]
        self.der_buses = list(rng.choice(three_ph, size=min(3, len(three_ph)), replace=False))
        self.p_kw = der_penetration * self.total_load_kw / len(self.der_buses)
        self.inv = []
        for i, b in enumerate(self.der_buses):
            dss.Circuit.SetActiveBus(b)
            vb = dss.Bus.kVBase() * 1000.0                      # line-to-neutral base (V)
            ir = self.p_kw * 1000.0 / (3 * vb)                    # rated current per phase (A), S = P
            for p in (1, 2, 3):
                dss.Command(f"New Isource.INV{i}_{p} phases=1 bus1={b}.{p} amps=0 angle=0")
            self.inv.append(dict(bus=b, vb=vb, ir=ir, cur=np.zeros(3, complex)))
        self.iters, self.ok = [], []
        self._base_solve()

    # ------------------------------------------------------------------ inverter control
    def _vterm(self, inv):
        dss.Circuit.SetActiveBus(inv["bus"])
        v = np.array(dss.Bus.Voltages()); nodes = dss.Bus.Nodes()
        vv = v[0::2] + 1j * v[1::2]
        return np.array([vv[nodes.index(p)] for p in (1, 2, 3)]) / inv["vb"]

    def _target(self, inv, v, fault):
        v1, v2 = seq(v)[1:]
        m1 = abs(v1); u1 = v1 / m1 if m1 > 1e-6 else 1.0
        imax = self.imax
        p_pu = 1.0                                             # active power in pu of rating
        id_ref = p_pu / max(m1, 0.05)                          # constant power
        iq, i2 = 0.0, 0j
        st = self.strategy if fault else "limit"
        if st == "mc" and m1 < MC_V:
            return np.zeros(3, complex)
        if st in ("kfactor", "kfactor_neg") and m1 < VRT_ON:
            iq = K * (VRT_ON - m1)
            if st == "kfactor_neg":
                i2 = K * abs(v2) * np.exp(1j * (np.angle(v2) + np.pi / 2))
        # priority currents (reactive, negative) limited first
        i1q = -1j * iq * u1                                    # lags V1 by 90 deg: injects reactive power
        pr = phases(i1q, i2)
        peak = np.abs(pr).max()
        if peak > imax:
            s = imax / peak; i1q, i2 = i1q * s, i2 * s; pr = pr * s
        # largest active current that keeps every phase within imax (bisection)
        lo, hi = 0.0, id_ref
        for _ in range(30):
            mid = (lo + hi) / 2
            if np.abs(phases(i1q + mid * u1, i2)).max() <= imax + 1e-9:
                lo = mid
            else:
                hi = mid
        return phases(i1q + lo * u1, i2) * inv["ir"]

    def _set(self, inv, i, cur):
        for p in range(3):
            dss.Command(f"Isource.INV{i}_{p + 1}.amps={abs(cur[p]):.6f} angle={np.degrees(np.angle(cur[p])):.6f}")
        inv["cur"] = cur

    def _iterate(self, fault):
        last = [self._vterm(inv) for inv in self.inv]
        for it in range(1, MAX_IT + 1):
            for i, inv in enumerate(self.inv):
                tgt = self._target(inv, last[i], fault)
                self._set(inv, i, DAMP * tgt + (1 - DAMP) * inv["cur"] if it > 1 else tgt)
            dss.Solution.Solve()
            now = [self._vterm(inv) for inv in self.inv]
            dv = max(np.abs(a - b).max() for a, b in zip(now, last))
            last = now
            if dv < TOL:
                return it, True
        return MAX_IT, False

    def _base_solve(self):
        dss.Command("Set ControlMode=STATIC")
        dss.Solution.Solve()
        if getattr(self, "inv", None):
            self._iterate(fault=False)

    # ------------------------------------------------------------------ fault sample
    def _apply_fault(self, bus, ftype, phs, rf, currents=True):
        if ftype == "SLG":
            cmd = f"Edit Fault.F1 phases=1 bus1={bus}.{phs[0]} bus2={bus}.0 r={rf}"
        elif ftype == "LL":
            cmd = f"Edit Fault.F1 phases=1 bus1={bus}.{phs[0]} bus2={bus}.{phs[1]} r={rf}"
        elif ftype == "LLG":
            cmd = f"Edit Fault.F1 phases=2 bus1={bus}.{phs[0]}.{phs[1]} bus2={bus}.0.0 r={rf}"
        else:
            cmd = f"Edit Fault.F1 phases=3 bus1={bus}.1.2.3 bus2={bus}.0.0.0 r={rf}"
        dss.Command(cmd + " enabled=yes")
        dss.Command("Set ControlMode=OFF")
        dss.Solution.Solve()
        it, ok = self._iterate(fault=True)
        converged = dss.Solution.Converged() and ok
        self.iters.append(it); self.ok.append(ok)
        self.inv_state = [(seq(inv["cur"] / inv["ir"])[1], seq(inv["cur"] / inv["ir"])[2]) for inv in self.inv]
        Vf, If = self._snapshot(currents)
        dss.Command("Fault.F1.enabled=no")
        self._base_solve()
        return Vf, If, converged


def generate(n_bus, n_samples, seed, out_path, der_penetration, strategy, imax=1.2, rf_range=(1e-3, 100.0)):
    import os
    out_path = os.path.abspath(out_path)
    f = InvFeeder(n_bus, der_penetration, strategy=strategy, imax=imax, seed=seed)
    rng = np.random.RandomState(seed)
    recs, inv1, inv2 = [], [], []
    for _ in range(n_samples):
        r = f.sample(rng, rf_range=rf_range)
        if r["converged"]:
            recs.append(r)
            inv1.append([abs(a) for a, _ in f.inv_state]); inv2.append([abs(b) for _, b in f.inv_state])
    np.savez_compressed(
        out_path,
        Vpre=np.stack([r["Vpre"] for r in recs]), Vf=np.stack([r["Vf"] for r in recs]),
        Ipre=np.stack([r["Ipre"] for r in recs]), If=np.stack([r["If"] for r in recs]),
        ftype=np.array([r["ftype"] for r in recs]), bus=np.array([r["bus"] for r in recs]),
        phases=np.array([",".join(map(str, r["phases"])) for r in recs]),
        rf=np.array([r["rf"] for r in recs]), scale=np.array([r["scale"] for r in recs]),
        node_names=np.array(list(f.node_index)), buses=np.array(f.buses),
        lines=np.array(f.lines), line_buses=np.array(f.line_buses),
        substation=simlib.SUBSTATION[n_bus], der_buses=np.array(f.der_buses, dtype=str),
        der_penetration=der_penetration, seed=seed, strategy=strategy, imax=imax,
        inv_I1=np.array(inv1), inv_I2=np.array(inv2),
        iters=np.array(f.iters), iter_ok=np.array(f.ok),
    )
    return len(recs), float(np.mean(f.ok)), float(np.mean(f.iters))
