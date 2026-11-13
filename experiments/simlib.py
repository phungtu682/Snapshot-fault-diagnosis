"""OpenDSS fault-snapshot generator shared by all distribution-feeder experiments.

Each sample stores complex per-node voltages (pu) and per-line terminal-1 currents (A)
before and during a fault, so that measurement noise, sensor subsets and feature
representations can be applied afterwards without re-simulating.
"""
import os
import numpy as np
import opendssdirect as dss

HERE = os.path.dirname(os.path.abspath(__file__))
FEEDER_DIR = os.path.join(HERE, "data", "feeders", "IEEETestCases")

FEEDERS = {
    13: os.path.join(FEEDER_DIR, "13Bus", "IEEE13Nodeckt.dss"),
    34: os.path.join(FEEDER_DIR, "34Bus", "ieee34Mod1.dss"),
    123: os.path.join(FEEDER_DIR, "123Bus", "IEEE123Master.dss"),
    240: os.path.join(HERE, "data", "feeders", "Iowa240", "model", "Master.dss"),
}
SUBSTATION = {13: "650", 34: "800", 123: "150", 240: "bus1"}
# Buses fed through underground cable (IEEE 13-node mtx606/mtx607); the other feeders are all overhead.
CABLE_BUSES = {13: {"675", "652"}, 34: set(), 123: set(), 240: set()}

FAULT_TYPES = ["SLG", "LL", "LLG", "3PH"]


def _is_internal(bus):
    """Source and regulator/switch internal buses are not physical measurement points."""
    return (bus in {"sourcebus", "eq_source_bus", "bus_xfmr", "610"} or bus.startswith("rg")
            or bus.endswith("r") or bus.startswith("t_bus"))  # t_bus*: LV side of service transformers (Iowa 240)


class Feeder:
    def __init__(self, n_bus, der_penetration=0.0, der_buses=None, seed=0):
        self.n_bus = n_bus
        dss.Basic.ClearAll()
        dss.Command(f'Compile "{FEEDERS[n_bus]}"')
        dss.Command("Set Maxiterations=100")
        dss.Command("Set Maxcontroliter=100")
        dss.Solution.Solve()

        self.buses = [b for b in dss.Circuit.AllBusNames() if not _is_internal(b)]
        self.bus_phases = {}
        for b in self.buses:
            dss.Circuit.SetActiveBus(b)
            self.bus_phases[b] = sorted(p for p in dss.Bus.Nodes() if p in (1, 2, 3))
        self.buses = [b for b in self.buses if self.bus_phases[b]]

        # node index table: (bus, phase) -> column in the voltage arrays
        all_nodes = dss.Circuit.AllNodeNames()
        self.node_index = {n: i for i, n in enumerate(all_nodes)}
        self.n_nodes = len(all_nodes)

        self.lines = list(dss.Lines.AllNames())
        self.line_buses = []
        for ln in self.lines:
            dss.Lines.Name(ln)
            self.line_buses.append((dss.Lines.Bus1().split(".")[0], dss.Lines.Bus2().split(".")[0]))

        self.total_load_kw = sum(_load_kw(ld) for ld in dss.Loads.AllNames())
        self.der_buses = []
        if der_penetration > 0:
            rng = np.random.RandomState(seed)
            three_ph = [b for b in self.buses if len(self.bus_phases[b]) == 3 and b != SUBSTATION[n_bus]]
            self.der_buses = der_buses or list(rng.choice(three_ph, size=min(3, len(three_ph)), replace=False))
            kw = der_penetration * self.total_load_kw / len(self.der_buses)
            for i, b in enumerate(self.der_buses):
                dss.Circuit.SetActiveBus(b)
                kv = dss.Bus.kVBase() * np.sqrt(3)
                # Model=7: constant-PQ inverter that becomes current-limited when voltage collapses
                dss.Command(f"New Generator.DER{i} bus1={b} phases=3 kV={kv:.4f} kW={kw:.1f} pf=1 "
                            f"model=7 Vminpu=0.5 Vmaxpu=1.2 Balanced=yes")

        dss.Command("New Fault.F1 phases=1 bus1=" + self.buses[0] + ".1 r=1 enabled=no")
        self._base_solve()

        sub, cables = SUBSTATION[n_bus], CABLE_BUSES[n_bus]
        w = np.array([0.10 if b == sub else (0.20 / len(cables) if b in cables else 0.0) for b in self.buses])
        n_over = sum(1 for b in self.buses if b != sub and b not in cables)
        over_share = 0.70 if cables else 0.90
        w = np.where(w == 0, over_share / n_over, w)
        self.bus_weights = w / w.sum()

    def _base_solve(self):
        dss.Command("Set ControlMode=STATIC")
        dss.Solution.Solve()

    def _snapshot(self, currents=True):
        mag = np.array(dss.Circuit.AllBusMagPu())
        volts = np.array(dss.Circuit.AllBusVolts())
        ang = np.angle(volts[0::2] + 1j * volts[1::2])
        V = (mag * np.exp(1j * ang)).astype(np.complex64)
        I = np.zeros((len(self.lines), 3), dtype=np.complex64)
        if not currents:
            return V, I
        for k, ln in enumerate(self.lines):
            dss.Circuit.SetActiveElement("Line." + ln)
            c = np.array(dss.CktElement.Currents())
            nph = dss.CktElement.NumPhases()
            nodes = dss.CktElement.NodeOrder()[:nph]
            cc = c[0:2 * nph:2] + 1j * c[1:2 * nph:2]
            for p, val in zip(nodes, cc):
                if p in (1, 2, 3):
                    I[k, p - 1] = val
        return V, I

    def sample(self, rng, rf_range=(1e-3, 100.0), fault_types=FAULT_TYPES):
        """Simulate one fault. Returns dict with pre/fault phasors and metadata."""
        dss.Command("Fault.F1.enabled=no")
        scale = rng.uniform(0.7, 1.3)
        dss.Solution.LoadMult(scale)
        self._base_solve()
        Vpre, Ipre = self._snapshot()

        ftype = fault_types[rng.randint(len(fault_types))]
        need = {"SLG": 1, "LL": 2, "LLG": 2, "3PH": 3}[ftype]
        ok = np.array([len(self.bus_phases[b]) >= need for b in self.buses])
        w = self.bus_weights * ok
        bus = self.buses[rng.choice(len(self.buses), p=w / w.sum())]
        phs = sorted(rng.choice(self.bus_phases[bus], size=need, replace=False).tolist())
        rf = float(np.exp(rng.uniform(np.log(rf_range[0]), np.log(rf_range[1]))))
        Vf, If, converged = self._apply_fault(bus, ftype, phs, rf)
        return dict(Vpre=Vpre, Vf=Vf, Ipre=Ipre, If=If, ftype=ftype, bus=bus,
                    phases=phs, rf=rf, scale=scale, converged=converged)

    def fault(self, bus, ftype, phs, rf, scale=1.0):
        """Deterministic fault at a given bus/phases/resistance -> (Vpre, Vf, converged)."""
        dss.Command("Fault.F1.enabled=no")
        dss.Solution.LoadMult(scale)
        self._base_solve()
        Vpre, _ = self._snapshot(currents=False)
        Vf, _, conv = self._apply_fault(bus, ftype, phs, rf, currents=False)
        return Vpre, Vf, conv

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
        # regulators and capacitors do not move during the fault interval
        dss.Command("Set ControlMode=OFF")
        dss.Solution.Solve()
        converged = dss.Solution.Converged()
        Vf, If = self._snapshot(currents)
        dss.Command("Fault.F1.enabled=no")
        self._base_solve()
        return Vf, If, converged


def _load_kw(name):
    dss.Loads.Name(name)
    return dss.Loads.kW()


def feeder_graph(n_bus):
    """Undirected edges between measurable buses (lines, transformers, switches).
    Internal regulator/source buses are bridged so the graph stays connected."""
    dss.Basic.ClearAll()
    dss.Command(f'Compile "{FEEDERS[n_bus]}"')
    raw = []
    for cls in ("Line", "Transformer"):
        names = dss.Lines.AllNames() if cls == "Line" else dss.Transformers.AllNames()
        for nm in names:
            dss.Circuit.SetActiveElement(f"{cls}.{nm}")
            bs = [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]
            raw += [(bs[i], bs[j]) for i in range(len(bs)) for j in range(i + 1, len(bs)) if bs[i] != bs[j]]
    adj = {}
    for a, b in raw:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    edges = set()
    for a in adj:
        if _is_internal(a):
            continue
        stack, seen = list(adj[a]), {a}
        while stack:  # walk through internal buses until a measurable bus is reached
            b = stack.pop()
            if b in seen:
                continue
            seen.add(b)
            if _is_internal(b):
                stack += list(adj.get(b, ()))
            else:
                edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def generate(n_bus, n_samples, seed, out_path, der_penetration=0.0, rf_range=(1e-3, 100.0)):
    out_path = os.path.abspath(out_path)  # OpenDSS Compile changes the working directory
    feeder = Feeder(n_bus, der_penetration=der_penetration, seed=seed)
    rng = np.random.RandomState(seed)
    recs = [feeder.sample(rng, rf_range=rf_range) for _ in range(n_samples)]
    recs = [r for r in recs if r["converged"]]
    np.savez_compressed(
        out_path,
        Vpre=np.stack([r["Vpre"] for r in recs]), Vf=np.stack([r["Vf"] for r in recs]),
        Ipre=np.stack([r["Ipre"] for r in recs]), If=np.stack([r["If"] for r in recs]),
        ftype=np.array([r["ftype"] for r in recs]), bus=np.array([r["bus"] for r in recs]),
        phases=np.array([",".join(map(str, r["phases"])) for r in recs]),
        rf=np.array([r["rf"] for r in recs]), scale=np.array([r["scale"] for r in recs]),
        node_names=np.array(list(feeder.node_index)), buses=np.array(feeder.buses),
        lines=np.array(feeder.lines), line_buses=np.array(feeder.line_buses),
        substation=SUBSTATION[n_bus], der_buses=np.array(feeder.der_buses, dtype=str),
        der_penetration=der_penetration, seed=seed,
    )
    return len(recs)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=13)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--der", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = time.time()
    n = generate(a.bus, a.n, a.seed, a.out, der_penetration=a.der)
    print(f"{a.out}: {n} converged samples in {time.time() - t:.0f}s")
