"""Figure: worked example of Discriminability-Margin Placement (DMP) on the IEEE 34-node feeder.

(a) the feeder, one fault event, and the noise-normalised margin every candidate bus sees;
(b) the class signatures of that event at a covering and at a non-covering bus;
(c) how the median margin decays with fault resistance and where it crosses tau;
(d) the ranking of candidate buses by the number of design-grid events they cover, and the set
    that the max-coverage integer programme selects.

Data: data/dmp34.npz from compute_dmp34.py (1520 OpenDSS design-grid simulations, 272 events).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
import placement as P            # noqa: E402
import simlib                    # noqa: E402

OUT = HERE
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 8, "axes.labelweight": "normal", "axes.titlesize": 8,
    "axes.titleweight": "bold", "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "legend.fontsize": 6.5,
    "pdf.fonttype": 42, "lines.linewidth": 1.0, "lines.markersize": 3.5,
})
VERM, BLUE, GREEN, ORANGE, GREY, BLACK = "#D55E00", "#0072B2", "#009E73", "#E69F00", "#7F7F7F", "#000000"
CL = {"SLG": "#B2182B", "LL": "#2166AC", "3PH": "#1B7837"}
TAU = 3.0

# ----------------------------------------------------------------- data
d = np.load(os.path.join(HERE, "data", "dmp34.npz"), allow_pickle=True)
BUS = [str(b) for b in d["buses"]]
IDX = {b: i for i, b in enumerate(BUS)}
MARG, SAG, EV, META = d["marg"], d["sag"], d["events"], d["meta"]
COLS, VP, VF = d["cols"], d["Vp"], d["Vf"]
SEL = [int(i) for i in d["sel"]]
SEL_B = [BUS[i] for i in SEL]
COV = MARG >= TAU
N_EV = len(EV)

XY = {}
for line in open(os.path.join(HERE, "..", "experiments", "data", "feeders", "IEEETestCases", "34Bus",
                              "IEEE34_BusXY.csv")):
    name, x, y = line.strip().split(",")
    XY[name] = (float(x), float(y))
EDGES = simlib.feeder_graph(34)

FAULT, RF0 = "836", 50.0
IEV = next(j for j, e in enumerate(EV) if str(e[0]) == FAULT and float(e[1]) == RF0)
MROW = MARG[IEV]
GOOD, BAD = "830", "800"
NCOV = COV.sum(1)                      # covering buses per event
# An event is class-informative only where at least two fault classes are feasible: a
# single-phase lateral cannot host an LL or 3PH fault, so no sensor can classify it.
_fd = simlib.Feeder(34)
FEAS = {b: sum(1 for c in P.CLASSES if len(_fd.bus_phases[b]) >= {"SLG": 1, "LL": 2, "3PH": 3}[c])
        for b in _fd.buses}
INFORM = np.array([FEAS[str(e[0])] >= 2 for e in EV])
N_INFORM = int(INFORM.sum())
UNCOVERABLE = int((NCOV[INFORM] == 0).sum())
N_TOTAL_COVERED = int(COV[:, SEL].any(1).sum())
N_INFORM_COVERED = int(COV[INFORM][:, SEL].any(1).sum())


# ----------------------------------------------------------------- helpers
def sig_cloud(bus, ev, draws=300, seed=0):
    rng = np.random.RandomState(seed)
    out = {}
    for c in ("SLG", "LL", "3PH"):
        rows = [j for j, m in enumerate(META) if str(m[0]) == ev[0] and float(m[2]) == ev[1] and str(m[1]) == c]
        S = []
        for j in rows:
            col = COLS[IDX[bus]]
            ok = col >= 0
            vp = np.zeros(3, dtype=complex)
            vf = np.zeros(3, dtype=complex)
            vp[ok], vf[ok] = VP[j][col[ok]], VF[j][col[ok]]
            for _ in range(max(1, draws // len(rows))):
                n1 = (1 + rng.normal(0, 0.0032951, 3)) * np.exp(1j * np.deg2rad(rng.normal(0, 0.18879, 3)))
                n2 = (1 + rng.normal(0, 0.0032951, 3)) * np.exp(1j * np.deg2rad(rng.normal(0, 0.18879, 3)))
                S.append(P.bus_signature(vp * n1, vf * n2))
        out[c] = np.array(S)
    return out


def best_plane(clouds):
    best, bv = (0, 1), -1.0
    keys = list(clouds)
    for i in range(5):
        for j in range(i + 1, 5):
            mu = {c: v[:, [i, j]].mean(0) for c, v in clouds.items()}
            sd = {c: v[:, [i, j]].std(0) + 1e-9 for c, v in clouds.items()}
            v = min(np.linalg.norm((mu[a] - mu[b]) / np.sqrt((sd[a] ** 2 + sd[b] ** 2) / 2))
                    for a in keys for b in keys if a < b)
            if v > bv:
                best, bv = (i, j), v
    return best


def bare(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.tick_params(length=2)


# ----------------------------------------------------------------- (a) network
def panel_a(ax):
    for a, b in EDGES:
        if a in XY and b in XY:
            ax.plot([XY[a][0], XY[b][0]], [XY[a][1], XY[b][1]], color="#4D4D4D", lw=1.1, zorder=1,
                    solid_capstyle="round")
    m = np.clip(MROW, 0.3, None)
    sc = ax.scatter([XY[b][0] for b in BUS], [XY[b][1] for b in BUS], c=m, s=30, cmap="YlOrRd",
                    norm=LogNorm(vmin=0.3, vmax=np.nanmax(MROW)), edgecolor="k", linewidth=0.4, zorder=3)
    fx, fy = XY[FAULT]
    ax.plot(fx, fy, marker="*", ms=16, color=VERM, mec="k", mew=0.5, zorder=6)
    # only the buses that matter for the narrative are labelled; the rest are left clean
    used = {b: set() for b in BUS}
    for a, b in EDGES:
        if a in XY and b in XY:
            dx, dy = XY[b][0] - XY[a][0], XY[b][1] - XY[a][1]
            da = "right" if dx > 0 else "left" if dx < 0 else "up" if dy > 0 else "down"
            db = {"right": "left", "left": "right", "up": "down", "down": "up"}[da]
            for n, d_ in ((a, da), (b, db)):
                if n in used:
                    used[n].add(d_)
    off = {"up": (0, 190, "center", "bottom"), "down": (0, -190, "center", "top"),
           "right": (130, 0, "left", "center"), "left": (-130, 0, "right", "center")}
    for b in ("800", "806", "812", "830", "834", "836", "844", "854", "858", "862", "888", "890"):
        side = "up" if b == FAULT else next(t for t in ("up", "down", "right", "left") if t not in used[b])
        if b in SEL_B or b == BAD:
            side = "down" if "down" not in used[b] else side
        dx, dy, ha, va = off[side]
        c, fw = "#333333", "normal"
        if b in SEL_B:
            c, fw = BLACK, "bold"
        if b == BAD:
            c, fw = BLACK, "bold"
        ax.text(XY[b][0] + dx, XY[b][1] + dy, b, fontsize=5.8, ha=ha, va=va, color=c, fontweight=fw,
                zorder=8, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    for b in SEL_B:
        ax.scatter(*XY[b], s=150, facecolor="none", edgecolor=BLUE, linewidth=1.4, zorder=5)
    ax.scatter(*XY[GOOD], s=150, facecolor="none", edgecolor=BLACK, linewidth=1.1, zorder=5)
    ax.scatter(*XY[BAD], s=150, facecolor="none", edgecolor=BLACK, linewidth=1.1, ls="--", zorder=5)
    # legend for the three symbols used in the panel
    lx, ly = -750, -600
    ax.plot(lx, ly, marker="*", ms=13, color=VERM, mec="k", mew=0.5, zorder=8)
    ax.text(lx + 150, ly, "fault event ($R_f$ = %g $\\Omega$)" % RF0, fontsize=6.2, va="center")
    ax.plot(lx, ly - 330, "o", ms=10, mfc="none", mec=BLUE, mew=1.4, zorder=8)
    ax.text(lx + 150, ly - 330, "DMP sensor", fontsize=6.2, va="center", color=BLACK)
    ax.plot(lx, ly - 660, "o", ms=10, mfc="none", mec=BLACK, mew=1.1, ls="--", zorder=8)
    ax.text(lx + 150, ly - 660, "buses compared in (b)", fontsize=6.2, va="center")
    ax.set_aspect("auto")
    ax.set_xlim(-900, 6350)
    ax.set_ylim(-1500, 1600)
    ax.axis("off")
    cax = ax.inset_axes([1.0, 0.06, 0.012, 0.88])
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label(r"Margin $m_{eb}$", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    ax.set_title("(a) Step 1: margin of one fault event at each bus (IEEE 34-node)",
                 loc="left", fontsize=8, pad=2)


# ----------------------------------------------------------------- (b) signatures
PLANE = (3, 4)          # (z0, z2): the two axes are shared by both panels so the comparison is direct


def panel_b(ax, bus, title):
    clouds = sig_cloud(bus, (FAULT, RF0))
    (i, j) = PLANE
    names = ["dip 1", "dip 2", "dip 3", r"$z_0$", r"$z_2$"]
    means = {c: v[:, [i, j]].mean(0) for c, v in clouds.items()}
    cen = np.mean(list(means.values()), axis=0)
    span = max((v[:, [i, j]].max(0) - v[:, [i, j]].min(0)).max() for v in clouds.values())
    coincident = max(np.linalg.norm(means[a] - means[b]) for a in means for b in means) < 0.05 * span
    for c, v in clouds.items():
        x, y = v[:, i], v[:, j]
        ax.scatter(x, y, s=2.2, color=CL[c], alpha=0.55, lw=0, zorder=2)
        ax.add_patch(Ellipse((x.mean(), y.mean()), 4 * x.std() + 1e-9, 4 * y.std() + 1e-9, fill=False,
                             ec=CL[c], lw=0.8, zorder=3))
        ax.plot(x.mean(), y.mean(), "o", ms=2.4, color=CL[c], zorder=4)
    arw = dict(arrowstyle="-|>", color="k", lw=0.6, mutation_scale=6)
    if coincident:
        # zoomed axes: circle the merged cloud and point to it
        ax.add_patch(Ellipse(tuple(cen), 0.03, 0.022, fill=False, ec="k", lw=0.7, ls="--", zorder=5))
        ax.annotate("all three fault types\noverlap within the noise", xy=(cen[0] + 0.012, cen[1] + 0.008),
                    xytext=(0.62, 0.62), textcoords="axes fraction", fontsize=6.4, ha="center", color="k",
                    arrowprops=arw)
    else:
        # circle each cluster and label it outside, with an arrow
        txt = {"LL": (0.42, 0.88), "SLG": (0.84, 0.66), "3PH": (0.44, 0.38)}
        for c in clouds:
            ax.add_patch(Ellipse(tuple(means[c]), 0.075, 0.06, fill=False, ec="k", lw=0.6, zorder=5))
            ax.annotate(c, xy=(means[c][0] + 0.03, means[c][1] + 0.02 * (1 if c != "3PH" else 1)),
                        xytext=txt[c], textcoords="axes fraction", fontsize=6.6, ha="center", va="center",
                        color="k", arrowprops=arw)
    ax.set_xlabel(names[i], fontsize=7)
    ax.set_ylabel(names[j], fontsize=7)
    ax.set_title(title, loc="left", fontsize=7.5, pad=3)
    ax.tick_params(labelsize=6)
    if coincident:
        ax.set_xlim(-0.01, 0.15)
        ax.set_ylim(-0.01, 0.10)
    else:
        ax.set_xlim(-0.05, 0.45)
        ax.set_ylim(-0.05, 0.45)
    bare(ax)
    mm = MROW[IDX[bus]]
    ok = mm >= TAU
    ax.text(0.98, 0.94, r"$m_{eb}$ = %.1f  (%s)" % (mm, "covered" if ok else "not covered"),
            transform=ax.transAxes, fontsize=6.4, ha="right", va="top", color="k",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#AAAAAA", lw=0.4))


# ----------------------------------------------------------------- (c) margin vs Rf
def panel_c(ax):
    rf = np.array([float(e[1]) for e in EV])
    levels = sorted(set(rf))
    for b, c, ls, mk in (("830", BLUE, "-", "o"), ("834", VERM, "-", "^"), ("800", BLACK, ":", "v")):
        med = [np.median(MARG[np.where(rf == r)[0], IDX[b]]) for r in levels]
        ax.plot(levels, np.clip(med, 0.05, None), color=c, ls=ls, marker=mk, ms=2.6, lw=1.0, label="bus " + b)
    ax.axhline(TAU, color="k", lw=0.8, ls="-.")
    ax.axhspan(0.05, TAU, color="#E6E6E6", lw=0, zorder=0)
    ax.text(0.009, TAU * 1.8, r"$\tau=3$", fontsize=6.4)
    ax.text(10, 0.9, "not covered\n" + r"($m_{eb}<\tau$)", fontsize=6.2, ha="center", va="center", color="k")
    ax.text(0.03, 600, r"covered ($m_{eb}\geq\tau$)", fontsize=6.2, ha="left", va="center", color="k")
    ax.grid(True, which="major", color="#BDBDBD", lw=0.45, ls=(0, (3, 3)), zorder=0)
    ax.grid(True, which="minor", axis="x", color="#E0E0E0", lw=0.3, ls=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Fault resistance $R_f$ ($\Omega$)")
    ax.set_ylabel("Median margin (noise units)")
    ax.set_ylim(0.05, 2000)
    ax.set_xlim(0.008, 130)
    ax.legend(loc="lower left", bbox_to_anchor=(0.04, 0.05), fontsize=6.4, handlelength=1.4, labelspacing=0.2, ncol=1, frameon=True,
              edgecolor="k", fancybox=False, framealpha=1).get_frame().set_linewidth(0.5)
    ax.tick_params(labelsize=6)
    ax.set_title(r"(c) Step 3: margin against $R_f$", loc="left", fontsize=7.5, pad=3)
    bare(ax)


# ----------------------------------------------------------------- (d) ranking
def panel_d(ax):
    cnt = COV[INFORM].sum(0)
    order = np.argsort(-cnt)
    n = len(order)
    y = np.arange(n)[::-1]
    col = [BLUE if BUS[i] in SEL_B else "#C8C8C8" for i in order]
    ax.barh(y, cnt[order], height=0.72, color=col, edgecolor="k", linewidth=0.25)
    ax.axvline(N_INFORM_COVERED, color=BLUE, lw=0.9, ls="--")
    ax.set_yticks(y, [BUS[i] for i in order], fontsize=4.3)
    for t, i in zip(ax.get_yticklabels(), order):
        if BUS[i] in SEL_B:
            t.set_color(BLUE)
            t.set_fontweight("bold")
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel(f"Type-informative events covered (of {N_INFORM})", fontsize=7)
    ax.set_ylabel("Candidate location", fontsize=7)
    ax.tick_params(length=1.5, labelsize=6)
    ax.set_xlim(0, N_INFORM * 1.75)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.text(0.98, 0.24, "DMP\n{%s}\n%d/%d events\n(%.0f%%)"
            % (", ".join(sorted(SEL_B)), N_INFORM_COVERED, N_INFORM, 100 * N_INFORM_COVERED / N_INFORM),
            transform=ax.transAxes, fontsize=5.6, ha="right", va="center", color=BLUE)
    ax.text(0.98, 0.06, "best of all 5984 sets\n%d events not coverable" % UNCOVERABLE,
            transform=ax.transAxes, fontsize=5.6, ha="right", va="center", color="#8B0000")
    ax.set_title("(d) Step 4: location ranking and DMP set", loc="left", fontsize=7.5, pad=3)
    bare(ax)


def make_figure():
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.0], hspace=0.22, wspace=0.42,
                          left=0.07, right=0.985, top=0.955, bottom=0.075)
    panel_a(fig.add_subplot(gs[0, :]))
    gsb = gs[1, 0].subgridspec(2, 1, hspace=0.45)
    panel_b(fig.add_subplot(gsb[0]), GOOD, "(b) Step 2: bus 830 (covered)")
    panel_b(fig.add_subplot(gsb[1]), BAD, "      bus 800 (not covered)")
    panel_c(fig.add_subplot(gs[1, 1]))
    panel_d(fig.add_subplot(gs[1, 2]))
    return fig


def main():
    fig = make_figure()
    fig.savefig(os.path.join(OUT, "png", "fig_dmp34_mechanism.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "pdf", "fig_dmp34_mechanism.pdf"), bbox_inches="tight")
    print("wrote", os.path.join(OUT, "pdf", "fig_dmp34_mechanism.pdf"))


if __name__ == "__main__":
    main()
