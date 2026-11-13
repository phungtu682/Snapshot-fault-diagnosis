"""Result figures in the style of published J. Reliab. Sci. Eng. papers: closed axes, inward ticks,
bold black labels, dense multi-panel layouts, shared legend below. Data: ../experiments/results."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "results")
OUT = HERE
for _d in ("pdf", "png"):
    os.makedirs(os.path.join(OUT, _d), exist_ok=True)
load = lambda n: json.load(open(os.path.join(RES, n + ".json")))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 9, "axes.labelweight": "bold", "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False, "legend.fontsize": 8, "pdf.fonttype": 42,
    "lines.linewidth": 1.2, "lines.markersize": 4,
})
# Okabe-Ito colour-blind safe set; the proposed method is always vermilion
VERM, BLUE, GREEN, PURP, SKY, BLACK, GREY = "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#000000", "#7F7F7F"
MODELS = {"RF": (BLUE, "o"), "SVM": (SKY, "s"), "LR": (GREY, "v"), "kNN": (PURP, "^"), "MLP": (GREEN, "D"),
          "XGB": ("#E69F00", "P"), "LGBM": (VERM, "*")}
# feature-set names used in the manuscript
N = {"AGG": "AGG", "PPA": "PPA", "SEQ": "SEQ", "FPS": "FPS"}  # display names


def save(fig, name):
    fig.savefig(os.path.join(OUT, "pdf", name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "png", name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_representation():
    e1 = load("e1_ladder")
    reps = ["AGG", "PPA", "SEQ", "FPS"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, s, t in zip(axs, ("all", "3bus"), ("(a) All 14 buses observed", "(b) Three observed buses")):
        for m, (c, mk) in MODELS.items():
            ax.plot(range(4), [100 * np.mean(e1[s][r]["models"][m]["acc"]) for r in reps], color=c, marker=mk, label=m,
                    ms=6 if m == "LGBM" else 4, lw=1.6 if m == "LGBM" else 0.9)
        ax.plot(range(4), [100 * (1 - e1[s][r]["bayes_lb"]) for r in reps], "k--", marker="_", ms=9, label="Cover-Hart bound")
        ax.axhline(100 * np.mean(e1[s]["noise_free"]["FPS"]["acc"]), color="k", lw=0.9, ls=":")
        ax.text(0.05, 98.6, "FPS, noise-free", fontsize=8, va="top")
        ax.set_xticks(range(4), [N[r] for r in reps])
        ax.set_xlabel("Feature set")
        ax.set_title(t, loc="left")
        ax.set_ylim(62, 102)
    axs[0].set_ylabel("Accuracy (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig_representation")


def fig_noise_rf():
    e3 = load("e3_rf_noise")
    edges = np.array(e3["rf_edges"])
    noise = e3["noise"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"wspace": 0.08})
    labels = [f"{a:.3g}" for a in edges[:-1]]
    for ax, rep, t in zip(axs, ("AGG", "FPS"), ("(a) AGG, three buses", "(b) FPS, three buses")):
        M = 100 * np.array(e3["3bus"][rep])[:, :-1]
        im = ax.imshow(M, aspect="equal", cmap="RdYlBu", vmin=30, vmax=100, origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] < 45 or M[i, j] > 92 else "black")
        ax.set_xticks(range(0, 10, 2), labels[::2], rotation=45)
        ax.set_yticks(range(len(noise)), [f"{100 * n:g}" for n in noise])
        ax.tick_params(top=False, right=False, length=2)
        ax.set_xlabel(r"$R_f$ bin lower edge ($\Omega$)")
        ax.set_title(t, loc="left")
    axs[0].set_ylabel(r"Noise $\sigma_m$ (%)")
    axs[1].set_yticklabels([])
    fig.canvas.draw()
    # colour bar exactly as tall as the five rows of cells
    p = axs[1].get_position()
    cax = fig.add_axes([p.x1 + 0.012, p.y0, 0.012, p.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.5, length=2, direction="in")
    cb.set_label("Accuracy (%)", fontsize=7.5)
    cb.outline.set_linewidth(0.6)
    # arrows: direction of increasing noise and fault resistance
    from matplotlib.patches import FancyArrowPatch

    def arrow(a, b):
        fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                       color="k", lw=0.7))
    # short arrows placed directly after each axis label
    inv = fig.transFigure.inverted()
    r = fig.canvas.get_renderer()
    for ax in axs:
        bb = ax.xaxis.label.get_window_extent(r).transformed(inv)
        y = (bb.y0 + bb.y1) / 2
        arrow((bb.x1 + 0.006, y), (bb.x1 + 0.045, y))
    bb = axs[0].yaxis.label.get_window_extent(r).transformed(inv)
    x = (bb.x0 + bb.x1) / 2
    arrow((x, bb.y1 + 0.015), (x, bb.y1 + 0.11))
    save(fig, "fig_noise_rf")


def fig_exhaustive():
    e4 = load("e4_placement")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    marks = [({"650", "675", "684"}, "Hand-picked", BLACK, "--"), ({"634", "680", "692"}, "DMP", VERM, "-"),
             ({"671", "680", "692"}, "DMP N-1", BLUE, "-")]
    for ax, rep, t in zip(axs, ("AGG", "SEQ"), ("(a) AGG", "(b) SEQ")):
        sc = e4[f"exhaustive_{rep}"]["scores"]
        v = 100 * np.array(list(sc.values()))
        ax.hist(v, bins=30, color="#9ECAE1", edgecolor="k", lw=0.3)
        for s, lab, c, ls in marks:
            val = next(100 * x for k, x in sc.items() if set(k.split(",")) == s)
            ax.axvline(val, color=c, ls=ls, lw=1.4, label=f"{lab} ({val:.1f}%)")
        ax.set_xlabel("Accuracy of a three-bus subset (%)")
        ax.set_title(t, loc="left")
        ax.legend(loc="upper left", fontsize=7)
    axs[0].set_ylabel("Number of subsets")
    fig.tight_layout()
    save(fig, "fig_exhaustive")


def fig_placement():
    e4b = load("e4b_dmp")
    # plain style of the published JRSE figures: few series, thin lines, boxed legend, no bold
    rc = {"axes.labelweight": "normal", "axes.titleweight": "normal", "xtick.top": False, "ytick.right": False,
          "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1}
    # proposed methods in strong red; baselines thin with hollow markers; random sets as a grey band
    # focus + context (Tufte; Rougier, "Ten simple rules for better figures"): the two proposed variants in
    # saturated colours, the five baselines in a muted palette (Paul Tol "muted") with thin lines
    methods = [("DMP (proposed)", "DMP (proposed)", "#B2182B", "-", "o", 1.4, True),
               ("DMP N-1 (proposed)", "DMP N-1 (proposed)", "#2166AC", "-", "s", 1.4, True),
               ("MRA ILP", "MRA ILP", "#44AA99", "-", "^", 0.8, False),
               ("Observability ILP", "Observability ILP", "#AA4499", "-", "v", 0.8, False),
               ("Topology", "Topology", "#DDAA33", "-", "D", 0.8, False),
               ("Information greedy", "Information greedy", "#999933", "-", "<", 0.8, False),
               ("Accuracy greedy", "Accuracy greedy", "#888888", "-", ">", 0.8, False)]
    feeders = [("ieee13", "IEEE 13-node"), ("ieee34", "IEEE 34-node"), ("ieee123", "IEEE 123-node")]
    metrics = [("acc", "Accuracy (%)"), ("hif", "HIF accuracy (%)"), ("n1", "N-1 accuracy (%)")]
    with plt.rc_context(rc):
        fig, axs = plt.subplots(3, 3, figsize=(6.9, 6.6), sharex="col")
        for j, (fd, name) in enumerate(feeders):
            ks = sorted(e4b[fd]["k"], key=int)
            kk = [int(k) for k in ks]
            for i, (mt, ylab) in enumerate(metrics):
                ax = axs[i, j]
                rnd = [100 * e4b[fd]["k"][k]["Random (10 draws)"][mt] for k in ks]
                ax.plot(kk, rnd, color="#7F7F7F", ls=":", lw=0.9, label="Random (mean of 10 sets)")
                for m, lab, c, ls, mk, lw, filled in methods:
                    v = [100 * e4b[fd]["k"][k][m][mt] for k in ks]
                    ax.plot(kk, v, color=c, ls=ls, marker=mk, lw=lw, ms=4 if lw > 1 else 3.0,
                            mfc=c if filled else "white", mew=0.7, alpha=1 if lw > 1 else 0.9, label=lab,
                            zorder=3 if lw > 1 else 2)
                ax.grid(True, color="#D0D0D0", lw=0.4, ls=(0, (3, 3)), zorder=0)
                ax.set_axisbelow(True)
                ax.set_xticks(kk)
                ax.tick_params(length=2.5, width=0.6)
                for s in ax.spines.values():
                    s.set_linewidth(0.6)
                ax.text(0.0, 1.03, f"({'abcdefghi'[3 * i + j]})", transform=ax.transAxes, va="bottom", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(ylab)
                if i == 0:
                    ax.set_title(name, fontsize=9)
                ax.set_xlabel("Number of PMUs $k$")
                ax.tick_params(labelbottom=True)
        h, l = axs[0, 0].get_legend_handles_labels()
        order = [1, 2, 3, 4, 5, 6, 7, 0]
        leg = fig.legend([h[o] for o in order], [l[o] for o in order], loc="lower center", ncol=4,
                         bbox_to_anchor=(0.5, -0.005), fontsize=7.5, handlelength=2.2, columnspacing=1.1)
        leg.get_frame().set_linewidth(0.6)
        fig.tight_layout(rect=(0, 0.065, 1, 1), h_pad=1.0, w_pad=0.8)
        save(fig, "fig_placement")


def fig_derfault():
    e9 = load("e9_derfault")
    lad = e9["ladder"]
    br = e9["breakdown"]
    gen = e9["generalisation_from_S1"]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.5))
    ax = axs[0]
    reps = ["AGG", "PPA", "SEQ", "FPS"]
    for m, (c, mk) in MODELS.items():
        ax.plot(range(4), [100 * np.mean(lad[f"0.01_{r}_{m}"]) for r in reps], color=c, marker=mk, lw=0.9, label=m)
    ax.set_xticks(range(4), [N[r] for r in reps])
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Classifiers, 1% TVE", loc="left")
    ax.legend(ncol=2, fontsize=6.5, loc="lower right", columnspacing=0.6, handletextpad=0.3)
    ax = axs[1]
    sc = sorted(br["AGG"]["by_scenario"])
    for r, c, mk in (("AGG", GREY, "v"), ("SEQ", BLUE, "o"), ("FPS", VERM, "s")):
        ax.plot(range(6), [100 * br[r]["by_scenario"][q] for q in sc], color=c, marker=mk, label=N[r])
    ax.set_xticks(range(6), sc)
    ax.set_ylim(80, 95)
    ax.set_xlabel("DER scenario")
    ax.set_title("(b) Trained on all", loc="left")
    ax.legend(ncol=3, fontsize=7, columnspacing=0.8, handletextpad=0.3, loc="upper center")
    ax = axs[2]
    sc2 = sorted(gen["AGG"])
    for r, c, mk in (("AGG", GREY, "v"), ("SEQ", BLUE, "o"), ("FPS", VERM, "s")):
        ax.plot(range(5), [100 * gen[r][s] for s in sc2], color=c, marker=mk, label=N[r])
    ax.set_xticks(range(5), sc2)
    ax.set_ylim(76, 95)
    ax.set_xlabel("Test scenario")
    ax.set_title("(c) Trained on S1 (no DER)", loc="left")
    ax.legend(ncol=3, fontsize=7, columnspacing=0.8, handletextpad=0.3, loc="upper center")
    fig.tight_layout()
    save(fig, "fig_derfault")


def fig_derfault_heat(name="fig_derfault"):
    """DER-Fault as three heat maps with square cells, same colour scale as the noise figure."""
    e9 = load("e9_derfault")
    lad, br, gen = e9["ladder"], e9["breakdown"], e9["generalisation_from_S1"]
    reps, reps3 = ["AGG", "PPA", "SEQ", "FPS"], ["AGG", "SEQ", "FPS"]
    models = ["RF", "XGB", "LGBM", "MLP", "SVM", "kNN", "LR"]
    sc, sc2 = sorted(br["AGG"]["by_scenario"]), sorted(gen["AGG"])
    A = np.array([[100 * np.mean(lad[f"0.01_{r}_{m}"]) for m in models] for r in reps])
    B = np.array([[100 * br[r]["by_scenario"][q] for q in sc] for r in reps3])
    C = np.array([[100 * gen[r][q] for q in sc2] for r in reps3])
    rc = {"axes.labelweight": "normal", "axes.titleweight": "normal"}
    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.9), gridspec_kw={"width_ratios": [7, 6, 5], "wspace": 0.28})
        panels = ((A, ["XGB" if m == "XGB" else "LGBM" if m == "LGBM" else m for m in models], [N[r] for r in reps],
                   "Classifier", "(a) Trained on S1–S6, all classifiers"),
                  (B, sc, [N[r] for r in reps3], "DER scenario", "(b) LightGBM, by scenario"),
                  (C, sc2, [N[r] for r in reps3], "Test scenario", "(c) LightGBM, trained on S1"))
        for ax, (M, xl, yl, xlab, t) in zip(axs, panels):
            im = ax.imshow(M, cmap="RdYlBu", vmin=60, vmax=100, aspect="equal")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] > 93 or M[i, j] < 66 else "black")
            ax.set_xticks(range(len(xl)), xl)
            ax.set_yticks(range(len(yl)), yl)
            ax.tick_params(top=False, right=False, length=0, labelsize=7)
            ax.set_xlabel(xlab)
            ax.set_title(t, loc="left", fontsize=8.5)
        fig.canvas.draw()
        p = axs[2].get_position()
        cax = fig.add_axes([p.x1 + 0.012, p.y0, 0.01, p.height])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6.5, length=2, direction="in")
        cb.set_label("Accuracy (%)", fontsize=7.5)
        cb.outline.set_linewidth(0.6)
        save(fig, name)


if __name__ == "__main__":
    # figures that appear in the paper (figure 1 and 2 are TikZ drawings, see tikz/)
    fig_noise_rf()
    fig_exhaustive()
    fig_placement()
