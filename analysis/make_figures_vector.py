"""
Figures 5-8 of the manuscript as vector graphics (PDF for pdflatex, EPS for production): sans-serif
lettering (Liberation Sans = Arial metrics), a figure width of 131 mm so that the lettering prints at
7-8 pt, panel labels instead of titles, and line styles/markers in addition to colours.

usage: python analysis/make_figures_vector.py [out_dir]      (default: paper/figures_vector)

  Fig5  benchmark networks (toy network with demands, Solomon C101 customers)
  Fig6  customer geometries of A-1-100, A-1-R-100, A-1-RC-100 after pre-processing
  Fig7  convergence profiles (protocol E traces)
  Fig8  sensitivity analyses
The data are read from data/ and results/ exactly as in analysis/make_tables.py.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import make_tables as mt  # noqa: E402  (helpers: load, best_known_values, group_of, ...)
from mvmd.instance import TOY_CUSTOMERS, TOY_DEPOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper", "figures_vector")
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
TEXTW = 131 * MM          # figure width (131 mm)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,       # embed TrueType fonts (no Type 3)
    "savefig.dpi": 600, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".eps"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=7.5, pad=3)


# ----------------------------------------------------------------------------
# Fig. 5: benchmark networks
# ----------------------------------------------------------------------------
def fig_networks():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 0.47 * TEXTW), gridspec_kw=dict(width_ratios=[1, 1.05]))
    ax = axes[0]
    for v in range(0, 8):
        ax.axhline(v, color="0.8", lw=0.4, zorder=0); ax.axvline(v, color="0.8", lw=0.4, zorder=0)
    ax.scatter([TOY_DEPOT[0] / 1000], [TOY_DEPOT[1] / 1000], marker="s", s=42, c="#d62728", edgecolors="k", linewidths=0.6, zorder=3)
    ax.annotate("0", (TOY_DEPOT[0] / 1000, TOY_DEPOT[1] / 1000), xytext=(-2, -9), textcoords="offset points", ha="center", fontsize=6.5)
    for i, x, y, d in TOY_CUSTOMERS:
        ax.scatter([x / 1000], [y / 1000], marker="o", s=34, c="#c6c6f5", edgecolors="#1f1fbf", linewidths=0.7, zorder=3)
        ax.annotate(f"{i} [{d}]", (x / 1000, y / 1000), xytext=(5, 3), textcoords="offset points", fontsize=6.2)
    ax.set_xlim(-0.4, 7.6); ax.set_ylim(-0.4, 7.0); ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    panel_label(ax, "(a) Toy network")

    ax = axes[1]
    raw = os.path.join(ROOT, "data", "raw", "solomon_c101.txt")
    rows = []
    for line in open(raw):
        p = line.split()
        if len(p) >= 7 and p[0].isdigit():
            rows.append([float(t) for t in p[:7]])
    rows = np.array(rows)
    dep = rows[rows[:, 0] == 0][0]; cus = rows[rows[:, 0] > 0]
    ax.grid(color="0.85", lw=0.4)
    ax.scatter(cus[:, 1] / 10, cus[:, 2] / 10, s=6, c="#1f1fbf", zorder=3)
    for r in cus:
        ax.annotate(f"{int(r[0])}", (r[1] / 10, r[2] / 10), xytext=(1.5, 1.5), textcoords="offset points", fontsize=3.6, color="0.25")
    ax.scatter([dep[1] / 10], [dep[2] / 10], marker="s", s=30, c="#d62728", edgecolors="k", linewidths=0.6, zorder=4)
    ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.3, 10.3); ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    panel_label(ax, "(b) Solomon C101 customers")
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig5")


# ----------------------------------------------------------------------------
# Fig. 6: geometries after pre-processing
# ----------------------------------------------------------------------------
def fig_geometries():
    names = [("A-1-100", "(a) C101 (A-1-100)"), ("A-1-R-100", "(b) R101 (A-1-R-100)"), ("A-1-RC-100", "(c) RC101 (A-1-RC-100)")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 0.40 * TEXTW))
    cmap = plt.get_cmap("tab20")
    counts = {}
    for ax, (nm, lab) in zip(axes, names):
        d = json.load(open(os.path.join(ROOT, "data", "instances", nm + ".json")))
        xy = {c["id"]: (c["x"] / 1000, c["y"] / 1000) for c in d["customers"]}
        dep = (d["depot"][0] / 1000, d["depot"][1] / 1000)
        rad = d["params"]["gamma"] * d["params"]["Ld"] / 1000
        ax.add_patch(plt.Circle(dep, rad, fill=False, ls="--", lw=0.7, color="#b22222"))
        s0 = np.array([xy[i] for i in d["S0"]])
        ax.scatter(s0[:, 0], s0[:, 1], s=9, facecolors="none", edgecolors="#b22222", linewidths=0.6, zorder=3)
        for k, cl in enumerate(d["clusters"]):
            col = cmap(k % 20)
            m = np.array([xy[i] for i in cl["members"]])
            ax.scatter(m[:, 0], m[:, 1], s=5, color=col, zorder=3)
            ax.scatter([cl["x"] / 1000], [cl["y"] / 1000], marker="^", s=22, color=col, edgecolors="k", linewidths=0.5, zorder=4)
        ax.scatter([dep[0]], [dep[1]], marker="s", s=28, c="#ffd700", edgecolors="k", linewidths=0.7, zorder=5)
        ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.3, 10.3); ax.set_aspect("equal")
        ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel("x (km)")
        counts[nm] = (len(d["S0"]), len(d["S1"]), len(d["clusters"]))
        s0n, s1n, cn = counts[nm]
        ax.set_title(f"{lab}\n$|S_0|={s0n}$, $|S_1|={s1n}$, {cn} clusters", fontsize=6.8, pad=3)
    axes[0].set_ylabel("y (km)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "Fig6")
    return counts


# ----------------------------------------------------------------------------
# Fig. 7: convergence profiles (same computation as make_tables.fig_convergence)
# ----------------------------------------------------------------------------
def fig_convergence():
    df = mt.load("protoE")
    bk = mt.best_known_values(df, mt.load("protoT"))
    style = {"I": dict(color="#4C72B0", ls=(0, (4, 1.6)), lw=1.1),
             "P": dict(color="#55A868", ls=(0, (6, 1.5, 1, 1.5)), lw=1.1),
             "OSD": dict(color="#C44E52", ls="-", lw=1.4)}
    panels = [("C", 100), ("R", 100), ("RC", 100), ("M", 300)]
    fig, axes = plt.subplots(len(panels), 2, figsize=(TEXTW, 1.55 * len(panels) * 1.0), squeeze=False)
    letters = "abcdefgh"
    for row, grp in enumerate(panels):
        budget = mt.BUDGET[grp[1]]
        for a in mt.ALGOS:
            sub = df[(df["algo"] == a) & (df["instance"].apply(mt.group_of) == grp)]
            grid_ops = np.linspace(0, budget, 121); curves_ops = []
            tmax = sub["time"].max(); grid_t = np.linspace(0, tmax, 121); curves_t = []
            for _, r in sub.iterrows():
                p = os.path.join(mt.RES, "traces", f"protoE_{r['instance']}_{a}_-_{r['seed']}.json")
                if not os.path.exists(p):
                    continue
                tr = json.load(open(p)); b = bk[r["instance"]]
                ops = np.array([t[0] for t in tr]); tim = np.array([t[1] for t in tr]); val = np.array([t[2] for t in tr])

                def at(grid, key):
                    out = np.empty(len(grid))
                    for q, gg in enumerate(grid):
                        idx = np.searchsorted(key, gg, side="right") - 1
                        out[q] = val[max(idx, 0)]
                    return out
                curves_ops.append(100 * (at(grid_ops, ops) - b) / b); curves_t.append(100 * (at(grid_t, tim) - b) / b)
            if curves_ops:
                axes[row, 0].plot(grid_ops, np.mean(curves_ops, axis=0), label=mt.NAMES[a], **style[a])
                axes[row, 1].plot(grid_t, np.mean(curves_t, axis=0), label=mt.NAMES[a], **style[a])
        lab = mt.group_label(grp)
        panel_label(axes[row, 0], f"({letters[2 * row]}) {lab}")
        panel_label(axes[row, 1], f"({letters[2 * row + 1]}) {lab}")
        axes[row, 0].set_xlabel("Elementary operations"); axes[row, 1].set_xlabel("CPU time (s)")
        for ax in (axes[row, 0], axes[row, 1]):
            ax.set_yscale("symlog", linthresh=0.1); ax.grid(color="0.85")
            ax.set_ylabel("Excess (%)")
    axes[0, 0].legend(frameon=False, handlelength=3.2)
    fig.tight_layout(h_pad=0.5, w_pad=1.0)
    save(fig, "Fig7")


# ----------------------------------------------------------------------------
# Fig. 8: sensitivity analyses (same data as make_tables.fig_sensitivity)
# ----------------------------------------------------------------------------
def fig_sensitivity():
    df = mt.load("sens")
    df["par"] = df["variant"].str.split("=").str[0]; df["val"] = df["variant"].str.split("=").str[1].astype(float)
    df["n"] = df["instance"].str.split("-").str[-1].astype(int)
    g = df.groupby(["par", "val", "n"]).agg(best=("best", "min")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 0.36 * TEXTW))
    labels = {"gamma": ("Radius factor $\\gamma$", 1.0, "(a)"), "Ld": ("Drone endurance $L_d$ (km)", 1000.0, "(b)"),
              "Qd": ("Drone capacity $Q_d$ (units)", 1.0, "(c)")}
    for ax, par in zip(axes, ["gamma", "Ld", "Qd"]):
        for n, mk, ls, col in ((15, "o", "-", "#1f77b4"), (30, "s", "--", "#ff7f0e")):
            s = g[(g["par"] == par) & (g["n"] == n)].sort_values("val")
            ax.plot(s["val"] / labels[par][1], s["best"], marker=mk, ms=3, ls=ls, lw=1.0, color=col, label=f"{n} customers")
        ax.set_xlabel(labels[par][0]); ax.grid(color="0.85")
        panel_label(ax, labels[par][2])
    axes[1].set_xticks([5, 7, 9, 11, 13, 15]); axes[2].set_xticks([4, 6, 8, 10, 12])
    axes[0].set_ylabel("Objective (CNY)"); axes[2].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=0.8)
    save(fig, "Fig8")


if __name__ == "__main__":
    fig_networks()
    c = fig_geometries()
    fig_convergence()
    fig_sensitivity()
    json.dump(c, open(os.path.join(OUT, "geometry_counts.json"), "w"))
    print("figures written to", OUT, c)
