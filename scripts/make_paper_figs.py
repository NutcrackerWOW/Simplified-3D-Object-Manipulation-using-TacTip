#!/usr/bin/env python3
"""Paper figures F2-F6 (IEEE format, light surface, print-safe).

Style follows the dataviz reference palette: categorical blue/aqua, sequential
blue ramp for magnitude, status red reserved for failure marks, ink/muted text
tokens, hairline grid. Single column = 3.5 in, double = 7.16 in.

  python scripts/make_paper_figs.py            # all
  python scripts/make_paper_figs.py --only F3
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
OUT = os.path.join(RESULTS, "paper_figs")

INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#4a3aa7"
CRIT = "#d03b3b"                       # status: failure marks only
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
       "#0d366b"]

plt.rcParams.update({
    "font.size": 7.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": SEC, "axes.titlesize": 8, "axes.titlecolor": INK,
    "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7,
    "figure.dpi": 120, "savefig.dpi": 300,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ results/paper_figs/{name}.png/.pdf")


# --------------------------------------------------------------- F1 setup
def fig1():
    from pinchlab.model import HandKinematics, set_posture
    from pinchlab.params import Posture, deg
    kin = HandKinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(0.0), deg(0.0), 0.024)
    set_posture(kin.plant, kin.plant_ctx, posture)
    chain = ["{s}_Base_1", "{s}_Proximal_F_1", "{s}_Middle_F_1",
             "{s}_Distal_F_1", "{s}_Tip_1"]
    pts = {}
    for s in "LR":
        pts[s] = np.array([kin.plant.EvalBodyPoseInWorld(
            kin.plant_ctx, kin.plant.GetBodyByName(n.format(s=s))
        ).translation() for n in chain]) * 1e3          # mm
    g = kin.gap(posture)
    mid = g.mid * 1e3
    r_dome = float(np.linalg.norm(pts["L"][-1] - g.p_WA * 1e3))

    fig = plt.figure(figsize=(7.16, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 0.95, 1.1], wspace=0.35)

    # (a) FK schematic at the reference grasp (x-z plane)
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(plt.Rectangle((-6, 250), 62, 16, facecolor=GRID,
                               edgecolor=AXIS, lw=0.6))
    for s in "LR":
        xz = pts[s][:, [0, 2]]
        ax.plot(xz[:, 0], xz[:, 1], "-", color=SEC, lw=2.2,
                solid_capstyle="round", zorder=2)
        ax.plot(xz[0:4, 0], xz[0:4, 1], "o", ms=3.2, mfc="white",
                mec=SEC, mew=0.8, zorder=3)
        ax.add_patch(plt.Circle((xz[-1, 0], xz[-1, 1]), r_dome,
                                facecolor="#eeede9", edgecolor=MUT,
                                lw=0.8, zorder=4))
    ax.add_patch(plt.Rectangle((mid[0] - 12, mid[2] - 6), 24, 12,
                               facecolor=SEQ[1], edgecolor=BLUE, lw=0.9,
                               zorder=5))
    # joint labels on the left (R) finger, leader lines clear of the plate
    for (x, z), lab in zip(pts["R"][:4, [0, 2]],
                           ["wave φ", "MCP", "PIP θ", "DIP (passive)"]):
        ax.plot([-9, x - 2], [z, z], color=AXIS, lw=0.5, zorder=1)
        ax.text(-10, z, lab, ha="right", va="center", fontsize=6,
                color=SEC)
    ax.annotate("", xy=(62, 165), xytext=(62, 195),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    ax.text(64, 180, "g", fontsize=7.5, style="italic", color=INK,
            va="center")
    ax.plot([46, 66], [140, 140], color=INK, lw=1.2)
    ax.text(56, 143, "20 mm", ha="center", fontsize=6, color=SEC)
    ax.set_xlim(-30, 72); ax.set_ylim(133, 270)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) pinch testbed", loc="left")

    # (b) contact decomposition frame
    ax = fig.add_subplot(gs[0, 1])
    for xc in (-17.5, 17.5):
        ax.add_patch(plt.Circle((xc, 0), 5.5, facecolor="#eeede9",
                                edgecolor=MUT, lw=0.8, zorder=2))
    ax.add_patch(plt.Rectangle((-12, -6), 24, 12, facecolor=SEQ[1],
                               edgecolor=BLUE, lw=0.9, zorder=3))
    # squeeze normal + gravity shear at the left contact, weight at center
    ax.annotate("", xy=(-3.5, 0), xytext=(-12, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.text(-7.5, -3.9, "f₁", color=BLUE, fontsize=7, ha="center")
    ax.annotate("", xy=(-12, 8.5), xytext=(-12, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.2))
    ax.text(-14.2, 6.5, "f₂", color="#0f7a54", fontsize=7, ha="center")
    ax.annotate("", xy=(0, -10), xytext=(0, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(1.6, -9, "mg", color=INK, fontsize=7)
    # grasp-frame compass, top right
    ox, oz = 24, 10
    ax.annotate("", xy=(ox + 7, oz), xytext=(ox, oz),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    ax.annotate("", xy=(ox, oz + 7), xytext=(ox, oz),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    ax.plot(ox, oz, marker="o", ms=5, mfc="white", mec=INK, mew=0.9)
    ax.plot(ox, oz, marker=".", ms=2, color=INK)
    ax.text(ox + 8, oz - 1, "e₁", fontsize=6.5, color=INK)
    ax.text(ox - 1, oz + 8.5, "e₂", fontsize=6.5, color=INK)
    ax.text(ox - 4.5, oz - 4.5, "e₃", fontsize=6.5, color=INK)
    ax.text(0, -17.5, "ρ = ‖(f₂, f₃)‖ / f₁     hold ⇔ ρ < μ_eff",
            fontsize=7, color=INK, ha="center")
    ax.set_xlim(-30, 36); ax.set_ylim(-21, 20)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) grasp frame & utilization", loc="left")

    # (c) thin grasp manifold: pad gap vs MCP flexion
    ax = fig.add_subplot(gs[0, 2])
    mcps = np.linspace(-2.0, 12.0, 57)
    gaps = np.array([kin.gap(Posture.symmetric(0.0, deg(m), 0.0)).distance
                     for m in mcps]) * 1e3
    ax.axhspan(19, 29, color="#e4efe9", zorder=1)
    ax.text(11.7, 27.8, "box graspable\n(gap 19–29 mm)", fontsize=6,
            color="#0f7a54", ha="right", va="top")
    ax.axvspan(12.5, 14.5, color="#f6dcdc", zorder=1)
    ax.text(13.5, 18, "fingers\ncross", fontsize=6, color=CRIT,
            ha="center")
    ax.plot(mcps, gaps, color=BLUE, lw=1.2, zorder=3)
    m_lo = float(np.interp(29, gaps[::-1], mcps[::-1]))
    m_hi = float(np.interp(19, gaps[::-1], mcps[::-1]))
    ax.annotate("", xy=(m_hi, 5), xytext=(m_lo, 5),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.9))
    ax.plot([m_lo, m_lo], [5, 29], color=AXIS, lw=0.6, ls=":")
    ax.plot([m_hi, m_hi], [5, 19], color=AXIS, lw=0.6, ls=":")
    ax.text(m_hi + 0.5, 4.4, "≈3° grasp window", fontsize=6.5, color=INK)
    i0 = int(np.argmin(np.abs(gaps)))
    ax.plot(mcps[i0], 0, "o", ms=3.5, color=BLUE)
    ax.text(mcps[i0] + 0.3, 1.2, "pads touch", fontsize=6, color=SEC)
    ax.set_xlim(-2, 14.5); ax.set_ylim(-4, 38)
    ax.set_xlabel("MCP flexion (°)"); ax.set_ylabel("pad gap (mm)")
    ax.grid(True, axis="y")
    ax.set_title("(c) thin grasp manifold (φ = 0, θ = 0)", loc="left")
    save(fig, "F1_setup")


# ---------------------------------------------------------------- F2 hero
def fig2():
    d = np.load(os.path.join(RESULTS, "fig_rolling_data.npz"))
    t, ph = d["t"], d["phase"]
    box_p, box_R = d["box_p"], d["box_R"]
    rot = np.rad2deg(d["rot_drift"]); drift = 1e3 * d["drift"]
    i0 = int(np.nonzero(rot > 0)[0][0])          # hold start
    R0 = box_R[i0]

    def pitch(i):                                 # rotation about world y
        Rr = R0.T @ box_R[i]
        return np.arctan2(Rr[0, 2], Rr[0, 0])

    # dome circle: radius = tip-to-contact distance at hold start
    r_dome = float(np.linalg.norm(d["tipL_p"][i0] - d["cL"][i0]))
    # snapshots: three through the creep phase, two through the roll-off
    t_fail = t[i0 + int(np.nonzero(drift[i0:] > 2.0)[0][0])] - t[i0]
    th_all = t[i0:] - t[i0]
    creep_ts = [0.0, 0.55 * t_fail, 0.97 * t_fail]
    idxs = [i0 + int(np.argmin(np.abs(th_all - a))) for a in creep_ts]
    idxs += [i0 + int(np.argmin(np.abs(rot[i0:] - a))) for a in (60., 95.)]
    labels = "abcde"

    fig = plt.figure(figsize=(7.16, 2.7))
    gs = fig.add_gridspec(2, 21, height_ratios=[1.55, 1.0], hspace=0.55)
    hx, hz = 1e3 * float(d["box_size"]) / 2, 1e3 * float(d["box_height"]) / 2
    for k, (i, lab) in enumerate(zip(idxs, labels)):
        ax = fig.add_subplot(gs[0, 4 * k:4 * k + 4])
        th = pitch(i)
        c, s = np.cos(th), np.sin(th)
        corners = np.array([[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz],
                            [-hx, -hz]])
        Rm = np.array([[c, s], [-s, c]])
        pts = corners @ Rm.T + 1e3 * box_p[i][[0, 2]]
        ax.fill(pts[:, 0], pts[:, 1], color=SEQ[1], lw=0.8,
                edgecolor=SEQ[5], zorder=3)
        for tip in ("tipL_p", "tipR_p"):
            p = 1e3 * d[tip][i][[0, 2]]
            ax.add_patch(plt.Circle(p, 1e3 * r_dome, facecolor="#f0efec",
                                    edgecolor=MUT, lw=0.8, zorder=2))
        ax.set_aspect("equal")
        x0 = 1e3 * box_p[i0][0]
        ax.set_xlim(x0 - 32, x0 + 32)
        ax.set_ylim(1e3 * box_p[i0][2] - 58, 1e3 * box_p[i0][2] + 24)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({lab})  t = {t[i] - t[i0]:.2f} s, "
                     f"{rot[i]:.0f}°", fontsize=7)
    # gravity arrow on first snapshot
    ax0 = fig.axes[0]
    ax0.annotate("", xy=(0.94, 0.35), xytext=(0.94, 0.72),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=SEC, lw=0.9))
    ax0.text(0.99, 0.53, "g", transform=ax0.transAxes, color=SEC,
             fontsize=7, ha="left", va="center", style="italic")

    th = t[i0:] - t[i0]
    axr = fig.add_subplot(gs[1, 0:10])
    axr.plot(th, rot[i0:], color=BLUE, lw=1.2)
    axr.set_xlabel("time in hold (s)"); axr.set_ylabel("box rotation (°)")
    axr.set_ylim(0, 200); axr.grid(True, axis="y")
    axd = fig.add_subplot(gs[1, 12:21])
    axd.plot(th, drift[i0:], color=BLUE, lw=1.2)
    axd.axhline(2.0, color=CRIT, lw=0.8, ls="--")
    axd.text(0.15, 3.0, "2 mm slip threshold", color=CRIT, fontsize=6.5)
    axd.set_xlabel("time in hold (s)"); axd.set_ylabel("drift (mm)")
    axd.set_ylim(0, 25); axd.grid(True, axis="y")
    for ax in (axr, axd):
        for i in idxs:
            ax.axvline(t[i] - t[i0], color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(0, th[-1])
    # snapshot letters above the rotation trace ("c-e" cluster shares one)
    for i, lab in list(zip(idxs, labels))[:2]:
        axr.annotate(lab, xy=(t[i] - t[i0], 200), xytext=(0, 2),
                     textcoords="offset points", color=MUT, fontsize=6.5,
                     ha="center", va="bottom", annotation_clip=False)
    axr.annotate("c–e", xy=(t[idxs[3]] - t[i0], 200), xytext=(0, 2),
                 textcoords="offset points", color=MUT, fontsize=6.5,
                 ha="center", va="bottom", annotation_clip=False)
    save(fig, "F2_rolling")


# ------------------------------------------------------- F3 shapes+materials
def fig3():
    shapes = [("disc (flat faces)", "shape_disc"),
              ("box 25×25×12", "material_fric_base"),
              ("prism 25×50×12", "shape_prism"),
              ("cylinder (rim)", "shape_cylinder"),
              ("disc on edge (key pinch)", "shape_disc_edge"),
              ("sphere", "shape_sphere")]
    mats = [("pair μd 0.45", "material_fric_low", "fric"),
            ("pair μd 0.85", "material_fric_base", "fric"),
            ("pair μd 1.03", "material_fric_high", "fric"),
            ("tip 25 kPa", "material_tip_soft", "stiff"),
            ("tip 50 kPa", "material_fric_base", "stiff"),
            ("tip 100 kPa", "material_tip_stiff", "stiff")]

    def mu(f):
        with open(os.path.join(RESULTS, f + ".json")) as fh:
            return json.load(fh)["mu_eff"]

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 3.4),
                               gridspec_kw=dict(hspace=0.75))
    # (a) shapes
    names = [n for n, _ in shapes]
    vals = [mu(f) for _, f in shapes]
    y = np.arange(len(names))[::-1]
    a.barh(y, vals, height=0.62, color=BLUE, zorder=3)
    a.axvline(0.85, color=INK, lw=0.8, ls=(0, (4, 2)))
    a.text(0.865, len(names) - 0.55, "Coulomb pair μd", fontsize=6.5,
           color=SEC, rotation=0)
    for yi, v in zip(y, vals):
        a.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=6.5,
               color=SEC)
    a.set_yticks(y, names, fontsize=7)
    a.set_xlim(0, 1.02); a.set_xlabel("μ_eff = (m·g/2) / f*")
    a.set_title("(a) shape study — 50 g, identical friction", loc="left")
    # (b) materials
    names = [n for n, _, _ in mats]
    vals = [mu(f) for _, f, _ in mats]
    cols = [BLUE if k == "fric" else AQUA for _, _, k in mats]
    pair = [0.45, 0.85, 1.03, None, None, None]
    y = np.arange(len(names))[::-1]
    for yi, v, c in zip(y, vals, cols):
        if v is None:
            b.text(0.02, yi, "no stable window", va="center",
                   fontsize=6.5, color=CRIT)
        else:
            b.barh(yi, v, height=0.62, color=c, zorder=3)
            b.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=6.5,
                   color=SEC)
    for yi, pm in zip(y, pair):
        if pm:
            b.plot(pm, yi, marker="D", ms=3.5, color=INK, zorder=4)
    # direct label on the top-most diamond instead of a legend box
    b.annotate("Coulomb pair μd", xy=(0.45, y[0]), xytext=(0.56, y[0] + 0.05),
               fontsize=6.5, color=SEC, va="center",
               arrowprops=dict(arrowstyle="-", color=MUT, lw=0.6))
    b.set_yticks(y, names, fontsize=7)
    b.set_xlim(0, 1.12); b.set_xlabel("μ_eff")
    b.set_title("(b) material study — box, 50 g "
                "(blue: friction, teal: tip stiffness)", loc="left",
                fontsize=7.5)
    for ax in (a, b):
        ax.grid(True, axis="x"); ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)
    save(fig, "F3_shapes_materials")


# --------------------------------------------------------- F4 envelope map
def fig4():
    from pinchlab.fit import mu_eff_from_boundary, classify_regimes
    conds = mu_eff_from_boundary(
        pd.read_csv(os.path.join(RESULTS, "sweepB_conditions.csv")))
    conds = classify_regimes(conds)
    conds.loc[(conds.regime == "friction") & (conds.pip_deg == -5.0),
              "regime"] = "fragile"
    waves = sorted(conds.wave_deg.unique())
    pips = sorted(conds.pip_deg.unique())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    vmin, vmax = 0.3, 0.9

    fig = plt.figure(figsize=(7.16, 2.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.22, 1.2],
                          wspace=0.4)
    for k, m in enumerate((0.03, 0.06)):
        ax = fig.add_subplot(gs[0, k])
        sub = conds[conds["mass"] == m]
        Z = np.full((len(pips), len(waves)), np.nan)
        for _, r in sub.iterrows():
            Z[pips.index(r.pip_deg), waves.index(r.wave_deg)] = r.mu_eff
        pc = ax.pcolormesh(np.arange(len(waves) + 1),
                           np.arange(len(pips) + 1), Z, cmap=cmap,
                           vmin=vmin, vmax=vmax, edgecolors="white",
                           linewidth=1.2)
        # regime hatches (hatch ink chosen against the cell lightness:
        # support cells are dark -> white, fragile cells light -> gray)
        for _, r in sub.iterrows():
            if r.regime in ("support", "fragile"):
                i, j = pips.index(r.pip_deg), waves.index(r.wave_deg)
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False,
                    hatch="///" if r.regime == "support" else "xxx",
                    edgecolor="white" if r.regime == "support" else SEC,
                    lw=0))
        ax.set_xticks(np.arange(len(waves)) + 0.5,
                      [f"{w:g}" for w in waves])
        ax.set_yticks(np.arange(len(pips)) + 0.5, [f"{p:g}" for p in pips])
        ax.set_xlabel("grasp tilt φ (°)")
        if k == 0:
            ax.set_ylabel("PIP flexion θ (°)")
        ax.set_title(f"({'ab'[k]}) μ_eff, {int(m*1000)} g", loc="left")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label("μ_eff", color=SEC)
    cb.outline.set_visible(False)
    cax.tick_params(length=0)
    fig.axes[0].text(0.02, -0.32,
                     "hatched: /// support regime,  xxx fragile "
                     "(seed-bimodal) — excluded from the fit",
                     transform=fig.axes[0].transAxes, fontsize=6.5,
                     color=SEC)

    # (c) force window vs mass at reference posture
    ax = fig.add_subplot(gs[0, 4])
    masses = np.array([30, 60])
    fstar = np.array([0.195, 0.49])
    ax.fill_between([20, 130], 2.5, 3.2, color="#f6dcdc", zorder=1)
    ax.plot([20, 130], [2.5, 2.5], color=CRIT, lw=1.0)
    ax.text(23, 2.62, "squeeze-out ejection", color=CRIT, fontsize=6.5)
    mm = np.linspace(22, 118, 60)
    # measured boundary + its creep-margin requirement
    from pinchlab.fit import mu_eff_model
    with open(os.path.join(RESULTS, "sweepB_fit.json")) as fh:
        coef = np.array(list(json.load(fh)["coefficients"].values()))
    fpred = np.array([(m / 1000 * 9.81 / 2) /
                      float(mu_eff_model(coef, [0.0], [0.0], m / 1000)[0])
                      for m in mm])
    ax.fill_between(mm, 2.0 * fpred, 2.5, where=2.0 * fpred < 2.5,
                    color="#e4efe9", zorder=1)
    ax.plot(mm, fpred, color=BLUE, lw=1.2)
    ax.plot(mm, 2.0 * fpred, color=AQUA, lw=1.2)
    # direct labels on the curves (no legend box)
    ax.text(64, 0.28, "boundary f*", color=BLUE, fontsize=6.5,
            rotation=14)
    ax.text(38, 0.85, "2.0× f* (creep-safe)", color="#0f7a54",
            fontsize=6.5, rotation=28)
    ax.text(60, 1.95, "stable window", color="#0f7a54", fontsize=6.5,
            style="italic")
    ax.plot(masses, fstar, "o", ms=4, color=BLUE, zorder=4)
    ax.plot([120], [1.5], marker="x", ms=5, color=CRIT, mew=1.4, zorder=4)
    ax.text(118, 1.12, "120 g: window\nclosed", color=CRIT, fontsize=6,
            ha="right", va="top")
    ax.set_xlim(20, 130); ax.set_ylim(0, 3.2)
    ax.set_xlabel("object mass (g)"); ax.set_ylabel("squeeze force f1 (N)")
    ax.set_title("(c) force window, reference posture", loc="left")
    ax.grid(True, axis="y")
    save(fig, "F4_envelope")


# -------------------------------------------------- F5 creep distributions
def fig5():
    with open(os.path.join(RESULTS, "phaseB_seeds.json")) as fh:
        pb = json.load(fh)
    with open(os.path.join(RESULTS, "creep_timestep_control.json")) as fh:
        cc = json.load(fh)
    static = pb["summary"]["B_static/fixed"]["fail_times_s"]
    moving = pb["summary"]["C_moving/scheduled"]["fail_times_s"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.9),
                               gridspec_kw=dict(wspace=0.35,
                                                width_ratios=[1.25, 1]))
    rng = np.random.default_rng(1)
    for y0, (vals, col, lab) in enumerate(
            [(static, BLUE, "motionless"), (moving, AQUA, "moving ±10°")]):
        yy = y0 + rng.uniform(-0.09, 0.09, len(vals))
        a.plot(vals, yy, "o", ms=3.4, color=col, alpha=0.9, zorder=3)
        m, s = np.mean(vals), np.std(vals)
        a.plot([m - s, m + s], [y0 - 0.28] * 2, color=INK, lw=0.9)
        a.plot(m, y0 - 0.28, marker="|", ms=5, color=INK, mew=1.2)
    a.set_yticks([0, 1], ["motionless", "moving ±10°"], fontsize=7)
    a.set_xlabel("failure time (s), margin 1.3")
    a.set_xlim(4, 12); a.set_ylim(-0.75, 1.5)
    a.grid(True, axis="x"); a.tick_params(left=False)
    a.spines["left"].set_visible(False)
    a.text(4.2, 1.32, "U = 57, p = 0.62 (n = 10+10)", fontsize=6.5,
           color=SEC)
    a.set_title("(a) rolling creep, 50 g", loc="left")

    # (b) time-step control: seed-matched failure times
    runs = {}
    for r in pb["runs"]:
        if r["block"] == "B_static" and r["seed"] < 5:
            runs.setdefault(r["seed"], {})[2.0] = r["fail_time_s"]
    for r in cc["runs"]:
        runs.setdefault(r["seed"], {})[r["time_step"] * 1e3] = r["fail_time_s"]
    dts = [2.0, 1.0, 0.5]
    for seed, vals in runs.items():
        ys = [dts.index(dt) for dt in dts if dt in vals]
        xs = [vals[dt] for dt in dts if dt in vals]
        b.plot(xs, ys, "-o", ms=3, lw=0.7, color=BLUE, alpha=0.85)
    b.set_yticks(range(len(dts)), [f"{d:g} ms" for d in dts], fontsize=7)
    b.set_xlabel("failure time (s)")
    b.set_ylim(-0.5, 2.5); b.set_xlim(4, 12)
    b.grid(True, axis="x"); b.tick_params(left=False)
    b.spines["left"].set_visible(False)
    b.set_title("(b) time-step control", loc="left")
    save(fig, "F5_creep")


# ------------------------------------------------------------- F6 slip
def fig6():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rescore_slip import (energies, load_episodes, floor_of,
                              fit_floor_model, scheduled_thr)
    from pinchlab.slipdetect import DetectorConfig
    eps = load_episodes()
    cfg = DetectorConfig()
    E = energies(eps, cfg, "tactip")
    pos = sorted([e for e in E if e["kind"] == "pos"],
                 key=lambda e: (e["wave"], e["pip"], e["mass"], e["seed"]))
    neg = sorted([e for e in E if e["kind"] == "neg"],
                 key=lambda e: (e["wave"], e["pip"], e["mass"], e["seed"]))
    train = pos[0::3] + neg[0::3]
    floors = [floor_of(e) for e in train]
    thr_global = 3.0 * max(floors)
    fcoef, _ = fit_floor_model(train)

    fig = plt.figure(figsize=(7.16, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.15, 0.75], wspace=0.42)

    # (a) example episode: energy trace + thresholds
    ep = next(e for e in pos if e["wave"] == 0.0 and e["seed"] == 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(ep["t"], np.maximum(ep["e"], 1e-12), color=MUT, lw=0.8)
    ax.axvline(ep["truth"], color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.text(ep["truth"] - 0.30, 2e-2, "kinematic slip", rotation=90,
            fontsize=6, color=INK, va="top", ha="right")
    ax.axhline(thr_global, color=CRIT, lw=1.0)
    ax.text(0.3, thr_global * 1.6, "global 3×max floor (misses)",
            fontsize=6.5, color=CRIT)
    ts = scheduled_thr(fcoef, ep)
    ax.axhline(ts, color=BLUE, lw=1.0)
    ax.text(0.3, ts * 1.8, "scheduled threshold", fontsize=6.5, color=BLUE)
    above = np.nonzero(ep["e"] > ts)[0]
    if len(above):
        ax.plot(ep["t"][above[0]], ep["e"][above[0]], marker="v", ms=5,
                color=BLUE, zorder=5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("band energy (N²)")
    ax.set_ylim(1e-9, 1e-1)
    ax.set_title("(a) slip episode, 100 Hz sensor", loc="left")

    # (b) floor spread across conditions
    ax = fig.add_subplot(gs[0, 1])
    conds = sorted({(e["wave"], e["pip"], e["mass"]) for e in E})
    xf, yf, ys = [], [], []
    for i, c in enumerate(conds):
        fl = [floor_of(e) for e in train if e["cond"] == c]
        if fl:
            xf.append(i); yf.append(3 * max(fl))
            ys.append(scheduled_thr(
                fcoef, next(e for e in train if e["cond"] == c)))
    ax.semilogy(xf, yf, "o", ms=3.4, color=MUT)
    ax.semilogy(xf, ys, "_", ms=7, color=BLUE, mew=1.4)
    ax.axhline(thr_global, color=CRIT, lw=1.0)
    ax.text(max(xf), thr_global * 0.5, "global 3×max floor", fontsize=6.5,
            color=CRIT, ha="right", va="top")
    ax.text(0.2, 7.5e-4, "3× measured floor", fontsize=6, color=SEC)
    ax.text(0.2, 4.2e-4, "scheduled model", fontsize=6, color=BLUE)
    ax.set_xlabel("condition (posture × mass)")
    ax.set_ylabel("threshold (N²)")
    ax.set_title("(b) 72× floor spread", loc="left")

    # (c) test errors per scheme
    ax = fig.add_subplot(gs[0, 2])
    schemes = [("global", 30), ("per-cond.", 2), ("scheduled", 1),
               ("adaptive", 0)]
    y = np.arange(len(schemes))[::-1]
    ax.barh(y, [s[1] for s in schemes], height=0.6, color=BLUE, zorder=3)
    for yi, (n, v) in zip(y, schemes):
        ax.text(v + 0.5, yi, str(v), va="center", fontsize=7, color=SEC)
    ax.set_yticks(y, [s[0] for s in schemes], fontsize=7)
    ax.set_xlabel("test errors / 60")
    ax.set_xlim(0, 33)
    ax.grid(True, axis="x"); ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    ax.set_title("(c) errors by scheme", loc="left")
    save(fig, "F6_slip")


FIGS = {"F1": fig1, "F2": fig2, "F3": fig3, "F4": fig4, "F5": fig5,
        "F6": fig6}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None, choices=tuple(FIGS))
    args = ap.parse_args()
    for name, fn in FIGS.items():
        if args.only and name != args.only:
            continue
        fn()


if __name__ == "__main__":
    main()
