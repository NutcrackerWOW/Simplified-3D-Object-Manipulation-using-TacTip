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
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 0.42, 0.95, 1.1],
                          wspace=0.38)

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

    # (b) what the two posture parameters look like (small FK icons)
    from pinchlab.model import set_posture as _setp

    def mini_hand(ax, posture, plane, box_rot_deg=0.0):
        _setp(kin.plant, kin.plant_ctx, posture)
        cols = {"xz": (0, 2), "yz": (1, 2)}[plane]
        for s in "LR":
            p = np.array([kin.plant.EvalBodyPoseInWorld(
                kin.plant_ctx, kin.plant.GetBodyByName(n.format(s=s))
            ).translation() for n in chain]) * 1e3
            xz = p[:, cols]
            ax.plot(xz[:, 0], xz[:, 1], "-", color=SEC, lw=1.5,
                    solid_capstyle="round", zorder=2)
            ax.add_patch(plt.Circle((xz[-1, 0], xz[-1, 1]), r_dome,
                                    facecolor="#eeede9", edgecolor=MUT,
                                    lw=0.6, zorder=3))
        gg = kin.gap(posture)
        m = (gg.mid * 1e3)[list(cols)]
        hw, hh = (12, 6) if plane == "xz" else (6, 6)
        th = np.deg2rad(box_rot_deg)
        c, s_ = np.cos(th), np.sin(th)
        Rm = np.array([[c, -s_], [s_, c]])
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        pts_b = corners @ Rm.T + m
        ax.fill(pts_b[:, 0], pts_b[:, 1], facecolor=SEQ[1],
                edgecolor=BLUE, lw=0.7, zorder=4)
        ax.set_aspect("equal"); ax.axis("off")

    gsb = gs[0, 1].subgridspec(2, 1, hspace=0.42)
    axp = fig.add_subplot(gsb[0])
    p_phi, _ = kin.solve_mcp_for_gap(deg(15.0), deg(0.0), 0.024)
    mini_hand(axp, p_phi, plane="yz", box_rot_deg=15.0)
    axp.set_xlim(-34, 34); axp.set_ylim(135, 215)
    axp.set_title("(b) posture params", loc="left")
    axp.text(0.5, -0.04, "grasp tilt φ = +15° (side)",
             transform=axp.transAxes, fontsize=6, color=INK,
             ha="center", va="top")
    axt = fig.add_subplot(gsb[1])
    p_th, _ = kin.solve_mcp_for_gap(deg(0.0), deg(8.0), 0.024)
    mini_hand(axt, p_th, plane="xz")
    axt.set_xlim(-14, 76); axt.set_ylim(135, 215)
    axt.text(0.5, -0.04, "PIP flexion θ = +8° (front)",
             transform=axt.transAxes, fontsize=6, color=INK,
             ha="center", va="top")
    # restore the reference posture for any later FK users
    _setp(kin.plant, kin.plant_ctx, posture)

    # (c) contact decomposition frame
    ax = fig.add_subplot(gs[0, 2])
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
    ax.text(0, -14.5, "ρ = ‖(f₂, f₃)‖ / f₁\nhold ⇔ ρ < μ_eff",
            fontsize=7, color=INK, ha="center", va="top")
    ax.set_xlim(-30, 36); ax.set_ylim(-24, 20)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(c) grasp frame & utilization", loc="left")

    # (d) thin grasp manifold: pad gap vs MCP flexion
    ax = fig.add_subplot(gs[0, 3])
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
    ax.set_title("(d) thin grasp manifold (φ = 0, θ = 0)", loc="left")
    save(fig, "F1_setup")


# ---------------------------------------------------------------- F2 hero
def fig2():
    d = np.load(os.path.join(RESULTS, "fig_rolling_data.npz"))
    t, ph = d["t"], d["phase"]
    box_p, box_R = d["box_p"], d["box_R"]
    rot = np.rad2deg(d["rot_drift"]); drift = 1e3 * d["drift"]
    i0 = int(np.nonzero(ph == 3)[0][0])          # hold start
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
    i_fail = i0 + int(np.nonzero(drift[i0:] > 2.0)[0][0])
    rot_budget = rot[i_fail]                     # rotation at roll-off

    # (bottom left) both failure coordinates, each normalized by its own
    # criterion: rotation spends its budget while drift idles.
    axr = fig.add_subplot(gs[1, 0:10])
    axr.axhline(1.0, color=CRIT, lw=0.8, ls="--", zorder=1)
    axr.plot(th, rot[i0:] / rot_budget, color=BLUE, lw=1.4, zorder=3)
    axr.plot(th, drift[i0:] / 2.0, color=AQUA, lw=1.4, zorder=3)
    axr.text(0.12, 0.50, f"rotation / {rot_budget:.0f}°\n(value at roll-off)",
             color=BLUE, fontsize=6.5)
    axr.text(2.15, 0.16, "drift / 2 mm (criterion)", color="#0f7a54",
             fontsize=6.5)
    axr.text(th[-1] - 0.04, 1.05, "failure criterion", color=CRIT,
             fontsize=6, ha="right", va="bottom")
    axr.set_xlabel("time in hold (s)")
    axr.set_ylabel("fraction of own\nfailure criterion")
    axr.set_ylim(0, 1.55); axr.grid(True, axis="y")
    axr.set_title("(f) failure coordinates, normalized", loc="left",
                  fontsize=7.5)

    # (bottom right) the contact interface itself: relative tangential
    # speed at each contact — stick through the creep, breaking only at
    # roll-off, while the patch centroid migrates (rolling).
    axd = fig.add_subplot(gs[1, 12:21])
    sl = np.maximum(1e3 * d["slide_L"][i0:], 1e-4)
    sr = np.maximum(1e3 * d["slide_R"][i0:], 1e-4)
    axd.semilogy(th, sl, color=BLUE, lw=1.0, zorder=3)
    axd.semilogy(th, sr, color=AQUA, lw=1.0, zorder=3)
    axd.axhline(5.0, color=CRIT, lw=0.8, ls="--", zorder=2)
    axd.text(0.12, 14.0, "5 mm/s sliding criterion", color=CRIT,
             fontsize=6.5)
    axd.text(1.15, 2.3e-3, "L contact", color=BLUE, fontsize=6.5)
    axd.text(2.45, 0.90, "R contact", color="#0f7a54", fontsize=6.5)
    axd.set_xlabel("time in hold (s)")
    axd.set_ylabel("interfacial slip (mm/s)")
    axd.set_ylim(1e-4, 4e2); axd.grid(True, axis="y")
    axd.set_title("(g) the interface sticks while the box rotates",
                  loc="left", fontsize=7.5)
    for ax in (axr, axd):
        for i in idxs:
            ax.axvline(t[i] - t[i0], color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(0, th[-1])
    # snapshot letters above both panels ("c-e" cluster shares one)
    for ax, ytop in ((axr, 1.55), (axd, 4e2)):
        for i, lab in list(zip(idxs, labels))[:2]:
            ax.annotate(lab, xy=(t[i] - t[i0], ytop), xytext=(0, 2),
                        textcoords="offset points", color=MUT, fontsize=6.5,
                        ha="center", va="bottom", annotation_clip=False)
        ax.annotate("c–e", xy=(t[idxs[3]] - t[i0], ytop), xytext=(0, 2),
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
    for yi, v, c, pm in zip(y, vals, cols, pair):
        if v is None:
            b.axhspan(yi - 0.31, yi + 0.31, color="#f6dcdc", zorder=2)
            b.text(0.50, yi, "no stable window (roll-out floor above "
                   "ejection ceiling)", va="center", fontsize=6,
                   color=CRIT, zorder=3)
        else:
            b.barh(yi, v, height=0.62, color=c, zorder=3)
            b.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=6.5,
                   color=SEC)
            # rolling discount, inside the bar (friction rows only)
            if pm:
                b.text(v - 0.02, yi, f"{v/pm:.2f}× pair", va="center",
                       ha="right", fontsize=5.8, color="white", zorder=4)
    for yi, pm in zip(y, pair):
        if pm:
            b.plot(pm, yi, marker="D", ms=3.5, color=INK, zorder=4)
    # direct label on the μd 0.85 row's diamond (clear space above it)
    b.annotate("Coulomb pair μd", xy=(0.85, y[1] + 0.10),
               xytext=(0.88, y[1] + 0.62), fontsize=6.5, color=SEC,
               va="center",
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
    gs = fig.add_gridspec(1, 7,
                          width_ratios=[1, 1, 0.05, 0.24, 1.05, 0.24, 1.15],
                          wspace=0.35)
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
        # cells with no stable boundary at 2-seed resolution: fill + mark
        for i in range(len(pips)):
            for j in range(len(waves)):
                if np.isnan(Z[i, j]):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                               facecolor="#efeee9", lw=0))
                    ax.text(j + 0.5, i + 0.5, "–", ha="center",
                            va="center", fontsize=7, color=MUT)
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
                     "(seed-bimodal) — excluded from the fit;  "
                     "“–”: no stable force found (seed-limited)",
                     transform=fig.axes[0].transAxes, fontsize=6.5,
                     color=SEC)

    # (c) flatness: μ_eff vs PIP flexion, one line per mass, bootstrap CI
    # of the across-tilt mean — the plot behind "flat in posture".
    ax = fig.add_subplot(gs[0, 4])
    rng = np.random.default_rng(7)
    for m, col, fill in ((0.03, SEQ[5], SEQ[1]), (0.06, SEQ[3], SEQ[0])):
        sub = conds[(conds["mass"] == m) & (conds.regime == "friction")
                    & (conds.pip_deg >= 0.0) & np.isfinite(conds.mu_eff)]
        th = sorted(sub.pip_deg.unique())
        mean, lo, hi = [], [], []
        for p in th:
            v = sub[sub.pip_deg == p].mu_eff.to_numpy()
            bs = np.array([rng.choice(v, len(v)).mean()
                           for _ in range(2000)])
            mean.append(v.mean())
            lo.append(np.percentile(bs, 2.5))
            hi.append(np.percentile(bs, 97.5))
        ax.fill_between(th, lo, hi, color=fill, alpha=0.75, lw=0, zorder=2)
        ax.plot(th, mean, "-o", ms=2.6, lw=1.1, color=col, zorder=3)
        dy = 0.035 if m == 0.03 else -0.045
        ax.text(th[-1] + 0.25, mean[-1] + dy, f"{int(m*1000)} g",
                color=col, fontsize=6.5, va="center")
    ax.set_xlabel("PIP flexion θ (°)"); ax.set_ylabel("μ_eff")
    ax.set_ylim(0.45, 1.0); ax.set_xlim(-0.5, 11.8)
    ax.grid(True, axis="y")
    ax.set_title("(c) flat in posture (95% CI)", loc="left")

    # (d) force window vs mass at reference posture
    ax = fig.add_subplot(gs[0, 6])
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
    # measured masses as points; the smooth curves are the fitted rule,
    # dashed to mark them as interpolation between three measured masses
    ax.plot(mm, fpred, color=BLUE, lw=1.1, ls=(0, (4, 2)))
    ax.plot(mm, 2.0 * fpred, color=AQUA, lw=1.1, ls=(0, (4, 2)))
    ax.text(64, 0.28, "boundary f* (fit)", color=BLUE, fontsize=6.5,
            rotation=14)
    ax.text(30, 0.72, "2.0× f* (creep-safe)", color="#0f7a54",
            fontsize=6.5, rotation=30)
    ax.text(58, 2.05, "stable window", color="#0f7a54", fontsize=6.5,
            style="italic")
    ax.plot(masses, fstar, "o", ms=4.5, color=BLUE, zorder=4)
    ax.text(23, 0.34, "measured", color=BLUE, fontsize=6)
    ax.plot([120], [1.5], marker="x", ms=5, color=CRIT, mew=1.4, zorder=4)
    ax.text(128, 1.28, "120 g:\nwindow closed\n(measured)", color=CRIT,
            fontsize=6, ha="right", va="top")
    ax.set_xlim(20, 130); ax.set_ylim(0, 3.2)
    ax.set_xlabel("object mass (g)"); ax.set_ylabel("squeeze force f1 (N)")
    ax.set_title("(d) force window, reference posture", loc="left")
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
    a.set_xlabel("failure time from trial start (s), margin 1.3")
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
    b.yaxis.tick_right()
    b.set_yticks(range(len(dts)), [f"Δt = {d:g} ms" for d in dts],
                 fontsize=7)
    b.set_xlabel("failure time (s)")
    b.set_ylim(-0.5, 2.5); b.set_xlim(4, 12)
    b.grid(True, axis="x"); b.tick_params(left=False, right=False)
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

    # (b) floor spread across conditions, sorted by floor height so the
    # 72x spread reads as one monotone sweep
    ax = fig.add_subplot(gs[0, 1])
    conds = sorted({(e["wave"], e["pip"], e["mass"]) for e in E})
    rows = []
    for c in conds:
        fl = [floor_of(e) for e in train if e["cond"] == c]
        if fl:
            rows.append((3 * max(fl), scheduled_thr(
                fcoef, next(e for e in train if e["cond"] == c))))
    rows.sort(key=lambda r: r[0])
    xf = np.arange(len(rows))
    yf = [r[0] for r in rows]; ys = [r[1] for r in rows]
    ax.semilogy(xf, yf, "o", ms=3.4, color=MUT,
                label="3× measured floor")
    ax.semilogy(xf, ys, "_", ms=7, color=BLUE, mew=1.4,
                label="scheduled model")
    ax.axhline(thr_global, color=CRIT, lw=1.0)
    ax.text(xf[-1], thr_global * 0.5, "global 3×max floor", fontsize=6.5,
            color=CRIT, ha="right", va="top")
    ax.legend(loc="upper left", fontsize=6, handletextpad=0.4,
              borderaxespad=0.2)
    ax.set_xlabel("condition, sorted by floor")
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


# ----------------------------------------------------- F7 the rule itself
def fig7():
    """mu_eff vs mass: measured conditions with seed-noise error bars, the
    one-line rule with a bootstrap CI band, and the 16 held-out probe
    brackets the rule must (and does) thread."""
    from pinchlab.fit import mu_eff_from_boundary, classify_regimes
    conds = mu_eff_from_boundary(
        pd.read_csv(os.path.join(RESULTS, "sweepB_conditions.csv")))
    conds = classify_regimes(conds)
    conds.loc[(conds.regime == "friction") & (conds.pip_deg == -5.0),
              "regime"] = "fragile"
    fr = conds[(conds.regime == "friction") & np.isfinite(conds.mu_eff)
               & (conds.pip_deg >= 0.0)]
    sig = {0.03: 0.015, 0.06: 0.0894}

    with open(os.path.join(RESULTS,
                           "sweepB_equation_validation.json")) as fh:
        val = json.load(fh)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    rng = np.random.default_rng(3)

    # measured conditions (jittered), seed-noise error bars
    for m in (0.03, 0.06):
        v = fr[fr["mass"] == m].mu_eff.to_numpy()
        x = 1e3 * m + rng.uniform(-1.6, 1.6, len(v))
        ax.errorbar(x, v, yerr=sig[m] * v, fmt="o", ms=2.6, color=BLUE,
                    ecolor=SEQ[2], elinewidth=0.6, capsize=0, zorder=3)

    # bootstrap CI band of the mass-only WLS rule over the fitted rows
    mg = np.linspace(25, 95, 60)
    masses = fr["mass"].to_numpy()
    mus = fr.mu_eff.to_numpy()
    w = 1.0 / np.array([sig[m] for m in masses]) ** 2
    boots = np.empty((2000, len(mg)))
    for b in range(2000):
        idx = rng.integers(0, len(mus), len(mus))
        A = np.stack([np.ones(len(idx)), masses[idx] - 0.045], axis=1)
        W = w[idx]
        coef, *_ = np.linalg.lstsq((A.T * W).T, mus[idx] * W, rcond=None)
        boots[b] = coef[0] + coef[1] * (mg / 1e3 - 0.045)
    ax.fill_between(mg, np.percentile(boots, 2.5, axis=0),
                    np.percentile(boots, 97.5, axis=0), color=SEQ[1],
                    alpha=0.6, lw=0, zorder=2)
    line = 0.7069 - 4.1958 * (mg / 1e3 - 0.045)
    ins = mg <= 62
    ax.plot(mg[ins], line[ins], color=INK, lw=1.3, zorder=4)
    ax.plot(mg[~ins], line[~ins], color=INK, lw=1.1, ls=(0, (4, 2)),
            zorder=4)
    ax.text(23.5, 0.44, "μ_eff = 0.71 − 4.2 (m − 0.045)", fontsize=7,
            color=INK)
    ax.text(29, 0.90, "measured\nconditions", fontsize=6.2, color=BLUE,
            ha="center")
    ax.text(70, 0.72, "fit ± 95% CI\n(dashed: beyond\nfitted masses)",
            fontsize=6.2, color=SEC)

    # held-out probes: held at 1.25x prediction, dropped at 0.6x, so the
    # true boundary lies inside [mu_pred/1.25, mu_pred/0.6] — the rule
    # must thread every bracket (16/16 do; 90 g is out-of-envelope).
    probes = [(p["mass"], (p["mass"] * 9.81 / 2) / p["f_pred"], True)
              for p in val["in_envelope"]]
    probes += [(p["mass"], (p["mass"] * 9.81 / 2) / p["f_pred"], False)
               for p in val["out_of_envelope"]
               if p.get("ok") and p["mass"] <= 0.1]
    seen = {}
    for m, mu_pred, in_env in probes:
        x = 1e3 * m + seen.get(m, 0.0)
        seen[m] = seen.get(m, 0.0) + 1.1
        lo, hi = mu_pred / 1.25, mu_pred / 0.6
        col = AQUA if in_env else MUT
        ax.plot([x, x], [lo, hi], color=col, lw=1.0, alpha=0.85, zorder=2)
        for yy in (lo, hi):
            ax.plot([x - 0.55, x + 0.55], [yy, yy], color=col, lw=1.0,
                    alpha=0.85, zorder=2)
    ax.text(41, 1.32, "held-out brackets:\nheld at 1.25×, dropped at 0.6×",
            fontsize=6.2, color="#0f7a54")
    ax.text(84, 0.94, "90 g probe\n(beyond envelope)", fontsize=6,
            color=MUT, ha="center")

    ax.set_xlim(22, 97); ax.set_ylim(0.3, 1.55)
    ax.set_xlabel("object mass (g)"); ax.set_ylabel("μ_eff")
    ax.grid(True, axis="y")
    ax.set_title("the rule, its uncertainty, and its held-out test",
                 loc="left")
    save(fig, "F7_rule")


# ------------------------------------------- F8 solver sensitivity controls
def fig8():
    """Failure time of the margin-1.3 motionless hold vs every solver knob
    that could have manufactured the creep. Flat rows are the result."""
    with open(os.path.join(RESULTS, "creep_sensitivity.json")) as fh:
        cs = json.load(fh)["summary"]
    with open(os.path.join(RESULTS, "creep_timestep_control.json")) as fh:
        cc = json.load(fh)["summary"]
    T0 = 1.23           # hold start (s from trial start)
    WIN = 14.3 - T0     # observation window, in-hold

    rows = [
        ("Δt = 2 ms (baseline)", cs["base_sap"]["fail_times_s"], 0),
        ("Δt = 1 ms", cc["1ms"]["fail_times_s"], 0),
        ("Δt = 0.5 ms", cc["0.5ms"]["fail_times_s"], 0),
        ("τ = 0.01 s", cs["sap_tau0.01"]["fail_times_s"], 0),
        ("τ = 0.1 s (default)", cs["sap_tau0.1"]["fail_times_s"], 0),
        ("τ = 0.5 s", cs["sap_tau0.5"]["fail_times_s"],
         5 - cs["sap_tau0.5"]["n_failed"]),
        ("HC d = 0.5", cs["lagged_d0.5_vs1e-5"]["fail_times_s"], 0),
        ("HC d = 1.5", cs["lagged_vs1e-5"]["fail_times_s"], 0),
        ("HC d = 3.0", cs["lagged_d3.0_vs1e-5"]["fail_times_s"], 0),
        ("v_s = 10⁻³ m/s", cs["lagged_vs1e-3"]["fail_times_s"], 0),
        ("v_s = 10⁻⁴ m/s", cs["lagged_vs1e-4"]["fail_times_s"], 0),
        ("v_s = 10⁻⁵ m/s", cs["lagged_vs1e-5"]["fail_times_s"], 0),
    ]
    groups = [("time step (kSap)", 0, 3), ("dissipation (kSap τ)", 3, 6),
              ("dissipation (kLagged HC, v_s = 10⁻⁵)", 6, 9),
              ("friction regularization (kLagged)", 9, 12)]

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    base = np.array(cs["base_sap"]["fail_times_s"]) - T0
    ax.axvspan(base.min(), base.max(), color=SEQ[0], alpha=0.55, zorder=1)
    y = 0
    ylabels, ypos = [], []
    for label, times, n_held in rows:
        tt = np.array(times) - T0
        ax.plot(tt, np.full(len(tt), y), "o", ms=3.2, color=BLUE,
                zorder=3)
        if n_held:
            ax.plot([WIN + 0.18 * k for k in range(n_held)],
                    [y] * n_held, marker=">", ms=4,
                    color=MUT, ls="none", zorder=3)
        ylabels.append(label); ypos.append(y)
        y -= 1
    for gname, i0g, i1g in groups:
        ax.text(-0.3, -(i0g) + 0.52, gname, fontsize=6.2, color=SEC,
                style="italic", ha="left")
        if i1g < len(rows):
            ax.axhline(-(i1g) + 0.5, color=GRID, lw=0.6, zorder=0)
    ax.plot([], [], "o", ms=3.2, color=BLUE, label="failure")
    ax.plot([], [], ">", ms=4, color=MUT, ls="none",
            label="held past window")
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.3)
    ax.set_yticks(ypos, ylabels, fontsize=6.5)
    ax.set_xlabel("failure time in hold (s), margin 1.3")
    ax.set_xlim(-0.4, 13.6)
    ax.set_ylim(-(len(rows) - 1) - 0.7, 1.1)
    ax.grid(True, axis="x"); ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    ax.set_title("creep failure vs solver parameters "
                 "(shaded: baseline range)", loc="left", fontsize=7.5)
    save(fig, "F8_sensitivity")


# ------------------------------------------------- F9 contact patch gallery
def fig9():
    """The patches themselves: hydroelastic pressure field on the L-tip
    contact for each shape (ordered by mu_eff), and the box patch migrating
    over the dome during a failing margin-1.3 hold."""
    d = np.load(os.path.join(RESULTS, "fig_patches_data.npz"))
    shapes = [("disc", "disc (flat face)", 0.823),
              ("box", "box", 0.564),
              ("prism", "prism", 0.564),
              ("cylinder", "cylinder (rim)", 0.529),
              ("disc_edge", "disc on edge", 0.429),
              ("sphere", "sphere", 0.411)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)

    pmax = max(float(d[f"{s}_press"].max()) for s, _, _ in shapes)
    fig = plt.figure(figsize=(7.16, 3.1))
    gs = fig.add_gridspec(2, 7, width_ratios=[1] * 6 + [0.07],
                          hspace=0.52, wspace=0.15)

    def panel(ax, verts, press, tip_p, title, sub):
        yz = 1e3 * (verts[:, 1:3] - tip_p[1:3])     # tip frame, mm
        ax.scatter(yz[:, 0], yz[:, 1], c=press / 1e3, s=1.2, cmap=cmap,
                   vmin=0, vmax=pmax / 1e3, rasterized=True, lw=0)
        ax.set_aspect("equal")
        ax.set_xlim(-9, 9); ax.set_ylim(-10, 8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(title, fontsize=6.6, loc="left", pad=2)
        ax.text(0.03, 0.03, sub, transform=ax.transAxes, fontsize=6,
                color=SEC)

    for k, (s, name, mu) in enumerate(shapes):
        ax = fig.add_subplot(gs[0, k])
        panel(ax, d[f"{s}_verts"], d[f"{s}_press"], d[f"{s}_tip_p"],
              f"({'abcdef'[k]}) {name}", f"μ_eff = {mu:.2f}")
        if k == 0:
            ax.set_ylabel("stable grasps,\nby μ_eff →", fontsize=6.5,
                          color=SEC)

    sc = None
    for k in range(5):
        if f"fail{k}_verts" not in d:
            continue
        ax = fig.add_subplot(gs[1, k])
        verts, press = d[f"fail{k}_verts"], d[f"fail{k}_press"]
        tip_p = d[f"fail{k}_tip_p"]
        yz = 1e3 * (verts[:, 1:3] - tip_p[1:3])
        sc = ax.scatter(yz[:, 0], yz[:, 1], c=press / 1e3, s=1.2,
                        cmap=cmap, vmin=0, vmax=pmax / 1e3,
                        rasterized=True, lw=0)
        c = 1e3 * (d[f"fail{k}_centroid"][1:3] - tip_p[1:3])
        if k == 0:
            c0z = c[1]
        ax.axhline(c0z, color=MUT, lw=0.5, ls=(0, (3, 2)), zorder=2)
        ax.plot(c[0], c[1], "+", ms=5, color=CRIT, mew=1.1, zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-9, 9); ax.set_ylim(-10, 8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(f"({'ghijk'[k]}) t = {float(d[f'fail{k}_t']):.1f} s",
                     fontsize=6.6, loc="left", pad=2)
        if k == 0:
            ax.set_ylabel("failing hold:\npatch migrates ↓", fontsize=6.5,
                          color=SEC)
    axl = fig.add_subplot(gs[1, 5])
    axl.axis("off")
    axl.text(0.05, 0.5, "box, margin 1.3\n(+ = patch centroid;\n"
             "the resultant &\ncentroid of these\nfields are the\n"
             "idealized TacTip\nsignal of §III)", fontsize=6, color=SEC,
             va="center")
    cax = fig.add_subplot(gs[:, 6])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("hydroelastic pressure (kPa)", color=SEC)
    cb.outline.set_visible(False)
    cax.tick_params(length=0)
    save(fig, "F9_patches")


FIGS = {"F1": fig1, "F2": fig2, "F3": fig3, "F4": fig4, "F5": fig5,
        "F6": fig6, "F7": fig7, "F8": fig8, "F9": fig9}


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
