"""Figures for trials, the stability map, boundary fits, and slip detection.

Static matplotlib PNGs for the paper pipeline. Colors follow the validated
reference palette (dataviz skill): categorical slot 1 (blue) = L finger,
slot 2 (aqua) = R finger; sequential single-hue blues for magnitude maps;
red reserved for slip/threshold marks. One measure per axis — different
units get their own stacked subplot, never a dual axis.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Reference palette (light mode)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
C_L = "#2a78d6"        # finger L  (categorical slot 1, blue)
C_R = "#1baf7a"        # finger R  (categorical slot 2, aqua)
C_REF = "#52514e"      # reference/setpoint lines
C_SLIP = "#e34948"     # slip / threshold marks (reserved)
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b"])
PHASE_TINTS = {0: "#eceff4", 1: "#fdf3e0", 2: "#fde7e2", 3: "#e9f3ea", 4: "#f4e0e0"}
PHASE_NAMES = {0: "close", 1: "grip", 2: "release", 3: "hold", 4: "failed"}


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXT_2)


def _shade_phases(ax, t, phase):
    change = np.nonzero(np.diff(phase) != 0)[0]
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(t) - 1]])
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=PHASE_TINTS.get(int(phase[s]), "#ffffff"),
                   alpha=0.55, lw=0, zorder=0)


def plot_trial(series: dict, title: str, out_png: str,
               setpoint: float | None = None, mu_pair: float = 1.09) -> str:
    t = np.asarray(series["t"])
    phase = np.asarray(series["phase"])
    fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for ax in axes:
        _shade_phases(ax, t, phase)

    ax = axes[0]
    ax.plot(t, series["f1_L"], color=C_L, lw=1.8, label="L")
    ax.plot(t, series["f1_R"], color=C_R, lw=1.8, label="R")
    if setpoint is not None:
        ax.axhline(setpoint, color=C_REF, lw=1.2, ls="--")
        ax.annotate(f"setpoint {setpoint:g} N", (t[-1], setpoint),
                    ha="right", va="bottom", fontsize=8, color=TEXT_2)
    _style(ax, "squeeze f1 (N)")
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=2)
    ax.set_title(title, fontsize=10, color=TEXT)

    ax = axes[1]
    ax.plot(t, series["f2_L"], color=C_L, lw=1.8, label="f2 L")
    ax.plot(t, series["f2_R"], color=C_R, lw=1.8, label="f2 R")
    ax.plot(t, series["f3_L"], color=C_L, lw=1.2, ls=":", label="f3 L")
    ax.plot(t, series["f3_R"], color=C_R, lw=1.2, ls=":", label="f3 R")
    _style(ax, "shear (N)")
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=4)

    ax = axes[2]
    ax.plot(t, np.clip(series["rho_L"], 0, 3), color=C_L, lw=1.8, label="L")
    ax.plot(t, np.clip(series["rho_R"], 0, 3), color=C_R, lw=1.8, label="R")
    ax.axhline(mu_pair, color=C_SLIP, lw=1.2, ls="--")
    ax.annotate(f"Coulomb pair μ = {mu_pair:g}", (t[-1], mu_pair),
                ha="right", va="bottom", fontsize=8, color=C_SLIP)
    _style(ax, "utilization ρ")
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=2)

    ax = axes[3]
    ax.plot(t, 1e3 * np.asarray(series["drift"]), color=C_L, lw=1.8)
    _style(ax, "drift (mm)")

    ax = axes[4]
    ax.plot(t, np.rad2deg(np.asarray(series["rot_drift"])), color=C_L, lw=1.8)
    _style(ax, "rotation (°)")

    ax = axes[5]
    ax.plot(t, 1e3 * np.asarray(series["slide_L"]), color=C_L, lw=1.8, label="L")
    ax.plot(t, 1e3 * np.asarray(series["slide_R"]), color=C_R, lw=1.8, label="R")
    _style(ax, "sliding (mm/s)")
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=2)
    ax.set_xlabel("time (s)", fontsize=9, color=TEXT)

    fig.align_ylabels(axes)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def _grid_pivot(df, value):
    waves = np.sort(df["wave_deg"].unique())
    pips = np.sort(df["pip_deg"].unique())
    M = np.full((len(pips), len(waves)), np.nan)
    for _, r in df.iterrows():
        i = np.searchsorted(pips, r["pip_deg"])
        j = np.searchsorted(waves, r["wave_deg"])
        M[i, j] = r[value]
    return waves, pips, M


def plot_map(conds_df, out_png: str, value: str = "mu_eff",
             label: str = "μ_eff") -> str:
    masses = np.sort(conds_df["mass"].unique())
    fig, axes = plt.subplots(1, len(masses), figsize=(4.3 * len(masses), 3.8),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    finite = conds_df[value].replace([np.inf, -np.inf], np.nan).dropna()
    vmin, vmax = (finite.min(), finite.max()) if len(finite) else (0, 1)
    for ax, m in zip(axes[0], masses):
        d = conds_df[conds_df["mass"] == m]
        waves, pips, M = _grid_pivot(d, value)
        pc = ax.pcolormesh(waves, pips, M, cmap=SEQ, vmin=vmin, vmax=vmax,
                           shading="nearest", edgecolors=SURFACE, linewidth=1.5)
        for i, p in enumerate(pips):
            for j, w in enumerate(waves):
                if np.isfinite(M[i, j]):
                    lum = (M[i, j] - vmin) / max(vmax - vmin, 1e-9)
                    ax.text(w, p, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="#ffffff" if lum > 0.55 else TEXT)
        ax.set_title(f"mass {m*1000:.0f} g", fontsize=10, color=TEXT)
        ax.set_xlabel("wave tilt (°)", fontsize=9, color=TEXT)
        ax.set_ylabel("PIP (°)", fontsize=9, color=TEXT)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        ax.set_facecolor("#efefec")   # NaN shows as neutral background
    cb = fig.colorbar(pc, ax=axes[0].tolist(), shrink=0.9, pad=0.02)
    cb.set_label(label, fontsize=9, color=TEXT)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_boundary(trials_df, cond_row, out_png: str) -> str:
    d = trials_df[trials_df["cond_id"] == cond_row["cond_id"]]
    d = d[d["outcome"] != "no_grasp"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    y = d["held"].astype(int) + rng.uniform(-0.04, 0.04, len(d))
    held = d["held"].astype(bool)
    ax.scatter(d["force_setpoint"][held], y[held], s=42, facecolor="none",
               edgecolor=C_L, lw=1.6, label="held", zorder=3)
    ax.scatter(d["force_setpoint"][~held], y[~held], s=42, marker="x",
               color=C_SLIP, lw=1.6, label="slipped/dropped", zorder=3)
    if np.isfinite(cond_row["f_star"]):
        ax.axvline(cond_row["f_star"], color=C_REF, lw=1.2, ls="--")
        ax.annotate(f" f* = {cond_row['f_star']:.2f} N",
                    (cond_row["f_star"], 0.5), fontsize=9, color=TEXT_2)
    ax.set_xscale("log")
    _style(ax, "held")
    ax.set_yticks([0, 1], ["slip", "hold"])
    ax.set_xlabel("squeeze setpoint (N)", fontsize=9, color=TEXT)
    ax.set_title(f"stability boundary — {cond_row['cond_id']}",
                 fontsize=10, color=TEXT)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def plot_slip_detection(t, x, energy, threshold, truth, detected,
                        title: str, out_png: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    ax.plot(t, x, color=C_L, lw=1.4)
    _style(ax, "tangential force ‖(f2,f3)‖ (N)")
    ax.set_title(title, fontsize=10, color=TEXT)
    ax = axes[1]
    ax.plot(t, energy, color=C_L, lw=1.4)
    ax.axhline(threshold, color=C_REF, ls="--", lw=1.2)
    ax.annotate(" threshold", (t[0], threshold), fontsize=8, color=TEXT_2,
                va="bottom")
    _style(ax, "band energy (N²)")
    ax.set_xlabel("time (s)", fontsize=9, color=TEXT)
    for ax in axes:
        if truth is not None and np.isfinite(truth):
            ax.axvline(truth, color=C_SLIP, lw=1.4)
        if detected is not None and np.isfinite(detected):
            ax.axvline(detected, color=C_SLIP, lw=1.4, ls=":")
    axes[0].annotate("slip truth —, detection ⋯", (0.99, 0.95),
                     xycoords="axes fraction", ha="right", fontsize=8,
                     color=C_SLIP)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_png
