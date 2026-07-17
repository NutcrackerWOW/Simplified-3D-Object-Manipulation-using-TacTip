"""Central parameter definitions for the pinch study.

All angles are radians and all lengths meters unless a name says otherwise.
Defaults implement the approved plan ("Concrete parameter defaults").
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

G = 9.81  # m/s^2

FLEX_JOINTS = ("MCP", "PIP", "DIP")
FINGERS = ("L", "R")
TIP_BODY = {"L": "L_Tip_1", "R": "R_Tip_1"}


@dataclass
class BoxSpec:
    """The manipulated object: a rigid-hydroelastic block.

    `size` is the grasp width (x) and depth (y). `height` (z) is shorter
    than the plan's original cube: a 25 mm cube's upper half jams against
    the Middle/Distal links (measured ±4 N — the links grip instead of the
    TacTip pads), which would bypass the tactile sensors entirely.
    """
    size: float = 0.025          # grasp width / depth (diameter for round shapes)
    height: float = 0.012        # vertical extent — keeps contact on the pads
    mass: float = 0.050          # kg
    mu_static: float = 1.0
    mu_dynamic: float = 0.8
    resolution_hint: float = 0.005
    shape: str = "box"           # box | prism | cylinder (vertical axis) |
                                 # disc (flat faces) | disc_edge (key pinch) | sphere


@dataclass
class ModelParams:
    time_step: float = 2e-3
    # TacTip pad material (existing proven constants from pinch_sim.py)
    hydro_modulus: float = 5e4     # Pa
    dissipation: float = 1.5       # s/m
    mu_static_tip: float = 1.2
    mu_dynamic_tip: float = 0.9
    mesh_resolution_hint: float = 0.003
    rotor_inertia: float = 1e-4    # kg·m² reflected; regularizes placeholder inertias
    # Solver-sensitivity knobs (creep sensitivity study). Baseline = kSap with
    # the plant's default stiction tolerance. Under kSap the friction
    # regularization sigma is hard-coded in Drake (1e-3, dimensionless);
    # kLagged/kSimilar regularize friction with the plant stiction tolerance,
    # which is what stiction_tolerance controls here.
    contact_approximation: str = "sap"       # sap | lagged | similar
    stiction_tolerance: float | None = None  # m/s; None = plant default
    # Tip relaxation_time (s): the dissipation knob kSap actually consumes
    # (linear Kelvin-Voigt; Drake default 0.1 s per geometry when unset).
    # hunt_crossley_dissipation above is consumed by kLagged/kSimilar/TAMSI.
    relaxation_time: float | None = None     # s; None = Drake default
    # Joint limits (deg) — plan decision #1
    flex_lower_deg: float = -5.0
    flex_upper_deg: float = 90.0
    wave_limit_deg: float = 15.0
    effort_limit: float = 10.0     # N·m
    velocity_limit: float = 20.0   # rad/s
    base_height: float = 0.25      # world z of the base weld (fingers point down)


@dataclass
class Posture:
    """Per-finger joint targets.

    Geometry note (measured via FK): equal L/R wave values SPLAY the tips
    apart in ±y (skewed pinch, feasible only ~±4°). Opposite values tilt
    both fingers the same world direction about the pinch axis — opposition
    is preserved across the full ±15°, and gravity's direction on the pads
    rotates with the tilt. `symmetric()` therefore uses TILT semantics:
    wave_l=+wave, wave_r=−wave.
    """
    wave_l: float = 0.0
    mcp_l: float = 0.02
    pip_l: float = 0.0
    wave_r: float = 0.0
    mcp_r: float = 0.02
    pip_r: float = 0.0

    @classmethod
    def symmetric(cls, wave: float, mcp: float, pip: float) -> "Posture":
        return cls(wave, mcp, pip, -wave, mcp, pip)

    def joint_map(self) -> dict:
        return {
            "L_Wave": self.wave_l, "L_MCP": self.mcp_l, "L_PIP": self.pip_l,
            "R_Wave": self.wave_r, "R_MCP": self.mcp_r, "R_PIP": self.pip_r,
        }

    def as_tuple(self):
        return (self.wave_l, self.mcp_l, self.pip_l,
                self.wave_r, self.mcp_r, self.pip_r)


@dataclass
class ControlParams:
    # Posture PD (per joint, N·m/rad). Gravity feedforward carries the
    # finger weight (without it, gravity sag ≈ load/kp eats the few-degree
    # contact window at tilted postures — measured failure mode).
    gravity_comp: bool = True
    kp_flex: float = 0.30
    kd_flex: float = 0.03
    kp_wave: float = 1.00
    kd_wave: float = 0.08
    # MCP gains while the force loop is active (weak, so the trim dominates)
    kp_mcp_force_mode: float = 0.02
    kd_mcp_force_mode: float = 0.010
    # DIP passive spring (plan decision #13: ~5° give at 1–2 N tip force)
    dip_spring_k: float = 0.35     # N·m/rad toward measured PIP angle
    dip_spring_c: float = 0.015    # N·m·s/rad
    # Outer force loop (integral trim on MCP torque)
    force_ki: float = 0.4          # N·m per (N·s)
    trim_min: float = -0.05        # N·m
    trim_max: float = 0.80         # N·m
    moment_arm: float = 0.08       # m, MCP→tip lever for feedforward
    force_filter_hz: float = 20.0  # low-pass on sensed force before the loop
    contact_force_on: float = 0.05  # N, both tips above this = contact made
    # State machine timing
    close_rate: float = 0.8        # rad/s MCP approach ramp
    grip_band: float = 0.10        # ±10% of setpoint counts as "at setpoint"
    grip_settle_time: float = 0.3  # s within band before release
    grip_timeout: float = 2.5      # s, proceed to release anyway
    release_time: float = 0.5      # s anti-gravity ramp-down
    hold_time: float = 3.0         # s post-release observation window


@dataclass
class TrialSpec:
    """One spawn→close→grip→release→hold episode."""
    posture: Posture = field(default_factory=Posture)
    box: BoxSpec = field(default_factory=BoxSpec)
    force_setpoint: float = 0.6     # N per-finger squeeze target
    time_step: float = 2e-3
    log_hz: float = 200.0
    seed: int = 0
    spawn_jitter: float = 0.0005    # m, ±uniform xyz jitter on box spawn
    open_gap_extra: float = 0.004   # m, tip gap beyond box size at spawn
    # Spawn below the dome-to-dome witness midpoint: at some postures the
    # witness sits high on the domes and a centered box jams its top corners
    # into the Distal links (measured). 3 mm keeps the top edge clear while
    # the pads still land on the faces.
    spawn_z_offset: float = -0.003  # m
    # Slip labels
    drift_threshold: float = 0.002  # m in tip-midpoint frame → slipped
    drop_margin: float = 0.15       # m below tips → dropped (early terminate)
    slide_speed: float = 0.005      # m/s sustained → incipient slip truth
    slide_sustain: float = 0.05     # s
    # Induced-slip protocol: setpoint decreases at this rate during HOLD
    # (0 = constant setpoint). Used by the slip-detector study.
    setpoint_ramp: float = 0.0      # N/s
    tag: str = ""

    def flat_row(self) -> dict:
        row = {
            "wave_l": self.posture.wave_l, "mcp_l": self.posture.mcp_l,
            "pip_l": self.posture.pip_l, "wave_r": self.posture.wave_r,
            "mcp_r": self.posture.mcp_r, "pip_r": self.posture.pip_r,
            "box_size": self.box.size, "box_mass": self.box.mass,
            "box_mu_s": self.box.mu_static, "force_setpoint": self.force_setpoint,
            "setpoint_ramp": self.setpoint_ramp,
            "time_step": self.time_step, "seed": self.seed, "tag": self.tag,
        }
        return row


def deg(x: float) -> float:
    return float(np.deg2rad(x))
