#!/usr/bin/env python3
"""
Pinch simulation: Complete.urdf two-finger hand with silicone hydroelastic fingertips.

L_Tip_1 is compliant_hydroelastic (50 kPa, silicone-like modulus + dissipation).
R_Tip_1 is rigid_hydroelastic (high-friction silicone surface). The compliant-rigid
pairing is Drake's numerically stable hydroelastic contact model for mesh geometry.
A three-phase pinch motion is simulated:

  Phase 1: all flex joints (MCP, PIP, DIP × L/R) ramp to the same reference
           angle simultaneously — synchronized, no joint leads another
  Phase 2: hold reference past contact point (PD error = squeeze force)
  Phase 3: synchronized ramp back to open

L_Wave and R_Wave are PD-locked at 0° so L_Base_1 and R_Base_1 never rotate.

The full animation is recorded into MeshCat (open the printed URL in a browser
and use the scrubber control to replay). ContactVisualizer shows force arrows
and contact-patch overlays live during the simulation.

A matplotlib figure of contact force, contact area, and max pressure is saved
to `pinch_contact_results.png`.

Usage
-----
  python pinch_sim.py                 # 15-second simulation
  python pinch_sim.py --duration 25
"""

import argparse
import os
import re

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    DiagramBuilder,
    DiscreteContactApproximation,
    JointActuatorIndex,
    LeafSystem,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    Role,
    RotationMatrix,
    Simulator,
    StartMeshcat,
)
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model", "Touch_Finger", "Complete.urdf",
)
PACKAGE_ROOT = os.path.dirname(URDF_PATH)  # dir containing meshes/

# ---------------------------------------------------------------------------
# Silicone / human fingertip material constants
#   Human fingertip pad elastic modulus: ~50–100 kPa
#   Soft silicone rubber:                ~40–500 kPa depending on grade
#   High friction from silicone-on-silicone contact
# ---------------------------------------------------------------------------
HYDRO_MODULUS = 5e4   # Pa   – compliant modulus (50 kPa, soft silicone pad)
DISSIPATION   = 1.5   # s/m  – Hunt-Crossley dissipation (damped, lossy contact)
MU_STATIC     = 1.2   # –    – static friction  (silicone-on-silicone)
MU_DYNAMIC    = 0.9   # –    – dynamic friction

# Reflected (rotor) inertia added to every actuated joint.  The URDF's link
# inertias are unrealistically tiny (~1e-7 kg·m², auto-generated placeholders);
# without regularization they make the dynamics extremely stiff and cause the
# contact solver to diverge.  ~1e-4 kg·m² is physically the motor-rotor + gearbox
# inertia and well-conditions the actuated DOFs.
ROTOR_INERTIA = 1e-4  # kg·m²

# Logging / recording rates
LOG_HZ    = 200   # contact-data sampling rate
VIZ_HZ    = 64    # ContactVisualizer publish rate


# ---------------------------------------------------------------------------
# URDF patching (in-memory – original file is never modified)
# ---------------------------------------------------------------------------

# L_Tip_1: compliant — pressure field generated from its volumetric mesh.
# Compliant-rigid is Drake's canonical, numerically stable hydroelastic pair;
# compliant-compliant mesh contact can produce NaN from degenerate tet intersections.
_L_TIP_PROX = (
    "<drake:proximity_properties>\n"
    "        <drake:compliant_hydroelastic/>\n"
    f"        <drake:mesh_resolution_hint value=\"0.003\"/>\n"
    f"        <drake:hydroelastic_modulus value=\"{HYDRO_MODULUS:.3g}\"/>\n"
    f"        <drake:hunt_crossley_dissipation value=\"{DISSIPATION}\"/>\n"
    f"        <drake:mu_static value=\"{MU_STATIC}\"/>\n"
    f"        <drake:mu_dynamic value=\"{MU_DYNAMIC}\"/>\n"
    "      </drake:proximity_properties>"
)

# R_Tip_1: rigid — acts as the contact surface; silicone friction applied.
_R_TIP_PROX = (
    "<drake:proximity_properties>\n"
    "        <drake:rigid_hydroelastic/>\n"
    f"        <drake:mu_static value=\"{MU_STATIC}\"/>\n"
    f"        <drake:mu_dynamic value=\"{MU_DYNAMIC}\"/>\n"
    "      </drake:proximity_properties>"
)


def _patch_urdf(content: str) -> str:
    """
    1. Swap .stl → .obj
    2. L_Tip_1: update compliant block with silicone hydroelastic properties.
    3. R_Tip_1: keep rigid_hydroelastic, update friction to silicone values.
    """
    content = content.replace(".stl", ".obj")

    # L_Tip_1: replace existing compliant block
    content = re.sub(
        r"<drake:proximity_properties>\s*"
        r"<drake:compliant_hydroelastic/>\s*"
        r"(?:<drake:[^/]+/>\s*)*"
        r"</drake:proximity_properties>",
        "      " + _L_TIP_PROX,
        content,
    )

    # R_Tip_1: update rigid block friction (keep rigid)
    content = re.sub(
        r"<drake:proximity_properties>\s*"
        r"<drake:rigid_hydroelastic/>\s*"
        r"(?:<drake:[^/]+/>\s*)*"
        r"</drake:proximity_properties>",
        "      " + _R_TIP_PROX,
        content,
    )

    return content


def _load_urdf(parser: Parser) -> None:
    parser.package_map().Add("Complete_description", PACKAGE_ROOT)
    with open(URDF_PATH) as fh:
        content = _patch_urdf(fh.read())
    parser.AddModelsFromString(content, "urdf")


# ---------------------------------------------------------------------------
# Collision filter: Drake filters self-collisions by default; explicitly
# re-allow contact between the two opposing fingertips.
# ---------------------------------------------------------------------------

def _allow_tip_contact(plant, scene_graph) -> None:
    l_geoms = GeometrySet(
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("L_Tip_1"))
    )
    r_geoms = GeometrySet(
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("R_Tip_1"))
    )
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().AllowBetween(l_geoms, r_geoms)
    )


# ---------------------------------------------------------------------------
# Pinch controller
# ---------------------------------------------------------------------------

class PinchController(LeafSystem):
    """
    Synchronized trajectory-tracking controller.

    All six flex joints (L/R × MCP, PIP, DIP) track the SAME reference
    angle at every instant, so MCP, PIP, and DIP always rotate by equal
    amounts — no joint leads or lags another.

    Reference profile (same for every flex joint):
      Phase 1 [0, t1):    ramp from Q_OPEN → Q_CLOSE at constant rate
      Phase 2 [t1, t2):   hold at Q_CLOSE (squeeze — PD error = contact force)
      Phase 3 [t2, end):  ramp back from Q_CLOSE → Q_OPEN (release)

    PD-locked at 0° throughout:
      L_Wave, R_Wave  →  L_Base_1, R_Base_1 never rotate
    """

    # All three phalanges flex together
    FLEX_JOINTS = ["L_MCP", "L_PIP", "L_DIP", "R_MCP", "R_PIP", "R_DIP"]

    # Wave joints only — MCP is now free
    LOCKED_JOINTS = [
        ("L_Wave", "L_Wave_actr"),
        ("R_Wave", "R_Wave_actr"),
    ]

    Q_OPEN  = 0.0    # rad – initial pose = origin (all joints at rotation 0)
    Q_CLOSE = 1.5    # rad – target reference past contact (ensures squeeze force)

    # With reflected rotor inertia (~1e-4 kg·m²) and SAP's implicit actuation
    # these gains are well within the stable range, so they can be high enough
    # for the fingers to track the reference tightly (no sag, no lag).
    KP_FLEX = 0.10   # N·m / rad     – flex tracking stiffness
    KD_FLEX = 0.01   # N·m·s / rad   – flex tracking damping
    KP_LOCK = 0.50   # N·m / rad     – wave-lock stiffness
    KD_LOCK = 0.05   # N·m·s / rad   – wave-lock damping

    def __init__(self, plant, t1: float, t2: float):
        LeafSystem.__init__(self)
        self._plant = plant
        self._t1    = t1
        self._t2    = t2

        # Angular ramp rate: travel from Q_OPEN to Q_CLOSE in exactly t1 seconds
        self._rate = (self.Q_CLOSE - self.Q_OPEN) / t1

        # (position_idx, velocity_idx, actuator_idx) for each flex joint
        self._flex = []
        for jname in self.FLEX_JOINTS:
            j = plant.GetJointByName(jname)
            self._flex.append((
                j.position_start(),
                j.velocity_start(),
                int(plant.GetJointActuatorByName(f"{jname}_actr").index()),
            ))

        # Same tuple structure for locked joints
        self._locked = []
        for jname, aname in self.LOCKED_JOINTS:
            j = plant.GetJointByName(jname)
            self._locked.append((
                j.position_start(),
                j.velocity_start(),
                int(plant.GetJointActuatorByName(aname).index()),
            ))

        nq = plant.num_positions()
        nv = plant.num_velocities()
        self.DeclareVectorInputPort("state", nq + nv)
        self.DeclareVectorOutputPort(
            "actuation", plant.num_actuators(), self._calc
        )

    def _q_ref(self, t: float) -> float:
        """Shared reference angle for all flex joints."""
        if t < self._t1:
            return self.Q_OPEN + self._rate * t
        elif t < self._t2:
            return self.Q_CLOSE
        else:
            return max(self.Q_OPEN, self.Q_CLOSE - self._rate * (t - self._t2))

    def _calc(self, context, output):
        t     = context.get_time()
        state = self.get_input_port(0).Eval(context)

        # Guard: if plant state is corrupt, output zero rather than propagate NaN
        if not np.all(np.isfinite(state)):
            output.SetFromVector(np.zeros(self._plant.num_actuators()))
            return

        nq    = self._plant.num_positions()
        q, v  = state[:nq], state[nq:]

        u     = np.zeros(self._plant.num_actuators())
        q_ref = self._q_ref(t)

        # All flex joints track the identical reference — synchronized rotation
        for qi, vi, ui in self._flex:
            u[ui] = self.KP_FLEX * (q_ref - q[qi]) - self.KD_FLEX * v[vi]

        # Wave joints: PD hold at 0°
        for qi, vi, ui in self._locked:
            u[ui] = self.KP_LOCK * (0.0 - q[qi]) - self.KD_LOCK * v[vi]

        output.SetFromVector(u)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run(duration: float = 15.0) -> None:
    # --- MeshCat ---
    meshcat = StartMeshcat()
    print(f"\n{'='*60}")
    print(f"  MeshCat URL: {meshcat.web_url()}")
    print(f"{'='*60}")
    print("Open the URL in a browser. Recording will be published after")
    print("the simulation; use the MeshCat time scrubber to replay.\n")

    # --- Diagram ---
    # Discrete SAP solver (implicit) robustly handles stiff hydroelastic contact
    # and PD actuation.  Combined with reflected rotor inertia (below) it avoids
    # the step-size collapse / NaN blow-ups seen with TAMSI and continuous RK3.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=2e-3)
    _load_urdf(Parser(plant))

    # Weld base to world, rotated so fingers point forward
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("base_link"),
        RigidTransform(RotationMatrix.MakeYRotation(-np.pi / 2)),
    )

    # SAP discrete contact + hydroelastic-with-fallback model
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.set_contact_model(ContactModel.kHydroelasticWithFallback)

    # Reflected inertia regularizes the URDF's tiny link inertias
    for i in range(plant.num_actuators()):
        plant.get_joint_actuator(JointActuatorIndex(i)).set_default_rotor_inertia(
            ROTOR_INERTIA
        )

    plant.Finalize()
    _allow_tip_contact(plant, scene_graph)

    # Phase boundaries
    t1 = duration / 3.0
    t2 = 2.0 * duration / 3.0

    # Pinch controller
    ctrl = builder.AddSystem(PinchController(plant, t1=t1, t2=t2))
    builder.Connect(plant.get_state_output_port(),    ctrl.get_input_port(0))
    builder.Connect(ctrl.get_output_port(0), plant.get_actuation_input_port())

    # Mesh visualizer (illustration = visual geometry, not collision proxy)
    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(
            role=Role.kIllustration,
            delete_on_initialization_event=False,
        ),
    )

    # Contact visualizer – force arrows and contact-patch overlays in MeshCat
    cv_params = ContactVisualizerParams()
    cv_params.publish_period = 1.0 / VIZ_HZ
    cv_params.force_threshold = 0.001  # N – suppress numerical noise
    ContactVisualizer.AddToBuilder(builder, plant, meshcat, cv_params)

    diagram  = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.25)  # slow playback for viewing

    # --- Initial pose: fingers slightly open so closing motion is visible ---
    ctx0       = simulator.get_mutable_context()
    plant_ctx0 = plant.GetMyMutableContextFromRoot(ctx0)
    q0 = plant.GetPositions(plant_ctx0).copy()
    for jname in PinchController.FLEX_JOINTS:
        q0[plant.GetJointByName(jname).position_start()] = PinchController.Q_OPEN
    plant.SetPositions(plant_ctx0, q0)

    # Publish initial state to MeshCat before recording starts
    diagram.ForcedPublish(ctx0)

    # --- Start recording ---
    meshcat.StartRecording(set_visualizations_while_recording=True)
    simulator.Initialize()

    # --- Data logging ---
    t_log, force_log, area_log, pressure_log = [], [], [], []
    record_dt = 1.0 / LOG_HZ

    rate_deg = np.rad2deg((PinchController.Q_CLOSE - PinchController.Q_OPEN) / t1)
    print(f"Phase 1 [0 – {t1:.1f}s]:       ramp close  ({rate_deg:.1f}°/s, all joints synchronized)")
    print(f"Phase 2 [{t1:.1f} – {t2:.1f}s]:  hold squeeze (q_ref = {np.rad2deg(PinchController.Q_CLOSE):.0f}°)")
    print(f"Phase 3 [{t2:.1f} – {duration:.1f}s]:  ramp open   ({rate_deg:.1f}°/s)")
    locked = ", ".join(j for j, _ in PinchController.LOCKED_JOINTS)
    print(f"PD-locked at 0°: {locked}")
    print(f"Flex joints (synchronized): {', '.join(PinchController.FLEX_JOINTS)}")
    print("\nSimulating… (Ctrl+C to stop early)\n")

    try:
        t = 0.0
        while t < duration:
            t = min(t + record_dt, duration)
            simulator.AdvanceTo(t)

            plant_ctx = plant.GetMyContextFromRoot(simulator.get_context())
            cr = plant.get_contact_results_output_port().Eval(plant_ctx)

            total_force   = 0.0
            total_area    = 0.0
            max_pressure  = 0.0

            for i in range(cr.num_hydroelastic_contacts()):
                info = cr.hydroelastic_contact_info(i)
                total_force += np.linalg.norm(info.F_Ac_W().translational())

                surf = info.contact_surface()
                # Area – Drake versions differ in attribute name
                for attr in ("total_area", "area"):
                    if hasattr(surf, attr):
                        total_area += getattr(surf, attr)()
                        break
                # Peak pressure – not available in all Drake builds
                for attr in ("maximum_pressure", "max_pressure", "maximum_pressure_magnitude"):
                    if hasattr(surf, attr):
                        max_pressure = max(max_pressure, getattr(surf, attr)())
                        break

            t_log.append(t)
            force_log.append(total_force)
            area_log.append(total_area)
            pressure_log.append(max_pressure)

    except KeyboardInterrupt:
        print("\nStopped early by user.")

    # --- Publish recording ---
    meshcat.StopRecording()
    meshcat.PublishRecording()
    print("\nRecording published — use the MeshCat time scrubber to replay.\n")

    # --- Plot ---
    _plot(t_log, force_log, area_log, pressure_log, t1, t2, duration)


# ---------------------------------------------------------------------------
# Results plot
# ---------------------------------------------------------------------------

def _plot(
    t, force, area, pressure,
    t1, t2, duration,
    out: str = "pinch_contact_results.png",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    has_area     = any(a > 0 for a in area)
    has_pressure = any(p > 0 for p in pressure)
    nrows = 1 + has_area + has_pressure

    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    pc = {"p1": "#b8d9f7", "p2": "#f7c6b8", "p3": "#b8f7c4"}
    q_close_deg = np.rad2deg(PinchController.Q_CLOSE)
    phase_legend = [
        Patch(color=pc["p1"], label="Phase 1 – synchronized ramp close (MCP=PIP=DIP)"),
        Patch(color=pc["p2"], label=f"Phase 2 – hold squeeze (q_ref = {q_close_deg:.0f}°)"),
        Patch(color=pc["p3"], label="Phase 3 – synchronized ramp open"),
    ]
    for ax in axes:
        ax.axvspan(0,  t1,       alpha=0.35, color=pc["p1"], lw=0)
        ax.axvspan(t1, t2,       alpha=0.35, color=pc["p2"], lw=0)
        ax.axvspan(t2, duration, alpha=0.35, color=pc["p3"], lw=0)
        ax.grid(True, alpha=0.35)

    row = 0
    axes[row].plot(t, force, color="tab:blue", linewidth=1.6)
    axes[row].set_ylabel("Contact force magnitude (N)", fontsize=10)
    axes[row].set_title(
        "Hydroelastic fingertip pinch — L_Tip_1 (compliant) vs R_Tip_1 (rigid)\n"
        f"E = {HYDRO_MODULUS/1e3:.0f} kPa, c = {DISSIPATION} s/m, μ_s = {MU_STATIC}",
        fontsize=10,
    )
    axes[row].legend(handles=phase_legend, fontsize=8, loc="upper left")
    row += 1

    if has_area:
        area_mm2 = [a * 1e6 for a in area]
        axes[row].plot(t, area_mm2, color="tab:green", linewidth=1.6)
        axes[row].set_ylabel("Contact area (mm²)", fontsize=10)
        row += 1

    if has_pressure:
        axes[row].plot(t, pressure, color="tab:red", linewidth=1.6)
        axes[row].set_ylabel("Max contact pressure (Pa)", fontsize=10)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Contact-results plot saved → {out}")
    try:
        plt.show()
    except Exception:
        pass  # headless / no display


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--duration", type=float, default=15.0,
        help="Simulation duration in seconds (default: 15)",
    )
    args = ap.parse_args()

    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    run(args.duration)


if __name__ == "__main__":
    main()
