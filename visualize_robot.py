#!/usr/bin/env python3
"""
Visualize and manipulate Complete.urdf (two-finger hand) in MeshCat using Drake.

Robot structure (two mirrored fingers, L = left, R = right):
  base_link
    --L_Wave (Z)--> L_Base_1
        --L_MCP (Y)--> L_Proximal_F_1
            --L_PIP (Y)--> L_Middle_F_1
                --L_DIP (Y)--> L_Distal_F_1
    --R_Wave (Z)--> R_Base_1
        --R_MCP (Y)--> R_Proximal_F_1
            --R_PIP (Y)--> R_Middle_F_1
                --R_DIP (Y)--> R_Distal_F_1

Usage:
  python visualize_robot.py              # interactive joint sliders (default)
  python visualize_robot.py --sim        # physics simulation under gravity
  python visualize_robot.py --animate    # automatic sine-wave joint animation
  python visualize_robot.py --duration 60
"""

import argparse
import os
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConstantVectorSource,
    DiagramBuilder,
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
from pydrake.multibody.plant import ApplyMultibodyPlantConfig, MultibodyPlantConfig
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.multibody.meshcat import JointSliders

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model", "Touch_Finger", "Complete.urdf",
)

# Root directory for the "Complete_description" package (parent of meshes/).
_PACKAGE_ROOT = os.path.dirname(URDF_PATH)


def _load_urdf(parser: Parser) -> None:
    """Load Complete.urdf, swapping .stl references to .obj meshes."""
    parser.package_map().Add("Complete_description", _PACKAGE_ROOT)
    with open(URDF_PATH) as fh:
        content = fh.read().replace(".stl", ".obj")
    parser.AddModelsFromString(content, "urdf")


def _allow_distal_collision(plant, scene_graph) -> None:
    """Explicitly allow collision between the two fingertip links."""
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
# Helpers
# ---------------------------------------------------------------------------

def _build_diagram(meshcat, time_step=0.0, zero_actuation=False):
    """Build a minimal Drake diagram with MultibodyPlant + MeshcatVisualizer."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)

    _load_urdf(Parser(plant))
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("base_link"),
        RigidTransform(RotationMatrix.MakeYRotation(-np.pi / 2)),
    )
    plant.Finalize()
    _allow_distal_collision(plant, scene_graph)

    if zero_actuation and plant.num_actuators() > 0:
        zero_u = builder.AddSystem(
            ConstantVectorSource(np.zeros(plant.num_actuators()))
        )
        builder.Connect(zero_u.get_output_port(), plant.get_actuation_input_port())

    MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(
            role=Role.kProximity,
            delete_on_initialization_event=False,
        ),
    )

    return builder.Build(), plant


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def kinematic_mode(meshcat):
    """
    Interactive kinematic visualization.

    JointSliders wires up sliders in the MeshCat 'Controls' panel
    (top-right corner of the browser tab).  Drag a slider to rotate the
    corresponding joint.  Ctrl+C in the terminal to quit.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    _load_urdf(Parser(plant))
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("base_link"),
        RigidTransform(RotationMatrix.MakeYRotation(-np.pi / 2)),
    )
    plant.Finalize()
    _allow_distal_collision(plant, scene_graph)

    print("Joints (sliders will appear in MeshCat):")
    for idx in plant.GetJointIndices():
        j = plant.get_joint(idx)
        if j.num_positions() > 0:
            print(f"  {j.name()}  ({j.num_positions()} dof)")

    MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(
            role=Role.kProximity,
            delete_on_initialization_event=False,
        ),
    )

    sliders = builder.AddSystem(JointSliders(meshcat=meshcat, plant=plant))
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    Simulator(diagram).Initialize()
    diagram.ForcedPublish(context)

    print("\nDrag sliders in the MeshCat browser panel to move joints.")
    print("Press Ctrl+C here to exit.\n")

    meshcat.AddButton("Stop Running", "Escape")
    try:
        while meshcat.GetButtonClicks("Stop Running") == 0:
            q = sliders.get_output_port().Eval(
                sliders.GetMyContextFromRoot(context)
            )
            plant.SetPositions(plant.GetMyMutableContextFromRoot(context), q)
            diagram.ForcedPublish(context)
            time.sleep(1 / 32.0)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        meshcat.DeleteButton("Stop Running")


def simulation_mode(meshcat, duration):
    """
    Discrete-time physics simulation.

    The robot is released from the zero-position with no applied torques and
    falls under gravity.  Watch it in the browser.
    """
    diagram, plant = _build_diagram(meshcat, time_step=1e-3, zero_actuation=True)

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    print(f"Running physics simulation for {duration}s (zero torques, gravity on).")
    print("Press Ctrl+C to stop early.\n")
    try:
        simulator.AdvanceTo(duration)
    except KeyboardInterrupt:
        print("\nStopped by user.")


def animate_mode(meshcat, duration):
    """
    Kinematic sine-wave animation — no physics, just pose sweeping.

    Each joint is driven by sin(t + offset) so they move out of phase,
    giving a fluid wave-like motion.  Runs until duration expires or Ctrl+C.
    """
    diagram, plant = _build_diagram(meshcat, time_step=0.0, zero_actuation=False)

    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)
    nq = plant.num_positions()

    print(f"Animating {nq} joints with offset sine waves for {duration}s.")
    print("Press Ctrl+C to stop early.\n")

    offsets = np.linspace(0.0, 2 * np.pi, nq, endpoint=False)
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            if t > duration:
                break
            plant.SetPositions(plant_ctx, np.sin(t + offsets))
            diagram.ForcedPublish(context)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped by user.")


class _PhasedController(LeafSystem):
    """
    Three-phase torque controller wired into the closing simulation.

      Phase 1 [0, t1):     close finger joints at +τ, Wave joints free (0 N·m)
      Phase 2 [t1, t2):    open finger joints at -τ, Wave joints free
      Phase 3 [t2, end):   close finger joints at +τ, Wave joints PD-held at 1°
    """
    _WAVE_TARGET = np.deg2rad(1.0)
    _KP_WAVE = 2.0   # N·m / rad
    _KD_WAVE = 0.2   # N·m·s / rad

    def __init__(self, plant, t1, t2, tau_close=0.05):
        LeafSystem.__init__(self)
        self._plant = plant
        self._t1 = t1
        self._t2 = t2
        self._tau = tau_close

        nq = plant.num_positions()
        nv = plant.num_velocities()

        lw = plant.GetJointByName("L_Wave")
        rw = plant.GetJointByName("R_Wave")
        self._lw_qi = lw.position_start()
        self._rw_qi = rw.position_start()
        self._lw_vi = lw.velocity_start()
        self._rw_vi = rw.velocity_start()

        self._lw_ui = int(plant.GetJointActuatorByName("L_Wave_actr").index())
        self._rw_ui = int(plant.GetJointActuatorByName("R_Wave_actr").index())

        _finger_joints = ["L_MCP", "L_PIP", "L_DIP", "R_MCP", "R_PIP", "R_DIP"]
        self._finger_uis = [
            int(plant.GetJointActuatorByName(f"{n}_actr").index())
            for n in _finger_joints
        ]
        self._finger_qi = [plant.GetJointByName(n).position_start() for n in _finger_joints]
        self._finger_vi = [plant.GetJointByName(n).velocity_start() for n in _finger_joints]

        self._state_port = self.DeclareVectorInputPort("state", nq + nv)
        self.DeclareVectorOutputPort("actuation", plant.num_actuators(), self._calc)

    def _calc(self, context, output):
        t = context.get_time()
        state = self._state_port.Eval(context)
        nq = self._plant.num_positions()
        q, v = state[:nq], state[nq:]

        u = np.zeros(self._plant.num_actuators())

        if t < self._t1:
            for i in self._finger_uis:
                u[i] = self._tau        # Phase 1: close
        elif t < self._t2:
            for i in self._finger_uis:
                u[i] = -self._tau  # open at constant torque
        else:
            for i in self._finger_uis:
                u[i] = self._tau        # Phase 3: close
            u[self._lw_ui] = (
                self._KP_WAVE * (self._WAVE_TARGET - q[self._lw_qi])
                - self._KD_WAVE * v[self._lw_vi]
            )
            # R_Wave targets opposite direction so both fingers wave symmetrically
            u[self._rw_ui] = (
                self._KP_WAVE * (-self._WAVE_TARGET - q[self._rw_qi])
                - self._KD_WAVE * v[self._rw_vi]
            )

        output.SetFromVector(u)


def closing_mode(meshcat, duration):
    """
    Three-phase finger simulation:
      1. Close MCP/PIP/DIP joints
      2. Open them back
      3. Close again while Wave joints are driven to 1°
    Records hydroelastic contact force, area, and Wave joint angles, then
    saves a plot to contact_results.png.
    """
    import matplotlib.pyplot as plt

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)

    _load_urdf(Parser(plant))
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("base_link"),
        RigidTransform(RotationMatrix.MakeYRotation(-np.pi / 2)),
    )
    # TAMSI is more numerically stable than SAP for mesh-based hydroelastic contact
    _cfg = MultibodyPlantConfig()
    _cfg.time_step = 1e-3
    _cfg.discrete_contact_approximation = "tamsi"
    ApplyMultibodyPlantConfig(_cfg, plant)
    plant.Finalize()
    _allow_distal_collision(plant, scene_graph)

    t1 = duration / 3.0
    t2 = 2.0 * duration / 3.0

    controller = builder.AddSystem(_PhasedController(plant, t1=t1, t2=t2))
    builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())

    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration, delete_on_initialization_event=False),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.1)
    simulator.Initialize()

    # Start fingers slightly open so there's visible closing motion from the start.
    plant_ctx0 = plant.GetMyMutableContextFromRoot(simulator.get_mutable_context())
    q0 = plant.GetPositions(plant_ctx0)
    for jname in ["L_MCP", "L_PIP", "L_DIP", "R_MCP", "R_PIP", "R_DIP"]:
        q0[plant.GetJointByName(jname).position_start()] = -0.15
    plant.SetPositions(plant_ctx0, q0)

    diagram.ForcedPublish(simulator.get_context())

    lw_qi = plant.GetJointByName("L_Wave").position_start()
    rw_qi = plant.GetJointByName("R_Wave").position_start()

    record_dt = 0.005  # 200 Hz
    t_log, force_log, area_log, lwave_log, rwave_log = [], [], [], [], []

    print(f"Phase 1 [0 – {t1:.1f}s]:  close fingers")
    print(f"Phase 2 [{t1:.1f} – {t2:.1f}s]: open fingers")
    print(f"Phase 3 [{t2:.1f} – {duration:.1f}s]: close + Wave joints → 1°")
    print("Press Ctrl+C to stop early.\n")

    try:
        t = 0.0
        while t < duration:
            t += record_dt
            simulator.AdvanceTo(t)

            plant_ctx = plant.GetMyContextFromRoot(simulator.get_context())
            q = plant.GetPositions(plant_ctx)
            lwave_log.append(np.rad2deg(q[lw_qi]))
            rwave_log.append(np.rad2deg(q[rw_qi]))

            cr = plant.get_contact_results_output_port().Eval(plant_ctx)
            total_force = 0.0
            total_area = 0.0
            for i in range(cr.num_hydroelastic_contacts()):
                info = cr.hydroelastic_contact_info(i)
                total_force += np.linalg.norm(info.F_Ac_W().translational())
                surf = info.contact_surface()
                for attr in ("total_area", "area"):
                    if hasattr(surf, attr):
                        total_area += getattr(surf, attr)()
                        break

            t_log.append(t)
            force_log.append(total_force)
            area_log.append(total_area)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    if not t_log:
        print("No data recorded.")
        return

    has_area = any(a > 0 for a in area_log)
    n_rows = 2 + (1 if has_area else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=True)

    colors = {"p1": "#d0e8ff", "p2": "#ffd0d0", "p3": "#d0ffd0"}
    for ax in axes:
        ax.axvspan(0,   t1,           alpha=0.4, color=colors["p1"])
        ax.axvspan(t1,  t2,           alpha=0.4, color=colors["p2"])
        ax.axvspan(t2,  duration,     alpha=0.4, color=colors["p3"])

    ax_f = axes[0]
    ax_f.plot(t_log, force_log, linewidth=1.5, color="tab:blue")
    ax_f.set_ylabel("Contact force (N)")
    ax_f.set_title("Hydroelastic fingertip contact — close / open / close+wave")
    ax_f.grid(True)

    row = 1
    if has_area:
        axes[row].plot(t_log, area_log, linewidth=1.5, color="tab:green")
        axes[row].set_ylabel("Contact area (m²)")
        axes[row].grid(True)
        row += 1

    ax_w = axes[row]
    ax_w.plot(t_log, lwave_log, linewidth=1.5, label="L_Wave", color="tab:orange")
    ax_w.plot(t_log, rwave_log, linewidth=1.5, label="R_Wave",
              color="tab:purple", linestyle="--")
    ax_w.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="target 1°")
    ax_w.set_ylabel("Wave joint (°)")
    ax_w.set_xlabel("Time (s)")
    ax_w.legend(fontsize=8)
    ax_w.grid(True)

    # phase labels in legend proxy
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(color=colors["p1"], label="Phase 1: close"),
        Patch(color=colors["p2"], label="Phase 2: open"),
        Patch(color=colors["p3"], label="Phase 3: close + wave"),
    ], fontsize=8, loc="upper left")

    plt.tight_layout()
    out = "contact_results.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Drake + MeshCat URDF visualizer (Complete two-finger hand)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--sim", action="store_true",
                       help="Physics simulation under gravity")
    group.add_argument("--animate", action="store_true",
                       help="Automatic sine-wave joint animation")
    group.add_argument("--close", action="store_true",
                       help="Close fingers at constant torque and plot hydroelastic contact")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Duration in seconds (default: 30)")
    args = ap.parse_args()

    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    meshcat = StartMeshcat()
    print(f"\n{'='*50}")
    print(f"  MeshCat URL: {meshcat.web_url()}")
    print(f"{'='*50}\n")
    print("Open the URL above in a browser to see the robot.\n")

    if args.sim:
        simulation_mode(meshcat, args.duration)
    elif args.animate:
        animate_mode(meshcat, args.duration)
    elif args.close:
        closing_mode(meshcat, args.duration)
    else:
        kinematic_mode(meshcat)


if __name__ == "__main__":
    main()
