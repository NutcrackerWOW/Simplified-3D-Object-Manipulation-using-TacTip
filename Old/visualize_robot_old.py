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
from pydrake.multibody.meshcat import JointSliders

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model", "Complete_Finger", "Complete.urdf",
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
    """Explicitly allow collision between the two distal fingertip links."""
    l_geoms = GeometrySet(
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("L_Distal_F_1"))
    )
    r_geoms = GeometrySet(
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("R_Distal_F_1"))
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
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Duration in seconds for --sim / --animate (default: 30)")
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
    else:
        kinematic_mode(meshcat)


if __name__ == "__main__":
    main()
