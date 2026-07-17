"""Scene construction: patched Touch_Finger hand + parametric box object.

URDF is patched in memory (file on disk never modified):
  * .stl → .obj mesh references (as in pinch_sim.py)
  * both tip proximity blocks → compliant hydroelastic (TacTip pad material)
  * the 8 `continuous` joints → `revolute` with the plan's limits
Hand is welded base-up so fingers point DOWN (world -z); the two fingers are
separated along world +x, so the pinch/grasp axis is horizontal ≈ x̂.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    DiagramBuilder,
    DiscreteContactApproximation,
    JointActuatorIndex,
    Parser,
    RigidTransform,
    RotationMatrix,
    SpatialInertia,
)
from pydrake.geometry import (
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    Box,
    Cylinder,
    ProximityProperties,
    Rgba,
    Sphere,
)
from pydrake.multibody.plant import CoulombFriction

from .params import BoxSpec, ModelParams, Posture, TIP_BODY, deg

URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "Touch_Finger", "Complete.urdf",
)
PACKAGE_ROOT = os.path.dirname(URDF_PATH)

_FLEX_JOINT_NAMES = ["L_MCP", "L_PIP", "L_DIP", "R_MCP", "R_PIP", "R_DIP"]
_WAVE_JOINT_NAMES = ["L_Wave", "R_Wave"]
ACTUATED_JOINTS = _WAVE_JOINT_NAMES + _FLEX_JOINT_NAMES


def _tip_proximity_block(mp: ModelParams) -> str:
    relax = ("" if mp.relaxation_time is None else
             f"        <drake:relaxation_time value=\"{mp.relaxation_time}\"/>\n")
    return (
        "<drake:proximity_properties>\n"
        "        <drake:compliant_hydroelastic/>\n"
        f"        <drake:mesh_resolution_hint value=\"{mp.mesh_resolution_hint}\"/>\n"
        f"        <drake:hydroelastic_modulus value=\"{mp.hydro_modulus:.3g}\"/>\n"
        f"        <drake:hunt_crossley_dissipation value=\"{mp.dissipation}\"/>\n"
        f"{relax}"
        f"        <drake:mu_static value=\"{mp.mu_static_tip}\"/>\n"
        f"        <drake:mu_dynamic value=\"{mp.mu_dynamic_tip}\"/>\n"
        "      </drake:proximity_properties>"
    )


def patch_urdf(content: str, mp: ModelParams) -> str:
    content = content.replace(".stl", ".obj")

    # Both tips compliant (plan decision #4). The file has exactly one
    # compliant block (L tip) and one rigid block (R tip).
    for pattern in (
        r"<drake:proximity_properties>\s*<drake:compliant_hydroelastic/>"
        r"\s*(?:<drake:[^/]+/>\s*)*</drake:proximity_properties>",
        r"<drake:proximity_properties>\s*<drake:rigid_hydroelastic/>"
        r"\s*(?:<drake:[^/]+/>\s*)*</drake:proximity_properties>",
    ):
        content = re.sub(pattern, "      " + _tip_proximity_block(mp), content)

    # Ensure revolute + our limits (idempotent: the file now ships revolute
    # joints with baked limits; ModelParams stays authoritative at runtime).
    # Transmission blocks also contain `<joint name=...>` but without a type
    # attribute, so anchoring on the type attribute is safe.
    def limit_joint(name: str, lower: float, upper: float) -> None:
        nonlocal content
        block_re = re.compile(
            rf'(<joint name="{name}" type=)"(?:continuous|revolute)">(.*?)(</joint>)',
            re.S)

        def repl(m):
            body = re.sub(r'\s*<limit [^>]*/>', "", m.group(2))
            limit = (f'\n    <limit lower="{lower:.6f}" upper="{upper:.6f}" '
                     f'effort="{mp.effort_limit}" velocity="{mp.velocity_limit}"/>')
            return f'{m.group(1)}"revolute">{limit}{body}{m.group(3)}'

        content = block_re.sub(repl, content, count=1)

    for name in _FLEX_JOINT_NAMES:
        limit_joint(name, deg(mp.flex_lower_deg), deg(mp.flex_upper_deg))
    for name in _WAVE_JOINT_NAMES:
        limit_joint(name, -deg(mp.wave_limit_deg), deg(mp.wave_limit_deg))

    return content


def _load_hand(plant, mp: ModelParams) -> None:
    parser = Parser(plant)
    parser.package_map().Add("Complete_description", PACKAGE_ROOT)
    with open(URDF_PATH) as fh:
        content = patch_urdf(fh.read(), mp)
    parser.AddModelsFromString(content, "urdf")


def _weld_hand(plant, mp: ModelParams) -> None:
    # base x̂ (finger extension at q=0) → world -ẑ: fingers point down.
    # base ẑ (finger separation) → world +x̂: pinch axis horizontal.
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("base_link"),
        RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                       [0.0, 0.0, mp.base_height]),
    )


def _add_box(plant, spec: BoxSpec):
    instance = plant.AddModelInstance("box_obj")
    r = spec.size / 2.0
    X_BG = RigidTransform()            # geometry pose in the body frame
    if spec.shape == "cylinder":       # vertical axis: pads grip the curved side
        inertia = SpatialInertia.SolidCylinderWithMass(
            spec.mass, r, spec.height, [0.0, 0.0, 1.0])
        shape = Cylinder(r, spec.height)
    elif spec.shape == "disc":         # axis along the pinch axis (world x):
        inertia = SpatialInertia.SolidCylinderWithMass(   # pads grip flat faces
            spec.mass, r, spec.height, [1.0, 0.0, 0.0])
        shape = Cylinder(r, spec.height)
        X_BG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2))
    elif spec.shape == "disc_edge":    # key pinch: disc upright like a wheel,
        inertia = SpatialInertia.SolidCylinderWithMass(   # pads on the thin rim
            spec.mass, r, spec.height, [0.0, 1.0, 0.0])   # across the diameter
        shape = Cylinder(r, spec.height)
        X_BG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2))
    elif spec.shape == "sphere":
        inertia = SpatialInertia.SolidSphereWithMass(spec.mass, r)
        shape = Sphere(r)
    elif spec.shape == "prism":        # elongated block: 2x depth along e3
        inertia = SpatialInertia.SolidBoxWithMass(
            spec.mass, spec.size, 2 * spec.size, spec.height)
        shape = Box(spec.size, 2 * spec.size, spec.height)
    else:
        inertia = SpatialInertia.SolidBoxWithMass(
            spec.mass, spec.size, spec.size, spec.height)
        shape = Box(spec.size, spec.size, spec.height)
    body = plant.AddRigidBody("box", instance, inertia)
    props = ProximityProperties()
    AddContactMaterial(
        friction=CoulombFriction(spec.mu_static, spec.mu_dynamic),
        properties=props,
    )
    AddRigidHydroelasticProperties(spec.resolution_hint, props)
    plant.RegisterCollisionGeometry(body, X_BG, shape, "box_collision", props)
    plant.RegisterVisualGeometry(body, X_BG, shape, "box_visual",
                                 np.array([0.85, 0.55, 0.15, 1.0]))
    return body


@dataclass
class Scene:
    builder: object
    plant: object
    scene_graph: object
    box_body: object  # None if box not added


def build_scene(mp: ModelParams | None = None,
                box: BoxSpec | None = None,
                time_step: float | None = None) -> Scene:
    """Hand (+ optional box) on a SAP discrete plant. Plant is finalized."""
    mp = mp or ModelParams()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step if time_step is not None else mp.time_step)
    _load_hand(plant, mp)
    _weld_hand(plant, mp)
    box_body = _add_box(plant, box) if box is not None else None

    approx = {"sap": DiscreteContactApproximation.kSap,
              "lagged": DiscreteContactApproximation.kLagged,
              "similar": DiscreteContactApproximation.kSimilar}
    plant.set_discrete_contact_approximation(approx[mp.contact_approximation])
    if mp.stiction_tolerance is not None:
        plant.set_stiction_tolerance(mp.stiction_tolerance)
    plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
    for i in range(plant.num_actuators()):
        plant.get_joint_actuator(JointActuatorIndex(i)).set_default_rotor_inertia(
            mp.rotor_inertia)
    plant.Finalize()
    return Scene(builder, plant, scene_graph, box_body)


def set_posture(plant, context, posture: Posture, dip_equals_pip: bool = True) -> None:
    for name, q in posture.joint_map().items():
        plant.GetJointByName(name).set_angle(context, q)
    if dip_equals_pip:
        plant.GetJointByName("L_DIP").set_angle(context, posture.pip_l)
        plant.GetJointByName("R_DIP").set_angle(context, posture.pip_r)


class HandKinematics:
    """FK/geometry queries on a hand-only scene (no box, no controller).

    Used for grasp feasibility, spawn placement, and the open-posture search.
    Tip "gap" is the signed distance between the two tip collision geometries
    (mesh proximity queries use convex hulls — fine for the dome-shaped tips).
    """

    def __init__(self, mp: ModelParams | None = None):
        self.mp = mp or ModelParams()
        scene = build_scene(self.mp, box=None)
        self.plant = scene.plant
        self.scene_graph = scene.scene_graph
        self.diagram = scene.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_ctx = self.plant.GetMyContextFromRoot(self.context)
        self.sg_ctx = self.scene_graph.GetMyContextFromRoot(self.context)
        gl = self.plant.GetCollisionGeometriesForBody(
            self.plant.GetBodyByName(TIP_BODY["L"]))
        gr = self.plant.GetCollisionGeometriesForBody(
            self.plant.GetBodyByName(TIP_BODY["R"]))
        assert len(gl) == 1 and len(gr) == 1, "expected one collision geom per tip"
        self.gid_l, self.gid_r = gl[0], gr[0]

    def tip_poses(self, posture: Posture):
        set_posture(self.plant, self.plant_ctx, posture)
        XL = self.plant.EvalBodyPoseInWorld(
            self.plant_ctx, self.plant.GetBodyByName(TIP_BODY["L"]))
        XR = self.plant.EvalBodyPoseInWorld(
            self.plant_ctx, self.plant.GetBodyByName(TIP_BODY["R"]))
        return XL, XR

    def gap(self, posture: Posture) -> "GapInfo":
        set_posture(self.plant, self.plant_ctx, posture)
        qo = self.scene_graph.get_query_output_port().Eval(self.sg_ctx)
        res = qo.ComputeSignedDistancePairClosestPoints(self.gid_l, self.gid_r)
        # Witness points are expressed in each geometry's frame; map to world.
        p_WA = qo.GetPoseInWorld(res.id_A) @ res.p_ACa
        p_WB = qo.GetPoseInWorld(res.id_B) @ res.p_BCb
        mid = 0.5 * (p_WA + p_WB)
        line = p_WB - p_WA
        norm = float(np.linalg.norm(line))
        if norm < 1e-9:
            tilt = 0.0
        else:
            cosx = abs(line[0]) / norm
            tilt = float(np.rad2deg(np.arccos(np.clip(cosx, -1.0, 1.0))))
        # The L finger starts on the +x side; once the fingers curl past each
        # other (physically impossible with collisions on) the tip x-order
        # swaps and "gap" no longer means an openable pinch aperture.
        XL = self.plant.EvalBodyPoseInWorld(
            self.plant_ctx, self.plant.GetBodyByName(TIP_BODY["L"]))
        XR = self.plant.EvalBodyPoseInWorld(
            self.plant_ctx, self.plant.GetBodyByName(TIP_BODY["R"]))
        crossed = XL.translation()[0] < XR.translation()[0]
        y_split = 0.5 * abs(p_WA[1] - p_WB[1])
        return GapInfo(float(res.distance), p_WA, p_WB, mid, tilt, crossed, y_split)

    def solve_mcp_for_gap(self, wave: float, pip: float, target_gap: float,
                          dip_equals_pip: bool = True):
        """On the pre-crossing manifold the pad gap decreases monotonically
        with MCP. Bisect symmetric MCP so gap(posture) == target_gap.
        Returns (posture, GapInfo) or (None, None) if infeasible."""
        lo = deg(self.mp.flex_lower_deg)          # most open MCP
        g_lo = self.gap(Posture.symmetric(wave, lo, pip))
        if g_lo.crossed or g_lo.distance < target_gap:
            return None, None                      # cannot open wide enough
        # Find an upper bracket: increase MCP until crossed or gap < target.
        hi = lo
        step = deg(5.0)
        upper = deg(self.mp.flex_upper_deg)
        while hi < upper:
            hi = min(hi + step, upper)
            g = self.gap(Posture.symmetric(wave, hi, pip))
            if g.crossed or g.distance < target_gap:
                break
        else:
            return None, None                      # never closes to target
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            g = self.gap(Posture.symmetric(wave, mid, pip))
            if g.crossed or g.distance < target_gap:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-5:
                break
        posture = Posture.symmetric(wave, lo, pip)
        return posture, self.gap(posture)

    def find_open_posture(self, grasp: Posture, target_gap: float) -> Posture:
        """Reduce both MCP angles equally (bisection) until the tip gap is
        ~target_gap: the opened posture used for spawn/close."""
        g0 = self.gap(grasp)
        assert not g0.crossed, "grasp posture is past finger crossing"
        if g0.distance >= target_gap:
            return grasp
        lo = 0.0
        hi = min(grasp.mcp_l, grasp.mcp_r) - deg(self.mp.flex_lower_deg)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            p = Posture(grasp.wave_l, grasp.mcp_l - mid, grasp.pip_l,
                        grasp.wave_r, grasp.mcp_r - mid, grasp.pip_r)
            if self.gap(p).distance < target_gap:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-4:
                break
        return Posture(grasp.wave_l, grasp.mcp_l - hi, grasp.pip_l,
                       grasp.wave_r, grasp.mcp_r - hi, grasp.pip_r)

    def grasp_feasibility(self, posture: Posture, box_size: float) -> dict:
        """Filter criteria from the plan (adapted to the discovered geometry):
        pre-crossing, pinch line within 30° of horizontal, pads land on the
        box faces (y-split within the face), gap near the box size."""
        g = self.gap(posture)
        ok = (
            not g.crossed
            and g.tilt_deg < 30.0
            and g.y_split <= box_size / 2 - 0.002
            and (box_size - 0.006) <= g.distance <= (box_size + 0.004)
        )
        return {"feasible": ok, "gap": g.distance, "tilt_deg": g.tilt_deg,
                "y_split": g.y_split, "crossed": g.crossed, "mid": g.mid}


@dataclass
class GapInfo:
    distance: float      # surface-to-surface (negative = interpenetration)
    p_WA: np.ndarray     # closest point on L tip, world
    p_WB: np.ndarray     # closest point on R tip, world
    mid: np.ndarray      # box spawn center
    tilt_deg: float      # pinch line vs world x̂
    crossed: bool        # fingers curled past each other
    y_split: float       # half |Δy| of the witness points
