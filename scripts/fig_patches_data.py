#!/usr/bin/env python3
"""Collect hydroelastic contact-patch data (vertex positions + pressure) for
the paper's patch-gallery figure:

  * each shape-study object in the reference grasp at 1.25x its own f*
    (stable hold, sampled ~1.3 s into the hold), L-tip patch;
  * the box during a failing margin-1.3 hold (0.49 N), L-tip patch sampled
    through the creep phase — the patch migrating over the dome shoulder.

  python scripts/fig_patches_data.py
"""

import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import (BoxSpec, ControlParams, ModelParams, TrialSpec,
                             TIP_BODY, deg)
from pinchlab.trial import get_kinematics
from pinchlab.model import build_scene, set_posture
from pinchlab.control import GripController

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

SHAPES = ("disc", "box", "prism", "cylinder", "disc_edge", "sphere")


def build_sim(spec, kin):
    from pydrake.all import Simulator, RigidTransform
    open_posture = kin.find_open_posture(
        spec.posture, spec.box.size + spec.open_gap_extra)
    feas = kin.grasp_feasibility(spec.posture, spec.box.size)
    spawn = np.asarray(feas["mid"], dtype=float)
    spawn[2] += spec.spawn_z_offset
    mp = ModelParams(time_step=spec.time_step)
    scene = build_scene(mp, spec.box)
    plant, builder = scene.plant, scene.builder
    ctrl = builder.AddSystem(GripController(
        plant, scene.box_body, spec.posture, open_posture,
        spec.force_setpoint, ControlParams(), period=spec.time_step))
    builder.Connect(plant.get_state_output_port(),
                    ctrl.GetInputPort("plant_state"))
    builder.Connect(plant.get_contact_results_output_port(),
                    ctrl.GetInputPort("contact_results"))
    builder.Connect(ctrl.GetOutputPort("actuation"),
                    plant.get_actuation_input_port())
    builder.Connect(ctrl.GetOutputPort("spatial_forces"),
                    plant.get_applied_spatial_force_input_port())
    diagram = builder.Build()
    sim = Simulator(diagram)
    context = sim.get_mutable_context()
    pctx = plant.GetMyMutableContextFromRoot(context)
    set_posture(plant, pctx, open_posture, dip_equals_pip=True)
    plant.SetFreeBodyPose(pctx, scene.box_body, RigidTransform(spawn))
    return sim, plant, pctx, ctrl, scene


def grab_patch(plant, pctx, ctrl, finger="L"):
    """L-tip contact surface: vertices (world), per-vertex pressure, plus the
    tip body pose so the figure can plot in the tip frame."""
    cr = plant.get_contact_results_output_port().Eval(pctx)
    tips = ctrl.extractor.tip_geoms[finger]
    for i in range(cr.num_hydroelastic_contacts()):
        info = cr.hydroelastic_contact_info(i)
        surf = info.contact_surface()
        if surf.id_M() in tips or surf.id_N() in tips:
            mesh = surf.poly_mesh_W()
            field = surf.poly_e_MN()
            n = mesh.num_vertices()
            verts = np.array([mesh.vertex(k) for k in range(n)])
            press = np.array([field.EvaluateAtVertex(k) for k in range(n)])
            X_tip = plant.EvalBodyPoseInWorld(
                pctx, plant.GetBodyByName(TIP_BODY[finger]))
            return dict(verts=verts, press=press,
                        centroid=np.asarray(surf.centroid()),
                        force=np.asarray(
                            info.F_Ac_W().translational()),
                        tip_p=X_tip.translation(),
                        tip_R=X_tip.rotation().matrix())
    return None


def run_shape(shape):
    with open(os.path.join(RESULTS, ("material_fric_base.json"
                                     if shape == "box" else
                                     f"shape_{shape}.json"))) as fh:
        f_star = json.load(fh)["f_star"]
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(0.0), deg(0.0), 0.024)
    spec = TrialSpec(posture=posture, box=BoxSpec(mass=0.05, shape=shape),
                     force_setpoint=1.25 * f_star, seed=0)
    sim, plant, pctx, ctrl, scene = build_sim(spec, kin)
    sim.AdvanceTo(2.5)     # hold starts ~1.2 s in; sample mid-hold
    patch = grab_patch(plant, pctx, ctrl)
    out = {f"{shape}_{k}": v for k, v in patch.items()}
    out[f"{shape}_fstar"] = f_star
    print(f"{shape}: {len(patch['verts'])} verts, "
          f"|F| = {np.linalg.norm(patch['force']):.2f} N", flush=True)
    return out


def run_failing():
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(0.0), deg(0.0), 0.024)
    spec = TrialSpec(posture=posture, box=BoxSpec(mass=0.05),
                     force_setpoint=0.49, seed=0)
    sim, plant, pctx, ctrl, scene = build_sim(spec, kin)
    t_hold0 = 1.23                      # measured (fig_rolling_data)
    out = {}
    for k, th in enumerate((0.05, 1.5, 3.0, 4.0, 4.3)):
        sim.AdvanceTo(t_hold0 + th)
        patch = grab_patch(plant, pctx, ctrl)
        if patch is None:
            print(f"fail t_hold={th}: no contact", flush=True)
            continue
        for key, v in patch.items():
            out[f"fail{k}_{key}"] = v
        out[f"fail{k}_t"] = th
        print(f"fail t_hold={th}: {len(patch['verts'])} verts, "
              f"centroid z {patch['centroid'][2]:.4f}", flush=True)
    return out


def main():
    data = {}
    ctx = mp_.get_context("spawn")
    with ctx.Pool(processes=len(SHAPES)) as pool:
        for out in pool.imap_unordered(run_shape, SHAPES):
            data.update(out)
    data.update(run_failing())
    np.savez_compressed(os.path.join(RESULTS, "fig_patches_data.npz"), **data)
    print("→ results/fig_patches_data.npz", flush=True)


if __name__ == "__main__":
    main()
