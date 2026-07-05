"""One pinch episode: spawn → close → grip → release → hold, with logging,
kinematic slip ground truth, and outcome labeling.

Outcomes:
  held     — object stayed within drift threshold through the hold window
  drift    — object moved > threshold relative to the fingertips (slipped)
  drop     — object fell away from the tips (gross slip)
  no_grasp — fingers never made two-tip contact (bad posture/spawn)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pydrake.all import Simulator

from .control import (GripController, PHASE_FAILED, PHASE_HOLD, PHASE_NAMES)
from .model import HandKinematics, build_scene
from .params import ControlParams, FINGERS, ModelParams, TrialSpec
from .tactile import decompose, grasp_frame

_KIN_CACHE: dict = {}


def get_kinematics() -> HandKinematics:
    if "kin" not in _KIN_CACHE:
        _KIN_CACHE["kin"] = HandKinematics(ModelParams())
    return _KIN_CACHE["kin"]


@dataclass
class TrialResult:
    spec: TrialSpec
    outcome: str
    held: bool
    slip_time: float          # first incipient slip after release (nan if none)
    drift_final: float
    rot_drift_final: float    # rad, box rotation since hold start
    grip_force_mean: dict     # finger → mean |F| during hold
    f1_mean: dict
    f2_mean: dict
    f3_mean: dict
    rho_mean: dict
    posture_measured: dict    # joint name → mean angle during hold
    series: dict | None = None

    def flat_row(self) -> dict:
        row = self.spec.flat_row()
        row.update({
            "outcome": self.outcome, "held": self.held,
            "slip_time": self.slip_time, "drift_final": self.drift_final,
            "rot_drift_final": self.rot_drift_final,
        })
        for f in FINGERS:
            row[f"f1_{f}"] = self.f1_mean[f]
            row[f"f2_{f}"] = self.f2_mean[f]
            row[f"f3_{f}"] = self.f3_mean[f]
            row[f"rho_{f}"] = self.rho_mean[f]
            row[f"grip_force_{f}"] = self.grip_force_mean[f]
        for name, val in self.posture_measured.items():
            row[f"q_{name}"] = val
        return row


def run_trial(spec: TrialSpec, meshcat=None, keep_series: bool = False,
              cp: ControlParams | None = None, posture_ref_fn=None,
              setpoint_fn=None, reflex: dict | None = None,
              hold_time: float | None = None,
              mp: ModelParams | None = None) -> TrialResult:
    cp = cp or ControlParams()
    kin = get_kinematics()
    if hold_time is None:
        hold_time = (cp.hold_time if spec.setpoint_ramp <= 0
                     else spec.force_setpoint / spec.setpoint_ramp + 2.0)

    feas = kin.grasp_feasibility(spec.posture, spec.box.size)
    if feas["crossed"]:
        return _empty_result(spec, "no_grasp")
    open_posture = kin.find_open_posture(
        spec.posture, spec.box.size + spec.open_gap_extra)
    spawn_center = np.asarray(feas["mid"], dtype=float)
    spawn_center[2] += spec.spawn_z_offset
    rng = np.random.default_rng(spec.seed)
    if spec.spawn_jitter > 0:
        spawn_center = spawn_center + rng.uniform(
            -spec.spawn_jitter, spec.spawn_jitter, 3)

    mp = mp or ModelParams(time_step=spec.time_step)
    scene = build_scene(mp, spec.box)
    plant, builder = scene.plant, scene.builder

    ctrl = builder.AddSystem(GripController(
        plant, scene.box_body, spec.posture, open_posture,
        spec.force_setpoint, cp, period=spec.time_step,
        setpoint_ramp=spec.setpoint_ramp, posture_ref_fn=posture_ref_fn,
        setpoint_fn=setpoint_fn, reflex=reflex))
    builder.Connect(plant.get_state_output_port(),
                    ctrl.GetInputPort("plant_state"))
    builder.Connect(plant.get_contact_results_output_port(),
                    ctrl.GetInputPort("contact_results"))
    builder.Connect(ctrl.GetOutputPort("actuation"),
                    plant.get_actuation_input_port())
    builder.Connect(ctrl.GetOutputPort("spatial_forces"),
                    plant.get_applied_spatial_force_input_port())

    if meshcat is not None:
        from pydrake.all import MeshcatVisualizer, MeshcatVisualizerParams, Role
        from pydrake.multibody.meshcat import (ContactVisualizer,
                                               ContactVisualizerParams)
        MeshcatVisualizer.AddToBuilder(
            builder, scene.scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration,
                                    delete_on_initialization_event=False))
        cv = ContactVisualizerParams()
        cv.publish_period = 1 / 64.0
        cv.force_threshold = 1e-3
        ContactVisualizer.AddToBuilder(builder, plant, meshcat, cv)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)

    # Initial pose: opened fingers (DIP tracks PIP), box floating at spawn.
    from .model import set_posture
    from pydrake.math import RigidTransform
    set_posture(plant, plant_ctx, open_posture, dip_equals_pip=True)
    plant.SetFreeBodyPose(plant_ctx, scene.box_body, RigidTransform(spawn_center))

    if meshcat is not None:
        diagram.ForcedPublish(context)
        meshcat.StartRecording(set_visualizations_while_recording=True)
        simulator.set_target_realtime_rate(1.0)

    extractor = ctrl.extractor
    tip_bodies = {f: plant.GetBodyByName(f"{f}_Tip_1") for f in FINGERS}
    act_joints = [f"{f}_{k}" for f in FINGERS for k in ("Wave", "MCP", "PIP", "DIP")]
    jidx = {n: plant.GetJointByName(n).position_start() for n in act_joints}

    t_max = ctrl.close_timeout + cp.grip_timeout + cp.release_time + hold_time + 1.5
    dt_log = 1.0 / spec.log_hz

    log: dict = {k: [] for k in [
        "t", "phase", "alpha", "trim_L", "trim_R", "mcp_ref", "setpoint",
        "fL", "fR", "cL", "cR", "inL", "inR",
        "f1_L", "f1_R", "f2_L", "f2_R", "f3_L", "f3_R", "rho_L", "rho_R",
        "box_p", "drift", "rot_drift", "slide_L", "slide_R", "q"]}

    outcome = None
    slip_time = np.nan
    slide_since = {"L": None, "R": None}
    rel0 = None          # box-minus-tip-midpoint baseline at hold start
    R0 = None            # box rotation baseline at hold start
    hold_t0 = None
    drift = 0.0
    rot_drift = 0.0

    t = 0.0
    try:
        while t < t_max:
            t = min(t + dt_log, t_max)
            simulator.AdvanceTo(t)

            dbg = ctrl.GetOutputPort("debug").Eval(
                ctrl.GetMyContextFromRoot(context))
            phase = int(dbg[0])

            cr = plant.get_contact_results_output_port().Eval(plant_ctx)
            readings = extractor.read(cr)
            frame = grasp_frame(readings)
            dec = decompose(readings, frame)

            X_box = plant.EvalBodyPoseInWorld(plant_ctx, scene.box_body)
            p_box = X_box.translation()
            p_tip = {f: plant.EvalBodyPoseInWorld(
                plant_ctx, tip_bodies[f]).translation() for f in FINGERS}
            tip_mid = 0.5 * (p_tip["L"] + p_tip["R"])
            rel = p_box - tip_mid

            # Sliding speed at each contact (kinematic incipient-slip truth):
            V_box = plant.EvalBodySpatialVelocityInWorld(plant_ctx, scene.box_body)
            slide = {}
            for f in FINGERS:
                if readings[f].in_contact and frame.valid:
                    c = readings[f].centroid_W
                    v_b = (V_box.translational()
                           + np.cross(V_box.rotational(), c - p_box))
                    V_tip = plant.EvalBodySpatialVelocityInWorld(
                        plant_ctx, tip_bodies[f])
                    v_t = (V_tip.translational()
                           + np.cross(V_tip.rotational(),
                                      c - p_tip[f]))
                    v_rel = v_b - v_t
                    v_tan = v_rel - np.dot(v_rel, frame.d) * frame.d
                    slide[f] = float(np.linalg.norm(v_tan))
                else:
                    slide[f] = 0.0

            if phase == PHASE_HOLD and hold_t0 is None:
                hold_t0 = t
                rel0 = rel.copy()
                R0 = X_box.rotation()
            if rel0 is not None:
                drift = float(np.linalg.norm(rel - rel0))
                rot_drift = float(np.abs(
                    (R0.inverse() @ X_box.rotation()).ToAngleAxis().angle()))

            # Incipient slip truth (only meaningful once object carries load).
            if rel0 is not None:
                for f in FINGERS:
                    if slide[f] > spec.slide_speed:
                        if slide_since[f] is None:
                            slide_since[f] = t
                        elif (t - slide_since[f] > spec.slide_sustain
                              and np.isnan(slip_time)):
                            slip_time = slide_since[f]
                    else:
                        slide_since[f] = None

            # ---- logging ----
            log["t"].append(t)
            log["phase"].append(phase)
            log["alpha"].append(dbg[5])
            log["trim_L"].append(dbg[3])
            log["trim_R"].append(dbg[4])
            log["mcp_ref"].append(dbg[6])
            log["setpoint"].append(dbg[7])
            log["fL"].append(readings["L"].force_W.copy())
            log["fR"].append(readings["R"].force_W.copy())
            log["cL"].append(readings["L"].centroid_W.copy())
            log["cR"].append(readings["R"].centroid_W.copy())
            log["inL"].append(readings["L"].in_contact)
            log["inR"].append(readings["R"].in_contact)
            for key, dd in (("f1", dec.f1), ("f2", dec.f2), ("f3", dec.f3),
                            ("rho", dec.rho)):
                log[f"{key}_L"].append(dd["L"])
                log[f"{key}_R"].append(dd["R"])
            log["box_p"].append(p_box.copy())
            log["drift"].append(drift)
            log["rot_drift"].append(rot_drift)
            log["slide_L"].append(slide["L"])
            log["slide_R"].append(slide["R"])
            log["q"].append([plant.GetPositions(plant_ctx)[jidx[n]]
                             for n in act_joints])

            # ---- termination ----
            if phase == PHASE_FAILED:
                outcome = "no_grasp"
                break
            min_tip_z = min(p_tip["L"][2], p_tip["R"][2])
            if p_box[2] < min_tip_z - spec.drop_margin:
                outcome = "drop"
                break
            if hold_t0 is not None and t - hold_t0 >= hold_time:
                outcome = "drift" if drift > spec.drift_threshold else "held"
                break
    except (RuntimeError, ValueError) as e:  # solver blow-up etc.
        outcome = f"error:{type(e).__name__}"

    if outcome is None:
        outcome = "no_grasp" if hold_t0 is None else (
            "drift" if drift > spec.drift_threshold else "held")
    # A drift beyond threshold at any point counts as slipped even if the
    # window ended via drop.
    if outcome == "held" and drift > spec.drift_threshold:
        outcome = "drift"

    if meshcat is not None:
        meshcat.StopRecording()
        meshcat.PublishRecording()

    arr = {k: np.asarray(v) for k, v in log.items()}
    held = outcome == "held"

    # Aggregates over the hold window (fallback: last 25% of samples).
    tarr = arr["t"]
    if hold_t0 is not None:
        mask = tarr >= hold_t0
    else:
        mask = tarr >= (tarr[-1] * 0.75 if len(tarr) else 0.0)
    if not mask.any():
        mask = np.ones_like(tarr, dtype=bool)

    def mmean(key):
        return {f: float(np.mean(arr[f"{key}_{f}"][mask]))
                if len(tarr) else np.nan for f in FINGERS}

    grip_force = {f: float(np.mean(
        np.linalg.norm(np.stack(arr[f"f{f}"]), axis=1)[mask]))
        if len(tarr) else np.nan for f in FINGERS}
    posture_meas = {}
    if len(tarr):
        qarr = np.asarray(arr["q"])
        for i, n in enumerate(act_joints):
            posture_meas[n] = float(np.mean(qarr[mask, i]))

    return TrialResult(
        spec=spec, outcome=outcome, held=held, slip_time=slip_time,
        drift_final=drift, rot_drift_final=rot_drift,
        grip_force_mean=grip_force,
        f1_mean=mmean("f1"), f2_mean=mmean("f2"), f3_mean=mmean("f3"),
        rho_mean=mmean("rho"),
        posture_measured=posture_meas,
        series=arr if keep_series else None,
    )


def _empty_result(spec: TrialSpec, outcome: str) -> TrialResult:
    z = {f: np.nan for f in FINGERS}
    return TrialResult(
        spec=spec, outcome=outcome, held=False, slip_time=np.nan,
        drift_final=np.nan, rot_drift_final=np.nan,
        grip_force_mean=dict(z), f1_mean=dict(z), f2_mean=dict(z),
        f3_mean=dict(z), rho_mean=dict(z), posture_measured={})
