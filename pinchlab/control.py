"""Grip controller: posture PD + tactile force-feedback squeeze + trial state
machine + anti-gravity spawn holding.

Implemented as a *discrete* LeafSystem whose outputs depend only on its own
discrete state: the plant's ContactResults output feeds the controller's
periodic update, and actuation is published from stored state — so wiring
contact results back to the plant closes the tactile loop without an
algebraic loop (one-tick sensing delay, as on real hardware).

Phases:  CLOSE → GRIP → RELEASE → HOLD   (FAILED if contact is never made)
  CLOSE:   strong PD; MCP reference ramps from the open posture into the box.
  GRIP:    MCP switches to weak PD + integral force trim regulating the
           sensed per-finger squeeze to the setpoint (plan decision #2:
           force loop on MCP only; PIP stays position-stiff).
  RELEASE: anti-gravity wrench on the box ramps 1→0 (the object is "let go"
           into gravity).
  HOLD:    force loop keeps running; trial layer watches for drift/drop.
DIP is never position-driven: it always gets spring torque toward the
measured PIP angle (plan decisions #1/#13).
"""

from __future__ import annotations

import numpy as np

from pydrake.common.value import Value
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.systems.framework import LeafSystem

from .params import ControlParams, FINGERS, Posture, deg
from .tactile import TactileExtractor, decompose, grasp_frame

PHASE_CLOSE, PHASE_GRIP, PHASE_RELEASE, PHASE_HOLD, PHASE_FAILED = range(5)
PHASE_NAMES = {0: "close", 1: "grip", 2: "release", 3: "hold", 4: "failed"}

# Discrete-state slots (before the stored actuation vector)
(_PHASE, _T0, _FL, _FR, _TRIM_L, _TRIM_R, _OK_SINCE, _ALPHA,
 _SP, _BOOST_UNTIL, _PX_L, _PX_R, _HP_L, _HP_R, _EN_L, _EN_R) = range(16)
_NHEAD = 16

DEBUG_LABELS = ["phase", "f_L_filt", "f_R_filt", "trim_L", "trim_R",
                "alpha", "mcp_ref", "setpoint", "reflex_energy"]


class GripController(LeafSystem):
    def __init__(self, plant, box_body, grasp: Posture, open_posture: Posture,
                 force_setpoint: float, cp: ControlParams | None = None,
                 period: float = 2e-3, setpoint_ramp: float = 0.0,
                 posture_ref_fn=None, setpoint_fn=None, reflex: dict | None = None):
        """posture_ref_fn(t_hold)→Posture and setpoint_fn(t_hold)→float apply
        during HOLD (Phase B trajectories / scheduled squeeze). `reflex` is a
        dict(threshold, highpass_hz, boost, boost_time) enabling the online
        vibration reflex: on band-energy > threshold the setpoint is
        multiplied by `boost` for `boost_time` seconds."""
        super().__init__()
        self.plant = plant
        self.cp = cp or ControlParams()
        self.grasp = grasp
        self.open_posture = open_posture
        self.force_setpoint = float(force_setpoint)
        self.setpoint_ramp = float(setpoint_ramp)
        self.posture_ref_fn = posture_ref_fn
        self.setpoint_fn = setpoint_fn
        self.reflex = reflex
        self.box_mass = box_body.default_mass()
        self.box_index = box_body.index()
        self.period = period
        self.extractor = TactileExtractor(plant, box_body)

        nq, nv, nu = plant.num_positions(), plant.num_velocities(), plant.num_actuators()
        assert nu == 8, f"expected 8 actuators, got {nu}"
        self.nq, self.nv, self.nu = nq, nv, nu

        def jinfo(name):
            j = plant.GetJointByName(name)
            a = plant.GetJointActuatorByName(f"{name}_actr")
            return (j.position_start(), j.velocity_start(), int(a.index()))

        self.j = {f: {kind: jinfo(f"{f}_{kind}")
                      for kind in ("Wave", "MCP", "PIP", "DIP")}
                  for f in FINGERS}
        # Scratch context for gravity feedforward (Wave/MCP/PIP only — DIP
        # stays a pure passive spring).
        self._grav_ctx = plant.CreateDefaultContext() if self.cp.gravity_comp else None
        self.grasp_map = grasp.joint_map()
        self.open_map = open_posture.joint_map()
        # Close phase: MCP ramps open→grasp(+overdrive so the PD presses in).
        self.mcp_overdrive = deg(8.0)
        close_travel = max(
            self.grasp_map["L_MCP"] - self.open_map["L_MCP"],
            self.grasp_map["R_MCP"] - self.open_map["R_MCP"], 0.0)
        self.close_timeout = (close_travel + self.mcp_overdrive) / self.cp.close_rate + 1.0

        init = np.zeros(_NHEAD + nu)
        init[_ALPHA] = 1.0
        init[_OK_SINCE] = -1.0
        init[_SP] = self.force_setpoint
        init[_BOOST_UNTIL] = -1.0
        self._xd_index = self.DeclareDiscreteState(init)
        self.DeclarePeriodicDiscreteUpdateEvent(period, 0.0, self._update)

        self._state_in = self.DeclareVectorInputPort("plant_state", nq + nv)
        from pydrake.multibody.plant import ContactResults
        self._contact_in = self.DeclareAbstractInputPort(
            "contact_results", Value(ContactResults()))

        state_ticket = {self.all_state_ticket()}
        self.DeclareVectorOutputPort(
            "actuation", nu,
            lambda ctx, out: out.SetFromVector(
                ctx.get_discrete_state(self._xd_index).value()[_NHEAD:]),
            prerequisites_of_calc=state_ticket)
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: Value([ExternallyAppliedSpatialForce()]),
            self._calc_spatial_forces,
            prerequisites_of_calc=state_ticket)
        self.DeclareVectorOutputPort(
            "debug", len(DEBUG_LABELS),
            self._calc_debug, prerequisites_of_calc=state_ticket)
        self._last_mcp_ref = 0.0  # for debug output only

    # ------------------------------------------------------------------ #
    def _calc_spatial_forces(self, context, output):
        alpha = context.get_discrete_state(self._xd_index).value()[_ALPHA]
        f = ExternallyAppliedSpatialForce()
        f.body_index = self.box_index
        f.p_BoBq_B = np.zeros(3)
        f.F_Bq_W = SpatialForce(
            np.zeros(3), np.array([0.0, 0.0, alpha * self.box_mass * 9.81]))
        output.set_value([f])

    def _calc_debug(self, context, output):
        xd = context.get_discrete_state(self._xd_index).value()
        output.SetFromVector(np.array([
            xd[_PHASE], xd[_FL], xd[_FR], xd[_TRIM_L], xd[_TRIM_R],
            xd[_ALPHA], self._last_mcp_ref, xd[_SP],
            max(xd[_EN_L], xd[_EN_R])]))

    # ------------------------------------------------------------------ #
    def _update(self, context, discrete_state):
        cp = self.cp
        t = context.get_time()
        dt = self.period
        xd = context.get_discrete_state(self._xd_index).value().copy()

        state = self._state_in.Eval(context)
        if not np.all(np.isfinite(state)):
            xd[_NHEAD:] = 0.0
            discrete_state.get_mutable_vector(self._xd_index).SetFromVector(xd)
            return
        q, v = state[:self.nq], state[self.nq:]

        contact = self._contact_in.Eval(context)
        readings = self.extractor.read(contact)
        # Regulated signal = squeeze component f1 (plan decision #2). Until
        # both tips touch (no grasp frame yet) fall back to the force norm,
        # which is what triggers the contact-made transition anyway.
        dec = decompose(readings, grasp_frame(readings))
        if dec.valid:
            raw = {f: max(dec.f1[f], 0.0) for f in FINGERS}
        else:
            raw = {f: float(np.linalg.norm(readings[f].force_W))
                   for f in FINGERS}

        # Low-pass the sensed force (plan: ~20 Hz before the loop).
        beta = min(1.0, dt * 2.0 * np.pi * cp.force_filter_hz)
        xd[_FL] += beta * (raw["L"] - xd[_FL])
        xd[_FR] += beta * (raw["R"] - xd[_FR])
        f_filt = {"L": xd[_FL], "R": xd[_FR]}

        # Online reflex detector state: one-pole high-pass of the tangential
        # force, then an energy EMA (windowed mean square).
        if self.reflex is not None:
            hz = self.reflex.get("highpass_hz", 15.0)
            win = self.reflex.get("window_s", 0.1)
            a = 1.0 / (1.0 + 2 * np.pi * hz * dt)
            g_ema = min(1.0, dt / win)
            for f, px, hp, en in (("L", _PX_L, _HP_L, _EN_L),
                                  ("R", _PX_R, _HP_R, _EN_R)):
                tan = (float(np.hypot(dec.f2[f], dec.f3[f]))
                       if dec.valid else 0.0)
                xd[hp] = a * (xd[hp] + tan - xd[px])
                xd[px] = tan
                xd[en] += g_ema * (xd[hp] * xd[hp] - xd[en])

        phase = int(xd[_PHASE])
        t0 = xd[_T0]

        # ---------------- phase transitions ----------------
        if phase == PHASE_CLOSE:
            in_contact = (f_filt["L"] > cp.contact_force_on
                          and f_filt["R"] > cp.contact_force_on)
            if in_contact:
                phase, t0 = PHASE_GRIP, t
                ff = 0.7 * self.force_setpoint * cp.moment_arm
                xd[_TRIM_L] = xd[_TRIM_R] = ff
                xd[_OK_SINCE] = -1.0
            elif t - t0 > self.close_timeout:
                phase, t0 = PHASE_FAILED, t
        elif phase == PHASE_GRIP:
            band = cp.grip_band * self.force_setpoint
            in_band = (abs(f_filt["L"] - self.force_setpoint) < band
                       and abs(f_filt["R"] - self.force_setpoint) < band)
            if in_band and xd[_OK_SINCE] < 0:
                xd[_OK_SINCE] = t
            elif not in_band:
                xd[_OK_SINCE] = -1.0
            settled = xd[_OK_SINCE] >= 0 and t - xd[_OK_SINCE] > cp.grip_settle_time
            if settled or t - t0 > cp.grip_timeout:
                phase, t0 = PHASE_RELEASE, t
        elif phase == PHASE_RELEASE:
            if t - t0 >= cp.release_time:
                phase, t0 = PHASE_HOLD, t

        # ---------------- anti-gravity ramp ----------------
        if phase in (PHASE_CLOSE, PHASE_GRIP, PHASE_FAILED):
            xd[_ALPHA] = 1.0
        elif phase == PHASE_RELEASE:
            xd[_ALPHA] = max(0.0, 1.0 - (t - t0) / cp.release_time)
        else:
            xd[_ALPHA] = 0.0

        # ---------------- setpoint & posture references ----------------
        setpoint = self.force_setpoint
        ref_map = self.grasp_map
        if phase == PHASE_HOLD:
            t_hold = t - t0
            if self.setpoint_fn is not None:
                setpoint = float(self.setpoint_fn(t_hold))
            elif self.setpoint_ramp > 0.0:
                setpoint = max(0.02,
                               self.force_setpoint - self.setpoint_ramp * t_hold)
            if self.posture_ref_fn is not None:
                ref_map = self.posture_ref_fn(t_hold).joint_map()
            if self.reflex is not None:
                energy = max(xd[_EN_L], xd[_EN_R])
                if energy > self.reflex["threshold"]:
                    xd[_BOOST_UNTIL] = t + self.reflex.get("boost_time", 1.0)
                if t < xd[_BOOST_UNTIL]:
                    setpoint *= self.reflex.get("boost", 1.5)
        xd[_SP] = setpoint

        # ---------------- actuation ----------------
        u = np.zeros(self.nu)
        tau_g = None
        if self._grav_ctx is not None:
            self.plant.SetPositions(self._grav_ctx, q)
            tau_g = self.plant.CalcGravityGeneralizedForces(self._grav_ctx)
        force_mode = phase in (PHASE_GRIP, PHASE_RELEASE, PHASE_HOLD)
        mcp_ref_dbg = 0.0
        for f in FINGERS:
            jw, jm, jp, jd = (self.j[f][k] for k in ("Wave", "MCP", "PIP", "DIP"))
            # Wave & PIP: strong PD to the (possibly trajectory) targets.
            u[jw[2]] = (cp.kp_wave * (ref_map[f + "_Wave"] - q[jw[0]])
                        - cp.kd_wave * v[jw[1]])
            u[jp[2]] = (cp.kp_flex * (ref_map[f + "_PIP"] - q[jp[0]])
                        - cp.kd_flex * v[jp[1]])
            if tau_g is not None:  # cancel finger weight on the active joints
                u[jw[2]] -= tau_g[jw[1]]
                u[jp[2]] -= tau_g[jp[1]]
            # DIP: passive spring toward measured PIP angle.
            u[jd[2]] = (cp.dip_spring_k * (q[jp[0]] - q[jd[0]])
                        - cp.dip_spring_c * v[jd[1]])
            # MCP: phase-dependent.
            mcp_grasp = ref_map[f + "_MCP"]
            if phase == PHASE_CLOSE:
                mcp_ref = min(self.open_map[f + "_MCP"] + cp.close_rate * (t - t0),
                              mcp_grasp + self.mcp_overdrive)
                u[jm[2]] = cp.kp_flex * (mcp_ref - q[jm[0]]) - cp.kd_flex * v[jm[1]]
                mcp_ref_dbg = mcp_ref
            elif force_mode:
                trim_slot = _TRIM_L if f == "L" else _TRIM_R
                err = setpoint - f_filt[f]
                xd[trim_slot] = float(np.clip(
                    xd[trim_slot] + cp.force_ki * err * dt,
                    cp.trim_min, cp.trim_max))
                u[jm[2]] = (cp.kp_mcp_force_mode * (mcp_grasp - q[jm[0]])
                            - cp.kd_mcp_force_mode * v[jm[1]]
                            + xd[trim_slot])
                mcp_ref_dbg = mcp_grasp
            else:  # FAILED: hold posture
                u[jm[2]] = cp.kp_flex * (mcp_grasp - q[jm[0]]) - cp.kd_flex * v[jm[1]]
            if tau_g is not None:
                u[jm[2]] -= tau_g[jm[1]]

        xd[_PHASE], xd[_T0] = phase, t0
        xd[_NHEAD:] = u
        self._last_mcp_ref = mcp_ref_dbg
        discrete_state.get_mutable_vector(self._xd_index).SetFromVector(xd)
