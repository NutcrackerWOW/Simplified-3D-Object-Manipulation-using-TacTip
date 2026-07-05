"""Idealized TacTip readings from hydroelastic contact, and the grasp-axis
force decomposition (plan decision #5).

Sign convention: `force_W` is the force the fingertip applies ON the object.
Decomposition basis (identical for both fingers, gravity-aligned):
    e1_f = grasp axis pointing from this finger's contact INTO the object
           (L: +d, R: −d, with d = (p_R − p_L)/‖·‖)
    e2   = world vertical with its d-component removed (weight-carrying shear)
    e3   = d × e2 (lateral shear)
so f1 = squeeze (positive when pressing in), f2 = vertical shear, f3 = lateral.
Utilization ρ = ‖(f2, f3)‖ / f1 — the friction-cone coordinate of the study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import FINGERS, TIP_BODY


@dataclass
class TipReading:
    in_contact: bool
    force_W: np.ndarray      # force on the object from this tip (3,)
    centroid_W: np.ndarray   # contact patch centroid (3,)
    area: float              # m² (0 if unavailable)


@dataclass
class GraspFrame:
    valid: bool
    d: np.ndarray            # grasp axis L→R
    e2: np.ndarray           # vertical ⊥ d
    e3: np.ndarray           # lateral = d × e2


@dataclass
class Decomposition:
    valid: bool
    f1: dict                 # finger → squeeze (N)
    f2: dict                 # finger → vertical shear (N)
    f3: dict                 # finger → lateral shear (N)
    rho: dict                # finger → ‖(f2,f3)‖/f1


class TactileExtractor:
    """Bound to a finalized plant with the box present."""

    def __init__(self, plant, box_body):
        self.plant = plant
        self.box_body = box_body
        self.box_geoms = set(plant.GetCollisionGeometriesForBody(box_body))
        self.tip_geoms = {
            f: set(plant.GetCollisionGeometriesForBody(
                plant.GetBodyByName(TIP_BODY[f])))
            for f in FINGERS
        }

    def read(self, contact_results) -> dict:
        """→ {finger: TipReading} for contacts between that tip and the box."""
        acc = {f: [np.zeros(3), np.zeros(3), 0.0]  # force, weighted centroid, area+|F|
               for f in FINGERS}
        weights = {f: 0.0 for f in FINGERS}
        areas = {f: 0.0 for f in FINGERS}

        for i in range(contact_results.num_hydroelastic_contacts()):
            info = contact_results.hydroelastic_contact_info(i)
            surf = info.contact_surface()
            id_m, id_n = surf.id_M(), surf.id_N()
            finger, sign = self._classify(id_m, id_n)
            if finger is None:
                continue
            # F_Ac_W acts on body A (owner of geometry M) at the centroid.
            f_on_box = sign * info.F_Ac_W().translational()
            c = surf.centroid()
            w = float(np.linalg.norm(f_on_box)) + 1e-12
            acc[finger][0] += f_on_box
            acc[finger][1] += w * c
            weights[finger] += w
            areas[finger] += float(surf.total_area()) if hasattr(surf, "total_area") else 0.0

        # Point-pair fallback contacts (should be rare for tip↔box).
        for i in range(contact_results.num_point_pair_contacts()):
            info = contact_results.point_pair_contact_info(i)
            ba, bb = int(info.bodyA_index()), int(info.bodyB_index())
            box_idx = int(self.box_body.index())
            tip_idx = {f: int(self.plant.GetBodyByName(TIP_BODY[f]).index())
                       for f in FINGERS}
            for f in FINGERS:
                if {ba, bb} == {box_idx, tip_idx[f]}:
                    # contact_force() is the force on body B.
                    f_on_b = info.contact_force()
                    f_on_box = f_on_b if bb == box_idx else -f_on_b
                    w = float(np.linalg.norm(f_on_box)) + 1e-12
                    acc[f][0] += f_on_box
                    acc[f][1] += w * np.asarray(info.contact_point())
                    weights[f] += w

        out = {}
        for f in FINGERS:
            if weights[f] > 1e-9:
                out[f] = TipReading(True, acc[f][0], acc[f][1] / weights[f], areas[f])
            else:
                out[f] = TipReading(False, np.zeros(3), np.zeros(3), 0.0)
        return out

    def _classify(self, id_m, id_n):
        """→ (finger, sign) where sign maps F_Ac_W to force-on-box."""
        for f in FINGERS:
            if id_m in self.tip_geoms[f] and id_n in self.box_geoms:
                return f, -1.0   # A = tip → force on box is the reaction
            if id_m in self.box_geoms and id_n in self.tip_geoms[f]:
                return f, +1.0   # A = box → F_Ac_W is already on the box
        return None, 0.0


def grasp_frame(readings: dict) -> GraspFrame:
    if not (readings["L"].in_contact and readings["R"].in_contact):
        return GraspFrame(False, np.zeros(3), np.zeros(3), np.zeros(3))
    d = readings["R"].centroid_W - readings["L"].centroid_W
    n = np.linalg.norm(d)
    if n < 1e-9:
        return GraspFrame(False, np.zeros(3), np.zeros(3), np.zeros(3))
    d = d / n
    up = np.array([0.0, 0.0, 1.0])
    e2 = up - np.dot(up, d) * d
    n2 = np.linalg.norm(e2)
    if n2 < 1e-6:  # grasp axis vertical — degenerate for this study
        e2 = np.array([0.0, 1.0, 0.0]) - d * d[1]
        e2 /= np.linalg.norm(e2)
    else:
        e2 /= n2
    e3 = np.cross(d, e2)
    return GraspFrame(True, d, e2, e3)


def decompose(readings: dict, frame: GraspFrame | None = None) -> Decomposition:
    frame = frame or grasp_frame(readings)
    if not frame.valid:
        z = {f: 0.0 for f in FINGERS}
        return Decomposition(False, dict(z), dict(z), dict(z), dict(z))
    e1 = {"L": frame.d, "R": -frame.d}
    f1, f2, f3, rho = {}, {}, {}, {}
    for f in FINGERS:
        F = readings[f].force_W
        f1[f] = float(np.dot(F, e1[f]))
        f2[f] = float(np.dot(F, frame.e2))
        f3[f] = float(np.dot(F, frame.e3))
        rho[f] = float(np.hypot(f2[f], f3[f]) / max(f1[f], 1e-6))
    return Decomposition(True, f1, f2, f3, rho)
