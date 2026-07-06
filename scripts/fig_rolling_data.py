#!/usr/bin/env python3
"""Collect pose time series of one rolling-failure trial for the paper's
hero figure (F2): 50 g box, reference posture, margin-1.3 force (0.49 N),
seed 0 — fails by rolling creep at ~5.4 s.

  python scripts/fig_rolling_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import BoxSpec, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


def main():
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(0.0), deg(0.0), 0.024)
    spec = TrialSpec(posture=posture, box=BoxSpec(mass=0.05),
                     force_setpoint=0.49, seed=0, tag="fig_rolling")
    res = run_trial(spec, keep_series=True, hold_time=14.3, log_poses=True)
    s = res.series
    print(f"outcome={res.outcome} slip_time={res.slip_time} "
          f"rot_final={np.rad2deg(res.rot_drift_final):.1f} deg")
    np.savez_compressed(
        os.path.join(RESULTS, "fig_rolling_data.npz"),
        t=np.asarray(s["t"]), phase=np.asarray(s["phase"]),
        box_p=np.asarray(s["box_p"]), box_R=np.asarray(s["box_R"]),
        tipL_p=np.asarray(s["tipL_p"]), tipR_p=np.asarray(s["tipR_p"]),
        cL=np.asarray(s["cL"]), cR=np.asarray(s["cR"]),
        drift=np.asarray(s["drift"]), rot_drift=np.asarray(s["rot_drift"]),
        f1_L=np.asarray(s["f1_L"]), f2_L=np.asarray(s["f2_L"]),
        f3_L=np.asarray(s["f3_L"]),
        outcome=res.outcome, box_size=spec.box.size,
        box_height=spec.box.height)
    print("→ results/fig_rolling_data.npz")


if __name__ == "__main__":
    main()
