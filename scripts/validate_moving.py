#!/usr/bin/env python3
"""M7 / Phase B: stable manipulation validation.

The hand grasps, then sweeps its wave tilt through a sinusoid while holding
the box — carrying the object through posture changes. Three controllers are
compared:

  fixed      constant squeeze chosen at the start posture
  scheduled  squeeze scheduled from the fitted μ_eff(θ) map (margin 1.3)
  reflex     scheduled + online vibration reflex (boost on detected slip)

  python scripts/validate_moving.py --tag sweepA
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.fit import basis
from pinchlab.params import BoxSpec, G, Posture, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial
from pinchlab import plots

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default="sweepA")
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--pip", type=float, default=0.0)
    ap.add_argument("--amp", type=float, default=10.0, help="wave amplitude, deg")
    ap.add_argument("--freq", type=float, default=0.15, help="Hz")
    ap.add_argument("--cycles", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=1.3)
    args = ap.parse_args()

    with open(os.path.join(RESULTS, f"{args.tag}_fit.json")) as fh:
        fit = json.load(fh)
    coef = np.array(list(fit["coefficients"].values()))

    def mu_eff(wave_deg, pip_deg):
        return float(basis(np.deg2rad(np.array([wave_deg])),
                           np.deg2rad(np.array([pip_deg])))[0] @ coef)

    def scheduled_sp(wave_deg):
        return (args.mass * G / 2.0) / mu_eff(wave_deg, args.pip) * args.margin

    kin = get_kinematics()
    posture0, _ = kin.solve_mcp_for_gap(deg(0.0), deg(args.pip), 0.024)
    mcp = posture0.mcp_l   # invariant across wave tilt (rigid rotation)

    def wave_traj(t_hold):
        return args.amp * np.sin(2 * np.pi * args.freq * t_hold)

    def posture_fn(t_hold):
        return Posture.symmetric(deg(wave_traj(t_hold)), mcp, deg(args.pip))

    hold_time = args.cycles / args.freq + 1.0
    sp0 = scheduled_sp(0.0)
    print(f"start setpoint (scheduled, margin {args.margin}) = {sp0:.3f} N; "
          f"hold {hold_time:.1f} s, wave ±{args.amp}° @ {args.freq} Hz")

    reflex_thr = 1e-4
    slip_json = os.path.join(RESULTS, "slip_eval.json")
    if os.path.exists(slip_json):
        with open(slip_json) as fh:
            reflex_thr = json.load(fh)["ideal"]["threshold_from_train"]

    modes = {
        "fixed": dict(setpoint_fn=None, reflex=None),
        "scheduled": dict(setpoint_fn=lambda th: scheduled_sp(wave_traj(th)),
                          reflex=None),
        "reflex": dict(setpoint_fn=lambda th: scheduled_sp(wave_traj(th)),
                       reflex=dict(threshold=reflex_thr, highpass_hz=15.0,
                                   boost=1.5, boost_time=1.0)),
    }

    summary = {}
    for name, kw in modes.items():
        spec = TrialSpec(posture=posture0, box=BoxSpec(mass=args.mass),
                         force_setpoint=sp0, seed=3, tag=f"phaseB_{name}")
        res = run_trial(spec, keep_series=True, posture_ref_fn=posture_fn,
                        hold_time=hold_time, **kw)
        s = res.series
        summary[name] = {
            "outcome": res.outcome,
            "drift_mm": round(1e3 * res.drift_final, 3),
            "rot_deg": round(float(np.rad2deg(res.rot_drift_final)), 2),
            "slip_time": None if np.isnan(res.slip_time) else round(res.slip_time, 3),
            "mean_f1": round(float(np.mean([res.f1_mean["L"], res.f1_mean["R"]])), 3),
        }
        print(f"{name:10s} -> {summary[name]}")
        plots.plot_trial(s, f"Phase B ({name}): wave ±{args.amp}° while holding "
                         f"{args.mass*1000:.0f} g",
                         os.path.join(RESULTS, f"phaseB_{name}.png"),
                         setpoint=None)

    with open(os.path.join(RESULTS, "phaseB_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsummary → results/phaseB_summary.json, figures → results/phaseB_*.png")


if __name__ == "__main__":
    main()
