#!/usr/bin/env python3
"""Run a single pinch trial, optionally in MeshCat.

Examples
--------
  python scripts/run_trial.py --force 0.6                     # headless
  python scripts/run_trial.py --force 0.6 --meshcat           # visual replay
  python scripts/run_trial.py --wave 10 --pip 4 --force 0.4
  python scripts/run_trial.py --force 0.35 --slip-study       # 1 ms + full-rate log
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import BoxSpec, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", type=float, default=0.0, help="wave tilt, deg")
    ap.add_argument("--pip", type=float, default=0.0, help="PIP angle, deg")
    ap.add_argument("--force", type=float, default=0.6, help="squeeze setpoint, N")
    ap.add_argument("--mass", type=float, default=0.05, help="box mass, kg")
    ap.add_argument("--size", type=float, default=0.025, help="box grasp width, m")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--meshcat", action="store_true", help="visualize + record")
    ap.add_argument("--slip-study", action="store_true",
                    help="1 ms timestep, every-step logging")
    ap.add_argument("--save-series", type=str, default="",
                    help="save time series to this .npz")
    args = ap.parse_args()

    kin = get_kinematics()
    box = BoxSpec(size=args.size, mass=args.mass)
    posture, g = kin.solve_mcp_for_gap(deg(args.wave), deg(args.pip),
                                       box.size - 0.001)
    if posture is None:
        sys.exit("infeasible posture: cannot land pads on the box")
    feas = kin.grasp_feasibility(posture, box.size)
    print(f"posture: wave ±{args.wave}°, PIP {args.pip}°, "
          f"MCP* {np.rad2deg(posture.mcp_l):.2f}° | gap {feas['gap']*1000:.1f} mm "
          f"tilt {feas['tilt_deg']:.1f}° feasible={feas['feasible']}")

    spec = TrialSpec(posture=posture, box=box, force_setpoint=args.force,
                     seed=args.seed)
    if args.slip_study:
        spec.time_step = 1e-3
        spec.log_hz = 1000.0

    meshcat = None
    if args.meshcat:
        from pydrake.all import StartMeshcat
        meshcat = StartMeshcat()
        print(f"MeshCat: {meshcat.web_url()}  (recording published at the end)")

    res = run_trial(spec, meshcat=meshcat, keep_series=True)
    print(f"\noutcome: {res.outcome} (held={res.held})")
    print(f"drift {res.drift_final*1000:.2f} mm | rot "
          f"{np.rad2deg(res.rot_drift_final):.1f}° | incipient slip at "
          f"{res.slip_time if not np.isnan(res.slip_time) else '—'} s")
    for f in ("L", "R"):
        print(f"  {f}: f1 {res.f1_mean[f]:+.3f}  f2 {res.f2_mean[f]:+.3f}  "
              f"f3 {res.f3_mean[f]:+.3f}  ρ {res.rho_mean[f]:.3f}")
    if args.save_series:
        np.savez_compressed(args.save_series, **res.series)
        print(f"series → {args.save_series}")
    if meshcat is not None:
        input("MeshCat recording published — press Enter to exit…")


if __name__ == "__main__":
    main()
