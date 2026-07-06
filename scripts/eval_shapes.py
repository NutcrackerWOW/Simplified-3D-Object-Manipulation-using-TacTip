#!/usr/bin/env python3
"""M8: shape study. Measure the minimal-stable-force boundary f* for other
object shapes at the reference posture and compare against the box-fitted
equation. Shapes (all 25 mm grasp width, 50 g):

  prism     25x50x12 mm block (2x depth: longer lever arm about the grasp axis)
  cylinder  d=25 mm vertical axis: pads grip the curved side (line contact)
  disc      d=25 mm, 12 mm thick, axis along the pinch axis: flat round faces
  sphere    d=25 mm: point-like contact, free to roll about every axis

  python scripts/eval_shapes.py --shape sphere        # one shape (parallelize
  python scripts/eval_shapes.py --shape all           # or sequential)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import G, BoxSpec, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
SHAPES = ("prism", "cylinder", "disc", "disc_edge", "sphere")


def stable(posture, box, force, seeds=(0, 1), tag="probe",
           mp=None) -> tuple[bool, list]:
    outs = []
    for s in seeds:
        spec = TrialSpec(posture=posture, box=box,
                         force_setpoint=force, seed=s, tag=tag)
        res = run_trial(spec, mp=mp)
        outs.append({"force": force, "seed": s, "outcome": res.outcome,
                     "drift_mm": round(1e3 * res.drift_final, 2),
                     "rot_deg": round(float(np.rad2deg(res.rot_drift_final)), 1)})
        print(f"  {tag} f={force:.3f} seed={s}: {res.outcome}", flush=True)
        if not res.held:
            return False, outs
    return True, outs


def boundary(posture, box, f_lo=0.05, f_hi=2.5, n_bisect=5, tag="probe",
             mp=None, seeds=(0, 1)):
    """Mass-scaled ascending anchor ladder, then log bisection below the
    first stable rung (same scheme as the sweep)."""
    trials = []
    w = box.mass * G / 2.0
    ladder = sorted({float(np.clip(3.0 * w * k, 0.3, f_hi)) for k in (1., 2., 4.)})
    anchor = None
    for rung in ladder:
        ok, outs = stable(posture, box, rung, tag=tag, mp=mp, seeds=seeds)
        trials += outs
        if ok:
            anchor = rung
            break
    if anchor is None:
        return np.nan, trials
    lo, hi = np.log(f_lo), np.log(anchor)
    for _ in range(n_bisect):
        mid = float(np.exp(0.5 * (lo + hi)))
        ok, outs = stable(posture, box, mid, tag=tag, mp=mp, seeds=seeds)
        trials += outs
        if ok:
            hi = np.log(mid)
        else:
            lo = np.log(mid)
    return float(np.exp(hi)), trials


def run_shape(job):
    """One shape row (picklable job dict — usable from a spawn Pool)."""
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(
        deg(job["wave"]), deg(job["pip"]), 0.024)
    shape = job["shape"]
    f_star, trials = boundary(
        posture, BoxSpec(mass=job["mass"], shape=shape),
        n_bisect=job["n_bisect"], tag=f"shape_{shape}",
        seeds=tuple(range(job["seeds"])))
    mu = (job["mass"] * G / 2.0) / f_star if np.isfinite(f_star) else np.nan
    out = {"shape": shape, "mass": job["mass"],
           "wave_deg": job["wave"], "pip_deg": job["pip"],
           "n_seeds": job["seeds"], "n_bisect": job["n_bisect"],
           "f_star": None if np.isnan(f_star) else round(f_star, 4),
           "mu_eff": None if np.isnan(mu) else round(mu, 4),
           "trials": trials}
    path = os.path.join(RESULTS, f"shape_{shape}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{shape}: f* = {f_star if np.isfinite(f_star) else 'none'} N  "
          f"mu_eff = {mu:.3f}" if np.isfinite(mu) else
          f"{shape}: no stable force found", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", type=str, default="all",
                    choices=SHAPES + ("all",))
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--wave", type=float, default=0.0)
    ap.add_argument("--pip", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-bisect", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelize across shapes (spawn pool)")
    args = ap.parse_args()

    shapes = SHAPES if args.shape == "all" else (args.shape,)
    jobs = [dict(shape=s, mass=args.mass, wave=args.wave, pip=args.pip,
                 seeds=args.seeds, n_bisect=args.n_bisect) for s in shapes]
    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp_
        ctx = mp_.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
            for _ in pool.imap_unordered(run_shape, jobs):
                pass
    else:
        for job in jobs:
            run_shape(job)


if __name__ == "__main__":
    main()
