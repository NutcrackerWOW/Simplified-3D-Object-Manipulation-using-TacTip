#!/usr/bin/env python3
"""Closed-loop rolling regulation: does actively regulating the rolling DOF
enlarge the stable window, i.e. how much of the 25-35 % rolling discount
does control buy back?

The regulator (pinchlab.control, `roll_reg`) is a simplified tactile
analogue of rolling-orientation pinch control: the object's pitch about
the lateral axis is observed as the antisymmetric vertical migration of
the two contact centroids (idealized-TacTip signal only) and both domes
are rolled to follow it via a rate-limited differential PIP trim.
Gains from the margin-1.3 tuning study: kp=1000 rad/m, limit 0.25 rad,
rate 0.7 rad/s, 5 Hz error filter.

  block A  creep arrest: margin-1.3 motionless hold (0.49 N, 14.3 s),
           10 seeds, regulation ON  (open-loop baseline: phaseB 0/10 held)
  block B  boundary: f* bisection (5 seeds, 7 steps) at 30/50/60 g,
           regulation ON and OFF, reference posture.

  python scripts/eval_closedloop.py --workers 12
"""

import argparse
import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import G, BoxSpec, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

RR = dict(kp=1000.0, limit=0.25, rate=0.7, filter_hz=5.0)
FORCE_M13 = 0.49
HOLD_M13 = 14.3


def fail_time(res):
    if not np.isnan(res.slip_time):
        return res.slip_time
    s = res.series
    over = np.nonzero(np.asarray(s["drift"]) > res.spec.drift_threshold)[0]
    return float(np.asarray(s["t"])[over[0]]) if len(over) else np.nan


def stable(posture, box, force, seeds, tag, roll_reg):
    outs = []
    for s in seeds:
        spec = TrialSpec(posture=posture, box=box, force_setpoint=force,
                         seed=s, tag=tag)
        res = run_trial(spec, roll_reg=roll_reg)
        outs.append({"force": round(force, 4), "seed": s,
                     "outcome": res.outcome})
        if not res.held:
            return False, outs
    return True, outs


def boundary(posture, box, seeds, n_bisect, tag, roll_reg,
             f_lo=0.05, f_hi=2.5):
    trials = []
    w = box.mass * G / 2.0
    ladder = sorted({float(np.clip(3.0 * w * k, 0.3, f_hi))
                     for k in (1., 2., 4.)})
    anchor = None
    for rung in ladder:
        ok, outs = stable(posture, box, rung, seeds, tag, roll_reg)
        trials += outs
        if ok:
            anchor = rung
            break
    if anchor is None:
        return np.nan, trials
    lo, hi = np.log(f_lo), np.log(anchor)
    for _ in range(n_bisect):
        mid = float(np.exp(0.5 * (lo + hi)))
        ok, outs = stable(posture, box, mid, seeds, tag, roll_reg)
        trials += outs
        if ok:
            hi = np.log(mid)
        else:
            lo = np.log(mid)
    return float(np.exp(hi)), trials


def run_job(job):
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(0.0), deg(0.0), 0.024)
    if job["kind"] == "arrest":
        spec = TrialSpec(posture=posture, box=BoxSpec(mass=0.05),
                         force_setpoint=FORCE_M13, seed=job["seed"],
                         tag="cl_arrest")
        res = run_trial(spec, keep_series=True, hold_time=HOLD_M13,
                        roll_reg=RR)
        ft = fail_time(res)
        out = dict(kind="arrest", seed=job["seed"], outcome=res.outcome,
                   held=bool(res.held),
                   fail_time_s=None if np.isnan(ft) else round(ft, 3),
                   rot_deg=round(float(np.rad2deg(res.rot_drift_final)), 2),
                   drift_mm=round(1e3 * res.drift_final, 3))
        print(f"arrest seed={job['seed']}: {res.outcome} "
              f"rot={out['rot_deg']}deg", flush=True)
        return out
    # boundary
    mass, loop = job["mass"], job["loop"]
    rr = RR if loop == "closed" else None
    tag = f"cl_bnd_{loop}_{int(mass*1000)}g"
    f_star, trials = boundary(posture, BoxSpec(mass=mass),
                              seeds=tuple(range(job["seeds"])),
                              n_bisect=job["n_bisect"], tag=tag, roll_reg=rr)
    mu = (mass * G / 2.0) / f_star if np.isfinite(f_star) else np.nan
    out = dict(kind="boundary", mass=mass, loop=loop,
               n_seeds=job["seeds"], n_bisect=job["n_bisect"],
               f_star=None if np.isnan(f_star) else round(f_star, 4),
               mu_eff=None if np.isnan(mu) else round(mu, 4),
               trials=trials)
    print(f"boundary {loop} {int(mass*1000)}g: f*="
          f"{f_star:.3f} mu_eff={mu:.3f}" if np.isfinite(f_star) else
          f"boundary {loop} {int(mass*1000)}g: none", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds-arrest", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-bisect", type=int, default=7)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    jobs = [dict(kind="boundary", mass=m, loop=lp, seeds=args.seeds,
                 n_bisect=args.n_bisect)
            for m in (0.03, 0.05, 0.06) for lp in ("closed",)]
    jobs += [dict(kind="boundary", mass=m, loop="open", seeds=args.seeds,
                  n_bisect=args.n_bisect) for m in (0.03, 0.06)]
    jobs += [dict(kind="arrest", seed=s) for s in range(args.seeds_arrest)]

    ctx = mp_.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run_job, jobs))

    arrest = [r for r in rows if r["kind"] == "arrest"]
    bnd = [r for r in rows if r["kind"] == "boundary"]
    summary = {
        "roll_reg": RR,
        "arrest": {"n": len(arrest),
                   "held": sum(r["held"] for r in arrest),
                   "rot_deg_max": max(r["rot_deg"] for r in arrest),
                   "fail_times_s": sorted(r["fail_time_s"] for r in arrest
                                          if r["fail_time_s"] is not None)},
        "boundaries": {f"{r['loop']}_{int(r['mass']*1000)}g":
                       {"f_star": r["f_star"], "mu_eff": r["mu_eff"]}
                       for r in bnd},
    }
    with open(os.path.join(RESULTS, "closedloop.json"), "w") as fh:
        json.dump({"summary": summary, "runs": rows}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("→ results/closedloop.json", flush=True)


if __name__ == "__main__":
    main()
