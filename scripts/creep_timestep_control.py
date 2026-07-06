#!/usr/bin/env python3
"""Numerical control for the rolling-creep finding: is the ~5-11 s failure
of a margin-1.3 motionless hold physics or integrator drift?

Rerun the phaseB_seeds B_static condition (50 g, reference posture, 0.49 N —
the same commanded force) at finer solver time steps and compare the
failure-time distribution against the 2 ms baseline (phaseB_seeds.json).
If creep were an integration artifact, halving/quartering the step should
shift or remove it.

  python scripts/creep_timestep_control.py --workers 10
"""

import argparse
import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import BoxSpec, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

FORCE = 0.49       # N — identical to the phaseB_seeds margin-1.3 command
HOLD = 14.3        # s — identical hold window
TIME_STEPS = (1e-3, 5e-4)


def fail_time(res) -> float:
    if not np.isnan(res.slip_time):
        return res.slip_time
    s = res.series
    over = np.nonzero(np.asarray(s["drift"]) > res.spec.drift_threshold)[0]
    if len(over):
        return float(np.asarray(s["t"])[over[0]])
    return np.nan


def run_one(job):
    dt, seed = job
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(0.0), deg(0.0), 0.024)
    spec = TrialSpec(posture=posture, box=BoxSpec(mass=0.05),
                     force_setpoint=FORCE, seed=seed, time_step=dt,
                     tag=f"creepctl_dt{dt:g}")
    res = run_trial(spec, keep_series=True, hold_time=HOLD)
    ft = fail_time(res)
    out = dict(time_step=dt, seed=seed, outcome=res.outcome,
               held=bool(res.held),
               fail_time_s=None if np.isnan(ft) else round(ft, 3),
               drift_mm=round(1e3 * res.drift_final, 3))
    print(f"dt={dt:g} seed={seed}: {res.outcome} t_fail={out['fail_time_s']}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    jobs = [(dt, s) for dt in TIME_STEPS for s in range(args.seeds)]
    print(f"{len(jobs)} runs (dt {TIME_STEPS} x {args.seeds} seeds), "
          f"force {FORCE} N, hold {HOLD} s", flush=True)
    ctx = mp_.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run_one, jobs))

    # 2 ms baseline from the seeded Phase B study
    with open(os.path.join(RESULTS, "phaseB_seeds.json")) as fh:
        base = json.load(fh)["summary"]["B_static/fixed"]["fail_times_s"]

    summary = {"2ms_baseline": base}
    for dt in TIME_STEPS:
        fts = sorted(r["fail_time_s"] for r in rows if r["time_step"] == dt
                     and r["fail_time_s"] is not None)
        n = sum(1 for r in rows if r["time_step"] == dt)
        summary[f"{dt*1e3:g}ms"] = {
            "n": n, "n_failed": len(fts), "fail_times_s": fts,
            "mean": round(float(np.mean(fts)), 2) if fts else None,
            "std": round(float(np.std(fts)), 2) if fts else None}
        print(f"dt={dt*1e3:g} ms: {len(fts)}/{n} failed, times {fts}",
              flush=True)
    print(f"2 ms baseline: mean {np.mean(base):.2f} ± {np.std(base):.2f} s")

    with open(os.path.join(RESULTS, "creep_timestep_control.json"), "w") as fh:
        json.dump({"force_N": FORCE, "hold_s": HOLD,
                   "summary": summary, "runs": rows}, fh, indent=2)
    print("→ results/creep_timestep_control.json", flush=True)


if __name__ == "__main__":
    main()
