#!/usr/bin/env python3
"""M7b: Phase B with seed repeats — turn single-run anecdotes into rates
and the creep failure time into a distribution.

  block A  margin 2.0, wave ±10° @ 0.15 Hz: fixed / scheduled / reflex,
           N seeds each → held-in-k/N rates for the design rule.
  block B  margin 1.3, motionless 14.3 s hold, N seeds → failure-time dist.
  block C  margin 1.3, wave ±10° carry, N seeds → failure-time dist.

  python scripts/phaseB_seeds.py --workers 10
  python scripts/phaseB_seeds.py --smoke          # 1 run, ~2 min
"""

import argparse
import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.fit import mu_eff_model
from pinchlab.params import BoxSpec, G, Posture, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


def fail_time(res) -> float:
    """Time the box is lost: kinematic slip if flagged, else drift-threshold
    crossing, else nan (held)."""
    if not np.isnan(res.slip_time):
        return res.slip_time
    s = res.series
    over = np.nonzero(np.asarray(s["drift"]) > res.spec.drift_threshold)[0]
    if len(over):
        return float(np.asarray(s["t"])[over[0]])
    return np.nan


def run_one(job):
    """One Phase B episode from a picklable job dict (spawn-pool worker)."""
    coef = np.array(job["coef"])
    mass, pip, margin = job["mass"], job["pip"], job["margin"]
    amp, freq = job["amp"], job["freq"]

    def mu_eff(wave_deg):
        return float(mu_eff_model(coef, np.array([wave_deg]),
                                  np.array([pip]), mass)[0])

    def scheduled_sp(wave_deg):
        return (mass * G / 2.0) / mu_eff(wave_deg) * margin

    kin = get_kinematics()
    posture0, _ = kin.solve_mcp_for_gap(deg(0.0), deg(pip), 0.024)
    mcp = posture0.mcp_l

    def wave_traj(t_hold):
        return amp * np.sin(2 * np.pi * freq * t_hold)

    posture_fn = (None if amp == 0.0 else
                  lambda th: Posture.symmetric(deg(wave_traj(th)), mcp,
                                               deg(pip)))
    mode = job["mode"]
    kw = dict(setpoint_fn=None, reflex=None)
    if mode in ("scheduled", "reflex"):
        kw["setpoint_fn"] = lambda th: scheduled_sp(wave_traj(th))
    if mode == "reflex":
        kw["reflex"] = dict(threshold=job["reflex_thr"], highpass_hz=15.0,
                            boost=1.5, boost_time=1.0)

    spec = TrialSpec(posture=posture0, box=BoxSpec(mass=mass),
                     force_setpoint=scheduled_sp(0.0), seed=job["seed"],
                     tag=f"phaseBseed_{job['block']}_{mode}")
    res = run_trial(spec, keep_series=True, posture_ref_fn=posture_fn,
                    hold_time=job["hold_time"], **kw)
    ft = fail_time(res)
    out = dict(block=job["block"], mode=mode, margin=margin, amp=amp,
               seed=job["seed"], outcome=res.outcome, held=bool(res.held),
               drift_mm=round(1e3 * res.drift_final, 3),
               rot_deg=round(float(np.rad2deg(res.rot_drift_final)), 2),
               fail_time_s=None if np.isnan(ft) else round(ft, 3),
               mean_f1=round(float(np.mean([res.f1_mean["L"],
                                            res.f1_mean["R"]])), 3))
    print(f"{job['block']:>8s} {mode:9s} m={margin} amp={amp:4.1f} "
          f"seed={job['seed']}: {res.outcome} "
          f"t_fail={out['fail_time_s']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default="sweepB")
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--pip", type=float, default=0.0)
    ap.add_argument("--amp", type=float, default=10.0)
    ap.add_argument("--freq", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(RESULTS, f"{args.tag}_fit.json")) as fh:
        coef = list(json.load(fh)["coefficients"].values())
    reflex_thr = 1e-4
    slip_json = os.path.join(RESULTS, "slip_eval.json")
    if os.path.exists(slip_json):
        with open(slip_json) as fh:
            reflex_thr = json.load(fh)["ideal"]["threshold_from_train"]

    hold_time = 2.0 / args.freq + 1.0   # 2 cycles + settle ≈ 14.3 s
    base = dict(coef=coef, mass=args.mass, pip=args.pip, freq=args.freq,
                reflex_thr=reflex_thr, hold_time=hold_time)
    jobs = []
    for seed in range(args.seeds):
        for mode in ("fixed", "scheduled", "reflex"):
            jobs.append(dict(base, block="A_m2.0", mode=mode, margin=2.0,
                             amp=args.amp, seed=seed))
        jobs.append(dict(base, block="B_static", mode="fixed", margin=1.3,
                         amp=0.0, seed=seed))
        jobs.append(dict(base, block="C_moving", mode="scheduled", margin=1.3,
                         amp=args.amp, seed=seed))
    if args.smoke:
        jobs = jobs[:1]
    print(f"{len(jobs)} runs, hold {hold_time:.1f} s, "
          f"{args.workers} workers", flush=True)

    ctx = mp_.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run_one, jobs))

    summary = {}
    for block in sorted({r["block"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows if r["block"] == block}):
            rr = [r for r in rows if r["block"] == block and r["mode"] == mode]
            fts = [r["fail_time_s"] for r in rr if r["fail_time_s"] is not None]
            summary[f"{block}/{mode}"] = {
                "n": len(rr), "held": sum(r["held"] for r in rr),
                "fail_times_s": sorted(fts),
                "fail_time_mean": round(float(np.mean(fts)), 2) if fts else None,
                "fail_time_std": round(float(np.std(fts)), 2) if fts else None,
                "drift_mm_max": max(r["drift_mm"] for r in rr),
            }
            print(f"{block}/{mode}: held {summary[f'{block}/{mode}']['held']}"
                  f"/{len(rr)}, fail times {sorted(fts)}", flush=True)

    out = {"config": {k: v for k, v in vars(args).items()},
           "hold_time_s": hold_time, "summary": summary, "runs": rows}
    with open(os.path.join(RESULTS, "phaseB_seeds.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("→ results/phaseB_seeds.json", flush=True)


if __name__ == "__main__":
    main()
