#!/usr/bin/env python3
"""M6b: expanded slip-detector study for statistical rigor.

The original eval_slip.py scores 3 TP / 3 TN — with zero errors in n trials
the rule-of-three only bounds the error rate at 3/n, so 6 test trials cannot
support a "<10% error" claim. This study generates 45 induced-slip + 45
clean-hold episodes across a posture x mass grid (friction regime only),
calibrates the threshold on a 1/3 training split, and scores the remaining
30 + 30 test episodes at both sensor rates.

  python scripts/eval_slip_big.py --workers 10
  python scripts/eval_slip_big.py --smoke          # 1 condition, ~2 min
"""

import argparse
import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import BoxSpec, TrialSpec, deg
from pinchlab.slipdetect import DetectorConfig, score_trials
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
CACHE = os.path.join(RESULTS, "slip_episodes_big")
KEEP_KEYS = ("t", "f2_L", "f2_R", "f3_L", "f3_R", "drift")

# 15 friction-regime conditions: 5 waves x 3 PIPs, mass cycling 30/50/60 g.
WAVES = (-15.0, -7.5, 0.0, 7.5, 15.0)
PIPS = (0.0, 4.0, 8.0)
MASSES = (0.03, 0.05, 0.06)


def truth_time(res) -> float:
    if not np.isnan(res.slip_time):
        return res.slip_time
    s = res.series
    over = np.nonzero(np.asarray(s["drift"]) > res.spec.drift_threshold)[0]
    if len(over):
        return float(np.asarray(s["t"])[over[0]])
    return np.nan


def run_episode(job):
    """Simulate (or load from cache) one episode; returns metadata + npz path."""
    key = (f"ep_{job['kind']}_w{job['wave']:+05.1f}_p{job['pip']:+05.1f}"
           f"_m{int(round(job['mass']*1000)):03d}_s{job['seed']}")
    path = os.path.join(CACHE, key + ".npz")
    if not os.path.exists(path):
        kin = get_kinematics()
        posture, _ = kin.solve_mcp_for_gap(
            deg(job["wave"]), deg(job["pip"]), 0.024)
        ramp = job["ramp"] if job["kind"] == "pos" else 0.0
        spec = TrialSpec(posture=posture, box=BoxSpec(mass=job["mass"]),
                         force_setpoint=job["f0"], setpoint_ramp=ramp,
                         time_step=1e-3, log_hz=1000.0, seed=job["seed"],
                         tag=f"slipbig_{job['kind']}")
        res = run_trial(spec, keep_series=True)
        truth = truth_time(res) if job["kind"] == "pos" else (
            np.nan if res.held else truth_time(res))
        series = {k: np.asarray(res.series[k]) for k in KEEP_KEYS}
        np.savez_compressed(path, truth=truth, outcome=res.outcome, **series)
    d = np.load(path, allow_pickle=False)
    out = dict(job)
    out.update(path=path, truth=float(d["truth"]), outcome=str(d["outcome"]))
    print(f"w={job['wave']:+5.1f} p={job['pip']:+4.1f} "
          f"m={job['mass']*1000:.0f}g s={job['seed']} {job['kind']}: "
          f"{out['outcome']} truth={out['truth']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f0", type=float, default=0.8)
    ap.add_argument("--ramp", type=float, default=0.15, help="N/s")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tactip-hz", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    conds = [(w, p, MASSES[i % len(MASSES)])
             for i, (w, p) in enumerate((w, p) for w in WAVES for p in PIPS)]
    if args.smoke:
        conds, args.seeds = conds[:1], 1
    jobs = [dict(wave=w, pip=p, mass=m, seed=s, kind=k,
                 f0=args.f0, ramp=args.ramp)
            for (w, p, m) in conds
            for s in range(args.seeds)
            for k in ("pos", "neg")]
    print(f"{len(jobs)} episodes ({len(conds)} conditions x {args.seeds} "
          f"seeds x pos/neg), {args.workers} workers", flush=True)

    ctx = mp_.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        episodes = list(pool.imap_unordered(run_episode, jobs))

    # Sanity: every positive needs a kinematic truth time.
    bad = [e for e in episodes if e["kind"] == "pos" and np.isnan(e["truth"])]
    if bad:
        bad_ids = [(e["wave"], e["pip"], e["mass"], e["seed"]) for e in bad]
        print(f"WARNING: {len(bad)} positive episodes never slipped "
              f"(excluded): {bad_ids}", flush=True)
        episodes = [e for e in episodes if e not in bad]

    def load(e):
        d = np.load(e["path"], allow_pickle=False)
        return {k: d[k] for k in KEEP_KEYS}, e["truth"]

    # Deterministic stratified split: every 3rd episode (per class) trains.
    order = sorted(episodes, key=lambda e: (e["wave"], e["pip"], e["mass"],
                                            e["seed"]))
    pos = [load(e) for e in order if e["kind"] == "pos"]
    neg = [load(e) for e in order if e["kind"] == "neg"]
    train = pos[0::3] + neg[0::3]
    test = [x for i, x in enumerate(pos) if i % 3] + \
           [x for i, x in enumerate(neg) if i % 3]
    n_pos_test = sum(1 for i in range(len(pos)) if i % 3)

    cfg = DetectorConfig(tactip_hz=args.tactip_hz)
    report = {"n_train": len(train), "n_test": len(test),
              "n_pos_test": n_pos_test, "n_neg_test": len(test) - n_pos_test,
              "conditions": len(conds), "seeds": args.seeds}
    for rate in ("ideal", "tactip"):
        cal = score_trials(train, cfg, rate=rate, threshold=None)
        scored = score_trials(test, cfg, rate=rate, threshold=cal["threshold"])
        scored["threshold_from_train"] = cal["threshold"]
        report[rate] = {k: v for k, v in scored.items() if k != "det_times"}
        # Rule-of-three 95% upper bounds when zero errors; else Wilson-ish
        # point estimate + 3 (crude but honest for small counts).
        report[rate]["miss_rate_ci95_upper"] = round(
            (scored["fn"] + 3.0) / max(n_pos_test, 1), 4)
        report[rate]["fp_rate_ci95_upper"] = round(
            (scored["fp"] + 3.0) / max(len(test) - n_pos_test, 1), 4)
        print(f"\n[{rate}] thr={cal['threshold']:.3g}  "
              f"tp={scored['tp']} fp={scored['fp']} fn={scored['fn']} "
              f"tn={scored['tn']}  latency mean={scored['latency_mean']:.3f}s "
              f"max={scored['latency_max']:.3f}s", flush=True)

    out = os.path.join(RESULTS, "slip_eval_big.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
