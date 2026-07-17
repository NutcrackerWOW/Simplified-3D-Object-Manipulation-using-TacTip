#!/usr/bin/env python3
"""Creep sensitivity sweep: is the 5-11 s rolling-creep failure an artifact
of the contact model's dissipative/regularization parameters?

Reruns the phaseB_seeds B_static condition (50 g, reference posture, 0.49 N,
margin 1.3, motionless 14.3 s hold) while varying, one at a time:

  * Hunt-Crossley dissipation (0.5 / 1.5 baseline / 3.0 s/m) — the parameter
    that sets a viscous creep timescale in the compliant contact model.
  * The discrete contact approximation and its friction regularization:
    - kSap (paper baseline): friction regularization sigma is hard-coded in
      Drake (1e-3, dimensionless); its residual creep velocity scales with
      the time step, so the existing 2x/4x time-step control already bounds
      it. Rerun here as the in-sweep baseline.
    - kLagged / kSimilar: regularized friction is parameterized directly by
      the plant stiction tolerance v_s; swept 1e-3 / 1e-4 (default) / 1e-5.
      If the creep were regularization slip, failure time would scale ~1/v_s.

If the failure-time distribution stays in the same band across all of these,
the creep is a property of the physical model, not the solver.

  python scripts/creep_sensitivity.py --workers 10
"""

import argparse
import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import BoxSpec, ModelParams, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

FORCE = 0.49       # N — identical to the phaseB_seeds margin-1.3 command
HOLD = 14.3        # s — identical hold window

# label → ModelParams overrides
#
# Batch-1 finding (kept in the merged JSON): hunt_crossley_dissipation is
# inert under kSap — dissipation_0.5/3.0 reproduced base_sap bit-identically,
# because kSap dissipates via the linear relaxation_time model. The
# dissipation axes below therefore vary relaxation_time under kSap and
# hunt_crossley_dissipation under kLagged (where it is live), the latter at
# the tightest regularization.
CONDITIONS = {
    "base_sap":        {},
    "dissipation_0.5": dict(dissipation=0.5),
    "dissipation_3.0": dict(dissipation=3.0),
    "lagged_vs1e-3":   dict(contact_approximation="lagged",
                            stiction_tolerance=1e-3),
    "lagged_vs1e-4":   dict(contact_approximation="lagged",
                            stiction_tolerance=1e-4),
    "lagged_vs1e-5":   dict(contact_approximation="lagged",
                            stiction_tolerance=1e-5),
    "similar_vs1e-4":  dict(contact_approximation="similar",
                            stiction_tolerance=1e-4),
    # batch 2
    "sap_tau0.01":     dict(relaxation_time=0.01),
    "sap_tau0.1":      dict(relaxation_time=0.1),   # = Drake default; control
    "sap_tau0.5":      dict(relaxation_time=0.5),
    "lagged_d0.5_vs1e-5": dict(contact_approximation="lagged",
                               stiction_tolerance=1e-5, dissipation=0.5),
    "lagged_d3.0_vs1e-5": dict(contact_approximation="lagged",
                               stiction_tolerance=1e-5, dissipation=3.0),
}


def fail_time(res) -> float:
    if not np.isnan(res.slip_time):
        return res.slip_time
    s = res.series
    over = np.nonzero(np.asarray(s["drift"]) > res.spec.drift_threshold)[0]
    if len(over):
        return float(np.asarray(s["t"])[over[0]])
    return np.nan


def run_one(job):
    label, seed = job
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(0.0), deg(0.0), 0.024)
    spec = TrialSpec(posture=posture, box=BoxSpec(mass=0.05),
                     force_setpoint=FORCE, seed=seed,
                     tag=f"creepsens_{label}")
    mp = ModelParams(time_step=spec.time_step, **CONDITIONS[label])
    res = run_trial(spec, keep_series=True, hold_time=HOLD, mp=mp)
    ft = fail_time(res)
    out = dict(condition=label, seed=seed, outcome=res.outcome,
               held=bool(res.held),
               fail_time_s=None if np.isnan(ft) else round(ft, 3),
               drift_mm=round(1e3 * res.drift_final, 3),
               rot_deg=round(float(np.rad2deg(res.rot_drift_final)), 2))
    print(f"{label:>16s} seed={seed}: {res.outcome} "
          f"t_fail={out['fail_time_s']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    out_path = os.path.join(RESULTS, "creep_sensitivity.json")
    done_rows = []
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done_rows = json.load(fh).get("runs", [])
    done = {(r["condition"], r["seed"]) for r in done_rows}

    jobs = [(label, s) for label in CONDITIONS for s in range(args.seeds)
            if (label, s) not in done]
    print(f"{len(jobs)} new runs ({len(done)} cached), "
          f"force {FORCE} N, hold {HOLD} s", flush=True)
    rows = list(done_rows)
    if jobs:
        ctx = mp_.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
            rows += list(pool.imap_unordered(run_one, jobs))

    with open(os.path.join(RESULTS, "phaseB_seeds.json")) as fh:
        base = json.load(fh)["summary"]["B_static/fixed"]["fail_times_s"]

    summary = {"phaseB_2ms_10seed_baseline": {
        "fail_times_s": base, "mean": round(float(np.mean(base)), 2),
        "std": round(float(np.std(base)), 2)}}
    for label in CONDITIONS:
        rr = [r for r in rows if r["condition"] == label]
        fts = sorted(r["fail_time_s"] for r in rr
                     if r["fail_time_s"] is not None)
        summary[label] = {
            "overrides": CONDITIONS[label],
            "n": len(rr), "n_failed": len(fts), "fail_times_s": fts,
            "mean": round(float(np.mean(fts)), 2) if fts else None,
            "std": round(float(np.std(fts)), 2) if fts else None}
        print(f"{label}: {len(fts)}/{len(rr)} failed, times {fts}",
              flush=True)

    with open(out_path, "w") as fh:
        json.dump({"force_N": FORCE, "hold_s": HOLD,
                   "summary": summary, "runs": rows}, fh, indent=2)
    print("→ results/creep_sensitivity.json", flush=True)


if __name__ == "__main__":
    main()
